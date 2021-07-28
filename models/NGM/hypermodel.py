import torch.nn.functional as F

from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.hungarian import hungarian
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from models.NGM.gnn import HyperGNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
from models.GMN.affinity_layer import GaussianAffinity, InnerpAffinity

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.geo_affinity_layer = GaussianAffinity(1, cfg.NGM.GAUSSIAN_SIGMA)
        self.feat_affinity_layer = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        self.feat_affinity_layer3 = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        self.tau = cfg.NGM.SK_TAU
        self.bi_stochastic = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.l2norm = nn.LocalResponseNorm(cfg.NGM.FEATURE_CHANNEL * 2, alpha=cfg.NGM.FEATURE_CHANNEL * 2, beta=0.5, k=0)

        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = HyperGNNLayer(
                    1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                    sk_channel=cfg.NGM.SK_EMB, sk_tau=self.tau
                )
            else:
                gnn_layer = HyperGNNLayer(
                    cfg.NGM.GNN_FEAT[i - 1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i - 1],
                    cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                    sk_channel=cfg.NGM.SK_EMB, sk_tau=self.tau
                )
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0), 1)

        self.weight2 = cfg.NGM.WEIGHT2
        self.weight3 = cfg.NGM.WEIGHT3
        self.rescale = cfg.PROBLEM.RESCALE

    def forward(self, data_dict):
        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']

            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, self.rescale)
        elif 'features' in data_dict:
            # synthetic data
            src, tgt = data_dict['features']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('unknown type string {}'.format(type))

        X = reshape_edge_feature(F_src, G_src, H_src)
        Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)
        dx = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
        dy = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]

        # affinity layer for 2-order affinity matrix
        if cfg.NGM.EDGE_FEATURE == 'cat':
            Ke, Kp = self.feat_affinity_layer(X, Y, U_src, U_tgt)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
             Ke, Kp = self.geo_affinity_layer(dx, dy, U_src, U_tgt)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))

        K = construct_aff_mat(Ke, torch.zeros_like(Kp), K_G, K_H)
        adj = (K > 0).to(K.dtype)

        # build 3-order affinity tensor
        hshape = list(adj.shape) + [adj.shape[-1]]
        order3A = adj.unsqueeze(1).expand(hshape) * adj.unsqueeze(2).expand(hshape) * adj.unsqueeze(3).expand(hshape)
        hyper_adj = order3A

        if cfg.NGM.ORDER3_FEATURE == 'cat':
            Ke_3, _ = self.feat_affinity_layer3(X, Y, torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), w1=0.5, w2=1)
            K_3 = construct_aff_mat(Ke_3, torch.zeros_like(Kp), K_G, K_H)
            H = (K_3.unsqueeze(1).expand(hshape) + K_3.unsqueeze(2).expand(hshape) + K_3.unsqueeze(3).expand(hshape)) * F.relu(self.weight3)
        elif cfg.NGM.ORDER3_FEATURE == 'geo':
            Ke_d, _ = self.geo_affinity_layer(dx, dy, torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))

            m_d_src = construct_aff_mat(dx.squeeze().unsqueeze(-1).expand_as(Ke_d), torch.zeros_like(Kp), K_G, K_H).cpu()
            m_d_tgt = construct_aff_mat(dy.squeeze().unsqueeze(-2).expand_as(Ke_d), torch.zeros_like(Kp), K_G, K_H).cpu()
            order3A = order3A.cpu()

            cum_sin = torch.zeros_like(order3A)
            for i in range(3):
                def calc_sin(t):
                    a = t.unsqueeze(i % 3 + 1).expand(hshape)
                    b = t.unsqueeze((i + 1) % 3 + 1).expand(hshape)
                    c = t.unsqueeze((i + 2) % 3 + 1).expand(hshape)
                    cos = torch.clamp((a.pow(2) + b.pow(2) - c.pow(2)) / (2 * a * b + 1e-15), -1, 1)
                    cos *= order3A
                    sin = torch.sqrt(1 - cos.pow(2)) * order3A
                    assert torch.sum(torch.isnan(sin)) == 0
                    return sin
                sin_src = calc_sin(m_d_src)
                sin_tgt = calc_sin(m_d_tgt)
                cum_sin += torch.abs(sin_src - sin_tgt)

            H = torch.exp(- 1 / cfg.NGM.SIGMA3 * cum_sin) * order3A
            H = H.cuda()
            order3A = order3A.cuda()
        elif cfg.NGM.ORDER3_FEATURE == 'none':
            H = torch.zeros_like(hyper_adj)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.ORDER3_FEATURE))

        hyper_adj = hyper_adj.cpu()
        hyper_adj_sum = torch.sum(hyper_adj, dim=tuple(range(2, 3 + 1)), keepdim=True) + 1e-10
        hyper_adj = hyper_adj / hyper_adj_sum
        hyper_adj = hyper_adj.to_sparse().coalesce().cuda()

        H = H.sparse_mask(hyper_adj)
        H = (H._indices(), H._values().unsqueeze(-1))

        if cfg.NGM.FIRST_ORDER:
            emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
        else:
            emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

        adj_sum = torch.sum(adj, dim=2, keepdim=True) + 1e-10
        adj = adj / adj_sum
        pack_M = [K.unsqueeze(-1), H]
        pack_A = [adj, hyper_adj]
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            pack_M, emb = gnn_layer(pack_A, pack_M, emb, ns_src, ns_tgt, norm=False)

        v = self.classifier(emb)
        s = v.view(v.shape[0], P_tgt.shape[1], -1).transpose(1, 2)

        ss = self.bi_stochastic(s, ns_src, ns_tgt)
        x = hungarian(ss, ns_src, ns_tgt)

        data_dict.update({
            'ds_mat': ss,
            'perm_mat': x,
            'aff_mat': K
        })

        return data_dict
