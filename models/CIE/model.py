import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn
from src.feature_align import feature_align
from src.gconv import Siamese_ChannelIndependentConv #, Siamese_GconvEdgeDPP, Siamese_GconvEdgeOri
from models.PCA.affinity_layer import Affinity
from src.lap_solvers.hungarian import hungarian

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter=cfg.CIE.SK_ITER_NUM, epsilon=cfg.CIE.SK_EPSILON, tau=cfg.CIE.SK_TAU)
        self.l2norm = nn.LocalResponseNorm(cfg.CIE.FEATURE_CHANNEL * 2, alpha=cfg.CIE.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.CIE.GNN_LAYER # numbur of GNN layers
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_ChannelIndependentConv(cfg.CIE.FEATURE_CHANNEL * 2, cfg.CIE.GNN_FEAT, 1)
            else:
                gnn_layer = Siamese_ChannelIndependentConv(cfg.CIE.GNN_FEAT, cfg.CIE.GNN_FEAT, cfg.CIE.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.CIE.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.CIE.GNN_FEAT * 2, cfg.CIE.GNN_FEAT))
                self.add_module('cross_graph_edge_{}'.format(i), nn.Linear(cfg.CIE.GNN_FEAT * 2, cfg.CIE.GNN_FEAT))
        self.rescale = cfg.PROBLEM.RESCALE

    def forward(self, data_dict, **kwargs):
        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
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
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('Unknown data type for this model.')

        P_src_dis = (P_src.unsqueeze(1) - P_src.unsqueeze(2))
        P_src_dis = torch.norm(P_src_dis, p=2, dim=3).detach()
        P_tgt_dis = (P_tgt.unsqueeze(1) - P_tgt.unsqueeze(2))
        P_tgt_dis = torch.norm(P_tgt_dis, p=2, dim=3).detach()

        Q_src = torch.exp(-P_src_dis / self.rescale[0])
        Q_tgt = torch.exp(-P_tgt_dis / self.rescale[0])

        emb_edge1 = Q_src.unsqueeze(-1)
        emb_edge2 = Q_tgt.unsqueeze(-1)

        # adjacency matrices
        A_src = torch.bmm(G_src, H_src.transpose(1, 2))
        A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))

        # U_src, F_src are features at different scales
        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)
        ss = []

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))

            # during forward process, the network structure will not change
            emb1, emb2, emb_edge1, emb_edge2 = gnn_layer([A_src, emb1, emb_edge1], [A_tgt, emb2, emb_edge2])

            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb1, emb2) # xAx^T

            s = self.sinkhorn(s, ns_src, ns_tgt)
            ss.append(s)

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                emb1 = new_emb1
                emb2 = new_emb2

                # edge cross embedding
                '''
                cross_graph_edge = getattr(self, 'cross_graph_edge_{}'.format(i))
                emb_edge1 = emb_edge1.permute(0, 3, 1, 2)
                emb_edge2 = emb_edge2.permute(0, 3, 1, 2)
                s = s.unsqueeze(1)
                new_emb_edge1 = cross_graph_edge(torch.cat((emb_edge1, torch.matmul(torch.matmul(s, emb_edge2), s.transpose(2, 3))), dim=1).permute(0, 2, 3, 1))
                new_emb_edge2 = cross_graph_edge(torch.cat((emb_edge2, torch.matmul(torch.matmul(s.transpose(2, 3), emb_edge1), s)), dim=1).permute(0, 2, 3, 1))
                emb_edge1 = new_emb_edge1
                emb_edge2 = new_emb_edge2
                '''

        data_dict.update({
            'ds_mat': ss[-1],
            'perm_mat': hungarian(ss[-1], ns_src, ns_tgt)
        })
        return data_dict