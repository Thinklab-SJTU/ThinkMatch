import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.lap_solvers.sinkhorn import Sinkhorn
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from models.NGM.gnn import GNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
from models.GMN.affinity_layer import InnerpAffinity, GaussianAffinity
from src.lap_solvers.hungarian import hungarian

from itertools import combinations
import numpy as np

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)

def pad_tensor(inp):
    assert type(inp[0]) == torch.Tensor
    it = iter(inp)
    t = next(it)
    max_shape = list(t.shape)
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
        except StopIteration:
            break
    max_shape = np.array(max_shape)

    padded_ts = []
    for t in inp:
        pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
        pad_pattern[::-2] = max_shape - np.array(t.shape)
        pad_pattern = tuple(pad_pattern.tolist())
        padded_ts.append(functional.pad(t, pad_pattern, 'constant', 0))

    return padded_ts


class Net(CNN):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        if cfg.NGM.EDGE_FEATURE == 'cat':
            self.affinity_layer = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            self.affinity_layer = GaussianAffinity(1, cfg.NGM.GAUSSIAN_SIGMA)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))
        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.l2norm = nn.LocalResponseNorm(cfg.NGM.FEATURE_CHANNEL * 2, alpha=cfg.NGM.FEATURE_CHANNEL * 2, beta=0.5, k=0)

        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = GNNLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=self.tau, edge_emb=cfg.NGM.EDGE_EMB)
            else:
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=self.tau, edge_emb=cfg.NGM.EDGE_EMB)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0), 1)

        self.sinkhorn2 = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, epsilon=cfg.NGM.SK_EPSILON, tau=cfg.NGM.MGM_SK_TAU)

    def forward(self, data_dict, **kwargs):
        # extract graph feature
        if 'images' in data_dict:
            # extract data
            data = data_dict['images']
            Ps = data_dict['Ps']
            ns = data_dict['ns']
            Gs = data_dict['Gs']
            Hs = data_dict['Hs']
            Gs_tgt = data_dict['Gs_tgt']
            Hs_tgt = data_dict['Hs_tgt']
            KGs = {k: v[0] for k, v in  data_dict['KGHs'].items()}
            KHs = {k: v[1] for k, v in  data_dict['KGHs'].items()}

            batch_size = data[0].shape[0]
            device = data[0].device

            data_cat = torch.cat(data, dim=0)
            P_cat = torch.cat(pad_tensor(Ps), dim=0)
            n_cat = torch.cat(ns, dim=0)
            node = self.node_layers(data_cat)
            edge = self.edge_layers(node)
            U = feature_align(node, P_cat, n_cat, self.rescale)
            F = feature_align(edge, P_cat, n_cat, self.rescale)
            feats = torch.cat((U, F), dim=1)
            feats = self.l2norm(feats)
            feats = torch.split(feats, batch_size, dim=0)
        elif 'features' in data_dict:
            # extract data
            data = data_dict['features']
            Ps = data_dict['Ps']
            ns = data_dict['ns']
            Gs = data_dict['Gs']
            Hs = data_dict['Hs']
            Gs_tgt = data_dict['Gs_tgt']
            Hs_tgt = data_dict['Hs_tgt']
            KGs = {k: v[0] for k, v in data_dict['KGHs'].items()}
            KHs = {k: v[1] for k, v in data_dict['KGHs'].items()}

            batch_size = data[0].shape[0]
            device = data[0].device

            feats = data
        else:
            raise ValueError('Unknown data type for this model.')

        # extract reference graph feature
        feat_list = []
        joint_indices = [0]
        iterator = zip(feats, Ps, Gs, Hs, Gs_tgt, Hs_tgt, ns)
        for idx, (feat, P, G, H, G_tgt, H_tgt, n) in enumerate(iterator):
            feat_list.append(
                (
                    idx,
                    feat,
                    P, G, H, G_tgt, H_tgt, n
                )
            )
            joint_indices.append(joint_indices[-1] + P.shape[1])

        joint_S = torch.zeros(batch_size, joint_indices[-1], joint_indices[-1], device=device)
        joint_S_diag = torch.diagonal(joint_S, dim1=1, dim2=2)
        joint_S_diag += 1

        pred_s = []
        pred_x = []
        indices = []

        for src, tgt in combinations(feat_list, 2):
            # pca forward
            src_idx, src_feat, P_src, G_src, H_src, _, __, n_src = src
            tgt_idx, tgt_feat, P_tgt, _, __, G_tgt, H_tgt, n_tgt = tgt
            K_G = KGs['{},{}'.format(src_idx, tgt_idx)]
            K_H = KHs['{},{}'.format(src_idx, tgt_idx)]
            s = self.__ngm_forward(src_feat, tgt_feat, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, K_G, K_H, n_src, n_tgt)

            if src_idx > tgt_idx:
                joint_S[:, joint_indices[tgt_idx]:joint_indices[tgt_idx+1], joint_indices[src_idx]:joint_indices[src_idx+1]] += s.transpose(1, 2)
            else:
                joint_S[:, joint_indices[src_idx]:joint_indices[src_idx+1], joint_indices[tgt_idx]:joint_indices[tgt_idx+1]] += s

        matching_s = []
        for b in range(batch_size):
            e, v = torch.symeig(joint_S[b], eigenvectors=True)
            topargs = torch.argsort(torch.abs(e), descending=True)[:joint_indices[1]]
            diff = e[topargs[:-1]] - e[topargs[1:]]
            if torch.min(torch.abs(diff)) > 1e-4:
                matching_s.append(len(data) * torch.mm(v[:, topargs], v[:, topargs].transpose(0, 1)))
            else:
                matching_s.append(joint_S[b])

        matching_s = torch.stack(matching_s, dim=0)

        for idx1, idx2 in combinations(range(len(data)), 2):
            s = matching_s[:, joint_indices[idx1]:joint_indices[idx1+1], joint_indices[idx2]:joint_indices[idx2+1]]
            s = self.sinkhorn2(s)

            pred_s.append(s)
            pred_x.append(hungarian(s))
            indices.append((idx1, idx2))

        data_dict.update({
            'ds_mat_list': pred_s,
            'perm_mat_list': pred_x,
            'graph_indices': indices,
        })
        return data_dict

    def __ngm_forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, K_G, K_H, ns_src, ns_tgt):
        U_src = src[:, :src.shape[1] // 2, :]
        F_src = src[:, src.shape[1] // 2:, :]
        U_tgt = tgt[:, :tgt.shape[1] // 2, :]
        F_tgt = tgt[:, tgt.shape[1] // 2:, :]

        if cfg.NGM.EDGE_FEATURE == 'cat':
            X = reshape_edge_feature(F_src, G_src, H_src)
            Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
            Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))

        #K_G = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(G_tgt.cpu().numpy(), G_src.cpu().numpy())]
        #K_H = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(H_tgt.cpu().numpy(), H_src.cpu().numpy())]
        #K_G = CSRMatrix3d(K_G).to(src.device)
        #K_H = CSRMatrix3d(K_H).transpose().to(src.device)

        # affinity layer
        Ke, Kp = self.affinity_layer(X, Y, U_src, U_tgt)

        K = construct_aff_mat(Ke, torch.zeros_like(Kp), K_G, K_H)

        A = (K > 0).to(K.dtype)

        if cfg.NGM.FIRST_ORDER:
            emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
        else:
            emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

        emb_K = K.unsqueeze(-1)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_K, emb = gnn_layer(A, emb_K, emb, ns_src, ns_tgt) #, norm=False)

        v = self.classifier(emb)
        s = v.view(v.shape[0], U_tgt.shape[2], -1).transpose(1, 2)

        ss = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)

        return ss
