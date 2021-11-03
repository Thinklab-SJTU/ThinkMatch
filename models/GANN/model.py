import torch
import torch.nn as nn
import torch.nn.functional as functional

import numpy as np

from src.lap_solvers.sinkhorn import Sinkhorn
from src.feature_align import feature_align
from models.PCA.affinity_layer import AffinityInp
from models.GANN.graduated_assignment import GA_GM
from src.lap_solvers.hungarian import hungarian
from src.utils.pad_tensor import pad_tensor

from itertools import combinations, product, chain

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.affinity_layer = AffinityInp(cfg.GANN.FEATURE_CHANNEL)
        self.tau = cfg.GANN.SK_TAU
        self.sinkhorn = Sinkhorn(max_iter=cfg.GANN.SK_ITER_NUM,
                                 tau=self.tau, epsilon=cfg.GANN.SK_EPSILON, batched_operation=False)
        self.l2norm = nn.LocalResponseNorm(cfg.GANN.FEATURE_CHANNEL * 2, alpha=cfg.GANN.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.univ_size = torch.tensor(cfg.GANN.UNIV_SIZE)
        self.quad_weight = cfg.GANN.QUAD_WEIGHT
        self.cluster_quad_weight = cfg.GANN.CLUSTER_QUAD_WEIGHT
        self.ga_mgmc = GA_GM(
            mgm_iter=cfg.GANN.MGM_ITER, cluster_iter=cfg.GANN.CLUSTER_ITER,
            sk_iter=cfg.GANN.SK_ITER_NUM, sk_tau0=cfg.GANN.INIT_TAU, sk_gamma=cfg.GANN.GAMMA,
            cluster_beta=cfg.GANN.BETA,
            converge_tol=cfg.GANN.CONVERGE_TOL, min_tau=cfg.GANN.MIN_TAU, projector0=cfg.GANN.PROJECTOR
        )
        self.rescale = cfg.PROBLEM.RESCALE

    def forward(self, data_dict, **kwargs):
        """
        wrapper function of forward pass, class information (predicted for training & ground truth for testing) is
        taken to output intra-class matching results
        """
        num_clusters = cfg.PROBLEM.NUM_CLUSTERS
        if cfg.PROBLEM.TYPE == '2GM':
            assert num_clusters == 1
        elif cfg.PROBLEM.TYPE == 'MGM':
            assert num_clusters == 1
        else:
            assert num_clusters > 1

        assert data_dict['batch_size'] == 1, "Only batch size == 1 is supported."

        U, cluster_v, Wds, mscum = self.real_forward(data_dict, num_clusters)

        if self.training:
            cls_indicator = cluster_v.cpu().numpy().tolist()
        else:
            assert 'cls' in data_dict
            gt_cls = data_dict['cls']
            cls_indicator = []
            for b in range(len(gt_cls[0])):
                cls_indicator.append([])
                for i in range(len(gt_cls)):
                    cls_indicator[b].append(gt_cls[i][b])

        sinkhorn_pairwise_preds, hungarian_pairwise_preds, multi_graph_preds, indices = \
            self.collect_intra_class_matching_wrapper(U[0], Wds[0], mscum[0], cls_indicator[0]) # we assume batch_size=1

        if cfg.PROBLEM.TYPE == '2GM':
            if self.training:
                data_dict.update({
                    'ds_mat': sinkhorn_pairwise_preds[0],
                    'perm_mat': hungarian_pairwise_preds[0],
                    'gt_perm_mat': multi_graph_preds[0],  # pseudo label during training
                })
            else:
                data_dict.update({
                    'perm_mat': multi_graph_preds[0],
                })
        else:
            if self.training:
                data_dict.update({
                    'ds_mat_list': sinkhorn_pairwise_preds,
                    'perm_mat_list': hungarian_pairwise_preds,
                    'gt_perm_mat_list': multi_graph_preds, # pseudo label during training
                    'graph_indices': indices,
                })
            else:
                data_dict.update({
                    'perm_mat_list': multi_graph_preds,
                    'graph_indices': indices,
                })

        if num_clusters > 1:
            data_dict['pred_cluster'] = cluster_v

        return data_dict


    def real_forward(self, data_dict, num_clusters, **kwargs):
        """
        the real forward function.
        :return U: stacked multi-matching matrix
        :return cluster_v: clustering indicator vector
        :return Wds: doubly-stochastic pairwise matching results
        :return mscum: cumsum of number of nodes in graphs
        """
        batch_size = data_dict['batch_size']
        num_graphs = data_dict['num_graphs']

        # extract graph feature
        if 'images' in data_dict:
            # real image data
            data = data_dict['images']
            Ps = data_dict['Ps']
            ns = data_dict['ns']
            As_src = data_dict['As']

            data_cat = torch.cat(data, dim=0)
            P_cat = torch.cat(pad_tensor(Ps), dim=0)
            n_cat = torch.cat(ns, dim=0)
            node = self.node_layers(data_cat)
            edge = self.edge_layers(node)
            U = feature_align(node, P_cat, n_cat, self.rescale)
            F = feature_align(edge, P_cat, n_cat, self.rescale)
            feats = torch.cat((U, F), dim=1)
            feats = self.l2norm(feats)
            feats[torch.isnan(feats)] = 0.

            feats = torch.split(feats, batch_size, dim=0)
        else:
            raise ValueError('Unknown data type for this model.')

        # store features and variables in feat_list
        feat_list = []
        iterator = zip(feats, Ps, As_src, ns)
        ms = torch.zeros(batch_size, num_graphs, dtype=torch.int, device=self.device)
        for idx, (feat, P, As_src, n) in enumerate(iterator):
            feat_list.append((idx, feat, P, As_src, n))
            ms[:, idx] = n
        msmax = torch.max(ms, dim=1).values
        mscum = torch.cumsum(ms, dim=1)
        mssum = mscum[:, -1]

        # compute multi-adjacency matrix A
        A = [torch.zeros(m.item(), m.item(), device=self.device) for m in mssum]
        for idx, feat, P, As_src, n in feat_list:
            edge_lens = torch.sqrt(torch.sum((P.unsqueeze(1) - P.unsqueeze(2)) ** 2, dim=-1)) * As_src
            median_lens = torch.median(torch.flatten(edge_lens, start_dim=-2), dim=-1).values
            median_lens = median_lens.unsqueeze(-1).unsqueeze(-1)
            A_ii = torch.exp(- edge_lens ** 2 / median_lens ** 2 / cfg.GANN.SCALE_FACTOR)
            if cfg.GANN.NORM_QUAD_TERM:
                A_ii = A_ii / n * self.univ_size
            diag_A_ii = torch.diagonal(A_ii, dim1=-2, dim2=-1)
            diag_A_ii[:] = 0

            for b in range(batch_size):
                start_idx = mscum[b, idx] - n[b]
                end_idx = mscum[b, idx]
                A[b][start_idx:end_idx, start_idx:end_idx] += A_ii[b, :n[b], :n[b]]

        # compute similarity matrix W
        Wds = [torch.zeros(m.item(), m.item(), device=self.device) for m in mssum]
        for src, tgt in product(feat_list, repeat=2):
            src_idx, src_feat, P_src, A_src, n_src = src
            tgt_idx, tgt_feat, P_tgt, A_tgt, n_tgt = tgt
            if src_idx < tgt_idx:
                continue
            W_ij = self.affinity_layer(src_feat.transpose(1, 2), tgt_feat.transpose(1, 2))
            for b in range(batch_size):
                start_x = mscum[b, src_idx] - n_src[b]
                end_x = mscum[b, src_idx]
                start_y = mscum[b, tgt_idx] - n_tgt[b]
                end_y = mscum[b, tgt_idx]
                W_ijb = W_ij[b, :n_src[b], :n_tgt[b]]
                if end_y - start_y >= end_x - start_x:
                    W_ij_ds = self.sinkhorn(W_ijb, dummy_row=True)
                else:
                    W_ij_ds = self.sinkhorn(W_ijb.t(), dummy_row=True).t()
                Wds[b][start_x:end_x, start_y:end_y] += W_ij_ds
                if src_idx != tgt_idx:
                    Wds[b][start_y:end_y, start_x:end_x] += W_ij_ds.t()

        # GANN
        U = [[] for _ in range(batch_size)]
        cluster_v = []
        for b in range(batch_size):
            if num_graphs == 2:
                univ_size = max(feat_list[0][-1][b], feat_list[1][-1][b])
            else:
                univ_size = data_dict['univ_size'][b]
            U0_b = torch.full((torch.sum(ms[b]), univ_size), 1 / univ_size.to(dtype=torch.float), device=self.device)
            U0_b += torch.randn_like(U0_b) / 1000

            U_b, cluster_v_b = self.ga_mgmc(A[b], Wds[b], U0_b, ms[b], univ_size, self.quad_weight, self.cluster_quad_weight, num_clusters)
            cluster_v.append(cluster_v_b)

            for i in range(num_graphs):
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = mscum[b, i-1]
                end_idx = mscum[b, i]
                U[b].append(U_b[start_idx:end_idx, :])

        cluster_v = torch.stack(cluster_v)

        return U, cluster_v, Wds, mscum

    @staticmethod
    def collect_intra_class_matching_wrapper(U, Wds, mscum, cls_list):
        """
        :param U: Stacked matching-to-universe matrix
        :param Wds: pairwise matching result in doubly-stochastic matrix
        :param mscum: cumsum of number of nodes in graphs
        :param cls_list: list of class information
        """
        # collect results
        pairwise_pred_s = []
        pairwise_pred_x = []
        mgm_pred_x = []
        indices = []
        unique_cls_list = set(cls_list)

        intra_class_iterator = []
        for cls in unique_cls_list:
            idx_range = np.where(np.array(cls_list) == cls)[0]
            intra_class_iterator.append(combinations(idx_range, 2))
        intra_class_iterator = chain(*intra_class_iterator)

        for idx1, idx2 in intra_class_iterator:
            start_x = mscum[idx1 - 1] if idx1 != 0 else 0
            end_x = mscum[idx1]
            start_y = mscum[idx2 - 1] if idx2 != 0 else 0
            end_y = mscum[idx2]
            if end_y - start_y >= end_x - start_x:
                s = Wds[start_x:end_x, start_y:end_y]
            else:
                s = Wds[start_y:end_y, start_x:end_x].t()

            pairwise_pred_s.append(s.unsqueeze(0))
            x = hungarian(s)
            pairwise_pred_x.append(x.unsqueeze(0))

            mgm_x = torch.mm(U[idx1], U[idx2].t())
            mgm_pred_x.append(mgm_x.unsqueeze(0))
            indices.append((idx1, idx2))

        return pairwise_pred_s, pairwise_pred_x, mgm_pred_x, indices
