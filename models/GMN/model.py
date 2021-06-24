import torch.nn as nn

from models.GMN.affinity_layer import InnerpAffinity as Affinity
from src.qap_solvers.spectral_matching import SpectralMatching
from src.qap_solvers.rrwm import RRWM
from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.hungarian import hungarian
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.affinity_layer = Affinity(cfg.GMN.FEATURE_CHANNEL)
        if cfg.GMN.GM_SOLVER == 'SM':
            self.gm_solver = SpectralMatching(max_iter=cfg.GMN.PI_ITER_NUM, stop_thresh=cfg.GMN.PI_STOP_THRESH)
        elif cfg.GMN.GM_SOLVER == 'RRWM':
            self.gm_solver = RRWM()
        self.sinkhorn = Sinkhorn(max_iter=cfg.GMN.BS_ITER_NUM, tau=1/cfg.GMN.VOTING_ALPHA, epsilon=cfg.GMN.BS_EPSILON, log_forward=False)
        self.l2norm = nn.LocalResponseNorm(cfg.GMN.FEATURE_CHANNEL * 2, alpha=cfg.GMN.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.rescale = cfg.PROBLEM.RESCALE

    def forward(self, data_dict, **kwargs):
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
            raise ValueError('Unknown data type for this model.')

        X = reshape_edge_feature(F_src, G_src, H_src)
        Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)

        # affinity layer
        Me, Mp = self.affinity_layer(X, Y, U_src, U_tgt)

        M = construct_aff_mat(Me, Mp, K_G, K_H)

        v = self.gm_solver(M, num_src=P_src.shape[1], ns_src=ns_src, ns_tgt=ns_tgt)
        s = v.view(v.shape[0], P_tgt.shape[1], -1).transpose(1, 2)

        s = self.sinkhorn(s, ns_src, ns_tgt)

        data_dict.update({
            'ds_mat': s,
            'perm_mat': hungarian(s, ns_src, ns_tgt),
            'aff_mat': M
        })
        return data_dict
