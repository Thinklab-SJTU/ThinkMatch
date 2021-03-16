# line 17 class Net(CNN) need to be modified ? 
import paddle
import paddle.nn as nn

from utils.sinkhorn import Sinkhorn
from utils.voting_layer import Voting
from GMN.displacement_layer import Displacement
from utils.feature_align import feature_align
from PCA.gconv import Siamese_Gconv
from PCA.affinity_layer import Affinity

from utils.config import cfg

import utils.backbone
CNN = eval('utils.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.bi_stochastic = Sinkhorn(max_iter=cfg.PCA.BS_ITER_NUM, epsilon=cfg.PCA.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.PCA.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.PCA.FEATURE_CHANNEL * 2, alpha=cfg.PCA.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.PCA.GNN_LAYER
        self.gnn_layer_list = nn.LayerList()
        self.aff_layer_list = nn.LayerList()
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL * 2, cfg.PCA.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT)
            self.gnn_layer_list.append(gnn_layer)
            self.aff_layer_list.append(Affinity(cfg.PCA.GNN_FEAT))
            if i == self.gnn_layer - 2: 
                # only second last layer will have cross-graph module
                self.cross_layer = (nn.Linear(cfg.PCA.GNN_FEAT * 2, cfg.PCA.GNN_FEAT))

    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, type='img'):
        if type == 'img' or type == 'image':
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
            U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
            F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
        elif type == 'feat' or type == 'feature':
            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('unknown type string {}'.format(type))

        # adjacency matrices
        A_src = paddle.bmm(G_src, H_src.transpose((0,2,1))
        A_tgt = paddle.bmm(G_tgt, H_tgt.transpose((0,2,1))

        emb1, emb2 = paddle.concat((U_src, F_src), axis=1).transpose((0,2,1)), paddle.concat((U_tgt, F_tgt), axis=1).transpose((0,2,1))

        for i in range(self.gnn_layer):
            # gnn layer
            emb1, emb2 = self.gnn_layer_list[i]([A_src, emb1], [A_tgt, emb2])
            # affinity layer
            s = self.aff_layer_list[i](emb1, emb2)
            s = self.voting_layer(s, ns_src, ns_tgt)
            s = self.bi_stochastic(s, ns_src, ns_tgt)

            if i == self.gnn_layer - 2:
                emb1_new = self.cross_layer(paddle.concat((emb1, paddle.bmm(s, emb2)), axis=-1))
                emb2_new = self.cross_layer(paddle.concat((emb2, paddle.bmm(s.transpose((0,2,1)), emb1)), axis=-1))
                emb1 = emb1_new
                emb2 = emb2_new

        d, _ = self.displacement_layer(s, P_src, P_tgt)
        return s, d
