#import torch
#import torch.nn as nn
import sys
sys.path.insert(0,"/data/lixinyang/mindspore3/ThinkMatch")
sys.path.insert(0,"/data/lixinyang/mindspore3/ThinkMatch/api/utils")
from api.lap_solvers.sinkhorn import Sinkhorn #done
from api.feature_align import feature_align #untested
from api.gconv import Siamese_Gconv #untested
from models.PCA.affinity_layer import Affinity #untested
from api.lap_solvers.hungarian import hungarian #untested
from models.PCA.specialLRN import SpecialLRN

from api.utils.config import cfg
#from config import cfg
from models.PCA.model_config import model_cfg

from api.backbone import *
CNN = eval(cfg.BACKBONE)

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.common.tensor import Tensor
import mindspore.context as context

context.set_context(device_target="GPU")

class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter=cfg.PCA.SK_ITER_NUM, epsilon=cfg.PCA.SK_EPSILON, tau=cfg.PCA.SK_TAU)
        self.l2norm = SpecialLRN()
        self.featurealign = feature_align()
        self.gnn_layer = cfg.PCA.GNN_LAYER
        #self.pointer_net = PointerNet(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT // 2, alpha=cfg.PCA.VOTING_ALPHA)
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL * 2, cfg.PCA.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT)
            setattr(self, 'gnn_layer_{}'.format(i), gnn_layer)
            setattr(self, 'affinity_{}'.format(i), Affinity(cfg.PCA.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                setattr(self, 'cross_graph_{}'.format(i), nn.Dense(cfg.PCA.GNN_FEAT * 2, cfg.PCA.GNN_FEAT))
        self.cross_iter = cfg.PCA.CROSS_ITER
        self.cross_iter_num = cfg.PCA.CROSS_ITER_NUM
        self.rescale = cfg.PROBLEM.RESCALE

        #print('node')
        #print(self.node_layers)
        #print('edge')
        #print(self.edge_layers)
        #self.aaaaa = Affinity(cfg.PCA.GNN_FEAT)
        #self.insert_child_to_cell('affinity_aaaaa', self.aaaaa)

    def reload_backbone(self):
        self.node_layers, self.edge_layers = self.get_backbone(True)

    def construct(self, data_dict, **kwargs):
        #print('check0')
        #print('model_edge_weight', self.edge_layers[6].weight[:])
        if 'img0' in data_dict:   #here

            # real image data
            # src, tgt = data_dict['images']
            # P_src, P_tgt = data_dict['Ps']
            # ns_src, ns_tgt = data_dict['ns']
            # A_src, A_tgt = data_dict['As']
            src, tgt = data_dict['img0'], data_dict['img1']
            P_src, P_tgt = data_dict['P1'], data_dict['P2']
            ns_src, ns_tgt = data_dict['n1'], data_dict['n2']
            A_src, A_tgt = data_dict['A1'], data_dict['A2']
            # print('src',src)
            # print('tgt',tgt)
            # print('P_src', P_src)
            # print('P_tgt', P_tgt)
            # print('ns_src', ns_src)
            # print('ns_tgt', ns_tgt)
            # print('A_src', A_src)
            # print('A_tgt', A_tgt)


            #print('check1.1')
            # extract feature(checked)
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)
            #print('check1.2')
            # feature normalization(checked)
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)
 #           print('check1.3')
            # arrange features(checked)
            U_src = self.featurealign(src_node, P_src, ns_src, self.rescale)
            F_src = self.featurealign(src_edge, P_src, ns_src, self.rescale)
            U_tgt = self.featurealign(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = self.featurealign(tgt_edge, P_tgt, ns_tgt, self.rescale)
#            print('check1.4')
            #print('check1')

        elif 'features' in data_dict:
            # synthetic data
            src, tgt = data_dict['features']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('Unknown data type for this model.')

        perm1 = tuple(range(3, len(U_src.shape)))
        perm1 = (0, 2, 1) + perm1
        perm2 = tuple(range(3, len(U_tgt.shape)))
        perm2 = (0, 2, 1) + perm2
        emb1, emb2 = P.Transpose()(P.Concat(1)((U_src, F_src)), perm1), P.Transpose()(P.Concat(1)((U_tgt, F_tgt)), perm2)
        ss = []
        #print('check2')
        if not self.cross_iter:#here

            # Vanilla PCA-GM
            for i in range(self.gnn_layer):
                #print('check2')
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                #print('emb1', emb1)
                #print('check3')
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                #print('affinity', s)
                #print('check4')
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)#checked
 #               print('check5')
                #print('sinkhorn',s)
                ss.append(s)
                if i == self.gnn_layer - 2:
                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    new_emb1 = cross_graph(P.Concat(-1)((emb1, P.BatchMatMul()(s, emb2))))

                    perm = tuple(range(3, len(s.shape)))
                    perm = (0, 2, 1) + perm
                    new_emb2 = cross_graph(P.Concat(-1)((emb2, P.BatchMatMul()(P.Transpose()(s, perm), emb1))))

                    emb1 = new_emb1
                    emb2 = new_emb2
                    #print('check6')
        else:

            # IPCA-GM
            for i in range(self.gnn_layer - 1):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])

            emb1_0, emb2_0 = emb1, emb2
            s = np.zeros((emb1.shape[0], emb1.shape[1], emb2.shape[1]))

            for x in range(self.cross_iter_num):
                i = self.gnn_layer - 2
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))

                emb1 = cross_graph(P.Concat(-1)((emb1_0, P.BatchMatMul()(s, emb2_0))))

                perm = tuple(range(3, len(s.shape)))
                perm = (0, 2, 1) + perm
                emb2 = cross_graph(P.Concat(-1)((emb2_0, P.BatchMatMul()(P.Transpose()(s, perm), emb1_0))))

                i = self.gnn_layer - 1
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                ss.append(s)
        #print('check7')
        #print('ss',ss[-1])
        data_dict.update({
            'ds_mat': ss[-1],
            'perm_mat': hungarian(ss[-1], ns_src, ns_tgt)
        })
        #print('check8')

        return data_dict
