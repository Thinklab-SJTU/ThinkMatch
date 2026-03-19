import torch
import itertools
import numpy as np
from torch_sparse import SparseTensor

from models.COMMONPLUS.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from models.COMMONPLUS.transformconv import TransformerConvLayer
from models.NGM.gnn import PYGNNLayer
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat, construct_sparse_aff_mat
from src.utils.pad_tensor import pad_tensor
from src.lap_solvers.sinkhorn import Sinkhorn, BinSinkhorn
from pygmtools.pytorch_backend import hungarian

from src.utils.config import cfg

from src.backbone import *

from src.loss_func import Distill_InfoNCE_Outlier, Distill_QuadraticContrast, Permutation_Bin_Loss

CNN = eval(cfg.BACKBONE)


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class InnerProduct(nn.Module):
    def __init__(self):
        super(InnerProduct, self).__init__()

    def _forward(self, X, Y):
        X = torch.nn.functional.normalize(X, dim=-1)
        Y = torch.nn.functional.normalize(Y, dim=-1)
        res = torch.matmul(X, Y.transpose(0, 1))
        return res

    def forward(self, Xs, Ys):
        return [self._forward(X, Y) for X, Y in zip(Xs, Ys)]


class Backbone(CNN):
    def __init__(self):
        super(Backbone, self).__init__()
        lantent_dim = cfg.COMMONPLUS.FEATURE_CHANNEL * 2
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=lantent_dim)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.vertex_affinity = InnerProduct()
        self.rescale = cfg.PROBLEM.RESCALE
        self.logit_scale = torch.ones([]) * np.log(1 / cfg.COMMONPLUS.SOFTMAXTEMP)
        self.sinkhorn = BinSinkhorn()
        self.bin_value = nn.Parameter(torch.ones([1]) * 0.2)  # 0.3最好
        self.sinkhorn_bin_value = nn.Parameter(torch.ones([1]) * 0.2)
        self.projection = nn.Sequential(
            nn.Linear(lantent_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Following NGMv2
        if cfg.COMMONPLUS.TRANSFORMER_GNN == True:
            self.TRANSFORMER_GNN = True
            self.attn = TransformerConvLayer(cfg.ATTN.IN_DIM, cfg.ATTN.OUT_DIM, depth=cfg.ATTN.DEPTH, head=cfg.ATTN.HEAD,
                                             sk_channel=cfg.ATTN.SK_EMB, sk_tau=cfg.NGM.SK_TAU, recurrence=cfg.ATTN.RECURRENCE)
            self.classifier = nn.Linear(cfg.ATTN.OUT_DIM[-1] + cfg.ATTN.SK_EMB, 1)
        else:
            self.TRANSFORMER_GNN = False
            self.gnn_layer = cfg.NGM.GNN_LAYER
            for i in range(self.gnn_layer):
                tau = cfg.NGM.SK_TAU
                if i == 0:
                    gnn_layer = PYGNNLayer(1, 1,
                                           cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                           sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                else:
                    gnn_layer = PYGNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                           cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                           sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)

    def forward(self, data_dict, online=True):
        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052)  # clamp temperature to be between 0.01 and 1

        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['batch_size']
        num_graphs = len(images)
        orig_graph_list = []

        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            # if cfg.backbone start with "Vit"
            if cfg.BACKBONE.startswith("Vit"):
                nodes, edges, glb = self.node_edge_layer(image)
            else:
                nodes = self.node_layers(image)
                edges = self.edge_layers(nodes)

            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features, following BBGM
            U = feature_align(nodes, p, n_p, self.rescale)
            F = feature_align(edges, p, n_p, self.rescale)
            U = concat_features(U, n_p)
            F = concat_features(F, n_p)
            node_features = torch.cat((U, F), dim=1)

            # GNN
            graph.x = node_features
            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)


        # L-QAP
        unary_affs_list = [
            self.vertex_affinity([self.projection(item.x) for item in g_1], [self.projection(item.x) for item in g_2])
            for (g_1, g_2) in lexico_iter(orig_graph_list)
        ]
        quadratic_affs_list = [
            self.vertex_affinity([self.projection(item.edge_attr) for item in g_1], [self.projection(item.edge_attr) for item in g_2])
            for (g_1, g_2) in lexico_iter(orig_graph_list)
        ]

        feature_dim = orig_graph_list[0][0].x.shape[-1]
        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]
        x_list, s_list = [], []
        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]

            Kp = torch.stack(pad_tensor(unary_affs), dim=0)  # .detach()
            Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)  # .detach()
            emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)

            if self.TRANSFORMER_GNN == True:
                # dense
                K = construct_aff_mat(Ke, Kp, kro_G, kro_H)
                A = (K > 0).to(K.dtype)
                emb = self.attn(emb, K, A, n_points[idx1], n_points[idx2])
            else:
                # sparse, PYGNN
                qap_emb = []
                for b in range(len(data_dict['KGHs_sparse'])):
                    kro_G, kro_H = data_dict['KGHs_sparse'][b] if num_graphs == 2 else data_dict['KGHs_sparse']['{},{}'.format(idx1, idx2)]
                    # sparse
                    K_value, row_idx, col_idx = construct_sparse_aff_mat(quadratic_affs[b], unary_affs[b], kro_G, kro_H)
                    # NGM qap solver
                    tmp_emb = emb[b].unsqueeze(0)
                    adj = SparseTensor(row=row_idx.long(), col=col_idx.long(), value=K_value,
                                       sparse_sizes=(Kp.shape[1] * Kp.shape[2], Kp.shape[1] * Kp.shape[2]))
                    for i in range(self.gnn_layer):
                        gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                        tmp_emb = gnn_layer(adj, tmp_emb, n_points[idx1], n_points[idx2], b)
                    qap_emb.append(tmp_emb.squeeze(0))
                emb = torch.stack(pad_tensor(qap_emb), dim=0)

            v = self.classifier(emb)
            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)
            s_bin, ss = self.sinkhorn(s, self.sinkhorn_bin_value, n_points[idx1], n_points[idx2])
            s_list.append(ss / ss.detach().max())

        # KB-QAP (Feature for contrastive learning)
        unary_affs_list = [
            self.vertex_affinity([self.projection(item.x) for item in g_1], [self.projection(item.x) for item in g_2])
            for (g_1, g_2) in lexico_iter(orig_graph_list)
        ]
        # prepare aligned node features for computing contrastive loss
        keypoint_number_list = []  # the number of keypoints in each image pair
        node_feature_list = []  # node features for computing contrastive loss
        node_feature_outlier_list = []
        node_feature_outlier_list_1, node_feature_outlier_list_2 = [], []

        node_feature_graph1 = torch.zeros([batch_size, data_dict['gt_perm_mat'].shape[1], feature_dim], device=node_features.device)
        node_feature_graph2 = torch.zeros([batch_size, data_dict['gt_perm_mat'].shape[2], feature_dim], device=node_features.device)
        # count the available keypoints in number list
        for index in range(batch_size):
            # empty tensor placed at the end of the node feature list, outliers are in the front with 0 alignment point
            node_feature_graph1[index, :orig_graph_list[0][index].x.shape[0]] = orig_graph_list[0][index].x
            node_feature_graph2[index, :orig_graph_list[1][index].x.shape[0]] = orig_graph_list[1][index].x
            id_graph1 = torch.zeros(data_dict['gt_perm_mat'][index].shape[0], device=node_features.device)
            id_graph1[:data_dict['ns'][0][index]] = 1
            # the outlier should be filtered with padding points
            id_graph1_index = (torch.sum(data_dict['gt_perm_mat'][index], dim=1) == 0) == id_graph1
            id_graph2 = torch.zeros(data_dict['gt_perm_mat'][index].shape[1], device=node_features.device)
            id_graph2[:data_dict['ns'][1][index]] = 1
            id_graph2_index = (torch.sum(data_dict['gt_perm_mat'][index], dim=0) == 0) == id_graph2
            node_feature_outlier_list_1.append(node_feature_graph1[index, id_graph1_index])
            node_feature_outlier_list_2.append(node_feature_graph2[index, id_graph2_index])
            keypoint_number_list.append(torch.sum(data_dict['gt_perm_mat'][index]))
        node_feature_outlier_list_1 = torch.cat(node_feature_outlier_list_1)
        node_feature_outlier_list_2 = torch.cat(node_feature_outlier_list_2)
        number = int(sum(keypoint_number_list))  # calculate the number of correspondence

        # pre-align the keypoints for further computing the contrastive loss
        node_feature_graph2 = torch.bmm(data_dict['gt_perm_mat'], node_feature_graph2)
        final_node_feature_graph1 = torch.zeros([number, feature_dim], device=node_features.device)
        final_node_feature_graph2 = torch.zeros([number, feature_dim], device=node_features.device)
        count = 0
        for index in range(batch_size):
            # the keypoint in graph 1 could also be outlier
            id = torch.sum(data_dict['gt_perm_mat'][index], dim=1)
            final_node_feature_graph1[count: count + int(keypoint_number_list[index])] = node_feature_graph1[index, id == 1]
            final_node_feature_graph2[count: count + int(keypoint_number_list[index])] = node_feature_graph2[index, id == 1]
            assert sum(id) == int(keypoint_number_list[index])
            count += int(keypoint_number_list[index])

        all_features = torch.cat([final_node_feature_graph1, final_node_feature_graph2, node_feature_outlier_list_1, node_feature_outlier_list_2])
        all_features = self.projection(all_features)
        projected_all_features = all_features
        node_feature_list.append(projected_all_features[:number])
        node_feature_list.append(projected_all_features[number: 2 * number])
        node_feature_outlier_list.append(projected_all_features[2 * number:node_feature_outlier_list_1.shape[0] + 2 * number])
        node_feature_outlier_list.append(projected_all_features[node_feature_outlier_list_1.shape[0] + 2 * number:])

        if online == False:
            # output of the momentum network
            x_list2 = []
            for unary_affs, (idx1, idx2) in zip(unary_affs_list, lexico_iter(range(num_graphs))):
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Kp_max = Kp.detach().max()
                if cfg.PROBLEM.FILTER == 'unfiltered':
                    x = hungarian(s_list[0], n_points[idx1], n_points[idx2], self.bin_value.detach().expand(Kp.shape[0], Kp.shape[1]) / Kp_max,
                                  self.bin_value.detach().expand(Kp.shape[0], Kp.shape[2]) / Kp_max)
                    x_Kp = hungarian(Kp, n_points[idx1], n_points[idx2], self.bin_value.detach().expand(Kp.shape[0], Kp.shape[1]),
                                     self.bin_value.detach().expand(Kp.shape[0], Kp.shape[2]))
                else:
                    x = hungarian(s_list[0], n_points[idx1], n_points[idx2])
                    x_Kp = hungarian(Kp, n_points[idx1], n_points[idx2])
                x_list.append(x)
                x_list2.append(x_Kp)
            return node_feature_list, node_feature_outlier_list, x_list, x_list2, Kp / Kp_max, s_list[0]
        elif online == True:
            # output of the online network
            x_list3 = []
            for unary_affs, (idx1, idx2) in zip(unary_affs_list, lexico_iter(range(num_graphs))):
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Kp_max = Kp.detach().max()
                Kp_norm = (Kp / Kp_max + s_list[0]) / 2.0
                if cfg.PROBLEM.FILTER == 'unfiltered':
                    x_fused = hungarian(Kp_norm, n_points[idx1], n_points[idx2], self.bin_value.detach().expand(Kp.shape[0], Kp.shape[1]) / Kp_max,
                                        self.bin_value.detach().expand(Kp.shape[0], Kp.shape[2]) / Kp_max)
                else:
                    x_fused = hungarian(Kp_norm, n_points[idx1], n_points[idx2])
                x_list3.append(x_fused)
            return node_feature_list, node_feature_outlier_list, self.bin_value, x_list3, s_bin


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.onlineNet = Backbone()
        self.momentumNet = Backbone()  # initialize the online network and momentum network
        self.momentum = cfg.COMMONPLUS.MOMENTUM  # momentum parameter for the momentum network
        self.backbone_params = list(self.onlineNet.backbone_params)  # used in train_eval.py
        self.warmup_step = cfg.COMMONPLUS.WARMUP_STEP  # warmup steps for the distillation
        self.epoch_iters = cfg.TRAIN.EPOCH_ITERS  # iterations for one epoch, specified by the training dataset

        self.model_pairs = [[self.onlineNet, self.momentumNet]]
        self.copy_params()  # initialize the momentum network

        assert cfg.PROBLEM.TYPE == '2GM'  # only support 2GM problem currently

    def forward(self, data_dict, training=False, iter_num=0, epoch=0):
        # calculate the distillation weight alpha
        if epoch * self.epoch_iters + iter_num >= self.warmup_step:
            alpha = cfg.COMMONPLUS.ALPHA
        else:
            alpha = cfg.COMMONPLUS.ALPHA * min(1, (epoch * self.epoch_iters + iter_num) / self.warmup_step)

        # output of the online network
        node_feature_list, node_feature_outlier_list, bin_value, x_list3, s_bin = self.onlineNet(data_dict)

        if training == True:
            # the momentum network is only using for training
            assert cfg.COMMONPLUS.DISTILL == True
            # obtain output of the momentum network
            with torch.no_grad():
                self._momentum_update()
                node_feature_m_list, node_feature_outlier_m_list, x_list, x_list2, Kp, ss = self.momentumNet(data_dict, online=False)
            target = data_dict['gt_perm_mat']
            if epoch * self.epoch_iters + iter_num >= self.warmup_step: # Eq.(36)
                consistent_GT = x_list[0].int() & x_list2[0].int() & data_dict['gt_perm_mat'].int()  # A,B,C = 1. GT position
                consistent_non_GT = x_list[0].int() & x_list2[0].int() & (1 - data_dict['gt_perm_mat'].int())  # A，B = 1, C = 0. AB position
                single_GT1 = (x_list[0].int()) & (x_list2[0].int() == 0) & data_dict['gt_perm_mat'].int()
                single_GT2 = (x_list[0].int() == 0) & (x_list2[0].int()) & data_dict['gt_perm_mat'].int()
                single_GT = (((x_list[0].int() == 0) & (x_list2[0].int()) & data_dict['gt_perm_mat'].int()) |
                             ((x_list[0].int()) & (x_list2[0].int() == 0) & data_dict['gt_perm_mat'].int()))  # A=1,C=1,B=0; B=1,C=1,A=0. GT position
                nonGT = (x_list[0].int() == 0) & (x_list2[0].int() == 0) & data_dict['gt_perm_mat'].int()  # A,B = 0，C=1. GT position
                target = data_dict['gt_perm_mat'].clone()
                target[consistent_GT.bool()] = 1.0
                target[single_GT2.bool()] = ((1 - cfg.COMMONPLUS.ALPHA) * data_dict['gt_perm_mat'].int() + cfg.COMMONPLUS.ALPHA * Kp)[single_GT2.bool()]
                target[single_GT1.bool()] = ((1 - cfg.COMMONPLUS.ALPHA) * data_dict['gt_perm_mat'].int() + cfg.COMMONPLUS.ALPHA * ss)[single_GT1.bool()]

                target[consistent_non_GT.bool()] = (Kp + ss)[consistent_non_GT.bool()] / 2.0
                target[nonGT.bool()] = (Kp + ss)[nonGT.bool()] / 2.0

            queue = node_feature_outlier_list
            queue_m = node_feature_outlier_m_list
            # loss function
            contrastloss = Distill_InfoNCE_Outlier()
            loss, bin_value_new = contrastloss(node_feature_list, node_feature_m_list, queue, queue_m, alpha,
                                               self.onlineNet.logit_scale, self.momentumNet.logit_scale)
            self.onlineNet.bin_value = nn.Parameter(self.onlineNet.bin_value * self.momentum + bin_value_new * (1. - self.momentum))
            crossloss = Distill_QuadraticContrast()
            loss = loss + crossloss(node_feature_list, node_feature_m_list, self.onlineNet.logit_scale, self.momentumNet.logit_scale)
            criterion = Permutation_Bin_Loss()
            loss_perm = criterion(s_bin, target, *data_dict['ns'])

            loss = loss + 0.1 * loss_perm

            if cfg.PROBLEM.TYPE == '2GM':
                data_dict.update({
                    'perm_mat': x_list3[0],
                    'loss': loss,
                    'ds_mat': None,
                })
        else:
            # directly output the results
            if cfg.PROBLEM.TYPE == '2GM':
                data_dict.update({
                    'perm_mat': x_list3[0],
                    'ds_mat': None,
                })
        return data_dict

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
