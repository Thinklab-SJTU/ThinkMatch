import torch
import itertools
import numpy as np

from models.COMMON.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from src.utils.pad_tensor import pad_tensor
from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.hungarian import hungarian

from src.utils.config import cfg

from src.backbone import *

from src.loss_func import *

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
    def __init__(self, output_dim):
        super(InnerProduct, self).__init__()
        self.d = output_dim

    def _forward(self, X, Y):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        X = torch.nn.functional.normalize(X, dim=-1)
        Y = torch.nn.functional.normalize(Y, dim=-1)
        res = torch.matmul(X, Y.transpose(0, 1))
        return res

    def forward(self, Xs, Ys):
        return [self._forward(X, Y) for X, Y in zip(Xs, Ys)]


class Backbone(CNN):
    def __init__(self):
        super(Backbone, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.COMMON.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.vertex_affinity = InnerProduct(256)
        self.rescale = cfg.PROBLEM.RESCALE
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.COMMON.SOFTMAXTEMP))

        self.projection = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

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

        unary_affs_list = [
            self.vertex_affinity([self.projection(item.x) for item in g_1], [self.projection(item.x) for item in g_2])
            for (g_1, g_2) in lexico_iter(orig_graph_list)
        ]

        # prepare aligned node features for computing contrastive loss
        keypoint_number_list = []  # the number of keypoints in each image pair
        node_feature_list = []  # node features for computing contrastive loss

        node_feature_graph1 = torch.zeros([batch_size, data_dict['gt_perm_mat'].shape[1], node_features.shape[1]],
                                         device=node_features.device)
        node_feature_graph2 = torch.zeros([batch_size, data_dict['gt_perm_mat'].shape[2], node_features.shape[1]],
                                         device=node_features.device)
        # count the available keypoints in number list
        for index in range(batch_size):
            node_feature_graph1[index, :orig_graph_list[0][index].x.shape[0]] = orig_graph_list[0][index].x
            node_feature_graph2[index, :orig_graph_list[1][index].x.shape[0]] = orig_graph_list[1][index].x
            keypoint_number_list.append(torch.sum(data_dict['gt_perm_mat'][index]))
        number = int(sum(keypoint_number_list))  # calculate the number of correspondence

        # pre-align the keypoints for further computing the contrastive loss
        node_feature_graph2 = torch.bmm(data_dict['gt_perm_mat'], node_feature_graph2)
        final_node_feature_graph1 = torch.zeros([number, node_features.shape[1]], device=node_features.device)
        final_node_feature_graph2 = torch.zeros([number, node_features.shape[1]], device=node_features.device)
        count = 0
        for index in range(batch_size):
            final_node_feature_graph1[count: count + int(keypoint_number_list[index])] \
                = node_feature_graph1[index, :int(keypoint_number_list[index])]
            final_node_feature_graph2[count: count + int(keypoint_number_list[index])] \
                = node_feature_graph2[index, :int(keypoint_number_list[index])]
            count += int(keypoint_number_list[index])
        node_feature_list.append(self.projection(final_node_feature_graph1))
        node_feature_list.append(self.projection(final_node_feature_graph2))

        if online == False:
            # output of the momentum network
            return node_feature_list
        elif online == True:
            # output of the online network
            x_list = []
            for unary_affs, (idx1, idx2) in zip(unary_affs_list, lexico_iter(range(num_graphs))):
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                # conduct hungarian matching to get the permutation matrix for evaluation
                x = hungarian(Kp, n_points[idx1], n_points[idx2])
                x_list.append(x)
            return node_feature_list, x_list


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.onlineNet = Backbone()
        self.momentumNet = Backbone()  # initialize the online network and momentum network
        self.momentum = cfg.COMMON.MOMENTUM  # momentum parameter for the momentum network
        self.backbone_params = list(self.onlineNet.backbone_params) # used in train_eval.py
        self.warmup_step = cfg.COMMON.WARMUP_STEP  # warmup steps for the distillation
        self.epoch_iters = cfg.TRAIN.EPOCH_ITERS  # iterations for one epoch, specified by the training dataset

        self.model_pairs = [[self.onlineNet, self.momentumNet]]
        self.copy_params()  # initialize the momentum network

        assert cfg.PROBLEM.TYPE == '2GM'  # only support 2GM problem currently

    def forward(self, data_dict, training=False, iter_num=0, epoch=0):
        # calculate the distillation weight alpha
        if epoch * self.epoch_iters + iter_num >= self.warmup_step:
            alpha = cfg.COMMON.ALPHA
        else:
            alpha = cfg.COMMON.ALPHA * min(1, (epoch * self.epoch_iters + iter_num) / self.warmup_step)

        # output of the online network
        node_feature_list, x_list = self.onlineNet(data_dict)

        if training == True:
            # the momentum network is only using for training
            assert cfg.COMMON.DISTILL == True

            # obtain output of the momentum network
            with torch.no_grad():
                self._momentum_update()
                node_feature_m_list = self.momentumNet(data_dict, online=False)
            # loss function
            contrastloss = Distill_InfoNCE()
            loss = contrastloss(node_feature_list, node_feature_m_list, alpha,
                                self.onlineNet.logit_scale, self.momentumNet.logit_scale)
            crossloss = Distill_QuadraticContrast()
            loss = loss + crossloss(node_feature_list, node_feature_m_list,
                                    self.onlineNet.logit_scale, self.momentumNet.logit_scale)

            if cfg.PROBLEM.TYPE == '2GM':
                data_dict.update({
                    'perm_mat': x_list[0],
                    'loss': loss,
                    'ds_mat': None,
                })
        else:
            # directly output the results
            if cfg.PROBLEM.TYPE == '2GM':
                data_dict.update({
                    'perm_mat': x_list[0],
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
