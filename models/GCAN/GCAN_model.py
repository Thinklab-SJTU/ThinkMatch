import itertools

from models.GCAN.positional_encoding_layer import positional_encoding_layer
from models.GCAN.GCA_module import GCA_module
from src.feature_align import feature_align
from src.utils.pad_tensor import pad_tensor, pad_tensor_varied
# from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.ILP import ILP_solver
from src.lap_solvers.sinkhorn import Sinkhorn as Sinkhorn_varied
from torch_geometric import utils as geometric_util
from scipy.linalg import block_diag
import numpy as np

from src.utils.config import cfg

from src.backbone_gcan import *
CNN = eval(cfg.BACKBONE)

def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

def pair_split(node_features, ns):
    batch_feature_src = []
    batch_feature_tgt = []
    partitions = []
    idx = 0
    for i in range(len(ns)):
        partitions.append(node_features[idx:idx + ns[i], :])
        idx = idx + ns[i]
    for i in range(0, len(ns), 2):
        batch_feature_src.append(partitions[i])
        batch_feature_tgt.append(partitions[i + 1])
    return batch_feature_src,  batch_feature_tgt

def get_graph_feature(batch_graphs_src, ns_src, batch_graphs_tgt, ns_tgt):
    adjacency_matrixs_list = []
    for idx in range(len(batch_graphs_src)):
        adjacency_src = geometric_util.to_dense_adj(
            batch_graphs_src[idx].edge_index, max_num_nodes=ns_src[idx]).squeeze().cpu()
        adjacency_matrixs_list.append(np.array(adjacency_src))
        adjacency_tgt = geometric_util.to_dense_adj(
            batch_graphs_tgt[idx].edge_index, max_num_nodes=ns_tgt[idx]).squeeze().cpu()
        adjacency_matrixs_list.append(np.array(adjacency_tgt))
    adjacency_matrixs = block_diag(*adjacency_matrixs_list).astype('float32')

    return torch.tensor(adjacency_matrixs).cuda()


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.positional_encoding = positional_encoding_layer(input_node_dim=cfg.GCAN.FEATURE_CHANNEL * 2)
        self.global_state_dim = cfg.GCAN.FEATURE_CHANNEL * 2
        cross_parameters = [self.global_state_dim, self.positional_encoding.num_node_features]
        self_parameters = [cfg.GCAN.NODE_HIDDEN_SIZE[-1]*2, int(cfg.GCAN.NODE_HIDDEN_SIZE[-1]/4), 4]
        self.GCA_module1 = GCA_module(cross_parameters,self_parameters)
        self.GCA_module2 = GCA_module(cross_parameters, self_parameters)

        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.GCAN.SK_TAU

        self.sinkhorn = Sinkhorn_varied(max_iter=cfg.GCAN.SK_ITER_NUM, tau=self.tau, epsilon=cfg.GCAN.SK_EPSILON)

    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        num_graphs = len(images)

        global_avg_list = []
        global_max_list = []
        orig_graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_avg_list.append(self.final_layers_avg(edges).reshape((nodes.shape[0], -1)))
            global_max_list.append(self.final_layers_max(edges).reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = torch.cat((U, F), dim=1)
            graph.x = node_features

            graph = self.positional_encoding(graph)
            orig_graph = graph.to_data_list()
            orig_graph_list.append(orig_graph)

        ns_src, ns_tgt = n_points
        P_src, P_tgt = points
        global_avg_src, global_avg_tgt = global_avg_list
        global_max_src, global_max_tgt = global_max_list
        batch_graphs_src, batch_graphs_tgt = orig_graph_list
        cross_attention_list = []
        ### global weights
        global_max_weights = torch.cat([global_max_src, global_max_tgt], axis=-1)
        global_max_weights = normalize_over_channels(global_max_weights)
        global_avg_weights = torch.cat([global_avg_src, global_avg_tgt], axis=-1)
        global_avg_weights = normalize_over_channels(global_avg_weights)
        ### src node features
        batch_feature_src = [item.x for item in batch_graphs_src]
        ### tgt node features
        batch_feature_tgt = [item.x for item in batch_graphs_tgt]
        ### adjacency
        adjacency_matrixs = get_graph_feature(batch_graphs_src, ns_src, batch_graphs_tgt, ns_tgt)
        ###GCAN
        cross_attention, node_features, ns = self.GCA_module1(batch_feature_src, batch_feature_tgt, global_avg_weights, global_max_weights, ns_src, ns_tgt,adjacency_matrixs)
        cross_attention_list = cross_attention_list + cross_attention
        batch_feature_src, batch_feature_tgt = pair_split(node_features, ns)
        cross_attention, node_features, ns = self.GCA_module2(batch_feature_src, batch_feature_tgt, global_avg_weights,
                                                          global_max_weights, ns_src, ns_tgt, adjacency_matrixs)
        cross_attention_list = [ori + 0.1*new for ori, new in zip(cross_attention_list, cross_attention)]

        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []

        for idx1, idx2 in lexico_iter(range(num_graphs)):
            if True:
                Kp = torch.stack(pad_tensor_varied(cross_attention_list,dummy=0), dim=0)
            else:
                Kp = torch.stack(pad_tensor(cross_attention_list), dim=0)

            s = Kp
            if self.training:
                if True:
                    ss = self.sinkhorn(s, n_points[idx1]+1, n_points[idx2]+1, dummy_row=True)
                    ilp_x = ILP_solver(ss, n_points[idx1] + 1, n_points[idx2] + 1, dummy=True)
                else:
                    ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                    ilp_x = ILP_solver(ss, n_points[idx1], n_points[idx2])
                s_list.append(ss)
                x_list.append(ilp_x)
                indices.append((idx1, idx2))
            else:

                if True:
                    ss = self.sinkhorn(s, n_points[idx1]+1, n_points[idx2]+1, dummy_row=True)
                    ilp_x = ILP_solver(ss, n_points[idx1] + 1, n_points[idx2] + 1, dummy=True)
                else:
                    ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                    ilp_x = ILP_solver(ss, n_points[idx1], n_points[idx2])

                s_list.append(ss)
                x_list.append(ilp_x)
                indices.append((idx1, idx2))
        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0]
            })

        return data_dict
