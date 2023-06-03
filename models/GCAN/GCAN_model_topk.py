import itertools

from models.GCAN.positional_encoding_layer import positional_encoding_layer
from models.GCAN.GCA_module import GCA_module
from src.feature_align import feature_align
from src.utils.pad_tensor import pad_tensor
from src.lap_solvers.sinkhorn import Sinkhorn
from models.PCA.affinity_layer import Affinity
from src.gconv import Siamese_Gconv
from models.AFAT.k_pred_net import Encoder, TensorNetworkModule, DenseAttentionModule
from torch_geometric import utils as geometric_util
from models.AFAT.sinkhorn_topk import soft_topk, greedy_perm
from src.lap_solvers.hungarian import hungarian
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

        self.trainings = True
        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.GCAN.SK_TAU
        self.univ_size = cfg.AFA.UNIV_SIZE
        self.k_factor = cfg.AFA.K_FACTOR
        self.k_gnn_layer = cfg.AFA.K_GNN_LAYER
        self.regression = cfg.AFA.REGRESSION
        self.afau = cfg.AFA.AFAU
        self.mean_k = cfg.AFA.MEAN_K

        self.sinkhorn = Sinkhorn(max_iter=cfg.GCAN.SK_ITER_NUM, tau=self.tau, epsilon=cfg.GCAN.SK_EPSILON)

        if self.regression:
            self.k_params_id = []
            if self.afau:
                self.encoder_k = Encoder()
                self.k_params_id += [id(item) for item in self.encoder_k.parameters()]

                self.maxpool = nn.MaxPool1d(kernel_size=self.univ_size)
                self.final_row = nn.Sequential(
                    nn.Linear(self.univ_size, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()
                )

                self.final_col = nn.Sequential(
                    nn.Linear(self.univ_size, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()
                )

                self.k_params_id += [id(item) for item in self.final_row.parameters()]
                self.k_params_id += [id(item) for item in self.final_col.parameters()]

                self.k_params = [
                    {'params': self.encoder_k.parameters()},
                    {'params': self.final_row.parameters()},
                    {'params': self.final_col.parameters()}
                ]
            else:
                self.k_params = []
                for i in range(self.k_gnn_layer):
                    if i == 0:
                        gnn_layer_k = Siamese_Gconv(cfg.GCAN.FEATURE_CHANNEL * 2, cfg.AFA.REG_HIDDEN_FEAT)
                    else:
                        gnn_layer_k = Siamese_Gconv(cfg.AFA.REG_HIDDEN_FEAT, cfg.AFA.REG_HIDDEN_FEAT)
                    self.add_module('_k_gnn_layer_{}'.format(i), gnn_layer_k)
                    self.add_module('_affinity_{}'.format(i), Affinity(cfg.AFA.REG_HIDDEN_FEAT))

                    self.k_params_id += [id(item) for item in eval('self._k_gnn_layer_{}'.format(i)).parameters()]
                    self.k_params_id += [id(item) for item in eval('self._affinity_{}'.format(i)).parameters()]
                    self.k_params.append({'params': eval('self._k_gnn_layer_{}'.format(i)).parameters()})
                    self.k_params.append({'params': eval('self._affinity_{}'.format(i)).parameters()})

                    if i == self.k_gnn_layer - 2:  # only second last layer will have cross-graph module
                        self.add_module('_cross_graph_{}'.format(i),
                                        nn.Linear(cfg.AFA.REG_HIDDEN_FEAT * 2, cfg.AFA.REG_HIDDEN_FEAT))
                        self.k_params_id += [id(item) for item in eval('self._cross_graph_{}'.format(i)).parameters()]
                        self.k_params.append({'params': eval('self._cross_graph_{}'.format(i)).parameters()})

                self._attn_pool_1 = DenseAttentionModule(cfg.AFA.REG_HIDDEN_FEAT)
                self._attn_pool_2 = DenseAttentionModule(cfg.AFA.REG_HIDDEN_FEAT)

                self.k_params_id += [id(item) for item in self._attn_pool_1.parameters()]
                self.k_params_id += [id(item) for item in self._attn_pool_2.parameters()]
                self.k_params.append({'params': self._attn_pool_1.parameters()})
                self.k_params.append({'params': self._attn_pool_2.parameters()})

                self._ntnet = TensorNetworkModule(filters=cfg.AFA.REG_HIDDEN_FEAT, tensor_neurons=cfg.AFA.TN_NEURONS)

                self.k_params_id += [id(item) for item in self._ntnet.parameters()]
                self.k_params.append({'params': self._ntnet.parameters()})

                self._final_reg = nn.Sequential(
                    nn.Linear(cfg.AFA.TN_NEURONS, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Sigmoid()
                )

                self.k_params_id += [id(item) for item in self._final_reg.parameters()]
                self.k_params.append({'params': self._final_reg.parameters()})


    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        num_graphs = len(images)
        batch_size = data_dict['batch_size']
        A_src, A_tgt = data_dict['As']

        global_avg_list = []
        global_max_list = []
        orig_graph_list = []
        node_feature_list = []
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
            node_feature_list.append(node_features.detach())
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
            Kp = torch.stack(pad_tensor(cross_attention_list), dim=0)

            s = Kp
            ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)

            gt_ks = torch.tensor(
                [torch.sum(data_dict['gt_perm_mat'][i]) for i in range(data_dict['gt_perm_mat'].shape[0])],
                dtype=torch.float32, device=ss.device)

            min_point_list = [int(min(n_points[0][b], n_points[1][b])) for b in range(data_dict['gt_perm_mat'].shape[0])]

            min_point_tensor = torch.tensor(min_point_list, dtype=torch.float32, device=ss.device)

            if self.regression:
                dummy_row = self.univ_size - ss.shape[1]
                dummy_col = self.univ_size - ss.shape[2]
                assert dummy_row >= 0 and dummy_col >= 0

                if not self.afau:

                    emb1 = torch.zeros((batch_size, int(torch.max(n_points[idx1])), 2 * cfg.GCAN.FEATURE_CHANNEL),
                                       dtype=torch.float32, device=ss.device)

                    emb2 = torch.zeros((batch_size, int(torch.max(n_points[idx2])), 2 * cfg.GCAN.FEATURE_CHANNEL),
                                       dtype=torch.float32, device=ss.device)

                    mask1 = torch.zeros((batch_size, int(torch.max(n_points[idx1]))),
                                        dtype=torch.float32, device=ss.device)
                    mask2 = torch.zeros((batch_size, int(torch.max(n_points[idx2]))),
                                        dtype=torch.float32, device=ss.device)

                    total_nodes_1 = 0
                    total_nodes_2 = 0
                    for b in range(batch_size):
                        emb1_one = node_feature_list[0][total_nodes_1: total_nodes_1 + n_points[idx1][b]]
                        emb1[b, 0: n_points[idx1][b]] = emb1_one
                        emb2_one = node_feature_list[1][total_nodes_2: total_nodes_2 + n_points[idx2][b]]
                        emb2[b, 0: n_points[idx2][b]] = emb2_one
                        total_nodes_1 += n_points[idx1][b]
                        total_nodes_2 += n_points[idx2][b]
                        mask1_one = torch.ones((n_points[idx1][b]),
                                               dtype=torch.float32, device=ss.device)
                        mask2_one = torch.ones((n_points[idx2][b]),
                                               dtype=torch.float32, device=ss.device)
                        mask1[b, 0: n_points[idx1][b]] = mask1_one
                        mask2[b, 0: n_points[idx2][b]] = mask2_one

                    for i in range(self.k_gnn_layer):
                        gnn_layer = getattr(self, '_k_gnn_layer_{}'.format(i))
                        emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                        affinity = getattr(self, '_affinity_{}'.format(i))
                        s = affinity(emb1, emb2)
                        s = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)

                        if i == self.k_gnn_layer - 2:
                            cross_graph = getattr(self, '_cross_graph_{}'.format(i))
                            new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                            new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                            emb1 = new_emb1
                            emb2 = new_emb2

                    global_emb1 = self._attn_pool_1(emb1, mask1)
                    global_emb2 = self._attn_pool_2(emb2, mask2)

                    sim = self._ntnet(global_emb1, global_emb2)
                    ks = self._final_reg(sim).squeeze(-1)

                else:
                    init_row_emb = torch.zeros((batch_size, int(torch.max(n_points[idx1])), self.univ_size),
                                               dtype=torch.float32, device=ss.device)

                    init_col_emb = torch.zeros((batch_size, int(torch.max(n_points[idx2])), self.univ_size),
                                               dtype=torch.float32, device=ss.device)

                    for b in range(batch_size):
                        index = torch.linspace(0, n_points[idx2][b].item() - 1, n_points[idx2][b].item(),
                                               dtype=torch.long, device=ss.device).unsqueeze(1)
                        init_col_emb_one = torch.zeros(int(torch.max(n_points[idx2])), self.univ_size,
                                                       dtype=torch.float32, device=ss.device).scatter_(1, index, 1)
                        init_col_emb[b] = init_col_emb_one

                    out_emb_row, out_emb_col = self.encoder_k(init_row_emb, init_col_emb, ss.detach())
                    out_emb_row = torch.nn.functional.pad(out_emb_row, (0, 0, 0, dummy_row),
                                                          value=float('-inf')).permute(0, 2, 1)
                    out_emb_col = torch.nn.functional.pad(out_emb_col, (0, 0, 0, dummy_col),
                                                          value=float('-inf')).permute(0, 2, 1)
                    global_row_emb = self.maxpool(out_emb_row).squeeze(-1)
                    global_col_emb = self.maxpool(out_emb_col).squeeze(-1)
                    k_row = self.final_row(global_row_emb).squeeze(-1)
                    k_col = self.final_col(global_col_emb).squeeze(-1)
                    if self.mean_k:
                        ks = (k_row + k_col) / 2
                    else:
                        ks = k_row
            else:
                ks = gt_ks / min_point_tensor

            if self.trainings:
                _, ss = soft_topk(ss, gt_ks.view(-1), cfg.GCAN.SK_ITER_NUM, self.tau, n_points[idx1], n_points[idx2],
                                  True)
            else:
                _, ss = soft_topk(ss, ks.view(-1) * min_point_tensor, cfg.GCAN.SK_ITER_NUM, self.tau, n_points[idx1],
                                  n_points[idx2], True)

            supervised_ks = gt_ks / min_point_tensor

            if self.regression:
                ks_loss = torch.nn.functional.mse_loss(ks, supervised_ks) * self.k_factor
                ks_error = torch.nn.functional.l1_loss(ks * min_point_tensor, gt_ks)
            else:
                ks_loss = 0.
                ks_error = 0.

            x = hungarian(ss, n_points[idx1], n_points[idx2])
            top_indices = torch.argsort(x.mul(ss).reshape(x.shape[0], -1), descending=True, dim=-1)
            x = torch.zeros(ss.shape, device=ss.device)
            x = greedy_perm(x, top_indices, ks.view(-1) * min_point_tensor)

            s_list.append(ss)
            x_list.append(x)
            indices.append((idx1, idx2))
        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0],
                'ks_loss': ks_loss,
                'ks_error': ks_error
            })

        return data_dict
