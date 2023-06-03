import itertools
from torch_sparse import spmm, SparseTensor

from models.BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat, construct_sparse_aff_mat
from src.utils.pad_tensor import pad_tensor
from models.NGM.gnn import GNNLayer, SPGNNLayer, PYGNNLayer
from LinSATNet import linsat_layer
from models.AFAT.sinkhorn_topk import greedy_perm
from src.lap_solvers.hungarian import hungarian

from src.utils.config import cfg

from src.backbone import *

CNN = eval(cfg.BACKBONE)


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.NGM.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.NGM.FEATURE_CHANNEL * 2
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.NGM.SK_TAU
        self.mgm_tau = cfg.NGM.MGM_SK_TAU
        self.sparse = cfg.NGM.SPARSE_MODEL
        self.gnn_layer = cfg.NGM.GNN_LAYER

        if not self.sparse:
            for i in range(self.gnn_layer):
                tau = cfg.NGM.SK_TAU
                if i == 0:
                    gnn_layer = GNNLayer(1, 1,
                                         cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                         sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                else:
                    gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                         cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                         sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                self.add_module('gnn_layer_{}'.format(i), gnn_layer)
        else:
            self.geometric = True
            if self.geometric:
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
            else:
                for i in range(self.gnn_layer):
                    tau = cfg.NGM.SK_TAU
                    if i == 0:
                        gnn_layer = SPGNNLayer(1, 1,
                                               cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                               sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                    else:
                        gnn_layer = SPGNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                               cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                               sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                    self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)

    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        A_src, A_tgt = data_dict['As']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['gt_perm_mat'].shape[0]
        num_graphs = len(images)

        global_list = []
        orig_graph_list = []
        node_feature_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = torch.cat((U, F), dim=1)
            node_feature_list.append(node_features.detach())
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []

        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            if not self.sparse:
                kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
                K = construct_aff_mat(Ke, Kp, kro_G, kro_H)
                if num_graphs == 2: data_dict['aff_mat'] = K

                if cfg.NGM.FIRST_ORDER:
                    emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
                else:
                    emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

                if cfg.NGM.POSITIVE_EDGES:
                    A = (K > 0).to(K.dtype)
                else:
                    A = (K != 0).to(K.dtype)

                emb_K = K.unsqueeze(-1)

                # NGM qap solver
                for i in range(self.gnn_layer):
                    gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                    emb_K, emb = gnn_layer(A, emb_K, emb, n_points[idx1], n_points[idx2])
            else:
                kro_G, kro_H = data_dict['KGHs_sparse'] if num_graphs == 2 else data_dict['KGHs_sparse']['{},{}'.format(idx1, idx2)]
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
                K_value, row_idx, col_idx = construct_sparse_aff_mat(Ke, Kp, kro_G, kro_H)

                if cfg.NGM.FIRST_ORDER:
                    emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
                else:
                    emb = torch.ones(cfg.BATCH_SIZE, Kp.shape[1] * Kp.shape[2], 1, device=K_value.device)

                # NGM qap solver
                if self.geometric:
                    adj = SparseTensor(row=row_idx.long(), col=col_idx.long(), value=K_value,
                                       sparse_sizes=(Kp.shape[1] * Kp.shape[2], Kp.shape[1] * Kp.shape[2]))
                    for i in range(self.gnn_layer):
                        gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                        emb = gnn_layer(adj, emb, n_points[idx1], n_points[idx2])
                else:
                    K_index = torch.cat((row_idx.unsqueeze(0), col_idx.unsqueeze(0)), dim=0).long()
                    A_value = torch.ones(K_value.shape, device=K_value.device)
                    tmp = torch.ones([Kp.shape[1] * Kp.shape[2]], device=K_value.device).unsqueeze(-1)
                    normed_A_value = 1 / torch.flatten(
                        spmm(K_index, A_value, Kp.shape[1] * Kp.shape[2], Kp.shape[1] * Kp.shape[2], tmp))
                    A_index = torch.linspace(0, Kp.shape[1] * Kp.shape[2] - 1, Kp.shape[1] * Kp.shape[2]).unsqueeze(0)
                    A_index = torch.repeat_interleave(A_index, 2, dim=0).long().to(K_value.device)

                    for i in range(self.gnn_layer):
                        gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                        emb = gnn_layer(K_value, K_index, normed_A_value, A_index, emb, n_points[idx1], n_points[idx2])

            v = self.classifier(emb)
            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)

            gt_ks = torch.tensor(
                [torch.sum(data_dict['gt_perm_mat'][i]) for i in range(data_dict['gt_perm_mat'].shape[0])],
                dtype=torch.float32, device=s.device)

            ss_out = torch.zeros(s.shape, dtype=torch.float32, device=s.device)
            for ii in range(s.shape[0]):
                p0 = n_points[idx1][ii]
                p1 = n_points[idx2][ii]
                constraint = torch.zeros(p0 + p1, p0 * p1,  ### p0 + p1 + 1
                                         dtype=torch.float32, device=s.device)
                b = torch.zeros(p0 + p1, dtype=torch.float32, device=s.device)  ### p0 + p1 + 1

                for cons_id in range(p0 + p1):
                    tmp = torch.zeros(p0, p1, dtype=torch.float32, device=s.device)
                    if cons_id < p0:
                        tmp[cons_id, 0:p1] = 1
                    else:
                        tmp[0:p0, cons_id - p0] = 1
                    constraint[cons_id, :] = tmp.reshape(-1)
                    b[cons_id] = 1

                E = torch.ones(1, p0 * p1, dtype=torch.float32, device=s.device)
                f = torch.zeros(1, dtype=torch.float32, device=s.device)
                f[0] = gt_ks[ii]

                ### tmp = torch.ones(p0, p1, dtype=torch.float32, device=s.device)
                ### constraint[-1, :] = tmp.reshape(-1)
                ### b[-1] = gt_ks[ii]

                input = s[ii, 0:p0, 0:p1].reshape(-1)
                ss_out[ii, 0:p0, 0:p1] = linsat_layer(input, A=constraint, b=b, E=E, f=f, max_iter=2 * cfg.NGM.SK_ITER_NUM,
                                                 tau=self.tau).reshape(p0, p1)

            x = hungarian(ss_out, n_points[idx1], n_points[idx2])
            top_indices = torch.argsort(x.mul(ss_out).reshape(x.shape[0], -1), descending=True, dim=-1)
            x = torch.zeros(ss_out.shape, device=ss_out.device)
            x = greedy_perm(x, top_indices, gt_ks)
            s_list.append(ss_out)
            x_list.append(x)
            indices.append((idx1, idx2))

        if cfg.PROBLEM.TYPE == '2GM' or cfg.PROBLEM.TYPE == 'IMT':
            data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0],
            })
        elif cfg.PROBLEM.TYPE == 'MGM':
            data_dict.update({
                'ds_mat_list': mgm_s_list,
                'perm_mat_list': mgm_x_list,
                'graph_indices': indices,
            })

        return data_dict
