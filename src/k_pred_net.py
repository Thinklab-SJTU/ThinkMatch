import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils.config import cfg


class Encoder(nn.Module):
    """
    AFA-U graph attention module to generate bipartite node embeddings.
    """
    model_params = {
        'embedding_dim': cfg.AFA.UNIV_SIZE,
        'head_num': cfg.AFA.HEAD_NUM,
        'qkv_dim': cfg.AFA.KQV_DIM,
        'ff_hidden_dim': cfg.AFA.FF_HIDDEN_DIM,
        'ms_hidden_dim': cfg.AFA.MS_HIDDEN_DIM,
        'ms_layer1_init': cfg.AFA.MS_LAYER1_INIT,
        'ms_layer2_init': cfg.AFA.MS_LAYER2_INIT,
        'sqrt_qkv_dim': math.sqrt(cfg.AFA.KQV_DIM),
    }

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(**self.model_params)])

    def forward(self, row_emb, col_emb, cost_mat):
        """
        Making a forward propagation pass to generate bipartite node embeddings.

        :param row_emb: Initial node features of the source graph.
        :param col_emb: Initial node features of the target graph.
        :param cost_mat: Edge weights of the bipartite graph.
        :return row_emb, col_emb: Aggregated node embeddings of the source graph, aggregated node embeddings of the target graph.
        """
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb


class EncoderLayer(nn.Module):
    """
    Encoding layer in AFA-U graph attention module.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        """
        Making a forward propagation pass in AFA-U graph attention module.

        :param row_emb: Initial node features of the source graph.
        :param col_emb: Initial node features of the target graph.
        :param cost_mat: Edge weights of the bipartite graph.
        :return row_emb, col_emb: Aggregated node embeddings of the source graph, aggregated node embeddings of the target graph.
        """

        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    """
    Encoding block for the source graph/the target graph in AFA-U graph attention module.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = CrossSet_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        """
        Making a forward propagation pass for the source graph/the target graph in AFA-U graph attention module.

        :param row_emb: Initial node features of the source graph.
        :param col_emb: Initial node features of the target graph.
        :param cost_mat: Edge weights of the bipartite graph.
        :return out: Aggregated node embeddings.
        """
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out = self.add_n_normalization_2(out1, out2)

        return out
        # shape: (batch, row_cnt, embedding)


class AddAndInstanceNormalization(nn.Module):
    """
    Add and instance normalization in AFA-U graph attention module.
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        """
        Making a forward propagation pass to add and normalize 2 instances.

        :param input1: Input node features of the source graph.
        :param input2: Input node features of the target graph.
        :return out: Added and normalized node embeddings.
        """
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        out = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return out


class FeedForward(nn.Module):
    """
    Feed forward operation in AFA-U graph attention module.
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        """
        Making a forward propagation pass.

        :param input1: Input node features.
        :return out: Refined node features.
        """
        # input.shape: (batch, problem, embedding)
        out = self.W2(F.relu(self.W1(input1)))

        return out


class CrossSet_MultiHeadAttention(nn.Module):
    """
    Cross-set multi-head attention layer in AFA-U graph attention module.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

    def forward(self, q, k, v, cost_mat):
        """
        Making a forward propagation pass in cross-set multi-head attention layer.

        :param k: Key vectors in attention mechanism.
        :param q: Query vectors in attention mechanism.
        :param v: Value vectors in attention mechanism.
        :param cost_mat: Edge weights of the bipartite graph.
        :return out: Refined node features.
        """
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt)

        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)

        two_scores_transposed = two_scores.transpose(1,2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1,2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)

        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        out1 = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out1.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out


class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module in AFA-I module to calculate similarity vector.
    """
    def __init__(self, filters, tensor_neurons):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.filters_3 = filters
        self.tensor_neurons = tensor_neurons
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.filters_3, self.filters_3, self.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 2*self.filters_3))
        self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = len(embedding_1)
        scoring = torch.matmul(embedding_1, self.weight_matrix.view(self.filters_3, -1))
        scoring = scoring.view(batch_size, self.filters_3, -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, embedding_2.view(batch_size, self.filters_3, 1)).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        return scores


class DenseAttentionModule(torch.nn.Module):
    """
    SimGNN Dense Attention Module in AFA-I module to make a pass on graph.
    """

    def __init__(self, filters):
        """
        :param args: Arguments object.
        """
        super(DenseAttentionModule, self).__init__()
        self.filters_3 = filters
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.filters_3, self.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, mask=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param mask: Mask matrix indicating the valid nodes for each graph.
        :return representation: A graph level representation matrix.
        """
        B, N, _ = x.size()

        if mask is not None:
            num_nodes = mask.view(B, N).sum(dim=1).unsqueeze(-1)
            mean = x.sum(dim=1) / num_nodes.to(x.dtype)
        else:
            mean = x.mean(dim=1)

        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))

        koefs = torch.sigmoid(torch.matmul(x, transformed_global.unsqueeze(-1)))
        weighted = koefs * x

        if mask is not None:
            weighted = weighted * mask.view(B, N, 1).to(x.dtype)

        return weighted.sum(dim=1)


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed