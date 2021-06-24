import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lap_solvers.sinkhorn import Sinkhorn

from collections import Iterable

class GNNLayer(nn.Module):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features,
                 sk_channel=0, sk_iter=20, sk_tau=0.05, edge_emb=False):
        super(GNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.sk = Sinkhorn(sk_iter, sk_tau)
            self.classifier = nn.Linear(self.out_nfeat, self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.sk = self.classifier = None

        if edge_emb:
            self.e_func = nn.Sequential(
                nn.Linear(self.in_efeat + self.in_nfeat, self.out_efeat),
                nn.ReLU(),
                nn.Linear(self.out_efeat, self.out_efeat),
                nn.ReLU()
            )
        else:
            self.e_func = None

        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            #nn.Linear(self.in_nfeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            #nn.Linear(self.out_nfeat // self.out_efeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
        )

        self.n_self_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU()
        )

    def forward(self, A, W, x, n1=None, n2=None, norm=True):
        """
        :param A: adjacent matrix in 0/1 (b x n x n)
        :param W: edge feature tensor (b x n x n x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        if self.e_func is not None:
            W1 = torch.mul(A.unsqueeze(-1), x.unsqueeze(1))
            W2 = torch.cat((W, W1), dim=-1)
            W_new = self.e_func(W2)
        else:
            W_new = W

        if norm is True:
            A = F.normalize(A, p=1, dim=2)

        x1 = self.n_func(x)
        x2 = torch.matmul((A.unsqueeze(-1) * W_new).permute(0, 3, 1, 2), x1.unsqueeze(2).permute(0, 3, 1, 2)).squeeze(-1).transpose(1, 2)
        x2 += self.n_self_func(x)

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(x2)
            n1_rep = torch.repeat_interleave(n1, self.sk_channel, dim=0)
            n2_rep = torch.repeat_interleave(n2, self.sk_channel, dim=0)
            x4 = x3.permute(0,2,1).reshape(x.shape[0] * self.sk_channel, n2.max(), n1.max()).transpose(1, 2)
            x5 = self.sk(x4, n1_rep, n2_rep, dummy_row=True).transpose(2, 1).contiguous()

            x6 = x5.reshape(x.shape[0], self.sk_channel, n1.max() * n2.max()).permute(0, 2, 1)
            x_new = torch.cat((x2, x6), dim=-1)
        else:
            x_new = x2

        return W_new, x_new


class HyperGNNLayer(nn.Module):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features, orders=3, eps=1e-10,
                 sk_channel=False, sk_iter=20, sk_tau=0.05):
        super(HyperGNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.eps = eps
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.sk = Sinkhorn(sk_iter, sk_tau)
            self.classifier = nn.Linear(self.out_nfeat, self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.sk = self.classifier = None

        # used by forward_dense
        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU(),
        )

        # used by forward_sparse
        for i in range(2, orders + 1):
            n_func = nn.Sequential(
                nn.Linear(self.in_nfeat, self.out_nfeat),
                nn.ReLU(),
                nn.Linear(self.out_nfeat, self.out_nfeat),
                nn.ReLU()
            )
            self.add_module('n_func_{}'.format(i), n_func)

        self.n_self_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU()
        )

    def forward(self, A, W, x, n1=None, n2=None, weight=None, norm=True):
        """wrapper function of forward (support dense/sparse)"""
        if not isinstance(A, Iterable):
            A = [A]
            W = [W]

        W_new = []
        if weight is None:
            weight = [1.] * len(A)
        assert len(weight) == len(A)
        for i, (_A, _W, w) in enumerate(zip(A, W, weight)):
            if type(_W) is tuple or (type(_W) is torch.Tensor and _W.is_sparse):
                _W_new, _x = self.forward_sparse(_A, _W, x, norm)
            else:
                _W_new, _x = self.forward_dense(_A, _W, x, norm)
            if i == 0:
                x2 = _x * w
            else:
                x2 += _x * w
            W_new.append(_W_new)

        x2 += self.n_self_func(x)

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(x2)
            n1_rep = torch.repeat_interleave(n1, self.sk_channel, dim=0)
            n2_rep = torch.repeat_interleave(n2, self.sk_channel, dim=0)
            x4 = x3.permute(0,2,1).reshape(x.shape[0] * self.sk_channel, n2.max(), n1.max()).transpose(1, 2)
            x5 = self.sk(x4, n1_rep, n2_rep, dummy_row=True).transpose(2, 1).contiguous()
            x6 = x5.reshape(x.shape[0], self.sk_channel, n1.max() * n2.max()).permute(0, 2, 1)
            x_new = torch.cat((x2, x6), dim=-1)
        else:
            x_new = x2

        return W_new, x_new

    def forward_sparse(self, A, W, x, norm=True):
        """
        :param A: adjacent tensor in 0/1 (b x {n x ... x n})
        :param W: edge feature tensor (b x {n x ... x n} x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        order = len(A.shape) - 1

        if type(W) is tuple:
            W_ind, W_val = W
        elif type(W) is torch.Tensor and W.is_sparse:
            W_ind = W._indices()
            W_val = W._values()
        else:
            raise ValueError('Unknown datatype {}'.format(type(W)))

        W_new_val = W_val
        if norm is True:
            assert not A.is_sparse, 'sparse normalization is currently not supported'
            A_sum = torch.sum(A, dim=tuple(range(2, order + 1)), keepdim=True)
            A = A / A_sum.expand_as(A)
            A[torch.isnan(A)] = 0

        if not A.is_sparse:
            A = A.to_sparse()

        assert A._values().shape[0] == W_new_val.shape[0]
        assert len(W_new_val.shape) == 2

        n_func = getattr(self, 'n_func_{}'.format(order))
        x1 = n_func(x)

        tp_val = torch.mul(A._values().unsqueeze(-1), W_new_val)
        for i in range(order - 1):
            tp_val = x1[W_ind[0, :], W_ind[-1-i, :], :] * tp_val

        assert torch.all(W_ind == A._indices())

        x_new = torch.zeros_like(x1)
        x_new.index_put_((W_ind[0, :], W_ind[1, :]), tp_val, True)

        return (W_ind, W_new_val), x_new


    def forward_dense(self, A, W, x, norm=True):
        """
        :param A: adjacent tensor in 0/1 (b x {n x ... x n})
        :param W: edge feature tensor (b x {n x ... x n} x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        order = len(A.shape) - 1
        W_new = W

        if norm is True:
            A_sum = torch.sum(A, dim=tuple(range(2, order + 1)), keepdim=True)
            A = A / A_sum.expand_as(A)
            A[torch.isnan(A)] = 0

        x1 = self.n_func(x)

        x_new = torch.mul(A.unsqueeze(-1), W_new)
        for i in range(order - 1):
            x1_shape = [x1.shape[0]] + [1] * (order - 1 - i) + list(x1.shape[1:])
            x_new = torch.sum(torch.mul(x_new, x1.view(*x1_shape)), dim=-2)

        return W_new, x_new


class HyperConvLayer(nn.Module):
    """
    Hypergarph convolutional layer inspired by "Dynamic Hypergraph Neural Networks"
    """
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features, eps=0.0001,
                 sk_channel=False, sk_iter=20, voting_alpha=20):
        super(HyperConvLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.eps = eps
        if sk_channel:
            assert out_node_features == out_edge_features + 1
            self.out_nfeat = out_node_features - 1
            self.sk = Sinkhorn(sk_iter, 1 / voting_alpha)
            self.classifier = nn.Linear(self.out_efeat, 1)
        else:
            assert out_node_features == out_edge_features
            self.out_nfeat = out_node_features
            self.sk = self.classifier = None

        self.ne_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_efeat),
            nn.ReLU())

        self.e_func = nn.Sequential(
            nn.Linear(self.out_efeat + self.in_efeat, self.out_efeat),
            nn.ReLU()
        )

        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat + self.out_efeat, self.out_nfeat),
            nn.ReLU()
        )

    def forward(self, H, E, x, n1=None, n2=None, norm=None):
        """
        :param H: connectivity (b x n x e)
        :param E: (hyper)edge feature (b x e x f)
        :param x: node feature (b x n x f)
        :param n1: number of nodes in graph1
        :param n2: number of nodes in graph2
        :param norm: do normalization (only supports dense tensor)
        :return: new edge feature, new node feature
        """
        H_node_sum = torch.sum(H, dim=1, keepdim=True)
        H_node_norm = H / H_node_sum
        H_node_norm[torch.isnan(H_node_norm)] = 0
        H_edge_sum = torch.sum(H, dim=2, keepdim=True)
        H_edge_norm = H / H_edge_sum
        H_edge_norm[torch.isnan(H_edge_norm)] = 0

        x_to_E = torch.bmm(H_node_norm.transpose(1, 2), self.ne_func(x))
        new_E = self.e_func(torch.cat((x_to_E, E), dim=-1))
        E_to_x = torch.bmm(H_edge_norm, new_E)
        new_x = self.n_func(torch.cat((E_to_x, x), dim=-1))

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(new_x)
            x5 = self.sk(x3.view(x.shape[0], n2.max(), n1.max()).transpose(1, 2), n1, n2, dummy_row=True).transpose(2, 1).contiguous()

            new_x = torch.cat((new_x, x5.view(x.shape[0], n1.max() * n2.max(), -1)), dim=-1)

        return new_E, new_x


class Attention(nn.Module):
    """
    Attention module
    """
    def __init__(self, feat1, feat2, hid_feat):
        super(Attention, self).__init__()
        self.in_feat1 = feat1
        self.in_feat2 = feat2
        self.hid_feat = hid_feat
        self.linear1 = nn.Linear(self.in_feat1, self.hid_feat)
        self.linear2 = nn.Linear(self.in_feat2, self.hid_feat)
        self.v = nn.Parameter(torch.empty(self.hid_feat).uniform_(-1 / self.hid_feat, 1 / self.hid_feat))

    def forward(self, t1, t2, H):
        """
        :param t1: tensor1 (b x n1 x f)
        :param t2: tensor2 (b x n2 x f)
        :param H: indicator tensor (sparse b x n1 x n2)
        """
        H_idx = H._indices()
        H_data = H._values()
        sparse_mask = torch.sparse_coo_tensor(H_idx, H_data.unsqueeze(-1).expand(-1, self.hid_feat), H.shape + (self.hid_feat,)).coalesce()
        x = (self.linear1(t1).unsqueeze(2) + self.linear2(t2).unsqueeze(1)).sparse_mask(sparse_mask) #+ t2W2.sparse_mask(sparse_mask)
        v = self.v.view(-1, 1)
        w_data = torch.matmul(x._values(), v).squeeze()
        w_softmax_data = torch.empty_like(w_data)
        for b in range(x.shape[0]):
            mask_b = (H_idx[0] == b)
            for r in range(x.shape[1]):
                mask_r = (H_idx[1] == r)
                mask = mask_b * mask_r
                w_softmax_data[mask] = nn.functional.softmax(w_data[mask], dim=0)
        w_softmax = torch.sparse_coo_tensor(H_idx, w_softmax_data, H.shape)
        return w_softmax
