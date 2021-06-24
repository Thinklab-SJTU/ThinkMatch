import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math


class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = Parameter(Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X, self.A)
        #M = torch.matmul(X, (self.A + self.A.transpose(0, 1)) / 2)
        M = torch.matmul(M, Y.transpose(1, 2))
        return M

class AffinityInp(nn.Module):
    """
    Affinity Layer to compute inner product affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(AffinityInp, self).__init__()
        self.d = d

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X, Y.transpose(1, 2))
        return M



class AffinityLR(nn.Module):
    def __init__(self, d, k=512):
        super(AffinityLR, self).__init__()
        self.d = d
        self.k = k
        self.A = Parameter(Tensor(self.d, self.k))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(self.A, self.A.transpose(0, 1))
        M = torch.matmul(X, M)
        M = torch.matmul(M, Y.transpose(1, 2))

        return self.relu(M.squeeze())

class AffinityMah(nn.Module):
    def __init__(self, d, k=100):
        super(AffinityMah, self).__init__()
        self.d = d
        self.k = k
        self.A = Parameter(Tensor(self.d, self.k))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(2)
        dxy = X - Y
        M = torch.matmul(self.A, self.A.transpose(0, 1))
        M = torch.matmul(dxy.unsqueeze(-2), M)
        M = torch.matmul(M, dxy.unsqueeze(-1))

        return self.relu(M.squeeze())


class AffinityFC(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    Affinity score is modeled by a fc neural network.
    Parameter: input dimension d, list of hidden layer dimension hds
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d, hds=None):
        super(AffinityFC, self).__init__()
        self.d = d
        if hds is None:
            self.hds = [1024,]
        else:
            self.hds = hds
        self.hds.append(1)
        fc_lst = []
        last_hd = self.d * 2
        for hd in self.hds:
            fc_lst.append(nn.Linear(last_hd, hd))
            fc_lst.append(nn.ReLU())
            last_hd = hd

        self.fc = nn.Sequential(*fc_lst[:-1])  # last relu omitted

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        cat_feat = torch.cat((X.unsqueeze(-2).expand(X.shape[0], X.shape[1], Y.shape[1], X.shape[2]),
                              Y.unsqueeze(-3).expand(Y.shape[0], X.shape[1], Y.shape[1], Y.shape[2])), dim=-1)
        result = self.fc(cat_feat).squeeze(-1)
        return result


class AffinityBiFC(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    Affinity score is modeled by a bilinear layer followed by a fc neural network.
    Parameter: input dimension d, biliear dimension bd, list of hidden layer dimension hds
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d, bd=1024, hds=None):
        super(AffinityBiFC, self).__init__()
        self.d = d
        self.bd = bd
        if hds is None:
            self.hds = []
        self.hds.append(1)

        self.A = Parameter(Tensor(self.d, self.d, self.bd))
        self.reset_parameters()

        fc_lst = []
        last_hd = self.bd
        for hd in self.hds:
            fc_lst.append(nn.Linear(last_hd, hd))
            fc_lst.append(nn.ReLU())
            last_hd = hd
        self.fc = nn.Sequential(*fc_lst[:-1])  # last relu omitted

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)

    def forward(self, X, Y):
        device = X.device
        assert X.shape[2] == Y.shape[2] == self.d
        bi_result = torch.empty(X.shape[0], X.shape [1], Y.shape[1], self.bd, device=device)
        for i in range(self.bd):
            tmp = torch.matmul(X, self.A[:, :, i])
            tmp = torch.matmul(tmp, Y.transpose(1, 2))
            bi_result[:, :, :, i] = tmp
        S = self.fc(bi_result)
        assert len(S.shape) == 3
        return S
