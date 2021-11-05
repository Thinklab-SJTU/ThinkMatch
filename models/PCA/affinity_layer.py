import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
#from torch.nn.parameter import Parameter
#from torch import Tensor
import math

import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Uniform
#from mindspore.common.tensor import Tensor
import mindspore.context as context

context.set_context(device_target="GPU")

class Affinity(nn.Cell):
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
        self.A = Parameter(Tensor(np.empty((self.d, self.d))))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.set_data(initializer(Uniform(stdv), self.A.shape))
        self.A.set_data(self.A + np.eye(self.d))

    def construct(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        perm = tuple(range(3,len(Y.shape)))
        perm = (0,2,1) + perm
        M = np.matmul(X, self.A)
        #M = torch.matmul(X, (self.A + self.A.transpose(0, 1)) / 2)
        M = np.matmul(M, P.Transpose()(Y,perm))
        return M

class AffinityInp(nn.Cell):
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

    def construct(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        perm = tuple(range(3, len(Y.shape)))
        perm = (0, 2, 1) + perm
        M = np.matmul(X, P.Transpose()(Y,perm))
        return M



class AffinityLR(nn.Cell):
    def __init__(self, d, k=512):
        super(AffinityLR, self).__init__()
        self.d = d
        self.k = k
        self.A = Parameter(Tensor(np.empty((self.d, self.k))))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.set_data(initializer(Uniform(stdv), self.A.shape))

    def construct(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        perm = tuple(range(2, len(self.A.shape)))
        perm = (1, 0) + perm
        M = np.matmul(self.A, P.Transpose()(self.A, perm))
        M = np.matmul(X, M)
        perm = tuple(range(3, len(Y.shape)))
        perm = (0, 2, 1) + perm
        M = np.matmul(M, P.Transpose()(Y, perm))

        return self.relu(P.ReduceMean(None)(M))

class AffinityMah(nn.Cell):
    def __init__(self, d, k=100):
        super(AffinityMah, self).__init__()
        self.d = d
        self.k = k
        self.A = Parameter(Tensor(np.empty((self.d, self.k))))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.set_data(initializer(Uniform(stdv), self.A.shape))

    def construct(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        X = P.ExpandDims()(X,1)
        Y = P.ExpandDims()(Y,2)
        dxy = X - Y
        perm = tuple(range(2, len(self.A.shape)))
        perm = (1, 0) + perm
        M = np.matmul(self.A, P.Transpose()(self.A, perm))
        M = np.matmul(P.ExpandDims()(dxy,-2), M)
        M = np.matmul(M, P.ExpandDims()(dxy,-1))

        return self.relu(P.ReduceMean(None)(M))


class AffinityFC(nn.Cell):
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
            fc_lst.append(nn.Dense(in_channels=last_hd, out_channels=hd))
            fc_lst.append(nn.ReLU())
            last_hd = hd

        self.fc = nn.SequentialCell([*fc_lst[:-1]])  # last relu omitted

    def construct(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        cat_feat = P.Concat(-1)((P.BroadcastTo((X.shape[0], X.shape[1], Y.shape[1], X.shape[2]))(P.ExpandDims()(X,-2)), P.BroadcastTo((Y.shape[0], X.shape[1], Y.shape[1], Y.shape[2]))(P.ExpandDims()(Y,-3)) ))
        result = self.fc(cat_feat).squeeze(-1)
        return result


class AffinityBiFC(nn.Cell):
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

        self.A = Parameter(Tensor(np.empty((self.d, self.d, self.bd))))
        self.reset_parameters()

        fc_lst = []
        last_hd = self.bd
        for hd in self.hds:
            fc_lst.append(nn.Dense(in_channels=last_hd, out_channels=hd))
            fc_lst.append(nn.ReLU())
            last_hd = hd
        self.fc = nn.SequentialCell([*fc_lst[:-1]])  # last relu omitted

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.set_data(initializer(Uniform(stdv), self.A.shape))

    def construct(self, X, Y):
        device = X.device
        assert X.shape[2] == Y.shape[2] == self.d
        bi_result = np.empty(X.shape[0], X.shape [1], Y.shape[1], self.bd)
        for i in range(self.bd):
            tmp = np.matmul(X, self.A[:, :, i])
            perm = tuple(range(3, len(Y.shape)))
            perm = (0, 2, 1) + perm
            tmp = np.matmul(tmp, P.Transpose()(Y,perm))
            bi_result[:, :, :, i] = tmp
        S = self.fc(bi_result)
        assert len(S.shape) == 3
        return S
