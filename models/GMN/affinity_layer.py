import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math


class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix via inner product from feature space.
    Me = X * Lambda * Y^T
    Mp = Ux * Uy^T
    Parameter: scale of weight d
    Input: edgewise (pairwise) feature X, Y
           pointwise (unary) feature Ux, Uy
    Output: edgewise affinity matrix Me
            pointwise affinity matrix Mp
    Weight: weight matrix Lambda = [[Lambda1, Lambda2],
                                    [Lambda2, Lambda1]]
            where Lambda1, Lambda2 > 0
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.lambda1 = Parameter(Tensor(self.d, self.d))
        self.lambda2 = Parameter(Tensor(self.d, self.d))
        self.relu = nn.ReLU()  # problem: if weight<0, then always grad=0. So this parameter is never updated!
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.lambda1.size(1) * 2)
        self.lambda1.data.uniform_(-stdv, stdv)
        self.lambda2.data.uniform_(-stdv, stdv)
        self.lambda1.data += torch.eye(self.d) / 2
        self.lambda2.data += torch.eye(self.d) / 2

    def forward(self, X, Y, Ux, Uy, w1=1, w2=1):
        assert X.shape[1] == Y.shape[1] == 2 * self.d
        lambda1 = self.relu(self.lambda1 + self.lambda1.transpose(0, 1)) * w1
        lambda2 = self.relu(self.lambda2 + self.lambda2.transpose(0, 1)) * w2
        weight = torch.cat((torch.cat((lambda1, lambda2)),
                            torch.cat((lambda2, lambda1))), 1)
        Me = torch.matmul(X.transpose(1, 2), weight)
        Me = torch.matmul(Me, Y)
        Mp = torch.matmul(Ux.transpose(1, 2), Uy)

        return Me, Mp
