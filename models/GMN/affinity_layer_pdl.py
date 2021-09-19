import paddle
import paddle.nn as nn
import math
import numpy as np
from numpy.random import uniform


class Affinity(nn.Module): """ Affinity Layer to compute the affinity matrix via inner product from feature space.
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
        # set parameters
        stdv = 1. / math.sqrt(self.d)
        tmp1 = uniform(low=-1*stdv, high=stdv, size=(self.d,self.d))
        tmp2 = uniform(low=-1*stdv, high=stdv, size=(self.d,self.d))
        tmp1 += np.eye(self.d) / 2.0
        tmp2 += np.eye(self.d) / 2.0 

        self.lambda1 = paddle.ParamAttr(initializer=nn.initializer.Assign(paddle.to_tensor(tmp1, dtype='float64')
        self.lambda2 = paddle.ParamAttr(initializer=nn.initializer.Assign(paddle.to_tensor(tmp2, dtype='float64')
        self.add_parameter('labmda1', self.lambda1)
        self.add_parameter('lambda2', self.lambda2)

        self.relu = nn.ReLU()  # problem: if weight<0, then always grad=0. So this parameter is never updated!


    def forward(self, X, Y, Ux, Uy, w1=1, w2=1):
        assert X.shape[1] == Y.shape[1] == 2 * self.d
        lambda1 = self.relu(self.lambda1 + self.lambda1.transpose((1,0))) * w1
        lambda2 = self.relu(self.lambda2 + self.lambda2.transpose((1,0))) * w2
        weight = paddle.concat((paddle.concat((lambda1, lambda2)),
                            paddle.concat((lambda2, lambda1))), 1)
        Me = paddle.matmul(X.transpose((0,2,1)), weight)
        Me = paddle.matmul(Me, Y)
        Mp = paddle.matmul(Ux.transpose((0,2,1)), Uy)

        return Me, Mp
