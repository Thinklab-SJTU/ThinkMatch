import paddle
import paddle.nn as nn
import math
import numpy as np
from numpy.random import uniform 


class Affinity(nn.Layer):
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
        # set parameters    
        stdv = 1. / math.sqrt(self.d)
        tmp = uniform(low=-1*stdv, high=stdv, size=(self.d,self.d))
        tmp += np.eye(self.d)
        a_attr = paddle.ParamAttr(initializer=nn.initializer.Assign(paddle.to_tensor(tmp,dtype='float64')))

        self.A = self.create_parameter([self.d,self.d],attr=a_attr)
        self.add_parameter('A',self.A)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        #M = paddle.matmul(X, self.A)
        M = paddle.matmul(X, (self.A + paddle.transpose(self.A, [1,0])) / 2)
        M = paddle.matmul(M, paddle.transpose(Y,[0,2,1]))
        return M
