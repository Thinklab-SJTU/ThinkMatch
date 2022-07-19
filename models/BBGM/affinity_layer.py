import torch
import torch.nn as nn


class InnerProductWithWeightsAffinity(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InnerProductWithWeightsAffinity, self).__init__()
        self.d = output_dim
        self.A = torch.nn.Linear(input_dim, output_dim)

    def _forward(self, X, Y, weights, use_global):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        coefficients = torch.tanh(self.A(weights))
        if use_global:
            res = torch.matmul(X * coefficients, Y.transpose(0, 1))
        else:
            res = torch.matmul(X, Y.transpose(0, 1))
        res = torch.nn.functional.softplus(res) - 0.5
        return res

    def forward(self, Xs, Ys, Ws, use_global=True):
        return [self._forward(X, Y, W, use_global) for X, Y, W in zip(Xs, Ys, Ws)]
