import torch
import torch.nn as nn


class RobustLoss(nn.Module):
    """
    RobustLoss Criterion computes a robust loss function.
    L = Sum(Phi(d_i - d_i^gt)),
        where Phi(x) = sqrt(x^T * x + epsilon)
    Parameter: a small number for numerical stability epsilon
               (optional) division taken to normalize the loss norm
    Input: displacement matrix d1
           displacement matrix d2
           (optional)dummy node mask mask
    Output: loss value
    """
    def __init__(self, epsilon=1e-5, norm=None):
        super(RobustLoss, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def forward(self, d1, d2, mask=None):
        # Loss = Sum(Phi(d_i - d_i^gt))
        # Phi(x) = sqrt(x^T * x + epsilon)
        if mask is None:
            mask = torch.ones_like(mask)
        x = d1 - d2
        if self.norm is not None:
            x = x / self.norm

        xtx = torch.sum(x * x * mask, dim=-1)
        phi = torch.sqrt(xtx + self.epsilon)
        loss = torch.sum(phi) / d1.shape[0]

        return loss


if __name__ == '__main__':
    d1 = torch.tensor([[1., 2.],
                       [2., 3.],
                       [3., 4.]], requires_grad=True)
    d2 = torch.tensor([[-1., -2.],
                       [-2., -3.],
                       [-3., -4.]], requires_grad=True)
    mask = torch.tensor([[1., 1.],
                         [1., 1.],
                         [0., 0.]])

    rl = RobustLoss()
    loss = rl(d1, d2, mask)
    loss.backward()
    print(d1.grad)
    print(d2.grad)
