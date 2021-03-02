import paddle
import paddle.nn as nn


class RobustLoss(nn.layer):
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
            mask = paddle.ones_like(mask)
        x = d1 - d2
        if self.norm is not None:
            x = x / self.norm

        xtx = paddle.sum(x * x * mask, axis=-1)
        phi = paddle.sqrt(xtx + self.epsilon)
        loss = paddle.sum(phi) / d1.shape[0]

        return loss


if __name__ == '__main__':
    d1 = paddle.to_tensor([[1., 2.],
                       [2., 3.],
                       [3., 4.]], stop_gradient=False)
    d2 = paddle.to_tensor([[-1., -2.],
                       [-2., -3.],
                       [-3., -4.]], stop_gradient=False)
    mask = paddle.to_tensor([[1., 1.],
                         [1., 1.],
                         [0., 0.]])

    rl = RobustLoss()
    loss = rl(d1, d2, mask)
    loss.backward()
    print(d1.grad)
    print(d2.grad)
