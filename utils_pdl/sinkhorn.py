import paddle
import paddle.nn as nn

class Sinkhorn(nn.Layer):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, epsilon=1e-4):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=paddle.float32):
        batch_size = s.shape[0]

        # global function that sets all tensors' device to the device of "s"
        paddle.set_device(s.place)
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            s = paddle.concat((s, paddle.full(dummy_shape, 0.).cuda()), axis=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        row_norm_ones = paddle.zeros(batch_size, s.shape[1], s.shape[1])  # size: row x row
        col_norm_ones = paddle.zeros(batch_size, s.shape[2], s.shape[2])  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s += self.epsilon

        for i in range(self.max_iter):
            if exp:
                s = paddle.exp(exp_alpha * s)
            if i % 2 == 1:
                # column norm
                sum = paddle.sum(paddle.multiply(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), axis=2)
            else:
                # row norm
                sum = paddle.sum(paddle.multiply(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), axis=2)

            tmp = paddle.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row and dummy_shape[1] > 0:
            s = s[:, :-dummy_shape[1]]

        return s
