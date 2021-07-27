import paddle
import paddle.nn as nn
from pdl_device_trans import place2str

class Sinkhorn(nn.Layer):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, epsilon=1e-4, tau=0.10, log_forward=True):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tau = tau
        self.log_forward = log_forward
        if not log_forward:
            print('Warning: Sinkhorn algorithm not in log scale is deprecated since logrithm is more stable')

    def forward(self, *input, **kwinput):
        if self.log_forward:
            return self.forward_log(*input, **kwinput)
        else:
            return self.forward_ori(*input, **kwinput) # deprecated

    def forward_log(self, s, nrows=None, ncols=None, dummy_row=False, dtype=paddle.float32):
        # global function that sets all tensors' device to the device of "s"
        device_str = place2str(s.place)
        paddle.set_device(device_str)
        # computing sinkhorn with row/column normalization in the log space.
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if s.shape[2] >= s.shape[1]:
            transposed = False
        else:
            s = s.transpose((0, 2, 1))
            transposed = True

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # operations are performed on log_s
        s = s / self.tau

        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            s = paddle.cat((s, paddle.full(dummy_shape, -float('inf')).cuda()), axis=1)
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                s[b, nrows[b]:, :] = -float('inf')
                s[b, :, ncols[b]:] = -float('inf')

        ret_log_s = paddle.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), dtype=s.dtype).cuda()
        ret_log_s.stop_gradient = False

        for b in range(batch_size):
            row_slice = slice(0, int(nrows[b]))
            col_slice = slice(0, int(ncols[b]))
            log_s = s[b, row_slice, col_slice]

            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = paddle.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum
                else:
                    log_sum = paddle.logsumexp(log_s, 0, keepdim=True)
                    log_s = log_s - log_sum

            ret_log_s[b, row_slice, col_slice] = log_s

        if dummy_row:
            if dummy_shape[1] > 0:
                ret_log_s = ret_log_s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

        if transposed:
            ret_log_s = ret_log_s.transpose((0, 2, 1))
        if matrix_input:
            ret_log_s.squeeze_(0)

        return paddle.exp(ret_log_s)


    def forward_ori(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=paddle.float32):
        batch_size = s.shape[0]

        # global function that sets all tensors' device to the device of "s"
        device_str = place2str(s.place)
        paddle.set_device(device_str)
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            s = paddle.concat((s, paddle.full(dummy_shape, 0.).cuda()), axis=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        row_norm_ones = paddle.zeros((batch_size, s.shape[1], s.shape[1]))  # size: row x row
        col_norm_ones = paddle.zeros((batch_size, s.shape[2], s.shape[2]))  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, int(nrows[b]) if nrows is not None else int(s.shape[2]))
            col_slice = slice(0, int(ncols[b]) if ncols is not None else int(s.shape[1]))
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
                row_slice = slice(0, int(nrows[b]) if nrows is not None else s.shape[2])
                col_slice = slice(0, int(ncols[b]) if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row and dummy_shape[1] > 0:
            s = s[:, :-dummy_shape[1]]

        return s

def ori_main():
    from permutation_loss import CrossEntropyLoss as CELoss
    import numpy as np
    s = np.array([ [[0.00000206, 0.39126232, 0.00000039, 0.60873520, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.00001232, 0.97780335, 0.00003701, 0.02214720, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.00000033, 0.86960614, 0.00001453, 0.13037905, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.00006901, 0.15446325, 0.00003045, 0.84543729, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]],
             [[0.00000026, 0.01666268, 0.31441244, 0.00000159, 0.00001586, 0.06408666, 0.03489333, 0.00000004, 0.00000220, 0.56992495, 0.00000003],
         [0.00003907, 0.11164442, 0.32763091, 0.00026938, 0.00067560, 0.17508876, 0.15032516, 0.00001204, 0.00007836, 0.23422408, 0.00001221],
         [0.00278927, 0.10608968, 0.30548191, 0.00254041, 0.00788011, 0.14229497, 0.19681570, 0.00188713, 0.00066216, 0.23343839, 0.00012028],
         [0.00004377, 0.10789249, 0.14842081, 0.00020574, 0.00099834, 0.13971062, 0.15795937, 0.00000933, 0.00013808, 0.44460961, 0.00001175],
         [0.00007237, 0.05559206, 0.49432558, 0.00070045, 0.00134125, 0.12144947, 0.12560618, 0.00003383, 0.00168359, 0.19915234, 0.00004287],
         [0.00008717, 0.05557152, 0.27164841, 0.00005802, 0.00015237, 0.14933392, 0.11947705, 0.00000457, 0.00002119, 0.40364328, 0.00000248],
         [0.00000019, 0.03239984, 0.22995634, 0.00000404, 0.00001444, 0.40359825, 0.08687798, 0.00000002, 0.00000089, 0.24714798, 0.00000003],
         [0.00000936, 0.05670571, 0.19046211, 0.00007296, 0.00014078, 0.35014644, 0.07453655, 0.00000283, 0.00001744, 0.32790479, 0.00000101],
         [0.00000482, 0.07666548, 0.20235837, 0.00004108, 0.00018885, 0.56168914, 0.03874058, 0.00000091, 0.00001905, 0.12029009, 0.00000170],
         [0.00000065, 0.00943021, 0.83199155, 0.00000466, 0.00002752, 0.02751676, 0.02095186, 0.00000020, 0.00003825, 0.11003817, 0.00000022],
         [0.00010970, 0.16183394, 0.27879959, 0.00037161, 0.00753538, 0.17999281, 0.10271225, 0.00001080, 0.00008524, 0.26853555, 0.00001317]],

] )
    gt_perm = np.array( 
        [        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    
        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]],
])


    bi_stochastic = paddle.to_tensor(s, dtype='float32', stop_gradient=False) 
    perm_mat = paddle.to_tensor(gt_perm, dtype='float32')
    n_rows = paddle.to_tensor([11,4])
    n_cols = paddle.to_tensor([11,4])
    
    model = Sinkhorn(max_iter=1, epsilon=1e-6, tau=0.05 )    
    criterion = CELoss()
    
    with paddle.set_grad_enabled(True):
        m = model(bi_stochastic, n_rows, n_cols)
        print('m', m)
        loss = criterion(m, perm_mat, n_rows, n_cols)

        paddle.autograd.backward([loss])
        print(m.grad)
        print('*' * 20)
        print(bi_stochastic.grad)

def simple_main():
    bs = Sinkhorn(max_iter=10, epsilon=1e-4, tau=1.)
    inp = paddle.to_tensor([[[1., 0, 1.],
                         [1., 0, 3.],
                         [2., 0, 1.],
                         [4., 0, 2.]]], stop_gradient=False, dtype='float32')
    outp = bs(inp, (3, 4))

    print(outp)
    l = paddle.sum(outp)
    l.backward()
    print(inp.grad * 1e10)
    '''
    outp2 = paddle.to_tensor([[0.1, 0.1, 1],
                          [2, 3, 4.]], stop_gradient=False)

    l = paddle.sum(outp2)
    l.backward()
    print(outp2.grad)
    '''

if __name__ == '__main__':
    simple_main()
    #ori_main()
