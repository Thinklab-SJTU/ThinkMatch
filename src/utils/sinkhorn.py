import torch
import torch.nn as nn


class Sinkhorn(nn.Module):
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

    def forward(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=torch.float32):
        batch_size = s.shape[0]

        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        out_ = s + self.epsilon

        for i in range(self.max_iter):
            if exp:
                out_ = torch.exp(exp_alpha * out_)
            if i % 2 == 1:
                # column norm
                sum = torch.sum(torch.mul(out_.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), out_.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(out_)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else out_.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else out_.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            out_ = out_ * tmp

        if dummy_row and dummy_shape[1] > 0:
            out_ = out_[:, :-dummy_shape[1]]

        return out_

if __name__ == '__main__' :
    from permutation_loss import CrossEntropyLoss as CELoss
    import numpy as np
    import torch.autograd as autograd
    s = np.array([[[0.00000026, 0.01666268, 0.31441244, 0.00000159, 0.00001586, 0.06408666, 0.03489333, 0.00000004, 0.00000220, 0.56992495, 0.00000003],
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

        [[0.00000206, 0.39126232, 0.00000039, 0.60873520, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.00001232, 0.97780335, 0.00003701, 0.02214720, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.00000033, 0.86960614, 0.00001453, 0.13037905, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.00006901, 0.15446325, 0.00003045, 0.84543729, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]]] )
    gt_perm = np.array( 
        [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
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

        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])

    bi_stochastic = torch.tensor(s, dtype=torch.float32, requires_grad=True) 
    perm_mat = torch.tensor(gt_perm, dtype=torch.float32)
    n_rows = torch.tensor([11,4])
    n_cols = torch.tensor([11,4])
    
    model = Sinkhorn()    
    criterion = CELoss()
    
    with torch.set_grad_enabled(True):
        m = model(bi_stochastic, n_rows, n_cols)
        loss = criterion(m, perm_mat, n_rows, n_cols)

        #torch.autograd.backward([loss])
        print(autograd.grad(loss, m, retain_graph=True)[0])
        print('*' * 20)
        print(autograd.grad(loss, bi_stochastic)[0])
