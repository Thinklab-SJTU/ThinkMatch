import torch
import torch.nn as nn


class Sinkhorn(nn.Module):
    """
    Sinkhorn algorithm turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, tau=0.05, epsilon=1e-4, log_forward=True, batched_operation=False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward
        if not log_forward:
            print('Warning: Sinkhorn algorithm without log forward is deprecated because log_forward is more stable.')
        self.batched_operation = batched_operation # batched operation may cause instability in backward computation,
                                                   # but will boost computation.

    def forward(self, *input, **kwinput):
        if self.log_forward:
            return self.forward_log(*input, **kwinput)
        else:
            return self.forward_ori(*input, **kwinput) # deprecated

    def forward_log(self, s, nrows=None, ncols=None, dummy_row=False, dtype=torch.float32):
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
            s = s.transpose(1, 2)
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
            s = torch.cat((s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                s[b, nrows[b]:, :] = -float('inf')
                s[b, :, ncols[b]:] = -float('inf')

        if self.batched_operation:
            log_s = s

            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    log_s = log_s - log_sum
                    log_s[torch.isnan(log_s)] = -float('inf')
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum
                    log_s[torch.isnan(log_s)] = -float('inf')

                # ret_log_s[b, row_slice, col_slice] = log_s

            if dummy_row and dummy_shape[1] > 0:
                log_s = log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                log_s.squeeze_(0)

            return torch.exp(log_s)
        else:
            ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

            for b in range(batch_size):
                row_slice = slice(0, nrows[b])
                col_slice = slice(0, ncols[b])
                log_s = s[b, row_slice, col_slice]

                for i in range(self.max_iter):
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                        log_s = log_s - log_sum
                    else:
                        log_sum = torch.logsumexp(log_s, 0, keepdim=True)
                        log_s = log_s - log_sum

                ret_log_s[b, row_slice, col_slice] = log_s

            if dummy_row:
                if dummy_shape[1] > 0:
                    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if transposed:
                ret_log_s = ret_log_s.transpose(1, 2)
            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)

        # ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

        # for b in range(batch_size):
        #    row_slice = slice(0, nrows[b])
        #    col_slice = slice(0, ncols[b])
        #    log_s = s[b, row_slice, col_slice]

    def forward_ori(self, s, nrows=None, ncols=None, dummy_row=False, dtype=torch.float32):
        # computing sinkhorn with row/column normalization.
        # This function is deprecated because forward_log is more numerically stable.
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        #s = s.to(dtype=dtype)

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # tau scaling
        ret_s = torch.zeros_like(s)
        for b, n in enumerate(nrows):
            ret_s[b, 0:n, 0:ncols[b]] = \
                nn.functional.softmax(s[b, 0:n, 0:ncols[b]] / self.tau, dim=-1)
        s = ret_s

        # add dummy elements
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            #s = torch.cat((s, torch.full(dummy_shape, self.epsilon * 10).to(s.device)), dim=1)
            #nrows = nrows + dummy_shape[1] # non in-place
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            ori_nrows = nrows
            nrows = ncols
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = self.epsilon

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device, dtype=s.dtype)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device, dtype=s.dtype)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        s += self.epsilon

        for i in range(self.max_iter):
            if i % 2 == 0:
                # column norm
                #ones = torch.ones(batch_size, s.shape[1], s.shape[1], device=s.device)
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                # ones = torch.ones(batch_size, s.shape[2], s.shape[2], device=s.device)
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row:
            if dummy_shape[1] > 0:
                s = s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = 0

        if matrix_input:
            s.squeeze_(0)

        return s


class GumbelSinkhorn(nn.Module):
    """
    GumbelSinkhorn Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, tau=1., epsilon=1e-4, batched_operation=False):
        super(GumbelSinkhorn, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter, tau, epsilon, batched_operation=batched_operation)

    def forward(self, s, nrows=None, ncols=None, sample_num=5, dummy_row=False, dtype=torch.float32):
        def sample_gumbel(t_like, eps=1e-20):
            """
            randomly sample standard gumbel variables
            """
            u = torch.empty_like(t_like).uniform_()
            return -torch.log(-torch.log(u + eps) + eps)

        s_rep = torch.repeat_interleave(s, sample_num, dim=0)
        s_rep = s_rep + sample_gumbel(s_rep)
        nrows_rep = torch.repeat_interleave(nrows, sample_num, dim=0)
        ncols_rep = torch.repeat_interleave(ncols, sample_num, dim=0)
        s_rep = self.sinkhorn(s_rep, nrows_rep, ncols_rep, dummy_row, dtype)
        #s_rep = torch.reshape(s_rep, (-1, sample_num, s_rep.shape[1], s_rep.shape[2]))
        return s_rep

def ori_main():
    from permutation_loss import CrossEntropyLoss as CELoss
    import numpy as np
    import torch.autograd as autograd
    s = np.array([        [[0.00000206, 0.39126232, 0.00000039, 0.60873520, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
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

    bi_stochastic = torch.tensor(s, dtype=torch.float32, requires_grad=True) 
    perm_mat = torch.tensor(gt_perm, dtype=torch.float32)
    n_rows = torch.tensor([11,4])
    n_cols = torch.tensor([11,4])
    
    model = Sinkhorn(max_iter=1, epsilon=1e-6, tau=0.05)    
    criterion = CELoss()
    
    with torch.set_grad_enabled(True):
        m = model(bi_stochastic, n_rows, n_cols)
        #print('m', m )
        loss = criterion(m, perm_mat, n_rows, n_cols)

        print(autograd.grad(loss, m, retain_graph=True)[0])
        print('*' * 20)
        print(autograd.grad(loss, bi_stochastic)[0])

def main2():
    from permutation_loss import CrossEntropyLoss as CELoss
    import torch.autograd as autograd
    import numpy as np
    s = np.array([[[1.8049, 2.3439, 2.4651, 1.8594, 1.9814, 2.3853, 2.3615, 1.7067,
          1.8722, 2.5069, 1.6636],
         [1.3733, 1.7487, 1.8090, 1.4526, 1.5144, 1.7648, 1.7514, 1.3004,
          1.3944, 1.7900, 1.2987],
         [1.1746, 1.3289, 1.3974, 1.1518, 1.2093, 1.3520, 1.3510, 1.1511,
          1.0950, 1.3896, 1.0161],
         [1.3182, 1.6900, 1.7068, 1.3744, 1.4415, 1.6869, 1.6781, 1.2315,
          1.3583, 1.7484, 1.2289],
         [1.1323, 1.4563, 1.5800, 1.2099, 1.2704, 1.4788, 1.4882, 1.0871,
          1.2929, 1.5229, 1.0764],
         [1.7170, 2.0360, 2.1081, 1.6829, 1.7439, 2.0749, 2.0575, 1.5492,
          1.6313, 2.1287, 1.5101],
         [1.8894, 2.4682, 2.5647, 2.0054, 2.0818, 2.5764, 2.5142, 1.7567,
          1.9421, 2.5725, 1.7585],
         [1.4733, 1.8995, 1.9525, 1.5602, 1.6122, 1.9655, 1.8971, 1.4113,
          1.4899, 1.9788, 1.3523],
         [1.4543, 1.9154, 1.9652, 1.5428, 1.6084, 2.0195, 1.8744, 1.3525,
          1.5034, 1.9345, 1.3858],
         [1.5393, 1.9910, 2.2001, 1.5995, 1.6933, 2.0311, 2.0128, 1.4697,
          1.7042, 2.1107, 1.4256],
         [1.3575, 1.7008, 1.7372, 1.4100, 1.5470, 1.7045, 1.6570, 1.2335,
          1.3235, 1.7298, 1.2302]],

        [[1.7626, 2.3512, 1.6856, 2.3854, 0.6395, 0.6395, 0.6395, 0.6395,
          0.6395, 0.6395, 0.6395],
         [1.2241, 1.7908, 1.2976, 1.5958, 0.5253, 0.5253, 0.5253, 0.5253,
          0.5253, 0.5253, 0.5253],
         [1.6531, 2.3713, 1.8403, 2.3027, 0.6398, 0.6398, 0.6398, 0.6398,
          0.6398, 0.6398, 0.6398],
         [1.2672, 1.6322, 1.2352, 1.7351, 0.5332, 0.5332, 0.5332, 0.5332,
          0.5332, 0.5332, 0.5332],
         [0.5032, 0.6248, 0.5176, 0.6115, 0.3395, 0.3395, 0.3395, 0.3395,
          0.3395, 0.3395, 0.3395],
         [0.5032, 0.6248, 0.5176, 0.6115, 0.3395, 0.3395, 0.3395, 0.3395,
          0.3395, 0.3395, 0.3395],
         [0.5032, 0.6248, 0.5176, 0.6115, 0.3395, 0.3395, 0.3395, 0.3395,
          0.3395, 0.3395, 0.3395],
         [0.5032, 0.6248, 0.5176, 0.6115, 0.3395, 0.3395, 0.3395, 0.3395,
          0.3395, 0.3395, 0.3395],
         [0.5032, 0.6248, 0.5176, 0.6115, 0.3395, 0.3395, 0.3395, 0.3395,
          0.3395, 0.3395, 0.3395],
         [0.5032, 0.6248, 0.5176, 0.6115, 0.3395, 0.3395, 0.3395, 0.3395,
          0.3395, 0.3395, 0.3395],
         [0.5032, 0.6248, 0.5176, 0.6115, 0.3395, 0.3395, 0.3395, 0.3395,
          0.3395, 0.3395, 0.3395]],

        [[1.3980, 1.6063, 1.6185, 1.5412, 1.3445, 2.2173, 1.3910, 1.4027,
          0.5368, 0.5368, 0.5368],
         [1.1221, 1.3153, 1.2481, 1.2071, 1.0745, 1.5944, 1.0880, 1.1316,
          0.4740, 0.4740, 0.4740],
         [1.3183, 1.5774, 1.4628, 1.4730, 1.3311, 1.9394, 1.2810, 1.3320,
          0.5229, 0.5229, 0.5229],
         [1.7918, 2.0835, 2.0797, 1.9870, 1.7042, 2.7986, 1.7174, 1.7756,
          0.6543, 0.6543, 0.6543],
         [1.9365, 2.2109, 2.1413, 2.1015, 1.8005, 2.9768, 1.7825, 1.9025,
          0.6671, 0.6671, 0.6671],
         [1.4757, 1.7783, 1.6997, 1.6660, 1.4849, 2.2995, 1.4180, 1.5291,
          0.5764, 0.5764, 0.5764],
         [1.4604, 1.6727, 1.6082, 1.5569, 1.3595, 2.1692, 1.3737, 1.4439,
          0.5575, 0.5575, 0.5575],
         [1.4950, 1.7653, 1.6878, 1.6370, 1.4346, 2.2808, 1.4004, 1.4969,
          0.5697, 0.5697, 0.5697],
         [0.5412, 0.5827, 0.5701, 0.5688, 0.5184, 0.7074, 0.5194, 0.5231,
          0.3395, 0.3395, 0.3395],
         [0.5412, 0.5827, 0.5701, 0.5688, 0.5184, 0.7074, 0.5194, 0.5231,
          0.3395, 0.3395, 0.3395],
         [0.5412, 0.5827, 0.5701, 0.5688, 0.5184, 0.7074, 0.5194, 0.5231,
          0.3395, 0.3395, 0.3395]]])



    perm_mat = torch.tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
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
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]] )
    bi_stochastic = torch.tensor(s, dtype=torch.float32, requires_grad=True) 
    n_rows = torch.tensor([11,4,8])
    n_cols = torch.tensor([11,4,8])
    model = Sinkhorn(max_iter=1, epsilon=1e-6, tau=0.05 )    
    criterion = CELoss()

    with torch.set_grad_enabled(True):
        m = model(bi_stochastic, n_rows, n_cols)
        #print('m', m )
        loss = criterion(m, perm_mat, n_rows, n_cols)
        print('loss', loss)

        print(autograd.grad(loss, m, retain_graph=True)[0])
        print('*' * 20)
        print(autograd.grad(loss, bi_stochastic)[0])

def simple_main():
    bs = Sinkhorn(max_iter=10, epsilon=1e-4, tau=1.)
    inp = torch.tensor([[[1., 0, 1.],
                         [1., 0, 3.],
                         [2., 0, 1.],
                         [4., 0, 2.]],
                        [[1., 0, 1.],
                         [1., 0, 3.],
                         [2., 0, 1.],
                         [4., 0, 2.]]], requires_grad=True, dtype=torch.float32)
    outp = bs(inp, (3, 4))

    print(outp)
    l = torch.sum(outp)
    l.backward()
    print(inp.grad * 1e10)

    '''
    outp2 = torch.tensor([[0.1, 0.1, 1],
                          [2, 3, 4.]], requires_grad=True)

    l = torch.sum(outp2)
    l.backward()
    print(outp2.grad)
    '''

if __name__ == '__main__':
    simple_main()
    #ori_main()
    #main2()
