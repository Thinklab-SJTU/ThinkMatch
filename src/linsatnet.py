import torch
import sys


def otnet_layer(x, A=None, b=None, C=None, d=None, E=None, f=None, tau=0.05, max_iter=100, dummy_val=0):
    """
    Project x with the constraints A x <= b, C x >= d, E x = f.
    All elements in A, b, C, d, E, f must be non-negative.

    :param x: (n_v), it can optionally have a batch size (b x n_v)
    :param A, C, E: (n_c x n_v)
    :param b, d, f: (n_c)
    :param tau: parameter to control hard/soft constraint
    :param max_iter: max number of iterations
    :return:
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
        vector_input = True
    elif len(x.shape) == 2:
        vector_input = False
    else:
        raise ValueError('input data shape not understood.')

    device = None
    for _ in (A, C, E):
        if _ is not None:
            device = _.device
            break
    assert device is not None

    batch_size = x.shape[0]
    num_var = x.shape[1]
    num_constr = 0
    def init_shape(mat, vec, num_constr):
        if mat is not None:
            assert vec is not None
            assert torch.all(mat >= 0), 'all constraints must be non-negative!'
            assert torch.all(vec >= 0), 'all constraints must be non-negative!'
            num_constr += mat.shape[0]
            assert vec.shape[0] == mat.shape[0]
            assert mat.shape[1] == num_var
        else:
            mat = torch.zeros(0, num_var, device=device)
            vec = torch.zeros(0, device=device)
        return mat, vec, num_constr
    A, b, num_constr = init_shape(A, b, num_constr)
    C, d, num_constr = init_shape(C, d, num_constr)
    E, f, num_constr = init_shape(E, f, num_constr)
    ori_A, ori_b, ori_C, ori_d, ori_E, ori_f = A, b, C, d, E, f

    A = torch.cat((A, b.unsqueeze(-1)), dim=-1) # n_c x n_v
    b = torch.stack((b, A[:, :-1].sum(dim=-1)), dim=-1) # n_c x 2

    k = torch.floor(C.sum(dim=-1) / d)
    C = torch.cat((C, (k * d).unsqueeze(-1)), dim=-1) # n_c x n_v
    d = torch.stack(((k + 1) * d, C[:, :-1].sum(dim=-1) - d), dim=-1) # n_c x 2

    E = torch.cat((E, torch.zeros_like(f).unsqueeze(-1)), dim=-1) # n_c x n_v
    f = torch.stack((f, E[:, :-1].sum(dim=-1) - f), dim=-1) # n_c x 2

    # merge constraints
    A = torch.cat((A, C, E), dim=0)
    b = torch.cat((b, d, f), dim=0)

    # add dummy variables
    dum_x1 = []
    dum_x2 = []
    for j in range(num_constr):
        dum_x1.append(torch.full((batch_size, 1), dummy_val, dtype=x.dtype, device=x.device))
        dum_x2.append(torch.full((batch_size, torch.sum(A[j] != 0)), dummy_val, dtype=x.dtype, device=x.device))

    # operations are performed on log scale
    log_x = x / tau
    log_dum_x1 = [d / tau for d in dum_x1]
    log_dum_x2 = [d / tau for d in dum_x2]
    last_log_x = log_x

    log_A = torch.log(A)
    log_b = torch.log(b)

    for i in range(max_iter):
        for j in range(num_constr):
            _log_x = torch.cat((log_x, log_dum_x1[j]), dim=-1) # batch x n_v

            nonzero_indices = torch.where(A[j] != 0)[0]

            log_nz_x = _log_x[:, nonzero_indices]
            log_nz_x = torch.stack((log_nz_x, log_dum_x2[j]), dim=1)  # batch x 2 x n_v

            log_nz_Aj = log_A[j][nonzero_indices].unsqueeze(0).unsqueeze(0)
            log_t = log_nz_x + log_nz_Aj

            log_sum = torch.logsumexp(log_t, 2, keepdim=True) # batch x 2 x 1
            log_t = log_t - log_sum + log_b[j].unsqueeze(0).unsqueeze(-1)

            log_sum = torch.logsumexp(log_t, 1, keepdim=True) # batch x 1 x n_v
            log_t = log_t - log_sum + log_nz_Aj
            log_nz_x = log_t - log_nz_Aj

            log_dum_x1[j] = log_nz_x[:, 0, -1:]
            log_dum_x2[j] = log_nz_x[:, 1, :]
            if A[j][-1] != 0:
                scatter_idx = nonzero_indices[:-1].unsqueeze(0).repeat(batch_size, 1)
                log_x = torch.scatter(log_x, -1, scatter_idx, log_nz_x[:, 0, :-1])
            else:
                scatter_idx = nonzero_indices.unsqueeze(0).repeat(batch_size, 1)
                log_x = torch.scatter(log_x, -1, scatter_idx, log_nz_x[:, 0, :])

        diff = torch.max(torch.norm((log_x - last_log_x).view(batch_size, -1), dim=-1))
        cv_Ab = torch.matmul(ori_A, torch.exp(log_x).t()).t() - ori_b.unsqueeze(0)
        cv_Cd = -torch.matmul(ori_C, torch.exp(log_x).t()).t() + ori_d.unsqueeze(0)
        cv_Ef = torch.abs(torch.matmul(ori_E, torch.exp(log_x).t()).t() - ori_f.unsqueeze(0))

        if diff <= 1e-3 and \
                torch.sum(cv_Ab[cv_Ab > 0]) + torch.sum(cv_Cd[cv_Cd > 0]) + torch.sum(cv_Ef[cv_Ef > 0]) < 1e-3:
            break
        last_log_x = log_x

    if torch.sum(cv_Ab[cv_Ab > 0]) + torch.sum(cv_Cd[cv_Cd > 0]) + torch.sum(cv_Ef[cv_Ef > 0]) > 0.1 * batch_size:
        print('Warning: non-zero constraint violation within max iterations. Add more iterations or infeasible?',
              file=sys.stderr)

    if vector_input:
        log_x.squeeze_(0)
    return torch.exp(log_x)


if __name__ == '__main__':
    # This example shows how to encode permutation constraint for 3x3 variables
    A = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1],
         [1, 0, 0, 1, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 1, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 1, 0, 0, 1]], dtype=torch.float32
    )
    b = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)
    s = torch.rand(9) # s could be the output of neural network
    x_gt = torch.tensor(
        [1, 0, 0,
         0, 1, 0,
         0, 0, 1], dtype=torch.float32
    )

    s = s.requires_grad_(True)
    opt = torch.optim.SGD([s], lr=0.1, momentum=0.9)

    niters = 10
    with torch.autograd.set_detect_anomaly(True):
        for i in range(niters):
            x = otnet_layer(s, A, b, tau=0.1, max_iter=10, dummy_val=0)
            cv = torch.matmul(A, x.t()).t() - b.unsqueeze(0)
            loss = ((x - x_gt) ** 2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            print(f'{i}/{niters}\n'
                  f'  OT obj={torch.sum(s * x)},\n'
                  f'  loss={loss},\n'
                  f'  sum(constraint violation)={torch.sum(cv[cv > 0])},\n'
                  f'  x={x},\n'
                  f'  constraint violation={cv}')
