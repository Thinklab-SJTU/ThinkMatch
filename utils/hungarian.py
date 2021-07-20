import time
import torch
import scipy.optimize as opt
import numpy as np


def hungarian(s: torch.Tensor, n1=None, n2=None):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :return: optimal permutation matrix
    """
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    '''
    st = time.time()
    perm_mat = s.cpu()
    et = time.time()
    print( '.cpu() costs {}'.format(et - st))
    perm_mat = perm_mat.detach()
    st = time.time()
    print( '.detach() costs {}'.format(st - et))
    perm_mat = perm_mat.numpy() * -1
    et = time.time()
    print( '.numpy() costs{}'.format(et - st))
    '''
    
    st = time.time()
    for b in range(batch_num):
        n1b = perm_mat.shape[1] if n1 is None else n1[b]
        n2b = perm_mat.shape[2] if n2 is None else n2[b]
        row, col = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
        perm_mat[b] = np.zeros_like(perm_mat[b])
        perm_mat[b, row, col] = 1
    et = time.time()
    #print('scipy totally costs {}s'.format(et-st))

    st = time.time()
    perm_mat = torch.from_numpy(perm_mat).to(device)
    et = time.time()
    #print('from_numpy() costs {}s'.format(et-st))

    return perm_mat
