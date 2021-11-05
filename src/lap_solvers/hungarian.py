#import torch
import scipy.optimize as opt
import numpy as np                  #numpy
from multiprocessing import Pool

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as ops
#import mindspore.numpy as np
from mindspore.common.tensor import Tensor
import mindspore.context as context

context.set_context(device_target="GPU")

def hungarian(s: Tensor, n1=None, n2=None, nproc=1):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :param nproc: number of parallel processes (default =1 for no parallel)
    :return: optimal permutation matrix
    """
    if len(s.shape) == 2:
        s = P.ExpandDims()(s, 0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))

    batch_num = s.shape[0]

    context.set_context(device_target="CPU")

    perm_mat = s.asnumpy() * -1
    if n1 is not None:
        n1 = n1.asnumpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.asnumpy()
    else:
        n2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(hung_kernel, zip(perm_mat, n1, n2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([hung_kernel(perm_mat[b], n1[b], n2[b]) for b in range(batch_num)])

    context.set_context(device_target="GPU")

    perm_mat = Tensor(perm_mat)

    if matrix_input:
        P.Squeeze(0)(perm_mat)

    return perm_mat

def hung_kernel(s: Tensor, n1=None, n2=None):
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    row, col = opt.linear_sum_assignment(s[:n1, :n2])
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat