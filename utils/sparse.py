import sys
import torch
from torch.autograd import Function
import numpy as np
import scipy.sparse as ssp

from sparse_torch import CSRMatrix3d, CSCMatrix3d

from torch.utils.cpp_extension import load
bilinear_diag = load(name='bilinear_diag', sources=['extension/bilinear_diag/bilinear_diag.cpp',
                                                    'extension/bilinear_diag/bilinear_diag_cuda.cu'],
                     extra_include_paths=[
                         '/usr/include/python{}.{}/'.format(sys.version_info.major, sys.version_info.minor)]
                     )


def to_sparse(x, dense_dim=1):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)[:, :len(x.shape) - dense_dim + 1]
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values)


def sbmm(t1, t2):
    """
    Perform bmm (Batch Matrix Matrix) for sparse x dense -> dense.
    """
    return SparseDenseDenseBMM.apply(t1, t2)


def sbmm_diag(t1, t2):
    """
    Perform bmm and diagonal for sparse x dense -> dense. The diagonalized result is returned in vector tensor.
    This is a wrapper function and does not support gradient.
    """
    assert t1.is_sparse != t2.is_sparse, 't1, t2 must be one sparse and one dense!'
    return sdd_bmm_diag_torch(t1, t2)


def sdsbmm(t1, t2):
    """
    Perform bmm for sparse x dense -> sparse.
    This is a wrapper function and does not support gradient.
    """
    assert (type(t1) == list) != (type(t2) == list) or t1.is_sparse != t2.is_sparse, \
        't1, t2 must be one sparse and one dense!'
    if type(t1) == list or t1.is_sparse:
        result = sds_bmm_torch(t1, t2)
    else:
        result = sds_bmm_torch(t2.transpose(1, 2), t1.transpose(1, 2)).transpose(1, 2)
    return result


def sssbmm_diag(m1, m2):
    """
    Perform bmm and diagonal for sparse x sparse -> sparse.
    This is a wrapper function and does not support gradient.
    """
    if (type(m1) == list and type(m1[0]) == torch.Tensor) or type(m1) == torch.Tensor:
        m1 = torch2ssp(m1)
    if (type(m2) == list and type(m2[0]) == torch.Tensor) or type(m2) == torch.Tensor:
        m2 = torch2ssp(m2)
    return sss_bmm_diag_spp(m1, m2)


'''
Torch API Functions
'''


class SparseDenseDenseBMM(Function):
    """
    bmm (Batch Matrix Matrix) for sparse x dense -> dense.
    with s_t1.shape = (b, x, s), d_t2.shape = (b, s, y), the output shape is (b, x, y)
    This is a work around utilizing torch.mm for sparse x dense -> dense. Forward and backward options are implemented.
    """
    @staticmethod
    def forward(ctx, t1, t2):
        """
        :param t1: tensor 1
        :param t2: tensor 2
        :return: bmm result in dense
        """
        ctx.save_for_backward(t1, t2)
        assert t1.is_sparse != t2.is_sparse, 't1, t2 must be one sparse and one dense!'
        if t1.is_sparse:
            result = sdd_bmm_torch(t1, t2)
        else:
            result = sdd_bmm_torch(t2.transpose(1, 2), t1.transpose(1, 2)).transpose(1, 2)
        return result

    @staticmethod
    def backward(ctx, dm):
        s_t1, d_t2 = ctx.saved_tensors
        dt1 = dt2 = None
        if ctx.needs_input_grad[0]:
            dt1 = torch.bmm(dm, d_t2.transpose(1, 2))
            dt1 = dense_to_sparse(dt1)
        if ctx.needs_input_grad[1]:
            dt2 = sdd_bmm_torch(s_t1.transpose(1, 2), dm)
        return dt1, dt2


def sdd_bmm_torch(s_t1, d_t2):
    """
    bmm (Batch Matrix Matrix) for sparse x dense -> dense. This function itself doesn't support gradient.
    with s_t1.shape = (b, x, s), d_t2.shape = (b, s, y), the output shape is (b, x, y)
    This is a work around utilizing torch.mm for sparse x dense -> dense
    :param s_t1: sparse tensor 1
    :param d_t2: dense tensor 2
    :return: bmm result in dense
    """
    device = s_t1.device
    batch_num = s_t1.shape[0]
    x = s_t1.shape[1]
    y = d_t2.shape[2]
    assert s_t1.shape[0] == d_t2.shape[0], 'Batch size mismatch.'
    assert s_t1.shape[2] == d_t2.shape[1], 'Matrix shape mismatch.'
    #outp = torch.empty(batch_num, x, y, dtype=s_t1.dtype, device=device)
    for b in range(batch_num):
        _s_t1 = get_batches(s_t1, b)
        outp = torch.mm(_s_t1, d_t2[b, :, :])#, out=outp[b, :, :])
    return outp.view(1, x, y)


def sdd_bmm_diag_torch(t1, t2):
    """
    Perform bmm and diagonal for sparse x dense -> dense. The diagonalized result is returned in vector tensor.
    With s_t1.shape = (b, x, s), d_t2.shape = (b, s, x), the output shape is (b, x).
    This method avoids a temporal (b, x, x) for memory efficiency.
    :param t1: tensor 1
    :param t2: tensor 2
    :return: bmm_diag result in dense
    """
    assert t1.shape[0] == t2.shape[0], 'Batch size mismatch.'
    assert t1.shape[2] == t2.shape[1] and t1.shape[1] == t2.shape[2], 'Matrix shape mismatch.'
    if t1.is_sparse:
        d_t1 = t1.transpose(1, 2).to_dense()
        outp = torch.sum(d_t1.mul_(t2), dim=1)
    else:
        d_t2 = t2.transpose(1, 2).to_dense()
        outp = torch.sum(d_t2.mul_(t1), dim=2)
    return outp


def sds_bmm_torch(s_t1, d_t2):
    """
    bmm (Batch Matrix Matrix) for sparse x dense -> sparse. This function doesn't support gradient.
    And sparse tensors cannot accept gradient due to the limitation of torch implementation.
    with s_t1.shape = (b, x, s), d_t2.shape = (b, s, y), the output shape is (b, x, y)
    This is a work around utilizing torch.smm for sparse x dense -> sparse
    :param s_t1: sparse tensor 1 (in list, representing batches)
    :param d_t2: dense tensor 2
    :return: bmm result in sparse (in list, representing batches)
    """
    device = d_t2.device
    assert type(s_t1) == list
    batch_num = len(s_t1)

    assert batch_num == d_t2.shape[0], 'Batch size mismatch.'

    outp = []
    for b in range(batch_num):
        # force cpu
        _s_t1 = s_t1[b].cpu()
        _d_t2 = d_t2[b].cpu()
        assert _s_t1.shape[1] == _d_t2.shape[0], 'Matrix shape mismatch.'
        _outp = torch.smm(_s_t1, _d_t2)  # CUDA version of smm is not implemented
        outp.append(_outp)

    return outp


def bilinear_diag_torch(s_t1: CSRMatrix3d, d_t2: torch.Tensor, s_t3: CSCMatrix3d, device=None):
    """
    Bilinear and diagonal in sequence, for diagonal(sparse x dense x sparse) -> dense vector.
    with s_t1.shape = (b, x, y), d_t2.shape = (b, y, y), d_t3.shape = (b, y, x), the output shape is (b, x).
    In this function, two sparse tensors (s1 and s3) are represented in CSR and CSC format to guarantee efficient
    computation.
    The main operation is implemented in a custom C++ extension, and will be ~1000x faster if CUDA is available.
    :param s_t1: CSR matrix 1
    :param d_t2: dense tensor 2
    :param s_t3: CSC matrix 3
    :param device: device. If not specified, it will be the same as input.
    :return: returned dense vector
    """
    if device is None:
        device = d_t2.device
    #dtype = d_t2.dtype

    batch_num = s_t1.shape[0]
    xlen = s_t1.shape[1]
    assert s_t1.shape[0] == d_t2.shape[0] == s_t3.shape[0], 'Batch size mismatch.'
    assert s_t1.shape[1] == s_t3.shape[2], 'Sparse matrix 1 & 3 shape mismatch.'
    assert s_t1.shape[2] == d_t2.shape[1] == d_t2.shape[2] == s_t3.shape[1], 'Matrix size mismatch.'

    outp = bilinear_diag.bilinear_diag(*s_t1.as_list(), d_t2, *s_t3.as_list(), batch_num, xlen)

    return outp.to(device)


def dense_to_sparse(d_t):
    """
    Convert a dense tensor to a sparse one.
    :param d_t: dense tensor
    :return: sparse tensor
    """
    dtype = d_t.dtype
    device = d_t.device
    req_grad = d_t.requires_grad

    indices = torch.nonzero(d_t)
    if len(indices.shape) == 0:  # if all elements are zeros
        return torch.sparse_coo_tensor([], [], d_t.shape, dtype=dtype, device=device, requires_grad=req_grad)
    indices = indices.t()
    values = d_t[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse_coo_tensor(indices, values, d_t.size(), dtype=dtype, device=device, requires_grad=req_grad)


def get_batches(s_t, b=None, device=None):
    """
    Get batches from a 3d sparse tensor.
    :param s_t: sparse tensor
    :param b: if None, return all batches in a list; else, return a specific batch
    :param device: device. If None, it will be the same as input
    :return: sparse tensor or list of sparse tensors
    """
    if device is None:
        device = s_t.device

    coo = s_t._indices()
    data = s_t._values()
    if b is not None:
        idx = (coo[0, :] == b).nonzero()
        _coo = coo[1:3, idx].view(2, -1)
        _data = data[idx].view(-1)
        outp = torch.sparse_coo_tensor(_coo, _data, s_t.shape[1:3], dtype=_data.dtype, device=device)
    else:
        batch_num = s_t.shape[0]
        outp = []
        for b in range(batch_num):
            idx = (coo[0, :] == b).nonzero()
            _coo = coo[1:3, idx].view(2, -1)
            _data = data[idx].view(-1)
            outp.append(torch.sparse_coo_tensor(_coo, _data, s_t.shape[1:3], dtype=_data.dtype, device=device))
    return outp


def slicing_torch(s_t, slice, preserve_dim=False):
    """
    A slicing function for torch sparse tensors.
    :param s_t: input sparse tensor
    :param slice: tensor containing indices, -1 stands for all.
                  For example, (1, -1) returns the second row of a 2d tensor.
    :param preserve_dim: If True, the dimension of the original tensor will be preserved,
                         i.e. 1 will be padded for those removed dimensions.
    :return: sliced sparse tensor
    """
    device = s_t.device
    dim = slice.shape[0]
    assert len(s_t.shape) == dim
    coo = s_t._indices()
    data = s_t._values()
    idx_flag = torch.ones(coo.shape[1], dtype=torch.uint8, device=device)
    for i in range(dim):
        s = slice[i]
        if s == -1:
            continue
        _idx_flag = (coo[i, :] == s).view(-1)
        idx_flag.mul_(_idx_flag)
    idx = idx_flag.nonzero().view(-1)
    if not preserve_dim:
        dim_flag = (slice == -1).nonzero().view(-1)
        if dim_flag.numel() == 0:
            coo = torch.tensor([[0]], dtype=coo.dtype, device=device)
            shape = torch.Size([1])
        else:
            coo = coo[:, idx]
            coo = coo[dim_flag, :]
            shape = torch.Size(torch.tensor(s_t.shape)[dim_flag])
    else:
        coo = coo[:, idx]
        coo.mul_((slice == -1).type(coo.dtype).view(-1, 1))
        _dtype = torch.int32
        shape = torch.Size(torch.tensor(s_t.shape, dtype=_dtype, device=device) * (slice == -1).type(_dtype)
                           + torch.ones(len(s_t.shape), dtype=_dtype, device=device) * (slice != -1).type(_dtype))
    data = data[idx]

    return torch.sparse_coo_tensor(coo, data, shape, dtype=s_t.dtype, device=s_t.device)


'''
scipy.sparse API Functions
'''


def sss_bmm_diag_spp(s_m1, s_m2):
    """
    bmm (Batch Matrix Matrix) for sparse x sparse -> sparse. The diagonalized result is returned in vector tensor.
    with s_m1.shape = (b, x, s), s_m2.shape = (b, s, x), the output shape is (b, x)
    This function doesn't support gradient.
    :param s_m1: sparse matrix 1
    :param s_m2: sparse matrix 2
    :return: result in sparse vector
    """
    if type(s_m1) != list:
        s_m1 = [s_m1]
    if type(s_m2) != list:
        s_m2 = [s_m2]
    assert len(s_m1) == len(s_m2), 'Batch size mismatch.'

    outp = []
    for _m1, _m2 in zip(s_m1, s_m2):
        assert _m1.shape[1] == _m2.shape[0] and _m1.shape[0] == _m2.shape[1], 'Matrix shape mismatch.'
        outp.append(_m1.dot(_m2).diagonal().tocoo())

    return outp


'''
Conversion Functions
'''


def ssp2torch(M, batch='dim', dtype=torch.float32, device=None):
    """
    Convert scipy.sparse matrix to torch sparse matrix. Since scipy.sparse has a dimension limit of 2, list of matrices
    is supported for batches.
    :param M: input scipy.sparse matrix
    :param batch: the type that represent batches in the output.
                  If batch='list', tensors are 2d and stored in list.
                  If batch='dim', tensors are 3d ane the first dimension represents batch size.
    :param dtype: output data type
    :param device: device
    :return: output torch sparse matrix
    """
    assert batch in ('list', 'dim')

    if type(M) != list:
        M = [M]
    batch_num = len(M)

    if batch == 'list':
        outp = []
        for i in range(batch_num):
            _M = M[i]
            _M = _M.tocoo()
            coo = np.array([_M.row, _M.col])
            data = _M.data
            outp.append(torch.sparse_coo_tensor(coo, data, _M.shape, dtype=dtype, device=device))
    else:
        batch, row, col, data = np.array([]), np.array([]), np.array([]), np.array([])
        for i in range(batch_num):
            _M = M[i]
            _M = _M.tocoo()
            batch = np.append(batch, np.ones(_M.nnz) * i)
            row = np.append(row, _M.row)
            col = np.append(col, _M.col)
            data = np.append(data, _M.data)

        coo = np.array([batch, row, col])
        outp = torch.sparse_coo_tensor(coo, data, torch.Size([batch_num] + list(_M.shape)), dtype=dtype, device=device)

    return outp


def torch2ssp(M):
    """
    Convert torch sparse matrix to scipy.sparse matrix. Since scipy.sparse has a dimension limit of 2, batches are
    represented in list in the output.
    :param M: input torch sparse matrix
    :return: output scipy.sparse matrix
    """
    if type(M) == list:
        batch_num = len(M)
        outp = []
        for b in range(batch_num):
            _M = M[b]
            _coo = _M._indices()
            _data = _M._values()
            outp.append(ssp.coo_matrix((_data, _coo), _M.shape))
    else:
        coo = M._indices()
        data = M._values()
        batch_num = M.shape[0]

        if len(M.shape) == 2:
            outp = ssp.coo_matrix((data, coo), M.shape)
        else:
            assert len(M.shape) == 3
            outp = []
            for b in range(batch_num):
                idx = (coo[0, :] == b).nonzero()
                _coo = coo[1:3, idx].view(2, -1)
                _data = data[idx].view(-1)
                outp.append(ssp.coo_matrix((_data, _coo), M.shape[1:3]))
    return outp


def recover_ssp(t_dict):
    """
    Recover scipy.sparse coo_matrix from a dictionary containing row, col and data tensors.
    :param t_dict: containing keys
                   'row', 'col', 'data', each corresponds to a bxn tensor
                   'shape', containing the MxN shape of each tensor
    :return: list of scipy.sparse matrix. list indices represent batches.
    """
    batch_size = t_dict['row'].shape[0]
    np_dict = {key: t_dict[key].numpy() for key in t_dict}
    ss = []
    max_shape = np.zeros((2,), dtype=np.int)
    for b in range(batch_size):
        shape = np_dict['shape'][b].astype(np.int)
        max_shape[0] = max(shape[0], max_shape[0])
        max_shape[1] = max(shape[1], max_shape[1])
    for b in range(batch_size):
        data = np_dict['data'][b]
        row = np_dict['row'][b]
        col = np_dict['col'][b]
        _ss = ssp.coo_matrix((data, (row, col)), shape=max_shape)
        ss.append(_ss)
    return ss


if __name__ == '__main__':
    t = torch.tensor([[[ 1,  2,  3,  4],
                       [11, 22, 33, 44]]])
    t = dense_to_sparse(t)
    s = slicing_torch(t, torch.tensor((0, 0, 1)), preserve_dim=True)
    print(s.to_dense())


    from torch.autograd import gradcheck
    input = (dense_to_sparse(torch.randn(1, 20, 30, dtype=torch.double, requires_grad=True)),
             torch.randn(1, 30, 40, dtype=torch.double, requires_grad=True))
    test = gradcheck(sbmm, input, eps=1e-6, atol=1e-4)
    print(test)
