import sys
import torch
import numpy as np
import scipy.sparse as ssp

from torch.utils.cpp_extension import load

sparse_dot = load(name='sparse_dot',
                  sources=['extension/sparse_dot/sparse_dot.cpp',
                           'extension/sparse_dot/csr_dot_csc_cuda.cu',
                           'extension/sparse_dot/csr_dot_diag_cuda.cu'],
                  extra_include_paths=[
                      '/usr/include/python{}.{}/'.format(sys.version_info.major, sys.version_info.minor)]
)


class CSXMatrix3d:
    def __init__(self, inp, shape, device=None):
        def from_ssp(inp_s: list, shape, device=None, sptype=self.sptype):
            """
            Load data from list of scipy.sparse matrix
            :param inp_s: list of input scipy.sparse matrix
            :param shape: output matrix shape.
            :param device: device. If not specified, it will be the same as input.
            :param sptype: sparse matrix type. Should be 'csr' or 'csc'
            """
            assert len(shape) == 3, 'Only 3-dimensional tensor (bxhxw) is supported'
            batch_num = shape[0]

            indices = []
            indptr = []
            data = []
            indptr_offset = 0

            for b in range(batch_num):
                if sptype == 'csc':
                    inp_s[b].eliminate_zeros()
                    sp = inp_s[b].tocsc().astype(dtype=inp_s[b].dtype)
                elif sptype == 'csr':
                    inp_s[b].eliminate_zeros()
                    sp = inp_s[b].tocsr().astype(dtype=inp_s[b].dtype)
                else:
                    raise ValueError('Sparse type not understood {}'.format(sptype))

                indices.append(sp.indices)
                indptr.append(sp.indptr[:-1] + indptr_offset)
                data.append(sp.data)
                indptr_offset += sp.indptr[-1]
            indptr.append(np.array([indptr_offset]))

            return from_tensors(*[np.concatenate(x) for x in (indices, indptr, data)], shape=shape,
                                device=device)

        def from_tensors(ind: torch.Tensor or np.ndarray, indp: torch.Tensor or np.ndarray,
                         data: torch.Tensor or np.ndarray, shape, device=None):
            """
            Load data from raw input tensors/arrays.
            :param ind: indices array/tensor
            :param indp: indptr array/tensor
            :param data: data array/tensor
            :param shape: output matrix shape.
            :param device: device. Optional
            :return: indices(Tensor), indptr(Tensor), data(Tensor), shape(tuple)
            """
            if type(ind) == torch.Tensor and device is None:
                device = ind.device

            if type(ind) is torch.Tensor:
                indices_t = ind.to(torch.int64).to(device)
            else:
                indices_t = torch.tensor(ind, dtype=torch.int64, device=device)
            if type(indp) is torch.Tensor:
                indptr_t = indp.to(torch.int64).to(device)
            else:
                indptr_t = torch.tensor(indp, dtype=torch.int64, device=device)
            if type(data) is torch.Tensor:
                data_t = data.to(dtype=data.dtype).to(device)
            else:
                data_t = torch.tensor(data, device=device)

            return indices_t, indptr_t, data_t, tuple(shape)

        if type(inp) == list and isinstance(inp[0], ssp.spmatrix):
            self.indices, self.indptr, self.data, self.shape = from_ssp(inp, shape, device)

        elif type(inp) == list:
            self.indices, self.indptr, self.data, self.shape = from_tensors(*inp, shape, device)

        else:
            raise ValueError('Data type {} not understood.'.format(type(inp)))

    def __getitem__(self, item):
        """
        Get item through slicing. The slicing is only supported on the batch dimention
        :param item: index or slice
        :return: new sparse matrix
        """
        if isinstance(item, int):
            indices, indptr, data = self.get_batch(item)
            return self.__class__([indices, indptr, data], shape=[1] + list(self.shape[1:3]))
        elif isinstance(item, slice):
            indices = []
            indptr = []
            data = []
            indptr_offset = int(0)
            batch_iter = range(item.start, item.stop, item.step if item.step is not None else 1)
            for b in batch_iter:
                _indices, _indptr, _data = self.get_batch(b)
                indices.append(_indices)
                indptr.append(_indptr[:-1] + indptr_offset)
                data.append(_data)
                indptr_offset = indptr_offset + _indptr[-1]
            assert isinstance(indptr_offset, torch.Tensor)
            indptr.append(indptr_offset.view(1))

            indices = torch.cat(indices)
            indptr = torch.cat(indptr)
            data = torch.cat(data)
            return self.__class__([indices, indptr, data], shape=[len(batch_iter)] + list(self.shape[1:3]))
        else:
            raise ValueError('Index type {} not supported.'.format(type(item)))

    def __len__(self):
        return self.shape[0]

    @property
    def device(self):
        return self.indices.device

    @property
    def sptype(self):
        raise NotImplementedError

    def transpose(self, keep_type=False):
        raise NotImplementedError

    def to(self, tgt):
        """
        Compatible to torch.Tensor.to()
        :param tgt: target, can be torch.device or torch.dtype
        :return: a new instance
        """
        if isinstance(tgt, torch.device):
            return self.__class__([x.to(tgt) for x in [self.indices, self.indptr, self.data]], self.shape)
        elif isinstance(tgt, torch.dtype):
            return self.__class__([self.indices, self.indptr, self.data.to(tgt)], self.shape)
        else:
            raise ValueError('Data type not understood.')

    def cuda(self):
        """
        Compatible to torch.Tensor.cuda()
        :return: a new instance on CUDA
        """
        return self.__class__([x.cuda() for x in [self.indices, self.indptr, self.data]], self.shape)

    def cpu(self):
        """
        Compatible to torch.Tensor.cpu()
        :return: a new instance on CPU
        """
        return self.__class__([x.cpu() for x in [self.indices, self.indptr, self.data]], self.shape)

    def numpy(self):
        """
        Return dense numpy array.
        :return: dense numpy array.
        """
        ret = [x.toarray() for x in self.as_ssp()]
        ret = np.stack(ret, axis=0)
        return ret

    def as_list(self, mask=None):
        """
        Return [indices, indptr, data] in a list.
        :param mask: Optional. It should be an iterable containing 3 items, each indicating its corresponding attribute
                     shall be masked out or not.
        :return: [indices, indptr, data] * mask
        """
        attrs = [self.indices, self.indptr, self.data]
        if mask is not None:
            ret = []
            for m, a in zip(mask, attrs):
                if m:
                    ret.append(a)
        else:
            ret = attrs
        return ret

    def as_ssp(self):
        """
        Return scipy.sparse matrix.
        :return: list of scipy.sparse matrix
        """
        ret = []
        for b in range(self.shape[0]):
            indice, indptr, data = self.get_batch(b)
            construct_func = ssp.csr_matrix if self.sptype == 'csr' else ssp.csc_matrix
            ret.append(
                construct_func(
                    (data.cpu().to(dtype=data.dtype).numpy(),
                     indice.cpu().numpy(),
                     indptr.cpu().numpy()),
                    shape=self.shape[1:3]
                )
            )
        return ret

    def get_batch(self, item):
        """
        Get a certain batch in tuple (indices, indptr, data)
        :param item: batch index
        :return: (indices, indptr, data)
        """
        if type(item) != int:
            raise IndexError('Only int indices is currently supported.')

        if self.sptype == 'csr':
            start_idx = item * self.shape[1]
            end_idx = (item + 1) * self.shape[1] + 1
        elif self.sptype == 'csc':
            start_idx = item * self.shape[2]
            end_idx = (item + 1) * self.shape[2] + 1
        else:
            raise ValueError('Data type not understood.')
        indptr = self.indptr[start_idx: end_idx].clone()
        indice = self.indices[indptr[0]: indptr[-1]].clone()
        data = self.data[indptr[0]: indptr[-1]].clone()
        indptr = indptr - indptr[0]
        return indice, indptr, data

    def shape_eq(self, other):
        ret = True
        for s_shape, o_shape in zip(self.shape, other.shape):
            if s_shape != o_shape:
                ret = False
                break
        return ret


class CSCMatrix3d(CSXMatrix3d):
    def __init__(self, inp, shape=None, device=None):
        if type(inp) == list and isinstance(inp[0], ssp.spmatrix):
            max_shape = [0, 0]
            for s in inp:
                max_shape[0] = max(max_shape[0], s.shape[0])
                max_shape[1] = max(max_shape[1], s.shape[1])
            if shape is None:
                shape = tuple([len(inp)] + max_shape)
            else:
                assert shape[0] == len(inp)
                assert shape[1] <= max_shape[0]
                assert shape[2] <= max_shape[1]

        elif type(inp) == list:
            assert shape is not None
            batch = shape[0]
            row = _max(inp[0])
            col = (len(inp[1]) - 1) // batch
            assert shape[1] >= row
            assert shape[2] == col

        super(CSCMatrix3d, self).__init__(inp, shape, device)

    @property
    def sptype(self):
        return 'csc'

    def transpose(self, keep_type=False):
        if not keep_type:
            shape_t = list(self.shape)
            tmp = shape_t[1]
            shape_t[1] = shape_t[2]
            shape_t[2] = tmp
            return CSRMatrix3d(self.as_list(), shape=shape_t, device=self.device)
        else:
            coo = []
            for sp in self.as_ssp():
                coo.append(sp.transpose().tocoo().astype(sp.dtype))
            return CSCMatrix3d(coo, device=self.device)

    def Tdot(self, other, *args, **kwargs):
        """
        The dot result of a TRANSPOSED CSC matrix and another CSC matrix.
        This is equivalent to CSR dot CSC.
        :param other: second CSC matrix
        :return: dot product in a new CSR matrix
        """
        t_csr = self.transpose()
        return dot(t_csr, other, *args, **kwargs)


class CSRMatrix3d(CSXMatrix3d):
    def __init__(self, inp, shape=None, device=None):
        if type(inp) == list and isinstance(inp[0], ssp.spmatrix):
            max_shape = [0, 0]
            for s in inp:
                max_shape[0] = max(max_shape[0], s.shape[0])
                max_shape[1] = max(max_shape[1], s.shape[1])
            if shape is None:
                shape = tuple([len(inp)] + max_shape)
            else:
                assert shape[0] == len(inp)
                assert shape[1] <= max_shape[0]
                assert shape[2] <= max_shape[1]

        elif type(inp) == list:
            assert shape is not None
            batch = shape[0]
            row = (len(inp[1]) - 1) // batch
            col = _max(inp[0])
            assert shape[1] == row
            assert shape[2] >= col

        super(CSRMatrix3d, self).__init__(inp, shape, device)

    @property
    def sptype(self):
        return 'csr'

    def transpose(self, keep_type=False):
        if not keep_type:
            shape_t = list(self.shape)
            tmp = shape_t[1]
            shape_t[1] = shape_t[2]
            shape_t[2] = tmp
            return CSCMatrix3d(self.as_list(), shape=shape_t, device=self.device)
        else:
            coo = []
            for sp in self.as_ssp():
                coo.append(sp.transpose().tocoo().astype(sp.dtype))
            return CSRMatrix3d(coo, device=self.device)

    def dot(self, other, *args, **kwargs):
        """
        Dot product of this CSR matrix and a CSC matrix.
        :param other: CSC matrix.
        :return: dot product in CSR matrix
        """
        return dot(self, other, *args, **kwargs)

    def dotdiag(self, other):
        """
        Dot product of this CSR matrix and a diagonal matrix from a vector.
        :param other: input vector.
        :return: dot product in CSR matrix
        """
        assert self.shape[0] == other.shape[0], 'Batch size mismatch'
        assert self.shape[2] == other.shape[1], 'Matrix shape mismatch'
        batch_size = self.shape[0]
        out_h = self.shape[1]
        out_w = self.shape[2]

        result = sparse_dot.csr_dot_diag(*self.as_list(), other, batch_size, out_h, out_w)
        ret = CSRMatrix3d(result, shape=self.shape)
        '''
        indptr = self.indptr.clone()
        indice = self.indices.clone()
        data = self.data.clone()

        for b in range(batch_size):
            start_idx = b * self.shape[1]
            end_idx = (b + 1) * self.shape[1] + 1
            indp_b = indptr[start_idx: end_idx]
            indx_b = indice[indp_b[0]: indp_b[-1]]
            data_b = data[indp_b[0]: indp_b[-1]]

            for j in range(self.shape[2]):
                data_b[indx_b == j] *= other[b, j]

        ret = CSRMatrix3d([indice, indptr, data], self.shape)
        '''
        return ret


def dot(csr: CSRMatrix3d, csc: CSCMatrix3d, dense=False):
    """
    Compute the dot product of one CSR matrix and one CSC matrix. The result will be returned in a new CSR or dense
    matrix. Note that CUDA implementations do not work when dense=False.
    :param csr: fist input CSR matrix
    :param csc: second input CSC matrix
    :param dense: output matrix in dense format
    :return: dot result in new csr matrix (dense=False) or
             dot result in dense matrix (dense=True)
    """
    assert type(csr) == CSRMatrix3d
    assert type(csc) == CSCMatrix3d
    assert csr.shape[0] == csc.shape[0], 'Batch size mismatch'
    batch_num = csr.shape[0]
    assert csr.shape[2] == csc.shape[1], 'Matrix size mismatch'
    out_h = csr.shape[1]
    out_w = csc.shape[2]

    if csr.indptr.device == torch.device('cpu'):
        new_indices, new_indptr, new_data = \
            sparse_dot.csr_dot_csc(*csr.as_list(), *csc.as_list(), batch_num, out_h, out_w)
        ret = CSRMatrix3d([new_indices, new_indptr, new_data], shape=(batch_num, out_h, out_w))
        if dense:
            ret = ret.numpy()
    else:
        if not dense:
            raise RuntimeWarning('Sparse dot product result in CUDA is not implemented.')
        ret = sparse_dot.csr_dot_csc_dense_cuda(*csr.as_list(), *csc.as_list(), batch_num, out_h, out_w)
    return ret


def concatenate(*mats: CSXMatrix3d, device=None):
    """
    Concatenate multiple sparse matrix in first (batch) dimension.
    :param mats: sequence of input matrix
    :return: concatenated matrix
    """
    device = mats[0].device if device is None else device

    mat_type = type(mats[0])
    mat_h = mats[0].shape[1]
    mat_w = mats[0].shape[2]
    batch_size = 0

    indptr_offset = 0
    indices = []
    indptr = []
    data = []
    for mat in mats:
        assert type(mat) == mat_type, 'Matrix type inconsistent'
        assert mat.shape[1] == mat_h, 'Matrix shape inconsistent in dimension 1'
        assert mat.shape[2] == mat_w, 'Matrix shape inconsistent in dimension 2'
        indices.append(mat.indices.clone().to(device))
        indptr.append(mat.indptr[:-1].clone().to(device) + indptr_offset)
        data.append(mat.data.clone().to(device))
        indptr_offset += mat.indptr[-1]
        indptr_offset = indptr_offset.to(device)
        batch_size += mat.shape[0]

    indptr.append(indptr_offset)

    indices = torch.cat(indices)
    indptr = torch.cat(indptr)
    data = torch.cat(data)

    return mat_type([indices, indptr, data], shape=(batch_size, mat_h, mat_w))


def _max(inp, *args, **kwargs):
    if type(inp) == np.ndarray:
        return np.max(inp, *args, **kwargs)
    elif type(inp) == torch.Tensor:
        return torch.max(inp, *args, **kwargs)
    else:
        raise ValueError('Data type {} not understood.'.format(type(inp)))
