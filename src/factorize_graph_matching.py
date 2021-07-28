import torch
from torch import Tensor
from torch.autograd import Function
from src.utils.sparse import bilinear_diag_torch
from src.sparse_torch import CSRMatrix3d, CSCMatrix3d
import scipy.sparse as ssp
import numpy as np


def construct_aff_mat(Ke: Tensor, Kp: Tensor, KroG: CSRMatrix3d, KroH: CSCMatrix3d,
                      KroGt: CSRMatrix3d=None, KroHt: CSCMatrix3d=None) -> Tensor:
    r"""
    Construct the complete affinity matrix with edge-wise affinity matrix :math:`\mathbf{K}_e`, node-wise matrix
    :math:`\mathbf{K}_p` and graph connectivity matrices :math:`\mathbf{G}_1, \mathbf{H}_1, \mathbf{G}_2, \mathbf{H}_2`

    .. math ::
        \mathbf{K}=\mathrm{diag}(\mathrm{vec}(\mathbf{K}_p)) +
        (\mathbf{G}_2 \otimes_{\mathcal{K}} \mathbf{G}_1) \mathrm{diag}(\mathrm{vec}(\mathbf{K}_e))
        (\mathbf{H}_2 \otimes_{\mathcal{K}} \mathbf{H}_1)^\top

    where :math:`\mathrm{diag}(\cdot)` means building a diagonal matrix based on the given vector,
    and :math:`\mathrm{vec}(\cdot)` means column-wise vectorization.
    :math:`\otimes_{\mathcal{K}}` denotes Kronecker product.

    This function supports batched operations. This formulation is developed by `"F. Zhou and F. Torre. Factorized
    Graph Matching. TPAMI 2015." <http://www.f-zhou.com/gm/2015_PAMI_FGM_Draft.pdf>`_

    :param Ke: :math:`(b\times n_{e_1}\times n_{e_2})` edge-wise affinity matrix.
     :math:`n_{e_1}`: number of edges in graph 1, :math:`n_{e_2}`: number of edges in graph 2
    :param Kp: :math:`(b\times n_1\times n_2)` node-wise affinity matrix.
     :math:`n_1`: number of nodes in graph 1, :math:`n_2`: number of nodes in graph 2
    :param KroG: :math:`(b\times n_1n_2 \times n_{e_1}n_{e_2})` kronecker product of
     :math:`\mathbf{G}_2 (b\times n_2 \times n_{e_2})`, :math:`\mathbf{G}_1 (b\times n_1 \times n_{e_1})`
    :param KroH: :math:`(b\times n_1n_2 \times n_{e_1}n_{e_2})` kronecker product of
     :math:`\mathbf{H}_2 (b\times n_2 \times n_{e_2})`, :math:`\mathbf{H}_1 (b\times n_1 \times n_{e_1})`
    :param KroGt: transpose of KroG (should be CSR, optional)
    :param KroHt: transpose of KroH (should be CSC, optional)
    :return: affinity matrix :math:`\mathbf K`

    .. note ::
        This function is optimized with sparse CSR and CSC matrices with GPU support for both forward and backward
        computation with PyTorch. To use this function, you need to install ``ninja-build``, ``gcc-7``, ``nvcc`` (which
        comes along with CUDA development tools) to successfully compile our customized CUDA code for CSR and CSC
        matrices. The compiler is automatically called upon requirement.

    For a graph matching problem with 5 nodes and 4 nodes,
    the connection of :math:`\mathbf K` and :math:`\mathbf{K}_p, \mathbf{K}_e` is illustrated as

    .. image :: ../../images/factorized_graph_matching.png

    where :math:`\mathbf K (20 \times 20)` is the complete affinity matrix, :math:`\mathbf{K}_p (5 \times 4)` is the
    node-wise affinity matrix, :math:`\mathbf{K}_e(16 \times 10)` is the edge-wise affinity matrix.
    """
    return RebuildFGM.apply(Ke, Kp, KroG, KroH, KroGt, KroHt)


def kronecker_torch(t1: Tensor, t2: Tensor) -> Tensor:
    r"""
    Compute the kronecker product of :math:`\mathbf{T}_1` and :math:`\mathbf{T}_2`.
    This function is implemented in torch API and is not efficient for sparse {0, 1} matrix.

    :param t1: input tensor 1
    :param t2: input tensor 2
    :return: kronecker product of :math:`\mathbf{T}_1` and :math:`\mathbf{T}_2`
    """
    batch_num = t1.shape[0]
    t1dim1, t1dim2 = t1.shape[1], t1.shape[2]
    t2dim1, t2dim2 = t2.shape[1], t2.shape[2]
    if t1.is_sparse and t2.is_sparse:
        tt_idx = torch.stack(t1._indices()[0, :] * t2dim1, t1._indices()[1, :] * t2dim2)
        tt_idx = torch.repeat_interleave(tt_idx, t2._nnz(), dim=1) + t2._indices().repeat(1, t1._nnz())
        tt_val = torch.repeat_interleave(t1._values(), t2._nnz(), dim=1) * t2._values().repeat(1, t1._nnz())
        tt = torch.sparse.FloatTensor(tt_idx, tt_val, torch.Size(t1dim1 * t2dim1, t1dim2 * t2dim2))
    else:
        t1 = t1.reshape(batch_num, -1, 1)
        t2 = t2.reshape(batch_num, 1, -1)
        tt = torch.bmm(t1, t2)
        tt = tt.reshape(batch_num, t1dim1, t1dim2, t2dim1, t2dim2)
        tt = tt.permute([0, 1, 3, 2, 4])
        tt = tt.reshape(batch_num, t1dim1 * t2dim1, t1dim2 * t2dim2)
    return tt


def kronecker_sparse(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    r"""
    Compute the kronecker product of :math:`\mathbf{T}_1` and :math:`\mathbf{T}_2`.
    This function is implemented in scipy.sparse API and runs on cpu.

    :param arr1: input array 1
    :param arr2: input array 2
    :return: kronecker product of :math:`\mathbf{T}_1` and :math:`\mathbf{T}_2`
    """
    s1 = ssp.coo_matrix(arr1)
    s2 = ssp.coo_matrix(arr2)
    ss = ssp.kron(s1, s2)
    return ss


class RebuildFGM(Function):
    r"""
    Rebuild sparse affinity matrix in the formula of the paper `"Factorized Graph Matching, in
    TPAMI 2015" <http://www.f-zhou.com/gm/2015_PAMI_FGM_Draft.pdf>`_

    See :func:`~src.factorize_graph_matching.construct_aff_mat` for detailed reference.
    """
    @staticmethod
    def forward(ctx, Ke: Tensor, Kp: Tensor, Kro1: CSRMatrix3d, Kro2: CSCMatrix3d,
                Kro1T: CSRMatrix3d=None, Kro2T: CSCMatrix3d=None) -> Tensor:
        """
        Forward function to compute the affinity matrix :math:`\mathbf K`.
        """
        ctx.save_for_backward(Ke, Kp)
        if Kro1T is not None and Kro2T is not None:
            ctx.K = Kro1T, Kro2T
        else:
            ctx.K = Kro1.transpose(keep_type=True), Kro2.transpose(keep_type=True)
        batch_num = Ke.shape[0]

        Kro1Ke = Kro1.dotdiag(Ke.transpose(1, 2).contiguous().view(batch_num, -1))
        Kro1KeKro2 = Kro1Ke.dot(Kro2, dense=True)

        K = torch.empty_like(Kro1KeKro2)
        for b in range(batch_num):
            K[b] = Kro1KeKro2[b] + torch.diag(Kp[b].transpose(0, 1).contiguous().view(-1))

        return K

    @staticmethod
    def backward(ctx, dK):
        r"""
        Backward function from the affinity matrix :math:`\mathbf K` to node-wise affinity matrix :math:`\mathbf K_e`
        and edge-wize affinity matrix :math:`\mathbf K_e`.
        """
        device = dK.device
        Ke, Kp = ctx.saved_tensors
        Kro1t, Kro2t = ctx.K
        dKe = dKp = None
        if ctx.needs_input_grad[0]:
            dKe = bilinear_diag_torch(Kro1t, dK.contiguous(), Kro2t)
            dKe = dKe.view(Ke.shape[0], Ke.shape[2], Ke.shape[1]).transpose(1, 2)
        if ctx.needs_input_grad[1]:
            dKp = torch.diagonal(dK, dim1=-2, dim2=-1)
            dKp = dKp.view(Kp.shape[0], Kp.shape[2], Kp.shape[1]).transpose(1, 2)

        return dKe, dKp, None, None, None, None
