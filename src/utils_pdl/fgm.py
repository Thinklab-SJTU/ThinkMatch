import paddle
from paddle import Tensor
from src.utils.sparse import bilinear_diag_torch
from src.sparse_torch import CSRMatrix3d, CSCMatrix3d
import scipy.sparse as ssp
import numpy as np


def construct_m(Me: Tensor, Mp: Tensor, KG: CSRMatrix3d, KH: CSCMatrix3d, KGt: CSRMatrix3d=None, KHt: CSCMatrix3d=None):
    """
    Construct full affinity matrix M with edge matrix Me, point matrix Mp and graph structures G1, H1, G2, H2
    :param Me: edge affinity matrix
    :param Mp: point affinity matrix
    :param KG: kronecker product of G2, G1
    :param KH: kronecker product of H2, H1
    :param KGt: transpose of KG (should be CSR, optional)
    :param KHt: transpose of KH (should be CSC, optional)
    :return: M
    """
    '''
    if device is None:
        device = Me.device
    batch_num = Me.shape[0]
    feat_size = G1.shape[1] * G2.shape[1]
    '''

    M = RebuildFGM.apply(Me, Mp, KG, KH, KGt, KHt)

    '''
        K1 = kronecker_paddle(G2, G1).to(device)
        K2 = kronecker_paddle(H2, H1).to(device)
        Me = Me.to(device)
        Mp = Mp.to(device)
        M = paddle.empty(batch_num, feat_size, feat_size, dtype=paddle.float32, device=device)
        for i in range(batch_num):
            M[i, :, :] = paddle.mm(paddle.mm(K1[i, :, :], paddle.diag(Me[i, :, :].t().contiguous().view(-1))), K2[i, :, :].t())
            M[i, :, :] += paddle.diag(Mp[i, :, :].t().contiguous().view(-1))
    '''
    return M


def kronecker_paddle(t1: Tensor, t2: Tensor):
    """
    Compute the kronecker product of t1 (*) t2.
    This function is implemented in paddle API and is not efficient for sparse {0, 1} matrix.
    :param t1: input tensor 1
    :param t2: input tensor 2
    :return: t1 (*) t2
    """
    batch_num = t1.shape[0]
    t1dim1, t1dim2 = t1.shape[1], t1.shape[2]
    t2dim1, t2dim2 = t2.shape[1], t2.shape[2]
    t1 = t1.reshape((batch_num, -1, 1))
    t2 = t2.reshape((batch_num, 1, -1))
    tt = paddle.bmm(t1, t2)
    tt = tt.reshape((batch_num, t1dim1, t1dim2, t2dim1, t2dim2))
    tt = tt.transpose([0, 1, 3, 2, 4])
    tt = tt.reshape((batch_num, t1dim1 * t2dim1, t1dim2 * t2dim2))
    return tt


def kronecker_sparse(arr1: np.ndarray, arr2: np.ndarray):
    """
    Compute the kronecker product of t1 (*) t2.
    This function is implemented in scipy.sparse API and runs on cpu.
    :param arr1: input array 1
    :param arr2: input array 2
    :return: list of t1 (*) t2 (for tensors in a batch)
    """
    s1 = ssp.coo_matrix(arr1)
    s2 = ssp.coo_matrix(arr2)
    ss = ssp.kron(s1, s2)
    return ss


class RebuildFGM(paddle.nn.Layer):
    """
    Rebuild sparse affinity matrix in the formula of CVPR12's paper "Factorized Graph Matching"
    """
    @staticmethod
    def forward(ctx, Me: Tensor, Mp: Tensor, K1: CSRMatrix3d, K2: CSCMatrix3d, K1t: CSRMatrix3d=None, K2t: CSCMatrix3d=None):
        ctx.save_for_backward(Me, Mp)
        if K1t is not None and K2t is not None:
            ctx.K = K1t, K2t
        else:
            ctx.K = K1.transpose(keep_type=True), K2.transpose(keep_type=True)
        batch_num = Me.shape[0]

        #print('rebuild fgm')
        #print('K1.shape', K1.shape)
        #print('K2.shape', K2.shape)

        K1Me = K1.dotdiag(Me.transpose((0, 2, 1)).reshape((batch_num, -1)))
        K1MeK2 = K1Me.dot(K2, dense=True)

        M = paddle.empty_like(K1MeK2)
        for b in range(batch_num):
            M[b] = K1MeK2[b] + paddle.diag(Mp[b].transpose((1, 0)).reshape(-1))

        return M

    @staticmethod
    def backward(ctx, dM):
        place = dM.place
        Me, Mp = ctx.saved_tensors
        K1t, K2t = ctx.K
        dMe = dMp = None
        if ctx.needs_input_grad[0]:
            dMe = bilinear_diag_paddle(K1t, dM, K2t)
            dMe = dMe.reshape((Me.shape[0], Me.shape[2], Me.shape[1])).transpose((0, 2, 1))
        if ctx.needs_input_grad[1]:
            dM_p = np.diagonal(dM.numpy(), axis1=-2, axis2=-1)
            dMp = paddle.to_tensor(dM_p, place=place)
            dMp = dMp.reshape((Mp.shape[0], Mp.shape[2], Mp.shape[1])).transpose((0, 2, 1))

        return dMe, dMp, None, None, None, None
