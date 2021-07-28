import torch
from torch import Tensor
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

import itertools
import numpy as np

from typing import Tuple


def build_graphs(P_np: np.ndarray, n: int, n_pad: int=None, edge_pad: int=None, stg: str='fc', sym: bool=True,
                 thre: int=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""
    Build graph matrix :math:`\mathbf G, \mathbf H` from point set :math:`\mathbf P`.
    This function supports only cpu operations in numpy.
    :math:`\mathbf G, \mathbf H` are constructed from adjacency matrix :math:`\mathbf A`:
    :math:`\mathbf A = \mathbf G \cdot \mathbf H^\top`

    :param P_np: :math:`(n\times 2)` point set containing point coordinates
    :param n: number of exact points in the point set
    :param n_pad: padded node length
    :param edge_pad: padded edge length
    :param stg: strategy to build graphs. Options: ``fc``, ``near``, ``tri``
    :param sym: True for a symmetric adjacency, False for half adjacency (A contains only the upper half)
    :param thre: The threshold value of 'near' strategy
    :return: :math:`A`, :math:`G`, :math:`H`, edge_num

    The possible options for ``stg``:
    ::

        'fc'(default): construct a fully-connected graph
        'near': construct a fully-connected graph, but edges longer than ``thre`` are removed
        'tri': apply Delaunay triangulation

    An illustration of :math:`\mathbf G, \mathbf H` with their connections to the graph, the adjacency matrix,
    the incident matrix is

    .. image:: ../../images/build_graphs_GH.png
    """

    assert stg in ('fc', 'tri', 'near'), 'No strategy named {} found.'.format(stg)

    if stg == 'tri':
        A = delaunay_triangulate(P_np[0:n, :])
    elif stg == 'near':
        A = fully_connect(P_np[0:n, :], thre=thre)
    else:
        A = fully_connect(P_np[0:n, :])
    edge_num = int(np.sum(A, axis=(0, 1)))
    assert n > 0 and edge_num > 0, 'Error in n = {} and edge_num = {}'.format(n, edge_num)

    if n_pad is None:
        n_pad = n
    if edge_pad is None:
        edge_pad = edge_num
    assert n_pad >= n
    assert edge_pad >= edge_num

    G = np.zeros((n_pad, edge_pad), dtype=np.float32)
    H = np.zeros((n_pad, edge_pad), dtype=np.float32)
    edge_idx = 0
    for i in range(n):
        if sym:
            range_j = range(n)
        else:
            range_j = range(i, n)
        for j in range_j:
            if A[i, j] == 1:
                G[i, edge_idx] = 1
                H[j, edge_idx] = 1
                edge_idx += 1

    return A, G, H, edge_num


def delaunay_triangulate(P: np.ndarray) -> np.ndarray:
    r"""
    Perform delaunay triangulation on point set P.

    :param P: :math:`(n\times 2)` point set
    :return: adjacency matrix :math:`A`
    """
    n = P.shape[0]
    if n < 3:
        A = fully_connect(P)
    else:
        try:
            d = Delaunay(P)
            #assert d.coplanar.size == 0, 'Delaunay triangulation omits points.'
            A = np.zeros((n, n))
            for simplex in d.simplices:
                for pair in itertools.permutations(simplex, 2):
                    A[pair] = 1
        except QhullError as err:
            print('Delaunay triangulation error detected. Return fully-connected graph.')
            print('Traceback:')
            print(err)
            A = fully_connect(P)
    return A


def fully_connect(P: np.ndarray, thre=None) -> np.ndarray:
    r"""
    Return the adjacency matrix of a fully-connected graph.

    :param P: :math:`(n\times 2)` point set
    :param thre: edges that are longer than this threshold will be removed
    :return: adjacency matrix :math:`A`
    """
    n = P.shape[0]
    A = np.ones((n, n)) - np.eye(n)
    if thre is not None:
        for i in range(n):
            for j in range(i):
                if np.linalg.norm(P[i] - P[j]) > thre:
                    A[i, j] = 0
                    A[j, i] = 0
    return A


def make_grids(start, stop, num) -> np.ndarray:
    r"""
    Make grids.

    This function supports only cpu operations in numpy.

    :param start: start index in all dimensions
    :param stop: stop index in all dimensions
    :param num: number of grids in each dimension
    :return: point set P
    """
    length = np.prod(num)
    P = np.zeros((length, len(num)), dtype=np.float32)
    assert len(start) == len(stop) == len(num)
    for i, (begin, end, n) in enumerate(zip(start, stop, num)):
        g = np.linspace(begin, end, n + 1)
        g -= (g[1] - g[0]) / 2
        g = g[1:]
        P[:, i] = np.reshape(np.repeat([g], length / n, axis=i), length)
    return P


def reshape_edge_feature(F: Tensor, G: Tensor, H: Tensor, device=None) -> Tensor:
    r"""
    Given point-level features extracted from images, reshape it into edge feature matrix :math:`X`,
    where features are arranged by the order of :math:`G`, :math:`H`.

    .. math::
        \mathbf{X}_{e_{ij}} = concat(\mathbf{F}_i, \mathbf{F}_j)

    where :math:`e_{ij}` means an edge connecting nodes :math:`i, j`

    :param F: :math:`(b\times d \times n)` extracted point-level feature matrix.
     :math:`b`: batch size. :math:`d`: feature dimension. :math:`n`: number of nodes.
    :param G: :math:`(b\times n \times e)` factorized adjacency matrix, where :math:`\mathbf A = \mathbf G \cdot \mathbf H^\top`. :math:`e`: number of edges.
    :param H: :math:`(b\times n \times e)` factorized adjacency matrix, where :math:`\mathbf A = \mathbf G \cdot \mathbf H^\top`
    :param device: device. If not specified, it will be the same as the input
    :return: edge feature matrix X :math:`(b \times 2d \times e)`
    """
    if device is None:
        device = F.device

    batch_num = F.shape[0]
    feat_dim = F.shape[1]
    point_num, edge_num = G.shape[1:3]
    X = torch.zeros(batch_num, 2 * feat_dim, edge_num, dtype=torch.float32, device=device)
    X[:, 0:feat_dim, :] = torch.matmul(F, G)
    X[:, feat_dim:2*feat_dim, :] = torch.matmul(F, H)

    return X
