import torch
from torch import Tensor
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

import itertools
import numpy as np


def build_graphs(P_np: np.ndarray, n: int, n_pad: int=None, edge_pad: int=None, stg: str='fc'):
    """
    Build graph matrix G,H from point set P. This function supports only cpu operations in numpy.
    G, H is constructed from adjacency matrix A: A = G * H^T
    :param P_np: point set containing point coordinates
    :param n: number of exact points in the point set
    :param n_pad: padded node length
    :param edge_pad: padded edge length
    :param stg: strategy to build graphs.
                'tri', apply Delaunay triangulation or not.
                'near', fully-connected manner, but edges which are longer than max(w, h) is removed
                'fc'(default), a fully-connected graph is constructed
    :param device: device. If not specified, it will be the same as the input
    :return: G, H, edge_num
    """

    assert stg in ('fc', 'tri', 'near'), 'No strategy named {} found.'.format(stg)

    if stg == 'tri':
        A = delaunay_triangulate(P_np[0:n, :])
    elif stg == 'near':
        A = fully_connect(P_np[0:n, :], thre=0.5*256)
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
        for j in range(n):
            if A[i, j] == 1:
                G[i, edge_idx] = 1
                H[j, edge_idx] = 1
                edge_idx += 1

    return G, H, edge_num


def delaunay_triangulate(P: np.ndarray):
    """
    Perform delaunay triangulation on point set P.
    :param P: point set
    :return: adjacency matrix A
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


def fully_connect(P: np.ndarray, thre=None):
    """
    Fully connect a graph.
    :param P: point set
    :param thre: edges that are longer than this threshold will be removed
    :return: adjacency matrix A
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


def reshape_edge_feature(F: Tensor, G: Tensor, H: Tensor, device=None):
    """
    Reshape edge feature matrix into X, where features are arranged in the order in G, H.
    :param F: raw edge feature matrix
    :param G: factorized adjacency matrix, where A = G * H^T
    :param H: factorized adjacancy matrix, where A = G * H^T
    :param device: device. If not specified, it will be the same as the input
    :return: X
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
