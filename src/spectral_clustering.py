import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple

def initialize(X: Tensor, num_clusters: int, method: str='plus') -> np.array:
    r"""
    Initialize cluster centers.

    :param X: matrix
    :param num_clusters: number of clusters
    :param method: denotes different initialization strategies: ``'plus'`` (default) or ``'random'``
    :return: initial state

    .. note::
        We support two initialization strategies: random initialization by setting ``method='random'``, or `kmeans++
        <https://en.wikipedia.org/wiki/K-means%2B%2B>`_ by setting ``method='plus'``.
    """
    if method == 'plus':
        init_func = _initialize_plus
    elif method == 'random':
        init_func = _initialize_random
    else:
        raise NotImplementedError
    return init_func(X, num_clusters)


def _initialize_random(X, num_clusters):
    """
    Initialize cluster centers randomly. See :func:`src.spectral_clustering.initialize` for details.
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def _initialize_plus(X, num_clusters):
    """
    Initialize cluster centers by k-means++. See :func:`src.spectral_clustering.initialize` for details.
    """
    num_samples = len(X)
    centroid_index = np.zeros(num_clusters)
    for i in range(num_clusters):
        if i == 0:
            choice_prob = np.full(num_samples, 1 / num_samples)
        else:
            centroid_X = X[centroid_index[:i]]
            dis = _pairwise_distance(X, centroid_X)
            dis_to_nearest_centroid = torch.min(dis, dim=1).values
            choice_prob = dis_to_nearest_centroid / torch.sum(dis_to_nearest_centroid)
            choice_prob = choice_prob.detach().cpu().numpy()

        centroid_index[i] = np.random.choice(num_samples, 1, p=choice_prob, replace=False)

    initial_state = X[centroid_index]
    return initial_state

def kmeans(
        X: Tensor,
        num_clusters: int,
        init_x: Union[Tensor, str]='plus',
        distance: str='euclidean',
        tol: float=1e-4,
        device=torch.device('cpu'),
) -> Tuple[Tensor, Tensor]:
    r"""
    Perform kmeans on given data matrix :math:`\mathbf X`.

    :param X: :math:`(n\times d)` input data matrix. :math:`n`: number of samples. :math:`d`: feature dimension
    :param num_clusters: (int) number of clusters
    :param init_x: how to initiate x (provide a initial state of x or define a init method) [default: 'plus']
    :param distance: distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: convergence threshold [default: 0.0001]
    :param device: computing device [default: cpu]
    :return: cluster ids, cluster centers
    """
    if distance == 'euclidean':
        pairwise_distance_function = _pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = _pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if type(init_x) is str:
        initial_state = initialize(X, num_clusters, method=init_x)
    else:
        initial_state = init_x

    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index, as_tuple=False).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        if center_shift ** 2 < tol:
            break

        if torch.isnan(initial_state).any():
            print('NAN encountered in clustering. Retrying...')
            initial_state = initialize(X, num_clusters)

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X: Tensor,
        cluster_centers: Tensor,
        distance: str='euclidean',
        device=torch.device('cpu')
) -> Tensor:
    r"""
    Kmeans prediction using existing cluster centers.

    :param X: matrix
    :param cluster_centers: cluster centers
    :param distance: distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: computing device [default: 'cpu']
    :return: cluster ids
    """
    if distance == 'euclidean':
        pairwise_distance_function = _pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = _pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def _pairwise_distance(data1, data2, device=torch.device('cpu')):
    """Compute pairwise Euclidean distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1) #.squeeze(-1)
    return dis


def _pairwise_cosine(data1, data2, device=torch.device('cpu')):
    """Compute pairwise cosine distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze(-1)
    return cosine_dis


def spectral_clustering(sim_matrix: Tensor, cluster_num: int, init: Tensor=None,
                        return_state: bool=False, normalized: bool=False):
    r"""
    Perform spectral clustering based on given similarity matrix.

    This function firstly computes the leading eigenvectors of the given similarity matrix, and then utilizes the
    eigenvectors as features and performs k-means clustering based on these features.

    :param sim_matrix: :math:`(n\times n)` input similarity matrix. :math:`n`: number of instances
    :param cluster_num: number of clusters
    :param init: the initialization technique or initial features for k-means
    :param return_state: whether return state features (can be further used for prediction)
    :param normalized: whether to normalize the similarity matrix by its degree
    :return: the belonging of each instance to clusters, state features (if ``return_state==True``)
    """
    degree = torch.diagflat(torch.sum(sim_matrix, dim=-1))
    if normalized:
        aff_matrix = (degree - sim_matrix) / torch.diag(degree).unsqueeze(1)
    else:
        aff_matrix = degree - sim_matrix
    e, v = torch.symeig(aff_matrix, eigenvectors=True)
    topargs = torch.argsort(torch.abs(e), descending=False)[1:cluster_num]
    v = v[:, topargs]

    if cluster_num == 2:
        choice_cluster = (v > 0).to(torch.int).squeeze(1)
    else:
        choice_cluster, initial_state = kmeans(v, cluster_num, init if init is not None else 'plus',
                                               distance='euclidean', tol=1e-6)

    choice_cluster = choice_cluster.to(sim_matrix.device)

    if return_state:
        return choice_cluster, initial_state
    else:
        return choice_cluster

