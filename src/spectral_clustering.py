import numpy as np
import torch

def initialize(X, num_clusters, method='plus'):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param method: plus (default) or random for different initialization strategies
    :return: (np.array) initial state
    """
    if method == 'plus':
        init_func = initialize_plus
    elif method == 'random':
        init_func = initialize_random
    else:
        raise NotImplementedError
    return init_func(X, num_clusters)


def initialize_random(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def initialize_plus(X, num_clusters):
    """
    initialize cluster centers by k-means++
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    centroid_index = np.zeros(num_clusters)
    for i in range(num_clusters):
        if i == 0:
            choice_prob = np.full(num_samples, 1 / num_samples)
        else:
            centroid_X = X[centroid_index[:i]]
            dis = pairwise_distance(X, centroid_X)
            dis_to_nearest_centroid = torch.min(dis, dim=1).values
            choice_prob = dis_to_nearest_centroid / torch.sum(dis_to_nearest_centroid)
            choice_prob = choice_prob.detach().cpu().numpy()

        centroid_index[i] = np.random.choice(num_samples, 1, p=choice_prob, replace=False)

    initial_state = X[centroid_index]
    return initial_state

def kmeans(
        X,
        num_clusters,
        init_x='plus',
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cpu'),
        max_iter=1000
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param init_x: how to initiate x (provide a initial state of x or define a init method) [default: 'plus']
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param max_iter: maximum number of iterations
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    #print('running k-means on {}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
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
    #tqdm_meter = tqdm(desc='[running kmeans]')
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

        # update tqdm meter
        #tqdm_meter.set_postfix(
        #    iteration='{iteration}',
        #    center_shift='{center_shift ** 2:0.6f}',
        #    tol='{tol:0.6f}'
        #)
        #tqdm_meter.update()
        if center_shift ** 2 < tol:
            break

        if torch.isnan(initial_state).any():
            print('NAN encountered in clustering. Retrying...')
            initial_state = initialize(X, num_clusters)

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    #print('predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
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


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
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


def spectral_clustering(sim_matrix, cluster_num, init=None, return_state=False, normalized=False):
    degree = torch.diagflat(torch.sum(sim_matrix, dim=-1))
    if normalized:
        aff_matrix = (degree - sim_matrix) / torch.diag(degree).unsqueeze(1)
    else:
        aff_matrix = degree - sim_matrix
    e, v = torch.symeig(aff_matrix, eigenvectors=True)
    topargs = torch.argsort(torch.abs(e), descending=False)[1:cluster_num]
    v = v[:, topargs]
    #v = v / v.norm(dim=0, keepdim=True)

    #print(topargs)
    #print(v)
    if cluster_num == 2:
        choice_cluster = (v > 0).to(torch.int).squeeze(1)
    else:
        choice_cluster, initial_state = kmeans(v, cluster_num, init if init is not None else 'plus',
                                           tol=1e-6, distance='euclidean')

    choice_cluster = choice_cluster.to(sim_matrix.device)

    if return_state:
        return choice_cluster, initial_state
    else:
        return choice_cluster

