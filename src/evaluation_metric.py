import torch
from itertools import combinations


def matching_accuracy(pmat_pred, pmat_gt, ns):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    Matching Accuracy is equivalent to the recall of matching.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: mean matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    #indices_pred = torch.argmax(pmat_pred, dim=-1)
    #indices_gt = torch.argmax(pmat_gt, dim=-1)

    #matched = (indices_gt == indices_pred).type(pmat_pred.dtype)
    match_num = 0
    total_num = 0
    acc = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        #match_num += torch.sum(matched[b, :ns[b]])
        #total_num += ns[b].item()
        acc[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_gt[b, :ns[b]])
        match_num += torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
        total_num += torch.sum(pmat_gt[b, :ns[b]])

    acc[torch.isnan(acc)] = 1

    #return match_num / total_num, match_num, total_num
    return acc, match_num, total_num

def matching_precision(pmat_pred, pmat_gt, ns):
    """
    Matching Precision between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: mean matching precision, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    match_num = 0
    total_num = 0
    precision = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        precision[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_pred[b, :ns[b]])
        match_num += torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
        total_num += torch.sum(pmat_pred[b, :ns[b]])

    precision[torch.isnan(precision)] = 1

    # return match_num / total_num, match_num, total_num
    return precision, match_num, total_num


def format_accuracy_metric(ps, rs, f1s):
    """
    Helper function for formatting precision, recall and f1 score metric
    :param ps: tensor containing precisions
    :param rs: tensor containing recalls
    :param f1s: tensor containing f1 scores
    :return: a formatted string with mean and variance of precision, recall and f1 score
    """
    return 'p = {:.4f}v{:.4f}, r = {:.4f}v{:.4f}, f1 = {:.4f}v{:.4f}' \
        .format(torch.mean(ps), torch.std(ps), torch.mean(rs), torch.std(rs), torch.mean(f1s), torch.std(f1s))

def format_metric(ms):
    """
    Helping function for formatting single metric
    :param ms: tensor containing metric
    :return: a formatted string containing mean and variance
    """
    return '{:.4f}+-{:.4f}'.format(torch.mean(ms), torch.std(ms))


def objective_score(pmat_pred, affmtx, ns=None):
    """
    Objective score given predicted permutation matrix and affinity matrix from the problem.
    :param pmat_pred: predicted permutation matrix
    :param affmtx: affinity matrix from the problem
    :param ns: number of exact pairs (no use here)
    :return: objective scores
    """
    batch_num = pmat_pred.shape[0]

    p_vec = pmat_pred.transpose(1, 2).contiguous().view(batch_num, -1, 1)
    obj_score = torch.matmul(torch.matmul(p_vec.transpose(1, 2), affmtx), p_vec).view(-1)

    return obj_score

def clustering_accuracy(pred_clusters, gt_classes):
    """
    Clustering accuracy for clusters.
    :param pred_clusters: predicted clusters
                          e.g. [[0,0,1,2,1,2]
                                [0,1,2,2,1,0]]
    :param gt_classes: ground truth classes
                       e.g. [['car','car','bike','bike','person','person'],
                             ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: clustering accuracy
    """
    num_clusters = torch.max(pred_clusters, dim=-1).values + 1
    batch_num = pred_clusters.shape[0]

    gt_classes_t = []

    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = torch.tensor(gt_classes_t).to(dtype=pred_clusters.dtype, device=pred_clusters.device)

    cluster_acc = torch.zeros(batch_num, device=pred_clusters.device)
    for b in range(batch_num):
        sum = 0
        for i in range(num_clusters[b]):
            for j, k in combinations(range(num_clusters[b]), 2):
                pred_i = (pred_clusters[b] == i).to(dtype=torch.float)
                gt_j = (gt_clusters[b] == j).to(dtype=torch.float)
                gt_k = (gt_clusters[b] == k).to(dtype=torch.float)
                sum += (torch.sum(pred_i * gt_j) * torch.sum(pred_i * gt_k)) / torch.sum(pred_i) ** 2
        for i in range(num_clusters[b]):
            for j, k in combinations(range(num_clusters[b]), 2):
                gt_i = (gt_clusters[b] == i).to(dtype=torch.float)
                pred_j = (pred_clusters[b] == j).to(dtype=torch.float)
                pred_k = (pred_clusters[b] == k).to(dtype=torch.float)
                sum += (torch.sum(gt_i * pred_j) * torch.sum(gt_i * pred_k)) / (torch.sum(pred_j) * torch.sum(pred_k))

        cluster_acc[b] = 1 - sum / num_clusters[b].to(dtype=torch.float)

    return cluster_acc

def clustering_purity(pred_clusters, gt_classes):
    """
    Clustering purity for clusters.
    :param pred_clusters: predicted clusters
                          e.g. [[0,0,1,2,1,2]
                                [0,1,2,2,1,0]]
    :param gt_classes: ground truth classes
                       e.g. [['car','car','bike','bike','person','person'],
                             ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: clustering purity
    """
    num_clusters = torch.max(pred_clusters, dim=-1).values + 1
    num_instances = pred_clusters.shape[1]
    batch_num = pred_clusters.shape[0]
    gt_classes_t = []
    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = torch.tensor(gt_classes_t).to(dtype=pred_clusters.dtype, device=pred_clusters.device)

    cluster_purity = torch.zeros(batch_num, device=pred_clusters.device)
    for b in range(batch_num):
        for i in range(num_clusters[b]):
            max_counts = torch.max(torch.unique(gt_clusters[b][pred_clusters[b] == i], return_counts=True)[-1]).to(dtype=torch.float)
            cluster_purity[b] += max_counts / num_instances

    return cluster_purity


def rand_index(pred_clusters, gt_classes):
    """
    Rand index measurement for clusters.
    :param pred_clusters: predicted clusters
                          e.g. [[0,0,1,2,1,2]
                                [0,1,2,2,1,0]]
    :param gt_classes: ground truth classes
                       e.g. [['car','car','bike','bike','person','person'],
                             ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: clustering purity
    """
    num_clusters = torch.max(pred_clusters, dim=-1).values + 1
    num_instances = pred_clusters.shape[1]
    batch_num = pred_clusters.shape[0]
    gt_classes_t = []
    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = torch.tensor(gt_classes_t).to(dtype=pred_clusters.dtype, device=pred_clusters.device)
    pred_pairs = pred_clusters.unsqueeze(-1) == pred_clusters.unsqueeze(-2)
    gt_pairs = gt_clusters.unsqueeze(-1) == gt_clusters.unsqueeze(-2)
    unmatched_pairs = torch.logical_xor(pred_pairs, gt_pairs).to(dtype=torch.float)
    rand_index = 1 - torch.sum(unmatched_pairs, dim=(-1,-2)) / (num_instances * (num_instances - 1))
    return rand_index
