import torch


def pck(x, x_gt, perm_mat, dist_threshs, ns):
    """
    Percentage of Correct Keypoints evaluation metric.
    :param x: candidate coordinates
    :param x_gt: ground truth coordinates
    :param perm_mat: permutation matrix or doubly stochastic matrix indicating correspondence
    :param dist_threshs: a iterable list of thresholds in pixel
    :param ns: number of exact pairs.
    :return: pck, matched num of pairs, total num of pairs
    """
    device = x.device
    batch_num = x.shape[0]
    thresh_num = dist_threshs.shape[1]

    indices = torch.argmax(perm_mat, dim=-1)

    dist = torch.zeros(batch_num, x_gt.shape[1], device=device)
    for b in range(batch_num):
        x_correspond = x[b, indices[b], :]
        dist[b, 0:ns[b]] = torch.norm(x_correspond - x_gt[b], p=2, dim=-1)[0:ns[b]]

    match_num = torch.zeros(thresh_num, device=device)
    total_num = torch.zeros(thresh_num, device=device)
    for b in range(batch_num):
        for idx in range(thresh_num):
            matches = (dist[b] < dist_threshs[b, idx])[0:ns[b]]
            match_num[idx] += torch.sum(matches).to(match_num.dtype)
            total_num[idx] += ns[b].to(total_num.dtype)

    return match_num / total_num, match_num, total_num


def matching_accuracy(pmat_pred, pmat_gt, ns):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can noly contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should noly contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    #indices_pred = torch.argmax(pmat_pred, dim=-1)
    #indices_gt = torch.argmax(pmat_gt, dim=-1)

    #matched = (indices_gt == indices_pred).type(pmat_pred.dtype)
    match_num = 0
    total_num = 0
    for b in range(batch_num):
        #match_num += torch.sum(matched[b, :ns[b]])
        #total_num += ns[b].item()
        match_num += torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
        total_num += torch.sum(pmat_gt[b, :ns[b]])

    return match_num / total_num, match_num, total_num


def objective_score(pmat_pred, affmtx, ns):
    """
    Objective score given predicted permutation matrix and affinity matrix from the problem.
    :param pmat_pred: predicted permutation matrix
    :param affmtx: affinity matrix from the problem
    :param ns: number of exact pairs
    :return: objective scores
    """
    batch_num = pmat_pred.shape[0]

    p_vec = pmat_pred.transpose(1, 2).contiguous().view(batch_num, -1, 1)
    obj_score = torch.matmul(torch.matmul(p_vec.transpose(1, 2), affmtx), p_vec).view(-1)

    return obj_score
