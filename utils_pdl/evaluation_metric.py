import paddle
from pdl_device_trans import place2str


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
    device = place2str(x.place )
    paddle.set_device(device)

    batch_num = x.shape[0]
    thresh_num = dist_threshs.shape[1]

    indices = paddle.argmax(perm_mat, axis=-1)

    dist = paddle.zeros(batch_num, x_gt.shape[1])
    for b in range(batch_num):
        x_correspond = x[b, indices[b], :]
        dist[b, 0:ns[b]] = paddle.norm(x_correspond - x_gt[b], p=2, axis=-1)[0:ns[b]]

    match_num = paddle.zeros(thresh_num)
    total_num = paddle.zeros(thresh_num)
    for b in range(batch_num):
        for idx in range(thresh_num):
            matches = (dist[b] < dist_threshs[b, idx])[0:ns[b]]
            match_num[idx] += paddle.sum(matches).astype(match_num.dtype)
            total_num[idx] += ns[b].astype(total_num.dtype)

    return match_num / total_num, match_num, total_num


def matching_accuracy(pmat_pred, pmat_gt, ns):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.place
    batch_num = pmat_pred.shape[0]

    pmat_gt = paddle.to_tensor(pmat_gt, place=device)

    assert paddle.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can noly contain 0/1 elements.'
    assert paddle.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should noly contain 0/1 elements.'
    assert paddle.all(paddle.sum(pmat_pred, axis=-1) <= 1) and paddle.all(paddle.sum(pmat_pred, axis=-2) <= 1)
    assert paddle.all(paddle.sum(pmat_gt, axis=-1) <= 1) and paddle.all(paddle.sum(pmat_gt, axis=-2) <= 1)

    #indices_pred = paddle.argmax(pmat_pred, axis=-1)
    #indices_gt = paddle.argmax(pmat_gt, axis=-1)

    #matched = (indices_gt == indices_pred).type(pmat_pred.dtype)
    match_num = 0
    total_num = 0
    for b in range(batch_num):
        #match_num += paddle.sum(matched[b, :ns[b]])
        #total_num += ns[b].item()
        match_num += paddle.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
        total_num += paddle.sum(pmat_gt[b, :ns[b]])

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

    p_vec = pmat_pred.transpose(1, 2).reshape(batch_num, -1, 1)
    obj_score = paddle.matmul(paddle.matmul(p_vec.transpose(1, 2), affmtx), p_vec).reshape(-1)

    return obj_score
