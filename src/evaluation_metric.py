#import torch
from itertools import combinations

import numpy

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.common.tensor import Tensor
import mindspore_hub as mshub
import mindspore.context as context

context.set_context(device_target="GPU")
context.set_auto_parallel_context(parallel_mode="data_parallel")

def unique_count(b):
    a = b.ravel()
    dic = {}
    for key in a.asnumpy():
        dic[key] = dic.get(key, 0) + 1
    uni = np.unique(a)
    out = np.zeros_like(uni)
    for i in range(len(uni)):
        out[i] = dic[uni.asnumpy()[i]]
    return out

def xor(a,b):
    assert a.shape == b.shape
    s = a.shape
    a = a.ravel()
    b = b.ravel()
    out = np.zeros_like(a)
    for i in range(len(a)):
        if ( ((a[i] == 0) & (b[i] == 0)) | ((a[i] != 0) & (b[i] != 0)) ):
            out[i] = 0
        else:
            out[i] = 1
    return out.reshape(s)

def matching_accuracy(pmat_pred, pmat_gt, ns):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    Matching Accuracy is equivalent to the recall of matching.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: mean matching accuracy, matched num of pairs, total num of pairs
    """

    batch_num = pmat_pred.shape[0]

    assert numpy.all(((pmat_pred == 0) + (pmat_pred == 1)).asnumpy()), 'pmat_pred can only contain 0/1 elements.'
    assert numpy.all(((pmat_gt == 0) + (pmat_gt == 1)).asnumpy()), 'pmat_gt should only contain 0/1 elements.'
    assert numpy.all((P.ReduceSum()(pmat_pred, -1) <= 1).asnumpy()) and torch.all((P.ReduceSum()(pmat_pred, -2) <= 1).asnumpy())
    assert numpy.all((P.ReduceSum()(pmat_gt, -1) <= 1).asnumpy()) and torch.all((P.ReduceSum()(pmat_gt, -2) <= 1).asnumpy())

    #indices_pred = torch.argmax(pmat_pred, dim=-1)
    #indices_gt = torch.argmax(pmat_gt, dim=-1)

    #matched = (indices_gt == indices_pred).type(pmat_pred.dtype)
    match_num = 0
    total_num = 0
    acc = np.zeros(batch_num)
    for b in range(batch_num):
        #match_num += torch.sum(matched[b, :ns[b]])
        #total_num += ns[b].item()
        acc[b] = P.ReduceSum()(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / P.ReduceSum()(pmat_gt[b, :ns[b]])
        match_num += P.ReduceSum()(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
        total_num += P.ReduceSum()(pmat_gt[b, :ns[b]])

    acc[np.isnan(acc)] = 1

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

    batch_num = pmat_pred.shape[0]

    assert numpy.all(((pmat_pred == 0) + (pmat_pred == 1)).asnumpy()), 'pmat_pred can only contain 0/1 elements.'
    assert numpy.all(((pmat_gt == 0) + (pmat_gt == 1)).asnumpy()), 'pmat_gt should only contain 0/1 elements.'
    assert numpy.all((P.ReduceSum()(pmat_pred, -1) <= 1).asnumpy()) and torch.all(
        (P.ReduceSum()(pmat_pred, -2) <= 1).asnumpy())
    assert numpy.all((P.ReduceSum()(pmat_gt, -1) <= 1).asnumpy()) and torch.all(
        (P.ReduceSum()(pmat_gt, -2) <= 1).asnumpy())


    match_num = 0
    total_num = 0
    precision = np.zeros(batch_num)
    for b in range(batch_num):
        precision[b] = P.ReduceSum()(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / P.ReduceSum()(pmat_pred[b, :ns[b]])
        match_num += P.ReduceSum()(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
        total_num += P.ReduceSum()(pmat_pred[b, :ns[b]])

    precision[np.isnan(precision)] = 1

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
        .format(np.mean(ps),np.std(ps), np.mean(rs), np.std(rs), np.mean(f1s), np.std(f1s))

def format_metric(ms):
    """
    Helping function for formatting single metric
    :param ms: tensor containing metric
    :return: a formatted string containing mean and variance
    """
    return '{:.4f}+-{:.4f}'.format(np.mean(ms), np.std(ms))


def objective_score(pmat_pred, affmtx, ns=None):
    """
    Objective score given predicted permutation matrix and affinity matrix from the problem.
    :param pmat_pred: predicted permutation matrix
    :param affmtx: affinity matrix from the problem
    :param ns: number of exact pairs (no use here)
    :return: objective scores
    """
    batch_num = pmat_pred.shape[0]

    perm1 = tuple(range(3, len(pmat_pred.shape)))
    perm1 = (0, 2, 1) + perm1
    p_vec = P.Reshape()(P.Transpose()(pmat_pred, perm1), (batch_num, -1, 1))

    perm2 = tuple(range(3, len(p_vec.shape)))
    perm2 = (0, 2, 1) + perm2
    obj_score = P.Reshape()(np.matmul(np.matmul(P.Transpose()(p_vec, perm2), affmtx), p_vec), (-1, ))

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
    num_clusters = np.max(pred_clusters, -1) + 1
    batch_num = pred_clusters.shape[0]

    gt_classes_t = []

    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))

    gt_clusters = Tensor(gt_classes_t,dtype=pred_clusters.dtype)

    cluster_acc = np.zeros(batch_num)
    for b in range(batch_num):
        sum = 0
        for i in range(num_clusters[b]):
            for j, k in combinations(range(num_clusters[b]), 2):
                pred_i = (pred_clusters[b] == i).astype(mindspore.float32)
                gt_j = (gt_clusters[b] == j).astype(mindspore.float32)
                gt_k = (gt_clusters[b] == k).astype(mindspore.float32)
                sum += (P.ReduceSum()(pred_i * gt_j) * P.ReduceSum()(pred_i * gt_k)) / P.ReduceSum()(pred_i) ** 2
        for i in range(num_clusters[b]):
            for j, k in combinations(range(num_clusters[b]), 2):
                gt_i = (gt_clusters[b] == i).astype(mindspore.float32)
                pred_j = (pred_clusters[b] == j).astype(mindspore.float32)
                pred_k = (pred_clusters[b] == k).astype(mindspore.float32)
                sum += (P.ReduceSum()(gt_i * pred_j) * P.ReduceSum()(gt_i * pred_k)) / (P.ReduceSum()(pred_j) * P.ReduceSum()(pred_k))

        cluster_acc[b] = 1 - sum / num_clusters[b].astype(mindspore.float32)

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
    num_clusters = P.ReduceSum()(pred_clusters, -1) + 1
    num_instances = pred_clusters.shape[1]
    batch_num = pred_clusters.shape[0]
    gt_classes_t = []
    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = Tensor(gt_classes_t, dtype=pred_clusters.dtype)

    cluster_purity = np.zeros(batch_num)
    for b in range(batch_num):
        for i in range(num_clusters[b]):
            max_counts = np.max(unique_count(gt_clusters[b][pred_clusters[b] == i]))
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
    num_clusters = np.max(pred_clusters, -1).values + 1
    num_instances = pred_clusters.shape[1]
    batch_num = pred_clusters.shape[0]
    gt_classes_t = []
    for b in range(batch_num):
        gt_classes_b_set = list(set(gt_classes[b]))
        gt_classes_t.append([])
        assert len(gt_classes_b_set) == num_clusters[b]
        for i in range(len(gt_classes[b])):
            gt_classes_t[b].append(gt_classes_b_set.index(gt_classes[b][i]))
    gt_clusters = Tensor(gt_classes_t, dtype=pred_clusters.dtype)
    pred_pairs =  P.ExpandDims()(pred_clusters, -1) == P.ExpandDims()(pred_clusters, -2)
    gt_pairs = P.ExpandDims()(gt_clusters, -1) == P.ExpandDims()(gt_clusters, -2)
    unmatched_pairs = xor(pred_pairs, gt_pairs)
    rand_index = 1 - P.ReduceSum()(unmatched_pairs, (-1,-2)) / (num_instances * (num_instances - 1))
    return rand_index
