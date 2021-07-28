import torch
from torch import Tensor
from itertools import combinations


def pck(x: Tensor, x_gt: Tensor, perm_mat: Tensor, dist_threshs: Tensor, ns: Tensor) -> Tensor:
    r"""
    Percentage of Correct Keypoints (PCK) evaluation metric.

    If the distance between predicted keypoint and the ground truth keypoint is smaller than a given threshold, than it
    is regraded as a correct matching.

    This is the evaluation metric used by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_

    :param x: :math:`(b\times n \times 2)` candidate coordinates. :math:`n`: number of nodes in input graph
    :param x_gt: :math:`(b\times n_{gt} \times 2)` ground truth coordinates. :math:`n_{gt}`: number of nodes in ground
     truth graph
    :param perm_mat: :math:`(b\times n \times n_{gt})` permutation matrix or doubly-stochastic matrix indicating
     node-to-node correspondence
    :param dist_threshs: :math:`(b\times m)` a tensor contains thresholds in pixel. :math:`m`: number of thresholds for
     each batch
    :param ns: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(m)` the PCK values of this batch

    .. note::
        An example of ``dist_threshs`` for 4 batches and 2 thresholds:
        ::

            [[10, 20],
             [10, 20],
             [10, 20],
             [10, 20]]
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

    return match_num / total_num


def matching_recall(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    r"""
    Matching Recall between predicted permutation matrix and ground truth permutation matrix.

    .. math::
        \text{matching recall} = \frac{tr(\mathbf{X}\cdot {\mathbf{X}^{gt}}^\top)}{\sum \mathbf{X}^{gt}}

    :param pmat_pred: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
    :param pmat_gt: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
    :param ns: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(b)` matching recall

    .. note::
        This function is equivalent to "matching accuracy" if the matching problem has no outliers.
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    acc = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        acc[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_gt[b, :ns[b]])

    acc[torch.isnan(acc)] = 1

    return acc


def matching_precision(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    r"""
    Matching Precision between predicted permutation matrix and ground truth permutation matrix.

    .. math::
        \text{matching precision} = \frac{tr(\mathbf{X}\cdot {\mathbf{X}^{gt}}^\top)}{\sum \mathbf{X}}

    :param pmat_pred: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
    :param pmat_gt: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
    :param ns: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(b)` matching precision

    .. note::
        This function is equivalent to "matching accuracy" if the matching problem has no outliers.
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    precision = torch.zeros(batch_num, device=device)
    for b in range(batch_num):
        precision[b] = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) / torch.sum(pmat_pred[b, :ns[b]])

    precision[torch.isnan(precision)] = 1

    return precision


def matching_accuracy(pmat_pred: Tensor, pmat_gt: Tensor, ns: Tensor) -> Tensor:
    r"""
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.

    .. math::
        \text{matching recall} = \frac{tr(\mathbf{X}\cdot {\mathbf{X}^{gt}}^\top)}{\sum \mathbf{X}^{gt}}

    This function is a wrapper of ``matching_recall``.

    :param pmat_pred: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
    :param pmat_gt: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
    :param ns: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes, and
     ``ns`` is required to specify the exact number of nodes of each instance in the batch.
    :return: :math:`(b)` matching accuracy

    .. note::
        If the graph matching problem has no outliers, it is proper to use this metric and papers call it "matching
        accuracy". If there are outliers, it is better to use ``matching_precision`` and ``matching_recall``.
    """
    return matching_recall(pmat_pred, pmat_gt, ns)


def format_accuracy_metric(ps: Tensor, rs: Tensor, f1s: Tensor) -> str:
    r"""
    Helper function for formatting precision, recall and f1 score metric

    :param ps: tensor containing precisions
    :param rs: tensor containing recalls
    :param f1s: tensor containing f1 scores
    :return: a formatted string with mean and variance of precision, recall and f1 score

    Example output:
    ::

        p = 0.7837±0.2799, r = 0.7837±0.2799, f1 = 0.7837±0.2799
    """
    return 'p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}' \
        .format(torch.mean(ps), torch.std(ps), torch.mean(rs), torch.std(rs), torch.mean(f1s), torch.std(f1s))

def format_metric(ms: Tensor) -> str:
    r"""
    Helping function for formatting single metric.

    :param ms: tensor containing metric
    :return: a formatted string containing mean and variance
    """
    return '{:.4f}±{:.4f}'.format(torch.mean(ms), torch.std(ms))


def objective_score(pmat_pred: Tensor, affmtx: Tensor) -> Tensor:
    r"""
    Objective score given predicted permutation matrix and affinity matrix from the problem.

    .. math::
        \text{objective score} = \mathrm{vec}(\mathbf{X})^\top \mathbf{K} \mathrm{vec}(\mathbf{X})

    where :math:`\mathrm{vec}(\cdot)` means column-wise vectorization.

    :param pmat_pred: predicted permutation matrix :math:`(\mathbf{X})`
    :param affmtx: affinity matrix of the quadratic assignment problem :math:`(\mathbf{K})`
    :return: objective scores

    .. note::
        The most general mathematical form of graph matching is known as Quadratic Assignment Problem (QAP), which is an
        NP-hard combinatorial optimization problem. Objective score reflects the power of the graph matching/QAP solver
        concerning the objective score of the QAP.
    """
    batch_num = pmat_pred.shape[0]

    p_vec = pmat_pred.transpose(1, 2).contiguous().view(batch_num, -1, 1)
    obj_score = torch.matmul(torch.matmul(p_vec.transpose(1, 2), affmtx), p_vec).view(-1)

    return obj_score

def clustering_accuracy(pred_clusters: Tensor, gt_classes: Tensor) -> Tensor:
    r"""
    Clustering accuracy for clusters.

    :math:`\mathcal{A}, \mathcal{B}, ...` are ground truth classes and :math:`\mathcal{A}^\prime, \mathcal{B}^\prime,
    ...` are predicted classes and :math:`k` is the number of classes:

    .. math::
        \text{clustering accuracy} = 1 - \frac{1}{k} \left(\sum_{\mathcal{A}} \sum_{\mathcal{A}^\prime \neq \mathcal{B}^\prime}
         \frac{|\mathcal{A}^\prime \cap \mathcal{A}| |\mathcal{B}^\prime \cap \mathcal{A}|}{|\mathcal{A}| |\mathcal{A}|} +
         \sum_{\mathcal{A}^\prime} \sum_{\mathcal{A} \neq \mathcal{B}}
         \frac{|\mathcal{A}^\prime \cap \mathcal{A}| |\mathcal{A}^\prime \cap \mathcal{B}|}{|\mathcal{A}| |\mathcal{B}|} \right)

    This metric is proposed by `"Wang et al. Clustering-aware Multiple Graph Matching via Decayed Pairwise Matching
    Composition. AAAI 2020." <https://ojs.aaai.org/index.php/AAAI/article/view/5528/5384>`_

    :param pred_clusters: :math:`(b\times n)` predicted clusters. :math:`n`: number of instances.
        ::

            e.g. [[0,0,1,2,1,2]
                  [0,1,2,2,1,0]]
    :param gt_classes: :math:`(b\times n)` ground truth classes
        ::

            e.g. [['car','car','bike','bike','person','person'],
                  ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: :math:`(b)` clustering accuracy
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

def clustering_purity(pred_clusters: Tensor, gt_classes: Tensor) -> Tensor:
    r"""
    Clustering purity for clusters.

    :math:`n` is the number of instances,
    :math:`\mathcal{C}_i` represent the predicted class :math:`i` and :math:`\mathcal{C}^{gt}_j` is ground truth class :math:`j`:

    .. math::
        \text{clustering purity} = \frac{1}{n} \sum_{i=1}^{k} \max_{j\in\{1,...,k\}} |\mathcal{C}_i \cap \mathcal{C}^{gt}_{j}|

    :param pred_clusters: :math:`(b\times n)` predicted clusters. :math:`n`: number of instances.
        ::

            e.g. [[0,0,1,2,1,2]
                  [0,1,2,2,1,0]]
    :param gt_classes: :math:`(b\times n)` ground truth classes
        ::

            e.g. [['car','car','bike','bike','person','person'],
                  ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: :math:`(b)` clustering purity
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


def rand_index(pred_clusters: Tensor, gt_classes: Tensor) -> Tensor:
    r"""
    Rand index measurement for clusters.

    Rand index is computed by the number of instances predicted in the same class with the same label :math:`n_{11}` and
    the number of instances predicted in separate classes and with different labels :math:`n_{00}`, normalized by the total
    number of instances pairs :math:`n(n-1)`:

    .. math::
        \text{rand index} = \frac{n_{11} + n_{00}}{n(n-1)}

    :param pred_clusters: :math:`(b\times n)` predicted clusters. :math:`n`: number of instances.
        ::

            e.g. [[0,0,1,2,1,2]
                  [0,1,2,2,1,0]]
    :param gt_classes: :math:`(b\times n)` ground truth classes
        ::

            e.g. [['car','car','bike','bike','person','person'],
                  ['bus','bus','cat', 'sofa',  'cat',  'sofa' ]]
    :return: :math:`(b)` clustering purity
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
