import torch
import torch.nn as nn
import torch.nn.functional as F
from src.lap_solvers.hungarian import hungarian
from torch import Tensor


class PermutationLoss(nn.Module):
    r"""
    Binary cross entropy loss between two permutations, also known as "permutation loss".
    Proposed by `"Wang et al. Learning Combinatorial Embedding Networks for Deep Graph Matching. ICCV 2019."
    <http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf>`_

    .. math::
        L_{perm} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} + (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(PermutationLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss += F.binary_cross_entropy(
                pred_dsmat[batch_slice],
                gt_perm[batch_slice],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class CrossEntropyLoss(nn.Module):
    r"""
    Multi-class cross entropy loss between two permutations.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged cross-entropy loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            gt_index = torch.max(gt_perm[batch_slice], dim=-1).indices
            loss += F.nll_loss(
                torch.log(pred_dsmat[batch_slice]),
                gt_index,
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class PermutationLossHung(nn.Module):
    r"""
    Binary cross entropy loss between two permutations with Hungarian attention. The vanilla version without Hungarian
    attention is :class:`~src.loss_func.PermutationLoss`.

    .. math::
        L_{hung} &=-\sum_{i\in\mathcal{V}_1,j\in\mathcal{V}_2}\mathbf{Z}_{ij}\left(\mathbf{X}^\text{gt}_{ij}\log \mathbf{S}_{ij}+\left(1-\mathbf{X}^{\text{gt}}_{ij}\right)\log\left(1-\mathbf{S}_{ij}\right)\right) \\
        \mathbf{Z}&=\mathrm{OR}\left(\mathrm{Hungarian}(\mathbf{S}),\mathbf{X}^\text{gt}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Hungarian attention highlights the entries where the model makes wrong decisions after the Hungarian step (which is
    the default discretization step during inference).

    Proposed by `"Yu et al. Learning deep graph matching with channel-independent embedding and Hungarian attention.
    ICLR 2020." <https://openreview.net/forum?id=rJgBd2NYPH>`_

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.

    A working example for Hungarian attention:

    .. image:: ../../images/hungarian_attention.png
    """
    def __init__(self):
        super(PermutationLossHung, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        dis_pred = hungarian(pred_dsmat, src_ns, tgt_ns)
        ali_perm = dis_pred + gt_perm
        ali_perm[ali_perm > 1.0] = 1.0 # Hung
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
        gt_perm = torch.mul(ali_perm, gt_perm)
        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_dsmat[b, :src_ns[b], :tgt_ns[b]],
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
        return loss / n_sum


class OffsetLoss(nn.Module):
    r"""
    OffsetLoss Criterion computes a robust loss function based on image pixel offset.
    Proposed by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_

    .. math::
        \mathbf{d}_i =& \sum_{j \in V_2} \left( \mathbf{S}_{i, j} P_{2j} \right)- P_{1i} \\
        L_{off} =& \sum_{i \in V_1} \sqrt{||\mathbf{d}_i - \mathbf{d}^{gt}_i||^2 + \epsilon}

    :math:`\mathbf{d}_i` is the displacement vector. See :class:`src.displacement_layer.Displacement` or more details

    :param epsilon: a small number for numerical stability
    :param norm: (optional) division taken to normalize the loss
    """
    def __init__(self, epsilon: float=1e-5, norm=None):
        super(OffsetLoss, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def forward(self, d1: Tensor, d2: Tensor, mask: float=None) -> Tensor:
        """
        :param d1: predicted displacement matrix
        :param d2: ground truth displacement matrix
        :param mask: (optional) dummy node mask
        :return: computed offset loss
        """
        # Loss = Sum(Phi(d_i - d_i^gt))
        # Phi(x) = sqrt(x^T * x + epsilon)
        if mask is None:
            mask = torch.ones_like(mask)
        x = d1 - d2
        if self.norm is not None:
            x = x / self.norm

        xtx = torch.sum(x * x * mask, dim=-1)
        phi = torch.sqrt(xtx + self.epsilon)
        loss = torch.sum(phi) / d1.shape[0]

        return loss


class FocalLoss(nn.Module):
    r"""
    Focal loss between two permutations.

    .. math::
        L_{focal} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left((1-\mathbf{S}_{i,j})^{\gamma} \mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} +
        \mathbf{S}_{i,j}^{\gamma} (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs, :math:`\gamma` is the focal loss
    hyper parameter.

    :param gamma: :math:`\gamma` parameter for focal loss
    :param eps: a small parameter for numerical stability

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self, gamma=0., eps=1e-15):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged focal loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            x = pred_dsmat[b, :src_ns[b], :tgt_ns[b]]
            y = gt_perm[b, :src_ns[b], :tgt_ns[b]]
            loss += torch.sum(
                - (1 - x) ** self.gamma * y * torch.log(x + self.eps)
                - x ** self.gamma * (1 - y) * torch.log(1 - x + self.eps)
            )
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class InnerProductLoss(nn.Module):
    r"""
    Inner product loss for self-supervised problems.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged inner product loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss -= torch.sum(pred_dsmat[batch_slice] * gt_perm[batch_slice])
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class HammingLoss(torch.nn.Module):
    r"""
    Hamming loss between two permutations.

    .. math::
        L_{hamm} = \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}_{i,j} (1-\mathbf{X}^{gt}_{i,j}) +  (1-\mathbf{X}_{i,j}) \mathbf{X}^{gt}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Firstly adopted by `"Rolinek et al. Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers.
    ECCV 2020." <https://arxiv.org/abs/2003.11657>`_

    .. note::
        Hamming loss is defined between two discrete matrices, and discretization will in general truncate gradient. A
        workaround may be using the `blackbox differentiation technique <https://arxiv.org/abs/1912.02175>`_.
    """
    def __init__(self):
        super(HammingLoss, self).__init__()

    def forward(self, pred_perm: Tensor, gt_perm: Tensor) -> Tensor:
        r"""
        :param pred_perm: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :return:
        """
        errors = pred_perm * (1.0 - gt_perm) + (1.0 - pred_perm) * gt_perm
        return errors.mean(dim=0).sum()
