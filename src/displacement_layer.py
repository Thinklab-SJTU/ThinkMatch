import torch
import torch.nn as nn
from torch import Tensor


class Displacement(nn.Module):
    r"""
    Displacement Layer computes the displacement vector for each point in the source image, with its corresponding point
    (or points) in target image.

    The output is a displacement matrix constructed from all displacement vectors.
    This metric measures the shift from source point to predicted target point, and can be applied for matching
    accuracy.

    Together with displacement matrix d, this function will also return a grad_mask, which helps to filter out dummy
    nodes in practice.

    .. math::
        \mathbf{d}_i = \sum_{j \in V_2} \left( \mathbf{S}_{i, j} P_{2j} \right)- P_{1i}

    Proposed by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_
    """
    def __init__(self):
        super(Displacement, self).__init__()

    def forward(self, s: Tensor, P_src: Tensor, P_tgt: Tensor, ns_gt: Tensor=None):
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` permutation or doubly stochastic matrix. :math:`b`: batch size.
         :math:`n_1`: number of nodes in source image. :math:`n_2`: number of nodes in target image
        :param P_src: :math:`(b\times n_1 \times 2)` point set on source image
        :param P_tgt: :math:`(b\times n_2 \times 2)` point set on target image
        :param ns_gt: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes,
         therefore ``ns_gt`` is required to specify the exact number of nodes of each instance in the batch.
        :return: displacement matrix d,
            mask for dummy nodes grad_mask. If ``ns_gt=None``, it will not be calculated and None is returned.
        """
        if ns_gt is None:
            max_n = s.shape[1]
            P_src = P_src[:, 0:max_n, :]
            grad_mask = None
        else:
            grad_mask = torch.zeros_like(P_src)
            for b, n in enumerate(ns_gt):
                grad_mask[b, 0:n] = 1

        d = torch.matmul(s, P_tgt) - P_src
        return d, grad_mask