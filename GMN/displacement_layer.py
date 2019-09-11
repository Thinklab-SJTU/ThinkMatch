import torch
import torch.nn as nn


class Displacement(nn.Module):
    """
    Displacement Layer computes the displacement vector for each point in the source image, with its corresponding point
    (or points) in target image. The output is a displacement matrix constructed from all displacement vectors.
    This metric measures the shift from source point to predicted target point, and can be applied for matching
    accuracy.
    Together with displacement matrix d, this function will also return a grad_mask, which helps to filter out dummy
    nodes in practice.
    d = s * P_tgt - P_src
    Input: permutation or doubly stochastic matrix s
           point set on source image P_src
           point set on target image P_tgt
           (optional) ground truth number of effective points in source image ns_gt
    Output: displacement matrix d
            mask for dummy nodes grad_mask. If ns_gt=None, it will not be calculated and None is returned.
    """
    def __init__(self):
        super(Displacement, self).__init__()

    def forward(self, s, P_src, P_tgt, ns_gt=None):
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
