import torch
from torch import Tensor


def geo_edge_feature(P: Tensor, G: Tensor, H: Tensor, norm_d=256, device=None):
    """
    Compute geometric edge features [d, cos(theta), sin(theta)]
    Adjacency matrix is formed by A = G * H^T
    :param P: point set (b x num_nodes x 2)
    :param G: factorized graph partition G (b x num_nodes x num_edges)
    :param H: factorized graph partition H (b x num_nodes x num_edges)
    :param norm_d: normalize Euclidean distance by norm_d
    :param device: device
    :return: feature tensor (b x 3 x num_edges)
    """
    if device is None:
        device = P.device

    p1 = torch.sum(torch.mul(P.unsqueeze(-2), G.unsqueeze(-1)), dim=1) # (b x num_edges x dim)
    p2 = torch.sum(torch.mul(P.unsqueeze(-2), H.unsqueeze(-1)), dim=1)

    d = torch.norm((p1 - p2) / (norm_d * torch.sum(G, dim=1)).unsqueeze(-1), dim=-1) # (b x num_edges)
                                                                                     # non-existing elements are nan

    cos_theta = (p1[:, :, 0] - p2[:, :, 0]) / (d * norm_d) # non-existing elements are nan
    sin_theta = (p1[:, :, 1] - p2[:, :, 1]) / (d * norm_d)

    return torch.stack((d, cos_theta, sin_theta), dim=1).to(device)
