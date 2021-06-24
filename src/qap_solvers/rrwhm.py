import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn as Sinkhorn


class RRWHM(nn.Module):
    """
    RRWHM solver for hyper graph matching, implemented by tensor power iteration with Sinkhorn reweighted jumps.
    Parameter: maximum iteration max_iter
    Input: input tensor H
           maximum size of source graph num_src
           sizes of source graph in batch ns_src
           sizes of target graph in batch ns_tgt
           (optional) initialization vector v0. If not specified, v0 will be initialized with all 1.
    Output: computed eigenvector v
    """
    def __init__(self, max_iter=50, sk_iter=20, alpha=0.2, beta=30):
        super(RRWHM, self).__init__()
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.sk = Sinkhorn(max_iter=sk_iter,log_forward=False)

    def forward(self, H, num_src, ns_src, ns_tgt, v0=None):
        order = len(H.shape) - 1
        sum_dims = [i+2 for i in range(order-1)]
        d = H.sum(dim=sum_dims, keepdim=True)
        dmax = d.max(dim=1, keepdim=True).values
        H = H / dmax

        batch_num = H.shape[0]
        mn = H.shape[1]
        if v0 is None:
            v0 = torch.zeros(batch_num, num_src, mn // num_src, dtype=H.dtype, device=H.device)
            for b in range(batch_num):
                v0[b, 0:ns_src[b], 0:ns_tgt[b]] = torch.tensor(1.) / (ns_src[b] * ns_tgt[b])

            v0 = v0.transpose(1, 2).reshape(batch_num, mn, 1)

        v = v0
        for i in range(self.max_iter):
            H_red = H.unsqueeze(-1)
            for o in range(order - 1):
                v_shape = [v.shape[0]] + [1] * (order - 1 - o) + list(v.shape[1:])
                H_red = torch.sum(torch.mul(H_red, v.view(*v_shape)), dim=-2)
            v = H_red
            last_v = v
            n = torch.norm(v, p=1, dim=1, keepdim=True)
            v = v / n
            s = v.view(batch_num, -1, num_src).transpose(1, 2)
            s = torch.exp(self.beta * s / s.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values)

            v = self.alpha * self.sk(s, ns_src, ns_tgt).transpose(1, 2).reshape(batch_num, mn, 1) + (1 - self.alpha) * v
            n = torch.norm(v, p=1, dim=1, keepdim=True)
            v = torch.matmul(v, 1 / n)

            if torch.norm(v - last_v) < 1e-5:
                break

        return v.view(batch_num, -1)
