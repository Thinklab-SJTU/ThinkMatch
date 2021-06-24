import torch
import torch.nn as nn
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.sinkhorn import Sinkhorn
from itertools import product
from src.spectral_clustering import spectral_clustering
from src.utils.pad_tensor import pad_tensor

import time

class Timer:
    def __init__(self):
        self.start_time = 0
    def tic(self):
        self.start_time = time.time()
    def toc(self, str=""):
        print_helper('{:.5f}sec {}'.format(time.time()-self.start_time, str))

DEBUG=False

def print_helper(*args):
    if DEBUG:
        print(*args)


class GA_GM(nn.Module):
    """
    Graduated Assignment solver for
     Graph Matching, Multi-Graph Matching and Multi-Graph Matching with a Mixture of Modes.

    This operation does not support batched input, and all input tensors should not have the first batch dimension.

    Parameter: maximum iteration mgm_iter
               sinkhorn iteration sk_iter
               initial sinkhorn regularization sk_tau0
               sinkhorn regularization decaying factor sk_gamma
               minimum tau value min_tau
               convergence tolerance conv_tal
    Input: multi-graph similarity matrix W
           initial multi-matching matrix U0
           number of nodes in each graph ms
           size of universe n_univ
           (optional) projector to doubly-stochastic matrix (sinkhorn) or permutation matrix (hungarian)
    Output: multi-matching matrix U
    """
    def __init__(self, mgm_iter=(200,), cluster_iter=10, sk_iter=20, sk_tau0=(0.5,), sk_gamma=0.5, cluster_beta=(1., 0.), converge_tol=1e-5, min_tau=(1e-2,), projector0=('sinkhorn',)):
        super(GA_GM, self).__init__()
        self.mgm_iter = mgm_iter
        self.cluster_iter = cluster_iter
        self.sk_iter = sk_iter
        self.sk_tau0 = sk_tau0
        self.sk_gamma = sk_gamma
        self.cluster_beta = cluster_beta
        self.converge_tol = converge_tol
        self.min_tau = min_tau
        self.projector0 = projector0

    def forward(self, A, W, U0, ms, n_univ, quad_weight=1., cluster_quad_weight=1., num_clusters=2):
        # gradient is not required for MGM module
        W = W.detach()

        num_graphs = ms.shape[0]
        U = U0
        m_indices = torch.cumsum(ms, dim=0)

        Us = []
        clusters = []

        # initialize U with no clusters
        cluster_M = torch.ones(num_graphs, num_graphs, device=A.device)
        cluster_M01 = cluster_M

        U = self.gagm(A, W, U, ms, n_univ, cluster_M, self.sk_tau0[0], self.min_tau[0], self.mgm_iter[0], self.projector0[0],
                      quad_weight=quad_weight, hung_iter=(num_clusters == 1))
        Us.append(U)

        # MGM problem
        if num_clusters == 1:
            return U, torch.zeros(num_graphs, dtype=torch.int)

        for beta, sk_tau0, min_tau, max_iter, projector0 in \
                zip(self.cluster_beta, self.sk_tau0, self.min_tau, self.mgm_iter, self.projector0):
            for i in range(self.cluster_iter):
                lastU = U

                # clustering step
                def get_alpha(scale=1., qw=1.):
                    Alpha = torch.zeros(num_graphs, num_graphs, device=A.device)
                    for idx1, idx2 in product(range(num_graphs), repeat=2):
                        if idx1 == idx2:
                            continue
                        start_x = m_indices[idx1 - 1] if idx1 != 0 else 0
                        end_x = m_indices[idx1]
                        start_y = m_indices[idx2 - 1] if idx2 != 0 else 0
                        end_y = m_indices[idx2]
                        A_i = A[start_x:end_x, start_x:end_x]
                        A_j = A[start_y:end_y, start_y:end_y]
                        W_ij = W[start_x:end_x, start_y:end_y]
                        U_i = U[start_x:end_x, :]
                        U_j = U[start_y:end_y, :]
                        X_ij = torch.mm(U_i, U_j.t())
                        Alpha_ij = torch.sum(W_ij * X_ij) \
                                   + torch.exp(-torch.norm(torch.chain_matmul(X_ij.t(), A_i, X_ij) - A_j) / scale) * qw
                        Alpha[idx1, idx2] = Alpha_ij
                    return Alpha
                Alpha = get_alpha(qw=cluster_quad_weight)

                last_cluster_M01 = cluster_M01
                cluster_v = spectral_clustering(Alpha, num_clusters, normalized=True)
                cluster_M01 = (cluster_v.unsqueeze(0) == cluster_v.unsqueeze(1)).to(dtype=Alpha.dtype)
                cluster_M = (1 - beta) * cluster_M01 + beta

                if beta == self.cluster_beta[0] and i == 0:
                    clusters.append(cluster_v)

                # matching step
                U = self.gagm(A, W, U, ms, n_univ, cluster_M, sk_tau0, min_tau, max_iter,
                              projector='hungarian' if i != 0 else projector0, quad_weight=quad_weight,
                              hung_iter=(beta == self.cluster_beta[-1]))

                print_helper('beta = {:.2f}, delta U = {:.4f}, delta M = {:.4f}'.format(beta, torch.norm(lastU - U), torch.norm(last_cluster_M01 - cluster_M01)))

                Us.append(U)
                clusters.append(cluster_v)

                if beta == 1:
                    break

                if torch.norm(lastU - U) < self.converge_tol and torch.norm(last_cluster_M01 - cluster_M01) < self.converge_tol:
                    break

        #return Us, clusters
        return  U, cluster_v

    def gagm(self, A, W, U0, ms, n_univ, cluster_M, init_tau, min_tau, max_iter, projector='sinkhorn', hung_iter=True, quad_weight=1.):
        num_graphs = ms.shape[0]
        U = U0
        m_indices = torch.cumsum(ms, dim=0)

        lastU = torch.zeros_like(U)

        sinkhorn_tau = init_tau
        #beta = 0.9
        iter_flag = True

        while iter_flag:
            for i in range(max_iter):
                lastU2 = lastU
                lastU = U

                # compact matrix form update of V
                UUt = torch.mm(U, U.t())
                cluster_weight = torch.repeat_interleave(cluster_M, ms.to(dtype=torch.long), dim=0)
                cluster_weight = torch.repeat_interleave(cluster_weight, ms.to(dtype=torch.long), dim=1)
                V = torch.chain_matmul(A, UUt * cluster_weight, A, U) * quad_weight * 2 + torch.mm(W * cluster_weight, U)
                V /= num_graphs

                U_list = []
                if projector == 'hungarian':
                    m_start = 0
                    for m_end in m_indices:
                        U_list.append(hungarian(V[m_start:m_end, :n_univ]))
                        m_start = m_end
                elif projector == 'sinkhorn':
                    if torch.all(ms == ms[0]):
                        if ms[0] <= n_univ:
                            U_list.append(
                                Sinkhorn(max_iter=self.sk_iter, tau=sinkhorn_tau, batched_operation=True) \
                                    (V.reshape(num_graphs, -1, n_univ), dummy_row=True).reshape(-1, n_univ))
                        else:
                            U_list.append(
                                Sinkhorn(max_iter=self.sk_iter, tau=sinkhorn_tau, batched_operation=True) \
                                    (V.reshape(num_graphs, -1, n_univ).transpose(1, 2), dummy_row=True).transpose(1, 2).reshape(-1, n_univ))
                    else:
                        V_list = []
                        n1 = []
                        m_start = 0
                        for m_end in m_indices:
                            V_list.append(V[m_start:m_end, :n_univ])
                            n1.append(m_end - m_start)
                            m_start = m_end
                        n1 = torch.tensor(n1)
                        U = Sinkhorn(max_iter=self.sk_iter, tau=sinkhorn_tau, batched_operation=True) \
                            (torch.stack(pad_tensor(V_list), dim=0), n1, dummy_row=True)
                        m_start = 0
                        for idx, m_end in enumerate(m_indices):
                            U_list.append(U[idx, :m_end - m_start, :])
                            m_start = m_end
                else:
                    raise NameError('Unknown projecter name: {}'.format(projector))

                U = torch.cat(U_list, dim=0)
                if num_graphs == 2:
                    U[:ms[0], :] = torch.eye(ms[0], n_univ, device=U.device)

                if torch.norm(U - lastU) < self.converge_tol or torch.norm(U - lastU2) == 0:
                    break

            if i == max_iter - 1: # not converged
                if hung_iter:
                    pass
                else:
                    U_list = [hungarian(_) for _ in U_list]
                    U = torch.cat(U_list, dim=0)
                    print_helper(i, 'max iter')
                    break

            # projection control
            if projector == 'hungarian':
                print_helper(i, 'hungarian')
                break
            elif sinkhorn_tau > min_tau:
                print_helper(i, sinkhorn_tau)
                sinkhorn_tau *= self.sk_gamma
            else:
                print_helper(i, sinkhorn_tau)
                if hung_iter:
                    projector = 'hungarian'
                else:
                    U_list = [hungarian(_) for _ in U_list]
                    U = torch.cat(U_list, dim=0)
                    break

        return U


class HiPPI(nn.Module):
    """
    HiPPI solver for multiple graph matching: Higher-order Projected Power Iteration in ICCV 2019

    This operation does not support batched input, and all input tensors should not have the first batch dimension.

    Parameter: maximum iteration mgm_iter
               sinkhorn iteration sk_iter
               sinkhorn regularization sk_tau
    Input: multi-graph similarity matrix W
           initial multi-matching matrix U0
           number of nodes in each graph ms
           size of universe d
           (optional) projector to doubly-stochastic matrix (sinkhorn) or permutation matrix (hungarian)
    Output: multi-matching matrix U
    """
    def __init__(self, max_iter=50, sk_iter=20, sk_tau=1/200.):
        super(HiPPI, self).__init__()
        self.max_iter = max_iter
        self.sinkhorn = Sinkhorn(max_iter=sk_iter, tau=sk_tau)
        self.hungarian = hungarian

    def forward(self, W, U0, ms, d, projector='sinkhorn'):
        num_graphs = ms.shape[0]

        U = U0
        for i in range(self.max_iter):
            lastU = U
            WU = torch.mm(W, U) #/ num_graphs
            V = torch.chain_matmul(WU, U.t(), WU) #/ num_graphs ** 2

            #V_median = torch.median(torch.flatten(V, start_dim=-2), dim=-1).values
            #V_var, V_mean = torch.var_mean(torch.flatten(V, start_dim=-2), dim=-1)
            #V = V - V_mean
            #V = V / torch.sqrt(V_var)

            #V = V / V_median

            U = []
            m_start = 0
            m_indices = torch.cumsum(ms, dim=0)
            for m_end in m_indices:
                if projector == 'sinkhorn':
                    U.append(self.sinkhorn(V[m_start:m_end, :d], dummy_row=True))
                elif projector == 'hungarian':
                    U.append(self.hungarian(V[m_start:m_end, :d]))
                else:
                    raise NameError('Unknown projector {}.'.format(projector))
                m_start = m_end
            U = torch.cat(U, dim=0)

            #print_helper('iter={}, diff={}, var={}, vmean={}, vvar={}'.format(i, torch.norm(U-lastU), torch.var(torch.sum(U, dim=0)), V_mean, V_var))

            if torch.norm(U - lastU) < 1e-5:
                print_helper(i)
                break

        return U