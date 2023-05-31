import torch
import scipy.optimize as opt
import numpy as np
from multiprocessing import Pool
from torch import Tensor
# import gurobipy as gp
# from gurobipy import GRB
from ortools.linear_solver import pywraplp
from ortools.linear_solver.linear_solver_natural_api import LinearExpr
from contextlib import contextmanager
import sys,os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def softmax(x,axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def ILP_solver(s: Tensor, n1: Tensor=None, n2: Tensor=None, nproc: int=1, dummy: bool=False) -> Tensor:
    r"""
    Solve optimal LAP permutation by Integer Linear Programming.

    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
    :param n1: :math:`(b)` number of objects in dim1
    :param n2: :math:`(b)` number of objects in dim2
    :param nproc: number of parallel processes (default: ``nproc=1`` for no parallel)
    :param dummy: whether to add dummy node in permutation matrix to match the outliers
    :return: :math:`(b\times n_1 \times n_2)` optimal permutation matrix

    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded.
    """
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))

    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy()
    if n1 is not None:
        n1 = n1.cpu().numpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.cpu().numpy()
    else:
        n2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_ilp_kernel, zip(perm_mat, n1, n2, dummy))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_ilp_kernel(perm_mat[b], n1[b], n2[b], dummy) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)

    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat

def _ilp_kernel(s: torch.Tensor, n1=None, n2=None, dummy=False):
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    # row, col = opt.linear_sum_assignment(s[:n1, :n2])
    row, col = ilp_gurobi(s[:n1, :n2], dummy)
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat

def ilp_gurobi(s, dummy=False):

    s_list = [[None for _ in range(s.shape[1])] for _ in range(s.shape[0])]
    s_sum = []
    # try:
    with suppress_stdout():
        # m = gp.Model("mip1")
        solver = pywraplp.Solver.CreateSolver('SCIP')
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            # s_list[i][j] = m.addVar(vtype=GRB.BINARY, name="{row}_{col}".format(row=i,col=j))
            s_list[i][j] = solver.BoolVar("{row}_{col}".format(row=i,col=j))
            s_sum.append(-s_list[i][j] * np.log(s[i,j]+1e-10))
            # s_sum += -s_list[i][j] * np.log(s[i,j]+0.8)
            # s_sum += -s_list[i][j] * s[i, j]
    # Set objective
    # m.setObjective(s_sum, GRB.MINIMIZE)
    solver.Minimize(solver.Sum(s_sum))

    if dummy==False:
        # Add row constraint
        for i in range(s.shape[0]):
            row_c = s_list[i][0]
            for j in range(1, s.shape[1]):
                row_c += s_list[i][j]
            # m.addConstr(row_c == 1, "row_{row}".format(row=i))
            solver.Add(row_c == 1, "row_{row}".format(row=i))

        # Add column constraint
        for j in range(s.shape[1]):
            col_c = s_list[0][j]
            for i in range(1, s.shape[0]):
                col_c += s_list[i][j]
            # m.addConstr(col_c == 1, "column_{col}".format(col=j))
            solver.Add(col_c == 1, "column_{col}".format(col=j))
    else:
        # Add row constraint
        for i in range(s.shape[0]-1):
            row_c = s_list[i][0]
            for j in range(1, s.shape[1]):
                row_c += s_list[i][j]
            # m.addConstr(row_c == 1, "row_{row}".format(row=i))
            solver.Add(row_c == 1, "row_{row}".format(row=i))
        # for i in range(s.shape[0]):
        #     row_c = s_list[i][0]
        #     for j in range(1, s.shape[1]):
        #         row_c += s_list[i][j]
        #     m.addConstr(row_c <= 1, "row_{row}".format(row=i))

        # Add column constraint
        for j in range(s.shape[1]-1):
            col_c = s_list[0][j]
            for i in range(1, s.shape[0]):
                col_c += s_list[i][j]
            # m.addConstr(col_c == 1, "column_{col}".format(col=j))
            solver.Add(col_c == 1, "column_{col}".format(col=j))
        # for j in range(s.shape[1]):
        #     col_c = s_list[0][j]
        #     for i in range(1, s.shape[0]):
        #         col_c += s_list[i][j]
        #     m.addConstr(col_c <= 1, "column_{col}".format(col=j))

    # Optimize model
    with suppress_stdout():
        # m.optimize()
        _ = solver.Solve()

    results = np.zeros(s.size)

    for i,v in enumerate(solver.variables()):
        results[i] = v.solution_value()
    results = results.reshape(s.shape)
    [row, column] = np.nonzero(results)
    # except gp.GurobiError as e:
    #     print('Error code ' + str(e.errno) + ': ' + str(e))

    # except AttributeError:
    #     print('Encountered an attribute error')

    return row, column