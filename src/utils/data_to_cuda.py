import torch
from src.sparse_torch.csx_matrix import CSRMatrix3d, CSCMatrix3d
import torch_geometric as pyg

def data_to_cuda(inputs):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
    if type(inputs) is list:
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is tuple:
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is dict:
        for key in inputs:
            inputs[key] = data_to_cuda(inputs[key])
    elif type(inputs) in [str, int, float]:
        inputs = inputs
    elif type(inputs) in [torch.Tensor, CSRMatrix3d, CSCMatrix3d]:
        inputs = inputs.cuda()
    else:
        try:
            pyg_datatypes = [pyg.data.Data, pyg.data.Batch, pyg.data.batch.DataBatch]
        except AttributeError:
            pyg_datatypes = [pyg.data.Data, pyg.data.Batch]
        if type(inputs) in pyg_datatypes:
            inputs = inputs.to('cuda')
        else:
            raise TypeError('Unknown type of inputs: {}'.format(type(inputs)))
    return inputs
