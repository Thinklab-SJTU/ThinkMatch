import torch
import numpy as np
import torch.nn.functional as functional

def pad_tensor(inp):
    """
    Pad a list of input tensors into a list of tensors with same dimension
    :param inp: input tensor list
    :return: output tensor list
    """
    assert type(inp[0]) == torch.Tensor
    it = iter(inp)
    t = next(it)
    max_shape = list(t.shape)
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
        except StopIteration:
            break
    max_shape = np.array(max_shape)

    padded_ts = []
    for t in inp:
        pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
        pad_pattern[::-2] = max_shape - np.array(t.shape)
        pad_pattern = tuple(pad_pattern.tolist())
        padded_ts.append(functional.pad(t, pad_pattern, 'constant', 0))

    return padded_ts