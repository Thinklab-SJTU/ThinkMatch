import torch
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.common.tensor import Tensor
import mindspore_hub as mshub
import mindspore.context as context

context.set_context(device_target="GPU")

def dataset2mindspore(inputs):
    #print("wtf",type(inputs))
    #print(type(inputs))
    if isinstance(inputs, dict):
    #    print("dict")
        for key in inputs.keys():
            #print(key)
            inputs[key] = dataset2mindspore(inputs[key])
    elif isinstance(inputs, list):
    #    print("list")
        for (key, _) in enumerate(inputs):
            inputs[key] = dataset2mindspore(inputs[key])
    elif isinstance(inputs, torch.Tensor):
    #    print("tensor")
        inputs = Tensor(inputs.numpy())

    return inputs