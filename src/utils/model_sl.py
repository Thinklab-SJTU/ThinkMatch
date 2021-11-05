#import torch
#from torch.nn import DataParallel
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

def save_model(model, path):
    #if isinstance(model, DataParallel):
    #    model = model.module

    save_checkpoint(model, path)

def load_model(model, path, strict=True):
    #if isinstance(model, DataParallel):
    #    module = model.module
    #else:
    #    module = model

    param_dict = load_checkpoint(path, strict_load=strict)
    param_not_load = load_param_into_net(model, param_dict, strict_load=strict)

    if len(param_not_load) > 0:
        print('Warning: Unloaded key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in param_not_load)))
