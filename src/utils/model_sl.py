import torch
from torch.nn import DataParallel


def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module

    torch.save(model.state_dict(), path)


def load_model(model, path, strict=True):
    if isinstance(model, DataParallel):
        module = model.module
    else:
        module = model
    missing_keys, unexpected_keys = module.load_state_dict(torch.load(path), strict=strict)
    if len(unexpected_keys) > 0:
        print('Warning: Unexpected key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        print('Warning: Missing key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in missing_keys)))
