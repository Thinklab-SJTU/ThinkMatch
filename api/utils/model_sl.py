from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

def save_model(model, path):
    save_checkpoint(model, path)

def load_model(model, path, strict=True):
    param_dict = load_checkpoint(path, strict_load=strict)
    param_not_load = load_param_into_net(model, param_dict, strict_load=strict)

    if len(param_not_load) > 0:
        print('Warning: Unloaded key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in param_not_load)))
