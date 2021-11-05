# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Convert pytorch weight to mindspore checkpoint."""
import os
import torch
import logging
from mindspore.train.serialization import save_checkpoint, load_checkpoint
from mindspore import Tensor
import numpy as np
import uuid


def pytorch2mindspore(pth_file):
    """Convert pytorch weight to mindspore checkpoint."""
    torch_para_dict = torch.load(pth_file)
    torch_weight_list = []
    torch_paras_name_list = []
    ms_params_list = []
    ms_para_name_list = []

    node_bn_layers = ['1','4','8','11','15','18','21','25','28']
    edge_bn_layers = ['0','4']
    bn_name_dict = {'weight':'gamma', 'bias':'beta', 'running_mean':'moving_mean', 'running_var':'moving_variance', 'num_batches_tracked': 'num_batches_tracked'}

    for index, name in enumerate(torch_para_dict):
        torch_paras_name_list.append(name)
        torch_weight = torch_para_dict[name]

        if name.split('.')[0] == 'node_layers':
            if name.split('.')[1] in node_bn_layers:
                ms_name = name[:name.rfind('.')] + '.' + bn_name_dict[name.split('.')[2]]
            else:
                ms_name = name
        elif name.split('.')[0] == 'edge_layers':
            if name.split('.')[1] in edge_bn_layers:
                ms_name = name[:name.rfind('.')] + '.' + bn_name_dict[name.split('.')[2]]
            else:
                ms_name = name
        else:
            ms_name = name
        torch_weight_list.append(torch_weight)
        ms_para_name_list.append(ms_name)

    for index, name in enumerate(ms_para_name_list):
        logging.debug('========================py_name: {}'.format(torch_paras_name_list[index]))
        logging.debug('========================ms_name: {}'.format(name))
        param_dict = {}
        param_dict['name'] = name
        parameter = torch_weight_list[index]
        param_dict['data'] = Tensor(parameter.detach().cpu().numpy())
        ms_params_list.append(param_dict)

    save_path = os.path.dirname(pth_file)
    #save_file_name = os.path.join(save_path, "torch2ms_" + uuid.uuid1().hex[:8] + ".ckpt")
    save_file_name = 'torch2ms.ckpt'
    save_checkpoint(ms_params_list, save_file_name)
    return save_file_name


def pytorch2mindspore_extend(pth_file, model):
    """Convert torchvison  weight to vega weight of ms."""
    init_para_dict = torch.load(pth_file)
    init_names_list = []
    init_weights_list = []
    for index, name in enumerate(init_para_dict):
        init_names_list.append(name)
        init_weights_list.append(init_para_dict[name])

    vega_names_list = []
    vega_weights_list = []
    valid_names_list = []

    for name in model.parameters_dict():
        if not name.endswith("num_batches_tracked"):
            vega_names_list.append(name)

    for index, name in enumerate(vega_names_list):
        init_name = init_names_list[index]
        # if index < 1:
        #     continue
        if name.endswith("weight") and ("conv" or "downsample" in name or "down_sample" in name) and init_name.endswith(
                "weight") and ("conv" in init_name or "downsample" in init_name or "down_sample" in init_name):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        elif name.endswith("moving_mean") and init_name.endswith("running_mean"):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        elif name.endswith("moving_variance") and init_name.endswith(
                "running_var"):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        elif name.endswith("gamma") and init_name.endswith("weight") and (
                "bn" in init_name or "downsample" in init_name or "down_sample" in init_name):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        elif name.endswith("beta") and init_name.endswith("bias") and (
                "bn" in init_name or "downsample" in init_name or "down_sample" in init_name):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        else:
            continue

    ms_params_list = []

    for index, name in enumerate(valid_names_list):
        param_dict = {}
        param_dict['name'] = name
        parameter = vega_weights_list[index]
        param_dict['data'] = Tensor(parameter.detach().numpy())
        ms_params_list.append(param_dict)
    save_path = os.path.dirname(pth_file)
    save_file_name = os.path.join(save_path, "torch2ms_" + uuid.uuid1().hex[:8] + ".ckpt")
    save_checkpoint(ms_params_list, save_file_name)
    return save_file_name


def adaptive_weight(ckpt_file, ms_model):
    """Adapte the weight shape."""
    parameter_dict = load_checkpoint(ckpt_file)
    net_parameter = ms_model.parameters_and_names()
    new_ms_params_list = []
    for index, paras in enumerate(net_parameter):
        net_para_name = paras[0]
        net_para_shape = paras[1].data.shape

        if net_para_name in parameter_dict:
            init_weight = parameter_dict[net_para_name].data
            init_para_shape = init_weight.shape

            if net_para_shape != init_para_shape:
                if "conv" in net_para_name:
                    new_weight = _adaptive_conv(init_weight, net_para_shape)
                elif "batch_norm" in net_para_name:
                    new_weight = _adaptive_bn(init_weight, net_para_shape)
                else:
                    continue
                logging.debug("parameter shape not match,para name: {}, init_shape:{}, net_para_shape:{}".
                              format(net_para_name, init_para_shape, net_para_shape))
            param_dict = {}
            param_dict['name'] = net_para_name
            param_dict['data'] = init_weight if net_para_shape == init_para_shape else new_weight
            new_ms_params_list.append(param_dict)
            # parameter_dict[net_para_name].data = new_weight
    save_path = os.path.dirname(ckpt_file)
    save_file_name = os.path.join(save_path, "adaptive_" + uuid.uuid1().hex[:8] + ".ckpt")
    save_checkpoint(new_ms_params_list, save_file_name)
    if ckpt_file.startswith("torch2ms_"):
        os.remove(ckpt_file)
    return save_file_name


def _adaptive_conv(init_weight, new_shape):
    new_weight = init_weight.asnumpy()
    init_shape = init_weight.shape
    if init_shape[0] >= new_shape[0]:
        new_weight = new_weight[0:new_shape[0]]
    else:
        new_weight = np.tile(new_weight, (int(new_shape[0] / init_shape[0]), 1, 1, 1))

    if init_shape[1] >= new_shape[1]:
        new_weight = new_weight[:, 0:new_shape[1]]
    else:
        new_weight = np.tile(new_weight, (1, int(new_shape[1] / init_shape[1]), 1, 1))
    return Tensor(new_weight)


def _adaptive_bn(init_weight, new_shape):
    new_weight = init_weight.asnumpy()
    init_shape = init_weight.shape
    if init_shape[0] >= new_shape[0]:
        new_weight = new_weight[0:new_shape[0]]
    else:
        new_weight = np.tile(new_weight, int(new_shape[0] / init_shape[0]))
    return Tensor(new_weight)

if __name__ == '__main__':
    print(pytorch2mindspore('params_0020.pt'))