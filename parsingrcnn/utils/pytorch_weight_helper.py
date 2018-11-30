"""Helper functions for loading pretrained weights from Pytorch .pth files
"""

import torch
import numpy as np


def new_bn_weight(ckpt_weights, ckpt_name, eps=0.00001):
    # bn + gn
    if ckpt_name.replace('weight', 'var_weight') not in ckpt_weights:
        if ckpt_name.replace('weight', 'running_var') in ckpt_weights:  # bn --> merge
            var = ckpt_weights[ckpt_name.replace('weight', 'running_var')].cpu().numpy()
            gamma = ckpt_weights[ckpt_name].cpu().numpy()
            new_weight = gamma / (np.power(var + eps, 0.5))
            print('===> batchnorm.weight need merge.  '),
        else:  # gn
            new_weight = ckpt_weights[ckpt_name].cpu().numpy()
    else:  # sn
        new_weight = ckpt_weights[ckpt_name].cpu().numpy()

    return new_weight


def new_bn_bias(ckpt_weights, ckpt_name, eps=0.00001):
    # bn + gn
    if ckpt_name.replace('bias', 'mean_weight') not in ckpt_weights:
        if ckpt_name.replace('bias', 'running_var') in ckpt_weights:  # bn --> merge
            mu = ckpt_weights[ckpt_name.replace('bias', 'running_mean')].cpu().numpy()
            var = ckpt_weights[ckpt_name.replace('bias', 'running_var')].cpu().numpy()
            gamma = ckpt_weights[ckpt_name.replace('bias', 'weight')].cpu().numpy()
            beta = ckpt_weights[ckpt_name].cpu().numpy()
            new_bias = beta - gamma * mu / (np.power(var + eps, 0.5))
            print('===> batchnorm.bias need merge.  '),
        else:  # gn
            new_bias = ckpt_weights[ckpt_name].cpu().numpy()
    else:  # sn
        new_bias = ckpt_weights[ckpt_name].cpu().numpy()

    return new_bias


def remove_pytorch_prefix(pytorch_weights, prefix_base=('module',)):
    ckpt_weights = {}
    for py_name in pytorch_weights:
        weights = pytorch_weights[py_name]

        new_name = py_name
        for i in prefix_base:
            if i in py_name:
                new_name = py_name.replace(i + '.', '')

        ckpt_weights[new_name] = weights

    return ckpt_weights


def load_pytorch_weight(net, pytorch_weight_file, first_conv='conv1', merge_bn=True):
    name_mapping, orphan_in_pytorch = net.pytorch_weight_mapping
    # print(name_mapping)

    pytorch_weights = torch.load(pytorch_weight_file)
    ckpt_weights = remove_pytorch_prefix(pytorch_weights)

    params = net.state_dict()
    for net_name, net_tensor in params.items():
        if net_name not in name_mapping:
            print('!!! {} has no pretrain weights.'.format(net_name))
            continue

        ckpt_name = name_mapping[net_name]
        if not isinstance(ckpt_name, str):  # maybe str, None or True
            print('!!! {} is not a string.'.format(net_name))
            continue

        # Step 1: change first conv layer channel
        if ckpt_name == '{}.weight'.format(first_conv):
            conv1 = ckpt_weights[ckpt_name].cpu().numpy().copy()
            conv1[:, [0, 2], :, :] = conv1[:, [2, 0], :, :]
            net_tensor.copy_(torch.Tensor(conv1))
            print('===> load weight: {} -----> {}, convert first conv weight channel'.
                  format(ckpt_name, net_name), net_tensor.shape)

        # Step 2: mapping weights
        if merge_bn and ('bn' in ckpt_name or 'downsample.1' in ckpt_name) and 'weight' in ckpt_name:
            # Step 2.1: mapping bn.weight, and merge if needed    
            new_weight = new_bn_weight(ckpt_weights, ckpt_name)
            net_tensor.copy_(torch.Tensor(new_weight))
        elif merge_bn and ('bn' in ckpt_name or 'downsample.1' in ckpt_name) and 'bias' in ckpt_name:
            # Step 2.2: mapping bn.bias, and merge if needed    
            new_bias = new_bn_bias(ckpt_weights, ckpt_name)
            net_tensor.copy_(torch.Tensor(new_bias))
        else:
            # Step 2.3: mapping the other weights   
            net_tensor.copy_(torch.Tensor(ckpt_weights[ckpt_name].cpu()))
        print('===> load weight: {} -----> {}'.format(ckpt_name, net_name), net_tensor.shape)
