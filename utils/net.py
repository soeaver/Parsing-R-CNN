import numpy as np

import torch
import torch.nn as nn

import models.ops as ops


def make_conv(in_channels, out_channels, kernel=3, stride=1, dilation=1, padding=None, groups=1, 
              use_dwconv=False, conv_type='normal', use_bn=False, use_gn=False, use_relu=False, 
              kaiming_init=True, suffix_1x1=False, inplace=True, eps=1e-5, gn_group=32):
    _padding = (dilation * kernel - dilation) // 2 if padding is None else padding
    if use_dwconv:
        assert in_channels == out_channels
        _groups = out_channels
    else:
        _groups = groups
    
    if conv_type == 'normal':
        conv_op = nn.Conv2d
    elif conv_type == 'deform':
        conv_op = ops.DeformConvPack
    elif conv_type == 'deformv2':
        conv_op = ops.ModulatedDeformConvPack
    elif conv_type == 'convws':
        conv_op = ops.Conv2dWS
    else:
        raise ValueError('{} type conv operation is not supported.'.format(conv))
    conv = conv_op(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=_padding,
                   dilation=dilation, groups=_groups, bias=False if use_bn or use_gn else True)
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not (use_bn or use_gn):
        nn.init.constant_(conv.bias, 0)
    module = [conv, ]
    
    if use_bn:
        module.append(nn.BatchNorm2d(out_channels, eps=eps))
    if use_gn:
        module.append(nn.GroupNorm(gn_group, out_channels, eps=eps))
    if use_relu:
        module.append(nn.ReLU(inplace=inplace))
        
    if suffix_1x1:
        module.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False if use_bn or use_gn else True)
        )
        if use_bn:
            module.append(nn.BatchNorm2d(out_channels, eps=eps))
        if use_gn:
            module.append(nn.GroupNorm(gn_group, out_channels, eps=eps))
        if use_relu:
            module.append(nn.ReLU(inplace=inplace))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


def make_fc(dim_in, hidden_dim, use_bn=False, use_gn=False, eps=1e-5, gn_group=32):
    if use_bn or use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        module = [fc, ]
        if use_bn:
            module.append(nn.BatchNorm1d(hidden_dim))
        if use_gn:
            module.append(nn.GroupNorm(gn_group, hidden_dim, eps=eps))
        return nn.Sequential(*module)
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


def make_norm(c, norm='bn', eps=1e-5, an_k=10):
    if norm == 'bn':
        return nn.BatchNorm2d(c, eps=eps)
    elif norm == 'affine':
        return ops.AffineChannel2d(c)
    elif norm == 'gn':
        group = 32 if c >= 32 else c
        assert c % group == 0
        return nn.GroupNorm(group, c, eps=eps)
    elif norm == 'an_bn':
        return ops.MixtureBatchNorm2d(c, an_k)
    elif norm == 'an_gn':
        group = 32 if c >= 32 else c
        assert c % group == 0
        return ops.MixtureGroupNorm(c, group, an_k)
    elif norm == 'none':
        return None
    else:
        return nn.BatchNorm2d(c, eps=eps)


def convert_bn2affine_model(module, process_group=None, channel_last=False, merge=True):
    """
    This function is learned from the NVIDIA/apex.
    It can be seen here:
    https://github.com/NVIDIA/apex/blob/master/apex/parallel/sync_batchnorm.py

    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `ops.AffineChannel2d`.
    """
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not isinstance(module, ops.MixtureBatchNorm2d):
        # print(module.weight.cpu().detach().numpy().shape)
        mod = ops.AffineChannel2d(module.num_features)
        mod.weight.data = module.weight.data.clone().detach()
        mod.bias.data = module.bias.data.clone().detach()
        freeze_params(mod)  # freeze affine params
        if merge:
            gamma = module.weight.data.clone().detach().numpy()
            beta = module.bias.data.clone().detach().numpy()
            mu = module.running_mean.data.clone().detach().numpy()
            var = module.running_var.data.clone().detach().numpy()
            eps = module.eps

            new_gamma = gamma / (np.power(var + eps, 0.5))  # new bn.weight
            new_beta = beta - gamma * mu / (np.power(var + eps, 0.5))  # new bn.bias

            mod.weight.data = torch.from_numpy(new_gamma)
            mod.bias.data = torch.from_numpy(new_beta)
    for name, child in module.named_children():
        mod.add_module(name, convert_bn2affine_model(child, process_group=process_group, channel_last=channel_last,
                                                     merge=merge))
    del module
    return mod


def convert_conv2syncbn_model(module, process_group=None):
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = ops.NaiveSyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine,
                                     module.track_running_stats)
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
            # keep reuqires_grad unchanged
            mod.weight.requires_grad = module.weight.requires_grad
            mod.bias.requires_grad = module.bias.requires_grad
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        mod.add_module(name, convert_conv2syncbn_model(child, process_group=process_group))
    del module
    return mod
  
  
def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


def mismatch_params_filter(s):
    l = []
    for i in s:
        if i.split('.')[-1] in ['num_batches_tracked', 'running_mean', 'running_var']:
            continue
        else:
            l.append(i)
    return l


def reduce_tensor(tensor, world_size=1):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt
