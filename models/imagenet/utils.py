from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.ops import Conv2dSamePadding, Conv2dWS


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def convert_conv2convsamepadding_model(module, process_group=None, channel_last=False):
    mod = module
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        if isinstance(module.bias, torch.Tensor):
            bias = True
        else:
            bias = False
        mod = Conv2dSamePadding(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                module.dilation, module.groups, bias=bias)
        mod.weight.data = module.weight.data.clone().detach()
        if bias:
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_conv2convsamepadding_model(child, process_group=process_group,
                                                                channel_last=channel_last))
    # TODO(jie) should I delete model explicitly?
    del module
    return mod


def convert_conv2convws_model(module, process_group=None, channel_last=False):
    mod = module
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        if isinstance(module.bias, torch.Tensor):
            bias = True
        else:
            bias = False
        mod = Conv2dWS(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding,
                       module.dilation, module.groups, bias=bias)
        mod.weight.data = module.weight.data.clone().detach()
        if bias:
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_conv2convws_model(child, process_group=process_group, channel_last=channel_last))
    # TODO(jie) should I delete model explicitly?
    del module
    return mod


class IntermediateLayerGetter(nn.ModuleDict):
    """
    This function is taken from the torchvision repo.
    It can be seen here:
    https://github.com/pytorch/vision/blob/master/torchvision/models/_utils.py

    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
