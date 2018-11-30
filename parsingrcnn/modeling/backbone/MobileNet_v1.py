import os
import numbers
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (MobileNet_v1_C4_body, MobileNet_v1_C5_body, ...)
# ---------------------------------------------------------------------------- #

def MobileNet_v1_C4_body():
    return MobileNet_v1_CX_body(cfg.MOBILENETV1.BLOCK_COUNTS[:3])


def MobileNet_v1_C5_body():
    return MobileNet_v1_CX_body(cfg.MOBILENETV1.BLOCK_COUNTS)


# ---------------------------------------------------------------------------- #
# Generic MobileNet_v1 components
# ---------------------------------------------------------------------------- #


class MobileNet_v1_CX_body(nn.Module):
    def __init__(self, block_counts=(2, 2, 6, 2)):
        super().__init__()
        self.block_counts = block_counts  # 3 or 4
        self.convX = len(block_counts) + 1  # 4 or 5
        self.num_layers = sum(block_counts) * 2 + 4

        num_of_channels = [32, 64, 128, 256, 512, 1024]	
        channels = [_make_divisible(ch * cfg.MOBILENETV1.WIDEN_FACTOR, 8) for ch in num_of_channels]

        if cfg.MOBILENETV1.ACTIVATION == 'relu6':
            activ = nn.ReLU6()
        else:
            activ = nn.ReLU(True)

        self.mobv1_1 = globals()[cfg.MOBILENETV1.STEM_FUNC](channels, activation=activ)
        dim_in = channels[1]

        self.mobv1_2, dim_in = add_stage(dim_in, channels[2], block_counts[0], dilation=1, stride_init=2, activation=activ)
        self.mobv1_3, dim_in = add_stage(dim_in, channels[3], block_counts[1], dilation=1, stride_init=2, activation=activ)
        self.mobv1_4, dim_in = add_stage(dim_in, channels[4], block_counts[2], dilation=1, stride_init=2, activation=activ)

        if len(block_counts) == 4:
            if cfg.MOBILENETV1.C5_DILATION != 1:
                stride = 1
            else:
                stride = 2
            self.mobv1_5, dim_in = add_stage(dim_in, channels[5], block_counts[3], dilation=cfg.MOBILENETV1.C5_DILATION, 
                                             stride_init=stride, activation=activ)
            self.spatial_scale = 1 / 32 * cfg.MOBILENETV1.C5_DILATION  # cfg.MOBILENETV1.C5_DILATION: 1
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.MOBILENETV1.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.MOBILENETV1.FREEZE_AT: 2
        assert cfg.MOBILENETV1.FREEZE_AT <= self.convX
        for i in range(1, cfg.MOBILENETV1.FREEZE_AT + 1):
            freeze_params(getattr(self, 'mobv1_%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.MOBILENETV1.USE_GN:
            mapping_to_detectron = {
                'mobv1_1.conv1.weight': 'conv1_w',
                'mobv1_1.gn1.weight': 'conv1_gn_s',
                'mobv1_1.gn1.bias': 'conv1_gn_b',
                'mobv1_1.conv2.weight': 'conv2_w',
                'mobv1_1.gn2.weight': 'conv2_gn_s',
                'mobv1_1.gn2.bias': 'conv2_gn_b',
                'mobv1_1.conv3.weight': 'conv3_w',
                'mobv1_1.gn3.weight': 'conv3_gn_s',
                'mobv1_1.gn3.bias': 'conv3_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'mobv1_1.conv1.weight': 'conv1_w',
                'mobv1_1.bn1.weight': 'conv1_bn_s',
                'mobv1_1.bn1.bias': 'conv1_bn_b',
                'mobv1_1.conv2.weight': 'conv2_w',
                'mobv1_1.bn2.weight': 'conv2_bn_s',
                'mobv1_1.bn2.bias': 'conv2_bn_b',
                'mobv1_1.conv3.weight': 'conv3_w',
                'mobv1_1.bn3.weight': 'conv3_bn_s',
                'mobv1_1.bn3.bias': 'conv3_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'conv2_b', 'conv3_b', 'fc_w', 'fc_b']

        for mobv1_id in range(2, self.convX + 1):
            stage_name = 'mobv1_%d' % mobv1_id  # mobv1_id = 2, 3, 4, 5
            mapping, orphans = basic_stage_detectron_mapping(stage_name, self.block_counts, mobv1_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        if cfg.MOBILENETV1.USE_GN:
            mapping_to_pytorch = {
                'mobv1_1.conv1.weight': 'conv1.weight',
                'mobv1_1.gn1.weight': 'bn1.weight',
                'mobv1_1.gn1.bias': 'bn1.bias',
                'mobv1_1.conv2.weight': 'conv2.weight',
                'mobv1_1.gn2.weight': 'bn2.weight',
                'mobv1_1.gn2.bias': 'bn2.bias',
                'mobv1_1.conv3.weight': 'conv3.weight',
                'mobv1_1.gn3.weight': 'bn3.weight',
                'mobv1_1.gn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']
        else:
            mapping_to_pytorch = {
                'mobv1_1.conv1.weight': 'conv1.weight',
                'mobv1_1.bn1.weight': 'bn1.weight',
                'mobv1_1.bn1.bias': 'bn1.bias',
                'mobv1_1.conv2.weight': 'conv2.weight',
                'mobv1_1.bn2.weight': 'bn2.weight',
                'mobv1_1.bn2.bias': 'bn2.bias',
                'mobv1_1.conv3.weight': 'conv3.weight',
                'mobv1_1.bn3.weight': 'bn3.weight',
                'mobv1_1.bn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']

        for mobv1_id in range(2, self.convX + 1):
            stage_name = 'mobv1_%d' % mobv1_id  # mobv1_id = 2, 3, 4, 5
            mapping, orphans = basic_stage_pytorch_mapping(stage_name, self.block_counts, mobv1_id)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.MOBILENETV1.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'mobv1_%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'mobv1_%d' % (i + 1))(x)
        return x


def add_stage(inplanes, outplanes, nblocks, dilation=1, stride_init=2, activation=nn.ReLU):
    """Make a stage consist of `nblocks` basic blocks.
    Returns:
        - stage module: an nn.Sequentail module of basic blocks
        - final output dimension
    """
    basic_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        basic_blocks.append(add_basic_block(
            inplanes, outplanes, dilation, stride, activation
        ))
        inplanes = outplanes
        stride = 1
                 
    return nn.Sequential(*basic_blocks), outplanes


def add_basic_block(inplanes, outplanes, dilation, stride, activation=nn.ReLU):
    """Return a basic block module, """

    trans_func = globals()[cfg.MOBILENETV1.TRANS_FUNC]
    baisck_block = trans_func(
        inplanes, outplanes, stride=stride, dilation=dilation, kernel=cfg.MOBILENETV1.KERNEL_SIZE,
        activation=activation
    )

    return baisck_block


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_stem(channels, activation=nn.ReLU):
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False)),
        ('bn1', mynn.AffineChannel2d(channels[0])),
        ('relu1', activation),
        ('conv2', nn.Conv2d(channels[0], channels[0], kernel_size=cfg.MOBILENETV1.KERNEL_SIZE, stride=1,
                            padding=cfg.MOBILENETV1.KERNEL_SIZE // 2, groups=channels[0], bias=False)),
        ('bn2', mynn.AffineChannel2d(channels[0])),
        ('relu2', activation),
        ('conv3', nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)),
        ('bn3', mynn.AffineChannel2d(channels[1])),
        ('relu3', activation)])
    )


def basic_gn_stem(channels, activation=nn.ReLU):
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False)),
        ('gn1', nn.GroupNorm(net_utils.get_group_gn(channels[0]), channels[0], eps=cfg.GROUP_NORM.EPSILON)),
        ('relu1', activation),
        ('conv2', nn.Conv2d(channels[0], channels[0], kernel_size=cfg.MOBILENETV1.KERNEL_SIZE, stride=1,
                            padding=cfg.MOBILENETV1.KERNEL_SIZE // 2, groups=channels[0], bias=False)),
        ('gn2', nn.GroupNorm(net_utils.get_group_gn(channels[0]), channels[0], eps=cfg.GROUP_NORM.EPSILON)),
        ('relu2', activation),
        ('conv3', nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)),
        ('gn3', nn.GroupNorm(net_utils.get_group_gn(channels[1]), channels[1], eps=cfg.GROUP_NORM.EPSILON)),
        ('relu3', activation)])
    )


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class basicblock_transformation(nn.Module):
    """ Basic Block """

    def __init__(self, inplanes, planes, stride=1, dilation=1, kernel=3, activation=nn.ReLU):
        super().__init__()
        self.inplanes, self.planes = int(inplanes), int(planes)
                 
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=kernel, padding=kernel // 2, stride=stride,
                               groups=inplanes, bias=False)
        self.bn1 = mynn.AffineChannel2d(inplanes)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = mynn.AffineChannel2d(planes)
                 
        self.activation = activation

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        return out


class basicblock_gn_transformation(nn.Module):
    """ Basic Block With GN"""

    def __init__(self, inplanes, planes, stride=1, dilation=1, kernel=3, activation=nn.ReLU):
        super().__init__()
        self.inplanes, self.planes = int(inplanes), int(planes)
                 
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=kernel, padding=kernel // 2, stride=stride,
                               groups=inplanes, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(inplanes), inplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(planes), planes, eps=cfg.GROUP_NORM.EPSILON)
 
        self.activation = activation

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.activation(out)

        return out


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #


def _make_divisible(v, divisor, min_value=None):
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

                 
def level_outplanes():
    num_of_channels = [32, 64, 128, 256, 512, 1024]
    channels = [_make_divisible(ch * cfg.MOBILENETV1.WIDEN_FACTOR, 8) for ch in num_of_channels]
    return [channels[2], channels[3], channels[4], channels[5]]
                 
                 
def basic_stage_detectron_mapping(module_name, block_counts, mobv1_id):
    """Construct weight mapping relation for a basic stage with `num_blocks` of
    basic blocks given the stage id: `mobv1_id`
    """

    num_blocks = block_counts[mobv1_id - 2]
    if cfg.MOBILENETV1.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'mobv1_%d' % (sum(block_counts[:mobv1_id - 2]) + blk_id + 1)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: mobv1_2, mobv1_3, mobv1_4, mobv1_5

        # conv branch
        for i in range(1, 3):
            dtt_bp = detectron_prefix + '_conv' + str(i)
            mapping_to_detectron[my_prefix + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'

    return mapping_to_detectron, orphan_in_detectron


def basic_stage_pytorch_mapping(module_name, block_counts, mobv1_id):
    """Construct weight mapping relation for a basic stage with `num_blocks` of
    basic blocks given the stage id: `mobv1_id`
    """

    num_blocks = block_counts[mobv1_id - 2]
    if cfg.MOBILENETV1.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_pytorch = {}
    orphan_in_pytorch = []

    for blk_id in range(num_blocks):
        pytorch_prefix = 'layer{}.{}'.format(mobv1_id - 1, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: mobv1_2, mobv1_3, mobv1_4, mobv1_5

        # conv branch
        for i in range(1, 3):   # i: 1, 2
            mapping_to_pytorch[my_prefix + '.conv%d.weight' % i] = pytorch_prefix + '.conv%d.weight' % i
            mapping_to_pytorch[
                my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = pytorch_prefix + '.bn%d.weight' % i
            mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = pytorch_prefix + '.bn%d.bias' % i

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
