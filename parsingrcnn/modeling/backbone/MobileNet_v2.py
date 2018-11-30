import os
import numbers
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (MobileNet_v2_C4_body, MobileNet_v2_C5_body, ...)
# ---------------------------------------------------------------------------- #

def MobileNet_v2_C4_body():
    return MobileNet_v2_CX_body(cfg.MOBILENETV2.BLOCK_COUNTS[:3])


def MobileNet_v2_C5_body():
    return MobileNet_v2_CX_body(cfg.MOBILENETV2.BLOCK_COUNTS)


# ---------------------------------------------------------------------------- #
# Generic MobileNet_v2 components
# ---------------------------------------------------------------------------- #


class MobileNet_v2_CX_body(nn.Module):
    def __init__(self, block_counts=(2, 3, (4, 3), (3, 1))):
        super().__init__()
        self.block_counts = block_counts  # 3 or 4
        self.convX = len(block_counts) + 1  # 4 or 5

        num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]	
        channels = [_make_divisible(ch * cfg.MOBILENETV2.WIDEN_FACTOR, 8) for ch in num_of_channels]
        stage_channels = [channels[2], channels[3], (channels[4], channels[5]), (channels[6], channels[7])]

        if cfg.MOBILENETV2.ACTIVATION == 'relu6':
            activ = nn.ReLU6()
        else:
            activ = nn.ReLU(True)

        self.mobv2_1 = globals()[cfg.MOBILENETV2.STEM_FUNC](channels, activation=activ)
        dim_in = channels[1]

        self.mobv2_2, dim_in = add_stage(dim_in, stage_channels[0], dim_in * cfg.MOBILENETV2.INVERTED_T, block_counts[0],
                                         dilation=1, stride_init=2, activation=activ)
        self.mobv2_3, dim_in = add_stage(dim_in, stage_channels[1], dim_in * cfg.MOBILENETV2.INVERTED_T, block_counts[1],
                                         dilation=1, stride_init=2, activation=activ)
        self.mobv2_4, dim_in = add_stage(dim_in, stage_channels[2], dim_in * cfg.MOBILENETV2.INVERTED_T, block_counts[2],
                                         dilation=1, stride_init=2, activation=activ)

        if len(block_counts) == 4:
            if cfg.MOBILENETV2.C5_DILATION != 1:
                stride = 1
            else:
                stride = 2
            self.mobv2_5, dim_in = add_stage(dim_in, stage_channels[3], dim_in * cfg.MOBILENETV2.INVERTED_T, block_counts[3],
                                             dilation=cfg.MOBILENETV2.C5_DILATION, stride_init=stride, activation=activ)
            self.spatial_scale = 1 / 32 * cfg.MOBILENETV2.C5_DILATION  # cfg.MOBILENETV2.C5_DILATION: 1
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.MOBILENETV2.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.MOBILENETV2.FREEZE_AT: 2
        assert cfg.MOBILENETV2.FREEZE_AT <= self.convX
        for i in range(1, cfg.MOBILENETV2.FREEZE_AT + 1):
            freeze_params(getattr(self, 'mobv2_%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.MOBILENETV2.USE_GN:
            mapping_to_detectron = {
                'mobv2_1.conv1.weight': 'conv1_w',
                'mobv2_1.gn1.weight': 'conv1_gn_s',
                'mobv2_1.gn1.bias': 'conv1_gn_b',
                'mobv2_1.conv2.weight': 'conv2_w',
                'mobv2_1.gn2.weight': 'conv2_gn_s',
                'mobv2_1.gn2.bias': 'conv2_gn_b',
                'mobv2_1.conv3.weight': 'conv3_w',
                'mobv2_1.gn3.weight': 'conv3_gn_s',
                'mobv2_1.gn3.bias': 'conv3_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'mobv2_1.conv1.weight': 'conv1_w',
                'mobv2_1.bn1.weight': 'conv1_bn_s',
                'mobv2_1.bn1.bias': 'conv1_bn_b',
                'mobv2_1.conv2.weight': 'conv2_w',
                'mobv2_1.bn2.weight': 'conv2_bn_s',
                'mobv2_1.bn2.bias': 'conv2_bn_b',
                'mobv2_1.conv3.weight': 'conv3_w',
                'mobv2_1.bn3.weight': 'conv3_bn_s',
                'mobv2_1.bn3.bias': 'conv3_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'conv2_b', 'conv3_b', 'fc_w', 'fc_b']

        for mobv2_id in range(2, self.convX + 1):
            stage_name = 'mobv2_%d' % mobv2_id  # mobv2_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_detectron_mapping(stage_name, self.block_counts, mobv2_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        if cfg.MOBILENETV2.USE_GN:
            mapping_to_pytorch = {
                'mobv2_1.conv1.weight': 'conv1.weight',
                'mobv2_1.gn1.weight': 'bn1.weight',
                'mobv2_1.gn1.bias': 'bn1.bias',
                'mobv2_1.conv2.weight': 'conv2.weight',
                'mobv2_1.gn2.weight': 'bn2.weight',
                'mobv2_1.gn2.bias': 'bn2.bias',
                'mobv2_1.conv3.weight': 'conv3.weight',
                'mobv2_1.gn3.weight': 'bn3.weight',
                'mobv2_1.gn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']
        else:
            mapping_to_pytorch = {
                'mobv2_1.conv1.weight': 'conv1.weight',
                'mobv2_1.bn1.weight': 'bn1.weight',
                'mobv2_1.bn1.bias': 'bn1.bias',
                'mobv2_1.conv2.weight': 'conv2.weight',
                'mobv2_1.bn2.weight': 'bn2.weight',
                'mobv2_1.bn2.bias': 'bn2.bias',
                'mobv2_1.conv3.weight': 'conv3.weight',
                'mobv2_1.bn3.weight': 'bn3.weight',
                'mobv2_1.bn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']

        for mobv2_id in range(2, self.convX + 1):
            stage_name = 'mobv2_%d' % mobv2_id  # mobv2_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_pytorch_mapping(stage_name, self.block_counts, mobv2_id)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.MOBILENETV2.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'mobv2_%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'mobv2_%d' % (i + 1))(x)
        return x


def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2, activation=nn.ReLU6):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    if isinstance(nblocks, numbers.Number):
        for _ in range(nblocks):
            res_blocks.append(add_residual_block(
                inplanes, outplanes, innerplanes, dilation, stride, activation
            ))
            inplanes = outplanes
            innerplanes = inplanes * cfg.MOBILENETV2.INVERTED_T
            stride = 1
        outplane = outplanes
    else:
        assert len(outplanes) == len(nblocks)
        for idx in range(len(nblocks)):
            for _ in range(nblocks[idx]):
                res_blocks.append(add_residual_block(
                    inplanes, outplanes[idx], innerplanes, dilation, stride, activation
                ))
                inplanes = outplanes[idx]
                innerplanes = inplanes * cfg.MOBILENETV2.INVERTED_T
                stride = 1
        outplane = outplanes[-1]

    return nn.Sequential(*res_blocks), outplane


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride, activation=nn.ReLU6):
    """Return a residual block module, including residual connection, """

    trans_func = globals()[cfg.MOBILENETV2.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride=stride, dilation=dilation, kernel=cfg.MOBILENETV2.KERNEL_SIZE,
        groups=cfg.MOBILENETV2.GROUPS, activation=activation
    )

    return res_block


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_stem(channels, activation=nn.ReLU6):
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False)),
        ('bn1', mynn.AffineChannel2d(channels[0])),
        ('relu1', activation),
        ('conv2', nn.Conv2d(channels[0], channels[0], kernel_size=cfg.MOBILENETV2.KERNEL_SIZE, stride=1,
                            padding=cfg.MOBILENETV2.KERNEL_SIZE // 2, groups=channels[0], bias=False)),
        ('bn2', mynn.AffineChannel2d(channels[0])),
        ('relu2', activation),
        ('conv3', nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)),
        ('bn3', mynn.AffineChannel2d(channels[1])),
        ('relu3', activation)])
    )


def basic_gn_stem(channels, activation=nn.ReLU6):
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False)),
        ('gn1', nn.GroupNorm(net_utils.get_group_gn(channels[0]), channels[0], eps=cfg.GROUP_NORM.EPSILON)),
        ('relu1', activation),
        ('conv2', nn.Conv2d(channels[0], channels[0], kernel_size=cfg.MOBILENETV2.KERNEL_SIZE, stride=1,
                            padding=cfg.MOBILENETV2.KERNEL_SIZE // 2, groups=channels[0], bias=False)),
        ('gn2', nn.GroupNorm(net_utils.get_group_gn(channels[0]), channels[0], eps=cfg.GROUP_NORM.EPSILON)),
        ('relu2', activation),
        ('conv3', nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)),
        ('gn3', nn.GroupNorm(net_utils.get_group_gn(channels[1]), channels[1], eps=cfg.GROUP_NORM.EPSILON)),
        ('relu3', activation)])
    )


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class linear_bottleneck_transformation(nn.Module):
    """ Linear Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, kernel=3, groups=(1, 1),
                 activation=nn.ReLU6):
        super().__init__()
        self.stride = stride
        self.inplanes, self.outplanes = int(inplanes), int(outplanes)
        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, padding=0, stride=1, groups=groups[0], bias=False)
        self.bn1 = mynn.AffineChannel2d(innerplanes)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel, stride=stride, dilation=dilation,
                               padding=(dilation * kernel - dilation) // 2, groups=innerplanes, bias=False)
        self.bn2 = mynn.AffineChannel2d(innerplanes)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, padding=0, stride=1, groups=groups[1], bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)
        self.activation = activation

    def forward(self, x):
        if self.stride == 1 and self.inplanes == self.outplanes:
            residual = x
        else:
            residual = None

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual
        else:
            pass

        return out


class linear_bottleneck_gn_transformation(nn.Module):
    """ Linear Bottleneck Residual Block With GN"""

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, kernel=3, groups=(1, 1),
                 activation=nn.ReLU6):
        super().__init__()
        self.stride = stride
        self.inplanes, self.outplanes = int(inplanes), int(outplanes)
        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, padding=0, stride=1, groups=groups[0], bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel, stride=stride, dilation=dilation,
                               padding=(dilation * kernel - dilation) // 2, groups=innerplanes, bias=False)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, padding=0, stride=1, groups=groups[1], bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.activation = activation

    def forward(self, x):
        if self.stride == 1 and self.inplanes == self.outplanes:
            residual = x
        else:
            residual = None

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual
        else:
            pass

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
    num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
    channels = [_make_divisible(ch * cfg.MOBILENETV2.WIDEN_FACTOR, 8) for ch in num_of_channels]
    return [channels[2], channels[3], channels[5], channels[7]]


def residual_stage_detectron_mapping(module_name, block_counts, mobv2_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `mobv2_id`
    """

    num_blocks = block_counts[mobv2_id - 2]
    if cfg.MOBILENETV2.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []

    layer_id = 1
    for _ in block_counts[:mobv2_id - 2]:
        if isinstance(_, numbers.Number):
            layer_id += _
        else:
            layer_id += sum(_)

    if isinstance(num_blocks, numbers.Number):
        for blk_id in range(num_blocks):
            detectron_prefix = 'mobv2_%d' % (layer_id + blk_id)
            my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: mobv2_2, mobv2_3, mobv2_4, mobv2_5

            # conv branch
            for i in range(1, 4):
                dtt_bp = detectron_prefix + '_conv' + str(i)
                mapping_to_detectron[my_prefix + '.conv%d.weight' % i] = dtt_bp + '_w'
                orphan_in_detectron.append(dtt_bp + '_b')
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'
    else:
        for idx in range(len(num_blocks)):
            for blk_id in range(num_blocks[idx]):
                detectron_prefix = 'mobv2_%d' % (layer_id + blk_id)
                # module_name: mobv2_2, mobv2_3, mobv2_4, mobv2_5
                my_prefix = '%s.%d' % (module_name, blk_id + sum(num_blocks[:idx]))

                # conv branch
                for i in range(1, 4):
                    dtt_bp = detectron_prefix + '_conv' + str(i)
                    mapping_to_detectron[my_prefix + '.conv%d.weight' % i] = dtt_bp + '_w'
                    orphan_in_detectron.append(dtt_bp + '_b')
                    mapping_to_detectron[
                        my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
                    mapping_to_detectron[
                        my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'
            layer_id += num_blocks[idx]

    return mapping_to_detectron, orphan_in_detectron


def residual_stage_pytorch_mapping(module_name, block_counts, mobv2_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `mobv2_id`
    """

    num_blocks = block_counts[mobv2_id - 2]
    if cfg.MOBILENETV2.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_pytorch = {}
    orphan_in_pytorch = []

    layer_id = 1
    for _ in block_counts[:mobv2_id - 2]:
        if isinstance(_, numbers.Number):
            layer_id += 1
        else:
            layer_id += len(_)       

    if isinstance(num_blocks, numbers.Number):
        for blk_id in range(num_blocks):
            pytorch_prefix = 'layer{}.{}'.format(layer_id, blk_id)
            my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: mobv2_2, mobv2_3, mobv2_4, mobv2_5

            # conv branch
            for i in range(1, 4):   # i: 1, 2, 3
                mapping_to_pytorch[my_prefix + '.conv%d.weight' % i] = pytorch_prefix + '.conv%d.weight' % i
                mapping_to_pytorch[
                    my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = pytorch_prefix + '.bn%d.weight' % i
                mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = pytorch_prefix + '.bn%d.bias' % i
    else:
        for idx in range(len(num_blocks)):
            for blk_id in range(num_blocks[idx]): 
                pytorch_prefix = 'layer{}.{}'.format(layer_id, blk_id)
                # module_name: mobv2_2, mobv2_3, mobv2_4, mobv2_5
                my_prefix = '%s.%d' % (module_name, blk_id + sum(num_blocks[:idx]))
                
                # conv branch
                for i in range(1, 4):   # i: 1, 2, 3
                    mapping_to_pytorch[my_prefix + '.conv%d.weight' % i] = pytorch_prefix + '.conv%d.weight' % i
                    mapping_to_pytorch[
                        my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = pytorch_prefix + '.bn%d.weight' % i
                    mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = pytorch_prefix + '.bn%d.bias' % i
            layer_id += 1

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
