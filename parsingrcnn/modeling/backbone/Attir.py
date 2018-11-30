import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (Attir50, Attir101, ...)
# ---------------------------------------------------------------------------- #

def Attir50_conv4_body():
    return Attir_convX_body((3, 4, 6))


def Attir50_conv5_body():
    return Attir_convX_body((3, 4, 6, 3))


def Attir101_conv4_body():
    return Attir_convX_body((3, 4, 23))


def Attir101_conv5_body():
    return Attir_convX_body((3, 4, 23, 3))


def Attir50_roi_conv5_head(dim_in, roi_xform_func, spatial_scale):
    return Attir_roi_conv5_head(dim_in, roi_xform_func, spatial_scale, (3, 4, 6, 3))


def Attir101_roi_conv5_head(dim_in, roi_xform_func, spatial_scale):
    return Attir_roi_conv5_head(dim_in, roi_xform_func, spatial_scale, (3, 4, 23, 3))


# ---------------------------------------------------------------------------- #
# Generic Attir components
# ---------------------------------------------------------------------------- #


class Attir_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts  # 3 or 4
        self.convX = len(block_counts) + 1  # 4 or 5
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2  #

        self.attir1 = globals()[cfg.ATTIRS.STEM_FUNC]()
        dim_in = 64
        # cfg.ATTIRS.NUM_GROUPS: 1     cfg.ATTIRS.WIDTH_PER_GROUP: 64
        dim_bottleneck = cfg.ATTIRS.NUM_GROUPS * cfg.ATTIRS.WIDTH_PER_GROUP
        self.attir2, dim_in = add_stage(dim_in, cfg.ATTIRS.WIDTH_OUTPLANE, dim_bottleneck, block_counts[0],
                                        dilation=1, stride_init=1)
        self.attir3, dim_in = add_stage(dim_in, cfg.ATTIRS.WIDTH_OUTPLANE * 2, dim_bottleneck * 2, block_counts[1],
                                        dilation=1, stride_init=2)
        self.attir4, dim_in = add_stage(dim_in, cfg.ATTIRS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4, block_counts[2],
                                        dilation=1, stride_init=2)
        if len(block_counts) == 4:
            if cfg.ATTIRS.C5_DILATION != 1:
                stride = 1
            else:
                stride = 2
            self.attir5, dim_in = add_stage(dim_in, cfg.ATTIRS.WIDTH_OUTPLANE * 8, dim_bottleneck * 8, block_counts[3],
                                            dilation=cfg.ATTIRS.C5_DILATION, stride_init=stride)
            self.spatial_scale = 1 / 32 * cfg.ATTIRS.C5_DILATION  # cfg.ATTIRS.C5_DILATION: 1
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.ATTIRS.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.ATTIRS.FREEZE_AT: 2
        assert cfg.ATTIRS.FREEZE_AT <= self.convX
        for i in range(1, cfg.ATTIRS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'attir%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.ATTIRS.USE_GN:
            mapping_to_detectron = {
                'attir1.conv1.weight': 'conv1_w',
                'attir1.gn1.weight': 'conv1_gn_s',
                'attir1.gn1.bias': 'conv1_gn_b',
                'attir1.conv2.weight': 'conv2_w',
                'attir1.gn2.weight': 'conv2_gn_s',
                'attir1.gn2.bias': 'conv2_gn_b',
                'attir1.conv3.weight': 'conv3_w',
                'attir1.gn3.weight': 'conv3_gn_s',
                'attir1.gn3.bias': 'conv3_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'attir1.conv1.weight': 'conv1_w',
                'attir1.bn1.weight': 'conv1_bn_s',
                'attir1.bn1.bias': 'conv1_bn_b',
                'attir1.conv2.weight': 'conv2_w',
                'attir1.bn2.weight': 'conv2_bn_s',
                'attir1.bn2.bias': 'conv2_bn_b',
                'attir1.conv3.weight': 'conv3_w',
                'attir1.bn3.weight': 'conv3_bn_s',
                'attir1.bn3.bias': 'conv3_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'conv2_b', 'conv3_b', 'fc_w', 'fc_b']

        for attir_id in range(2, self.convX + 1):
            stage_name = 'attir%d' % attir_id  # attir_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, attir_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        if cfg.ATTIRS.USE_GN:
            mapping_to_pytorch = {
                'attir1.conv1.weight': 'conv1.weight',
                'attir1.gn1.weight': 'bn1.weight',
                'attir1.gn1.bias': 'bn1.bias',
                'attir1.conv2.weight': 'conv2.weight',
                'attir1.gn2.weight': 'bn2.weight',
                'attir1.gn2.bias': 'bn2.bias',
                'attir1.conv3.weight': 'conv3.weight',
                'attir1.gn3.weight': 'bn3.weight',
                'attir1.gn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']
        else:
            mapping_to_pytorch = {
                'attir1.conv1.weight': 'conv1.weight',
                'attir1.bn1.weight': 'bn1.weight',
                'attir1.bn1.bias': 'bn1.bias',
                'attir1.conv2.weight': 'conv2.weight',
                'attir1.bn2.weight': 'bn2.weight',
                'attir1.bn2.bias': 'bn2.bias',
                'attir1.conv3.weight': 'conv3.weight',
                'attir1.bn3.weight': 'bn3.weight',
                'attir1.bn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']

        for attir_id in range(2, self.convX + 1):
            stage_name = 'attir%d' % attir_id  # attir_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_pytorch_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, attir_id)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.ATTIRS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'attir%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'attir%d' % (i + 1))(x)
        return x


class Attir_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, block_counts):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.block_counts = block_counts

        dim_bottleneck = cfg.ATTIRS.NUM_GROUPS * cfg.ATTIRS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.attir5, self.dim_out = add_stage(dim_in, cfg.ATTIRS.WIDTH_OUTPLANE * 8, dim_bottleneck * 8, block_counts[3],
                                              dilation=1, stride_init=stride_init)
        assert self.dim_out == cfg.ATTIRS.WIDTH_OUTPLANE * 8
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
            residual_stage_detectron_mapping(self.attir5, 'attir5', self.block_counts, 5)
        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch, orphan_in_pytorch = \
            residual_stage_pytorch_mapping(self.attir5, 'attir5', self.block_counts, 5)
        return mapping_to_pytorch, orphan_in_pytorch

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        attir5_feat = self.attir5(x)
        x = self.avgpool(attir5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, attir5_feat
        else:
            return x


def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(add_residual_block(
            inplanes, outplanes, innerplanes, dilation, stride
        ))
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*res_blocks), outplanes


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride):
    """Return a residual block module, including residual connection, """
    if stride != 1 or inplanes != outplanes:
        shortcut_func = globals()[cfg.ATTIRS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.ATTIRS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride, dilation=dilation, group=cfg.ATTIRS.NUM_GROUPS, downsample=downsample
    )

    return res_block


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)),
        ('bn1', mynn.AffineChannel2d(32)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)),
        ('bn2', mynn.AffineChannel2d(32)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)),
        ('bn3', mynn.AffineChannel2d(64)),
        ('relu3', nn.ReLU(inplace=True)),
        # ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))]))
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))])
    )


def basic_gn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)),
        ('gn1', nn.GroupNorm(net_utils.get_group_gn(32), 32, eps=cfg.GROUP_NORM.EPSILON)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)),
        ('gn2', nn.GroupNorm(net_utils.get_group_gn(32), 32, eps=cfg.GROUP_NORM.EPSILON)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)),
        ('gn3', nn.GroupNorm(net_utils.get_group_gn(64), 64, eps=cfg.GROUP_NORM.EPSILON)),
        ('relu3', nn.ReLU(inplace=True)),
        # ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))]))
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))])
    )


# ------------------------------------------------------------------------------
# various downsample shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        mynn.AffineChannel2d(outplanes),
    )


def basic_gn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes, eps=cfg.GROUP_NORM.EPSILON)
    )


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, downsample=None):
        super().__init__()

        ratio = cfg.ATTIRS.COMPRESSION_RATIO
        group_2 = max(1, group // ratio)
        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.bn1 = mynn.AffineChannel2d(innerplanes)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=stride,
                               padding=1 * dilation, dilation=dilation, groups=group, bias=False)
        self.bn2 = mynn.AffineChannel2d(innerplanes)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)

        if stride == 1 and innerplanes < 512:  # for C2, C3, C4 stages
            self.conv_att1 = nn.Conv2d(inplanes, innerplanes // ratio, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_att1 = mynn.AffineChannel2d(innerplanes // ratio)
            self.subsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # self.conv_att2 = nn.Conv2d(innerplanes // ratio, innerplanes // ratio, kernel_size=3, stride=2, padding=1,
            #                            bias=False)
            # self.bn_att2 = mynn.AffineChannel2d(innerplanes // ratio)
            self.conv_att3 = nn.Conv2d(innerplanes // ratio, innerplanes // ratio, kernel_size=3, stride=1,
                                       padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)
            self.bn_att3 = mynn.AffineChannel2d(innerplanes // ratio)
            self.conv_att4 = nn.Conv2d(innerplanes // ratio, innerplanes, kernel_size=1, stride=1, padding=0,
                                       bias=False)
            self.bn_att4 = mynn.AffineChannel2d(innerplanes)
            self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.innerplanes = innerplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.stride == 1 and self.innerplanes < 512:
            att = self.conv_att1(x)
            att = self.bn_att1(att)
            att = self.relu(att)
            # att = self.conv_att2(att)
            # att = self.bn_att2(att)
            # att = self.relu(att)
            att = self.subsample(att)
            att = self.conv_att3(att)
            att = self.bn_att3(att)
            att = self.relu(att)
            att = F.upsample(att, size=out.size()[2:], mode='bilinear')
            att = self.conv_att4(att)
            att = self.bn_att4(att)
            att = self.sigmoid(att)

            out = out * att

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class bottleneck_gn_transformation(nn.Module):
    """ Bottleneck Residual Block With GN"""

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, downsample=None):
        super().__init__()

        ratio = cfg.ATTIRS.COMPRESSION_RATIO
        group_2 = max(1, group // ratio)
        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=stride,
                               padding=1 * dilation, dilation=dilation, groups=group, bias=False)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes, eps=cfg.GROUP_NORM.EPSILON)

        if stride == 1 and innerplanes < 512:  # for C2, C3, C4 stages
            self.conv_att1 = nn.Conv2d(inplanes, innerplanes // ratio, kernel_size=1, stride=1, padding=0, bias=False)
            self.gn_att1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes // ratio), innerplanes // ratio,
                                        eps=cfg.GROUP_NORM.EPSILON)
            self.subsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # self.conv_att2 = nn.Conv2d(innerplanes // ratio, innerplanes // ratio, kernel_size=3, stride=2, padding=1,
            #                            bias=False)
            # self.gn_att2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes // ratio), innerplanes // ratio,
            #                             eps=cfg.GROUP_NORM.EPSILON)
            self.conv_att3 = nn.Conv2d(innerplanes // ratio, innerplanes // ratio, kernel_size=3, stride=1,
                                       padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)
            self.gn_att3 = nn.GroupNorm(net_utils.get_group_gn(innerplanes // ratio), innerplanes // ratio,
                                        eps=cfg.GROUP_NORM.EPSILON)
            self.conv_att4 = nn.Conv2d(innerplanes // ratio, innerplanes, kernel_size=1, stride=1, padding=0,
                                       bias=False)
            self.gn_att4 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
            self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.innerplanes = innerplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        if self.stride == 1 and self.innerplanes < 512:
            att = self.conv_att1(x)
            att = self.gn_att1(att)
            att = self.relu(att)
            # att = self.conv_att2(att)
            # att = self.gn_att2(att)
            # att = self.relu(att)
            att = self.subsample(att)
            att = self.conv_att3(att)
            att = self.gn_att3(att)
            att = self.relu(att)
            att = F.upsample(att, size=out.size()[2:], mode='bilinear')
            att = self.conv_att4(att)
            att = self.gn_att4(att)
            att = self.sigmoid(att)

            out = out * att

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def residual_stage_detectron_mapping(module_ref, module_name, block_counts, attir_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `attir_id`
    """

    num_blocks = block_counts[attir_id - 2]
    if cfg.ATTIRS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'attir%d' % (sum(block_counts[:attir_id - 2]) + blk_id + 1)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: attir2, attir3, attir4, attir5

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_match_conv'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'

        # conv branch
        for i in range(1, 4):
            dtt_bp = detectron_prefix + '_conv' + str(i)
            mapping_to_detectron[my_prefix + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'

        # attir module
        attir_list = [1, 3, 4]
        for i in attir_list:
            dtt_bp = detectron_prefix + '_att_conv' + str(i)
            mapping_to_detectron[my_prefix + '.conv_att%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '_att%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '_att%d.bias' % i] = dtt_bp + norm_suffix + '_b'

    return mapping_to_detectron, orphan_in_detectron


def residual_stage_pytorch_mapping(module_ref, module_name, block_counts, attir_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `attir_id`
    """

    num_blocks = block_counts[attir_id - 2]
    if cfg.ATTIRS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_pytorch = {}
    orphan_in_pytorch = []
    for blk_id in range(num_blocks):
        pytorch_prefix = 'layer{}.{}'.format(attir_id - 1, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: attir2, attir3, attir4, attir5

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = pytorch_prefix + '.downsample'  # short for "pytorch_branch_prefix"
            mapping_to_pytorch[my_prefix + '.downsample.0.weight'] = dtt_bp + '.0.weight'
            mapping_to_pytorch[my_prefix + '.downsample.1.weight'] = dtt_bp + '.1.weight'
            mapping_to_pytorch[my_prefix + '.downsample.1.bias'] = dtt_bp + '.1.bias'

        # conv branch
        for i in range(1, 4):
            mapping_to_pytorch[my_prefix + '.conv%d.weight' % i] = pytorch_prefix + '.conv%d.weight' % i
            mapping_to_pytorch[
                my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = pytorch_prefix + '.bn%d.weight' % i
            mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = pytorch_prefix + '.bn%d.bias' % i

        # attir module
        attir_list = [1, 3, 4]
        for i in attir_list:
            mapping_to_pytorch[my_prefix + '.conv_att%d.weight' % i] = pytorch_prefix + '.conv_att%d.weight' % i
            mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '_att%d.weight' % i] = \
                pytorch_prefix + '.bn_att%d.weight' % i
            mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '_att%d.bias' % i] = \
                pytorch_prefix + '.bn_att%d.bias' % i

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
