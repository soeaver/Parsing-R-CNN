import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
from parsingrcnn.model.dcn.deform_conv import DeformConv2d
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils
import parsingrcnn.modeling.nonlocal_helper as nonlocal_helper


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (Air50, Air101, ...)
# ---------------------------------------------------------------------------- #

def Air14_conv4_body():
    return Air_convX_body((1, 1, 1))


def Air14_conv5_body():
    return Air_convX_body((1, 1, 1, 1))


def Air26_conv4_body():
    return Air_convX_body((2, 2, 2))


def Air26_conv5_body():
    return Air_convX_body((2, 2, 2, 2))


def Air50_conv4_body():
    return Air_convX_body((3, 4, 6))


def Air50_conv5_body():
    return Air_convX_body((3, 4, 6, 3))


def Air101_conv4_body():
    return Air_convX_body((3, 4, 23))


def Air101_conv5_body():
    return Air_convX_body((3, 4, 23, 3))


def Air152_conv5_body():
    return Air_convX_body((3, 8, 36, 3))


def Air14_roi_conv5_head(dim_in, roi_xform_func, spatial_scale):
    return Air_roi_conv5_head(dim_in, roi_xform_func, spatial_scale, (1, 1, 1, 1))


def Air26_roi_conv5_head(dim_in, roi_xform_func, spatial_scale):
    return Air_roi_conv5_head(dim_in, roi_xform_func, spatial_scale, (2, 2, 2, 2))


def Air50_roi_conv5_head(dim_in, roi_xform_func, spatial_scale):
    return Air_roi_conv5_head(dim_in, roi_xform_func, spatial_scale, (3, 4, 6, 3))


def Air101_roi_conv5_head(dim_in, roi_xform_func, spatial_scale):
    return Air_roi_conv5_head(dim_in, roi_xform_func, spatial_scale, (3, 4, 23, 3))


# ---------------------------------------------------------------------------- #
# Generic Air components
# ---------------------------------------------------------------------------- #


class Air_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts  # 3 or 4
        self.convX = len(block_counts) + 1  # 4 or 5
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2  #

        self.air1 = globals()[cfg.AIRS.STEM_FUNC]()
        dim_in = 64
        # cfg.AIRS.NUM_GROUPS: 1     cfg.AIRS.WIDTH_PER_GROUP: 64
        dim_bottleneck = cfg.AIRS.NUM_GROUPS * cfg.AIRS.WIDTH_PER_GROUP
        self.air2, dim_in = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE, dim_bottleneck, block_counts[0],
                                      dilation=1, stride_init=1)
        self.air3, dim_in = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE * 2, dim_bottleneck * 2, block_counts[1],
                                      dilation=1, stride_init=2, use_deform_stage=(True and cfg.FPN.FPN_ON))
        self.air4, dim_in = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4, block_counts[2],
                                      dilation=1, stride_init=2, use_nonlocal_stage=True, 
                                      use_deform_stage=(True and cfg.FPN.FPN_ON))
        if len(block_counts) == 4:
            if cfg.AIRS.C5_DILATION != 1:
                stride = 1
            else:
                stride = 2
            self.air5, dim_in = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE * 8, dim_bottleneck * 8, block_counts[3],
                                          dilation=cfg.AIRS.C5_DILATION, stride_init=stride,
                                          use_deform_stage=True, all_deform=True)
            self.spatial_scale = 1 / 32 * cfg.AIRS.C5_DILATION  # cfg.AIRS.C5_DILATION: 1
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.AIRS.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.AIRS.FREEZE_AT: 2
        assert cfg.AIRS.FREEZE_AT <= self.convX
        for i in range(1, cfg.AIRS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'air%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.AIRS.USE_GN:
            mapping_to_detectron = {
                'air1.conv1.weight': 'conv1_w',
                'air1.gn1.weight': 'conv1_gn_s',
                'air1.gn1.bias': 'conv1_gn_b',
                'air1.conv2.weight': 'conv2_w',
                'air1.gn2.weight': 'conv2_gn_s',
                'air1.gn2.bias': 'conv2_gn_b',
                'air1.conv3.weight': 'conv3_w',
                'air1.gn3.weight': 'conv3_gn_s',
                'air1.gn3.bias': 'conv3_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'air1.conv1.weight': 'conv1_w',
                'air1.bn1.weight': 'conv1_bn_s',
                'air1.bn1.bias': 'conv1_bn_b',
                'air1.conv2.weight': 'conv2_w',
                'air1.bn2.weight': 'conv2_bn_s',
                'air1.bn2.bias': 'conv2_bn_b',
                'air1.conv3.weight': 'conv3_w',
                'air1.bn3.weight': 'conv3_bn_s',
                'air1.bn3.bias': 'conv3_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'conv2_b', 'conv3_b', 'fc_w', 'fc_b']

        for air_id in range(2, self.convX + 1):
            stage_name = 'air%d' % air_id  # air_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, air_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        if cfg.AIRS.USE_GN:
            mapping_to_pytorch = {
                'air1.conv1.weight': 'conv1.weight',
                'air1.gn1.weight': 'bn1.weight',
                'air1.gn1.bias': 'bn1.bias',
                'air1.conv2.weight': 'conv2.weight',
                'air1.gn2.weight': 'bn2.weight',
                'air1.gn2.bias': 'bn2.bias',
                'air1.conv3.weight': 'conv3.weight',
                'air1.gn3.weight': 'bn3.weight',
                'air1.gn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']
        else:
            mapping_to_pytorch = {
                'air1.conv1.weight': 'conv1.weight',
                'air1.bn1.weight': 'bn1.weight',
                'air1.bn1.bias': 'bn1.bias',
                'air1.conv2.weight': 'conv2.weight',
                'air1.bn2.weight': 'bn2.weight',
                'air1.bn2.bias': 'bn2.bias',
                'air1.conv3.weight': 'conv3.weight',
                'air1.bn3.weight': 'bn3.weight',
                'air1.bn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']

        for air_id in range(2, self.convX + 1):
            stage_name = 'air%d' % air_id  # air_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_pytorch_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, air_id)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.AIRS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'air%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'air%d' % (i + 1))(x)
        return x


class Air_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, block_counts):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.block_counts = block_counts

        dim_bottleneck = cfg.AIRS.NUM_GROUPS * cfg.AIRS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.air5, self.dim_out = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE * 8, dim_bottleneck * 8, block_counts[3],
                                            dilation=1, stride_init=stride_init)
        assert self.dim_out == cfg.AIRS.WIDTH_OUTPLANE * 8
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
            residual_stage_detectron_mapping(self.air5, 'air5', self.block_counts, 5)
        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch, orphan_in_pytorch = \
            residual_stage_pytorch_mapping(self.air5, 'air5', self.block_counts, 5)
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
        air5_feat = self.air5(x)
        x = self.avgpool(air5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, air5_feat
        else:
            return x


def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2, use_nonlocal_stage=False,
              use_deform_stage=False, all_deform=False):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        if cfg.AIRS.USE_NONLOCAL and use_nonlocal_stage and _ == nblocks - 2:
            use_nonlocal = True
        else:
            use_nonlocal = False
            
        if cfg.AIRS.USE_DEFORM and use_deform_stage and (_ == nblocks - 1 or all_deform):
            use_deform = True
        else:
            use_deform = False
            
        res_blocks.append(add_residual_block(
            inplanes, outplanes, innerplanes, dilation, stride, use_nonlocal=use_nonlocal, use_deform=use_deform
        ))
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*res_blocks), outplanes


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride, use_nonlocal=False, use_deform=False):
    """Return a residual block module, including residual connection, """
    if stride != 1 or inplanes != outplanes:
        shortcut_func = globals()[cfg.AIRS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.AIRS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride, dilation=dilation, group=cfg.AIRS.NUM_GROUPS, 
        downsample=downsample, use_nonlocal=use_nonlocal, use_deform=use_deform
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

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, 
                 downsample=None, use_nonlocal=False, use_deform=False):
        super().__init__()
        self.stride = stride
        self.use_nonlocal = use_nonlocal
        self.use_deform = use_deform

        group_2 = max(1, group // 2)
        self.conv1_1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.bn1_1 = mynn.AffineChannel2d(innerplanes)
        if self.use_deform:
            self.conv1_2_offset = nn.Conv2d(
                innerplanes, 72, kernel_size=3, stride=stride, padding=1, bias=True)
            self.conv1_2 = DeformConv2d(
                innerplanes, innerplanes, kernel_size=3, stride=stride,
                padding=1 * dilation, dilation=dilation, num_deformable_groups=4)
        else:
            self.conv1_2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=stride,
                                     padding=1 * dilation, dilation=dilation, groups=group, bias=False)

        self.conv2_1 = nn.Conv2d(inplanes, innerplanes // 2, kernel_size=1, stride=1, bias=False)
        self.bn2_1 = mynn.AffineChannel2d(innerplanes // 2)
        if self.use_deform:
            self.conv2_2_offset = nn.Conv2d(
                innerplanes // 2, 72, kernel_size=3, stride=stride, padding=1, bias=True)
            self.conv2_2 = DeformConv2d(
                innerplanes // 2, innerplanes // 2, kernel_size=3, stride=stride,
                padding=1 * dilation, dilation=dilation, num_deformable_groups=4)
        else:
            self.conv2_2 = nn.Conv2d(innerplanes // 2, innerplanes // 2, kernel_size=3, stride=stride,
                                     padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)
        self.bn2_2 = mynn.AffineChannel2d(innerplanes // 2)
        if self.use_deform:
            self.conv2_3_offset = nn.Conv2d(
                innerplanes // 2, 72, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv2_3 = DeformConv2d(
                innerplanes // 2, innerplanes // 2, kernel_size=3, stride=1,
                padding=1 * dilation, dilation=dilation, num_deformable_groups=4)
        else:
            self.conv2_3 = nn.Conv2d(innerplanes // 2, innerplanes // 2, kernel_size=3, stride=1,
                                     padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)

        self.cat_bn = mynn.AffineChannel2d(innerplanes + (innerplanes // 2))
        self.conv3 = nn.Conv2d(innerplanes + (innerplanes // 2), outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        if self.use_nonlocal:
            self.non_local = nonlocal_helper.SpaceNonLocal(outplanes, outplanes // 2, outplanes)

    def forward(self, x):
        residual = x

        branch1 = self.conv1_1(x)
        branch1 = self.bn1_1(branch1)
        branch1 = self.relu(branch1)
        if self.use_deform:
            offset1 = self.conv1_2_offset(branch1)
            branch1 = self.conv1_2(branch1, offset1)
        else:
            branch1 = self.conv1_2(branch1)

        branch2 = self.conv2_1(x)
        branch2 = self.bn2_1(branch2)
        branch2 = self.relu(branch2)
        if self.use_deform:
            offset2 = self.conv2_2_offset(branch2)
            branch2 = self.conv2_2(branch2, offset2)
        else:
            branch2 = self.conv2_2(branch2)
        branch2 = self.bn2_2(branch2)
        branch2 = self.relu(branch2)
        if self.use_deform:
            offset3 = self.conv2_3_offset(branch2)
            branch2 = self.conv2_3(branch2, offset3)
        else:
            branch2 = self.conv2_3(branch2)

        out = torch.cat((branch1, branch2), 1)
        out = self.cat_bn(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        if self.use_nonlocal:
            out = self.non_local(out)

        return out


class bottleneck_gn_transformation(nn.Module):
    """ Bottleneck Residual Block With GN"""

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, downsample=None):
        super().__init__()

        group_2 = max(1, group // 2)
        self.conv1_1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.gn1_1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv1_2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=stride,
                                 padding=1 * dilation, dilation=dilation, groups=group, bias=False)

        self.conv2_1 = nn.Conv2d(inplanes, innerplanes // 2, kernel_size=1, stride=1, bias=False)
        self.gn2_1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes // 2), innerplanes // 2,
                                  eps=cfg.GROUP_NORM.EPSILON)
        self.conv2_2 = nn.Conv2d(innerplanes // 2, innerplanes // 2, kernel_size=3, stride=stride,
                                 padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)
        self.gn2_2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes // 2), innerplanes // 2,
                                  eps=cfg.GROUP_NORM.EPSILON)
        self.conv2_3 = nn.Conv2d(innerplanes // 2, innerplanes // 2, kernel_size=3, stride=1,
                                 padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)

        self.cat_gn = nn.GroupNorm(net_utils.get_group_gn(innerplanes + (innerplanes // 2)),
                                   innerplanes + (innerplanes // 2), eps=cfg.GROUP_NORM.EPSILON)
        self.conv3 = nn.Conv2d(innerplanes + (innerplanes // 2), outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        branch1 = self.conv1_1(x)
        branch1 = self.gn1_1(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.conv1_2(branch1)

        branch2 = self.conv2_1(x)
        branch2 = self.gn2_1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_2(branch2)
        branch2 = self.gn2_2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_3(branch2)

        out = torch.cat((branch1, branch2), 1)
        out = self.cat_gn(out)
        out = self.relu(out)

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

def residual_stage_detectron_mapping(module_ref, module_name, block_counts, air_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `air_id`
    """

    num_blocks = block_counts[air_id - 2]
    if cfg.AIRS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'air%d' % (sum(block_counts[:air_id - 2]) + blk_id + 1)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: air2, air3, air4, air5

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_match_conv'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'

        # conv branch1
        for i in range(1, 3):
            dtt_bp = detectron_prefix + '_conv1_' + str(i)
            mapping_to_detectron[my_prefix + '.conv1_%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            if i < 2:
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '1_%d.weight' % i] = dtt_bp + \
                                                                                              norm_suffix + '_s'
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '1_%d.bias' % i] = dtt_bp + norm_suffix + '_b'

        # conv branch2
        for i in range(1, 4):
            dtt_bp = detectron_prefix + '_conv2_' + str(i)
            mapping_to_detectron[my_prefix + '.conv2_%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            if i < 3:
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '2_%d.weight' % i] = dtt_bp + \
                                                                                              norm_suffix + '_s'
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '2_%d.bias' % i] = dtt_bp + norm_suffix + '_b'

        # cat_bn
        mapping_to_detectron[my_prefix + '.cat_' + norm_suffix[1:] + '.weight'] = detectron_prefix + '_cat' + \
                                                                                  norm_suffix + '_s'
        mapping_to_detectron[my_prefix + '.cat_' + norm_suffix[1:] + '.bias'] = detectron_prefix + '_cat' + \
                                                                                norm_suffix + '_b'

        # conv3
        mapping_to_detectron[my_prefix + '.conv3.weight'] = detectron_prefix + '_conv3' + '_w'
        orphan_in_detectron.append(detectron_prefix + '_conv3' + '_b')
        mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '3.weight'] = detectron_prefix + '_conv3' + \
                                                                               norm_suffix + '_s'
        mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '3.bias'] = detectron_prefix + '_conv3' + \
                                                                             norm_suffix + '_b'
        # nonlocal weight mapping
        try:
            if getattr(module_ref[blk_id], 'non_local'):
                mapping_to_detectron.update({
                    '{}.non_local.theta.weight'.format(my_prefix): '{}_theta_w'.format(detectron_prefix),
                    '{}.non_local.theta.bias'.format(my_prefix): '{}_theta_b'.format(detectron_prefix),
                    '{}.non_local.phi.weight'.format(my_prefix): '{}_phi_w'.format(detectron_prefix),
                    '{}.non_local.phi.bias'.format(my_prefix): '{}_phi_b'.format(detectron_prefix),
                    '{}.non_local.g.weight'.format(my_prefix): '{}_g_w'.format(detectron_prefix),
                    '{}.non_local.g.bias'.format(my_prefix): '{}_g_b'.format(detectron_prefix),
                    '{}.non_local.out.weight'.format(my_prefix): '{}_out_w'.format(detectron_prefix),
                    '{}.non_local.out.bias'.format(my_prefix): '{}_out_b'.format(detectron_prefix)
                })
                if cfg.NONLOCAL.USE_BN:
                    mapping_to_detectron.update({
                        '{}.non_local.bn.weight'.format(my_prefix): '{}_bn_s'.format(detectron_prefix),
                        '{}.non_local.bn.bias'.format(my_prefix): '{}_bn_b'.format(detectron_prefix),
                        '{}.non_local.bn.running_mean'.format(my_prefix): '{}_bn_running_mean'.format(detectron_prefix),
                        '{}.non_local.bn.running_var'.format(my_prefix): '{}_bn_running_var'.format(detectron_prefix)
                    })
                if cfg.NONLOCAL.USE_AFFINE:
                    mapping_to_detectron.update({
                        '{}.non_local.affine.weight'.format(my_prefix): '{}_bn_s'.format(detectron_prefix),
                        '{}.non_local.affine.bias'.format(my_prefix): '{}_bn_b'.format(detectron_prefix)
                    })
        except:
            pass
        # deform conv weight mapping
        try:
            if getattr(module_ref[blk_id], 'conv1_2_offset'):
                mapping_to_detectron.update({
                    '{}.conv1_2_offset.weight'.format(my_prefix): '{}_conv1_2_offset_w'.format(detectron_prefix),
                    '{}.conv1_2_offset.bias'.format(my_prefix): '{}_conv1_2_offset_b'.format(detectron_prefix),
                    '{}.conv2_2_offset.weight'.format(my_prefix): '{}_conv2_2_offset_w'.format(detectron_prefix),
                    '{}.conv2_2_offset.bias'.format(my_prefix): '{}_conv2_2_offset_b'.format(detectron_prefix),
                    '{}.conv2_3_offset.weight'.format(my_prefix): '{}_conv2_3_offset_w'.format(detectron_prefix),
                    '{}.conv2_3_offset.bias'.format(my_prefix): '{}_conv2_3_offset_b'.format(detectron_prefix),
                })
        except:
            pass

    return mapping_to_detectron, orphan_in_detectron


def residual_stage_pytorch_mapping(module_ref, module_name, block_counts, air_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `air_id`
    """

    num_blocks = block_counts[air_id - 2]
    if cfg.AIRS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_pytorch = {}
    orphan_in_pytorch = []
    for blk_id in range(num_blocks):
        pytorch_prefix = 'layer{}.{}'.format(air_id - 1, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: air2, air3, air4, air5

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = pytorch_prefix + '.downsample'  # short for "pytorch_branch_prefix"
            mapping_to_pytorch[my_prefix + '.downsample.0.weight'] = dtt_bp + '.0.weight'
            mapping_to_pytorch[my_prefix + '.downsample.1.weight'] = dtt_bp + '.1.weight'
            mapping_to_pytorch[my_prefix + '.downsample.1.bias'] = dtt_bp + '.1.bias'

        # conv branch1
        for i in range(1, 3):
            dtt_bp = pytorch_prefix
            mapping_to_pytorch[my_prefix + '.conv1_%d.weight' % i] = dtt_bp + '.conv1_%d.weight' % i
            if i < 2:
                mapping_to_pytorch[
                    my_prefix + '.' + norm_suffix[1:] + '1_%d.weight' % i] = dtt_bp + '.bn1_%d.weight' % i
                mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '1_%d.bias' % i] = dtt_bp + '.bn1_%d.bias' % i

        # conv branch2
        for i in range(1, 4):
            dtt_bp = pytorch_prefix
            mapping_to_pytorch[my_prefix + '.conv2_%d.weight' % i] = dtt_bp + '.conv2_%d.weight' % i
            if i < 3:
                mapping_to_pytorch[
                    my_prefix + '.' + norm_suffix[1:] + '2_%d.weight' % i] = dtt_bp + '.bn2_%d.weight' % i
                mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '2_%d.bias' % i] = dtt_bp + '.bn2_%d.bias' % i

        # cat_bn
        mapping_to_pytorch[my_prefix + '.cat_' + norm_suffix[1:] + '.weight'] = pytorch_prefix + '.bn_concat.weight'
        mapping_to_pytorch[my_prefix + '.cat_' + norm_suffix[1:] + '.bias'] = pytorch_prefix + '.bn_concat.bias'

        # conv3
        mapping_to_pytorch[my_prefix + '.conv3.weight'] = pytorch_prefix + '.conv.weight'
        mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '3.weight'] = pytorch_prefix + '.bn.weight'
        mapping_to_pytorch[my_prefix + '.' + norm_suffix[1:] + '3.bias'] = pytorch_prefix + '.bn.bias'
        # nonlocal weight mapping
        try:
            if getattr(module_ref[blk_id], 'non_local'):
                mapping_to_pytorch.update({
                    '{}.non_local.theta.weight'.format(my_prefix): '{}.non_local.theta.weight'.format(pytorch_prefix),
                    '{}.non_local.theta.bias'.format(my_prefix): '{}.non_local.theta.bias'.format(pytorch_prefix),
                    '{}.non_local.phi.weight'.format(my_prefix): '{}.non_local.phi.weight'.format(pytorch_prefix),
                    '{}.non_local.phi.bias'.format(my_prefix): '{}.non_local.phi.bias'.format(pytorch_prefix),
                    '{}.non_local.g.weight'.format(my_prefix): '{}.non_local.g.weight'.format(pytorch_prefix),
                    '{}.non_local.g.bias'.format(my_prefix): '{}.non_local.g.bias'.format(pytorch_prefix),
                    '{}.non_local.out.weight'.format(my_prefix): '{}.non_local.out.weight'.format(pytorch_prefix),
                    '{}.non_local.out.bias'.format(my_prefix): '{}.non_local.out.bias'.format(pytorch_prefix),
                })
                if cfg.NONLOCAL.USE_BN:
                    mapping_to_pytorch.update({
                        '{}.non_local.bn.weight'.format(my_prefix): '{}.non_local.bn.weight'.format(pytorch_prefix),
                        '{}.non_local.bn.bias'.format(my_prefix): '{}.non_local.bn.bias'.format(pytorch_prefix),
                        '{}.non_local.bn.running_mean'.format(my_prefix):
                            '{}.non_local.bn.running_mean'.format(pytorch_prefix),
                        '{}.non_local.bn.running_var'.format(my_prefix):
                            '{}.non_local.bn.running_var'.format(pytorch_prefix)
                    })
                if cfg.NONLOCAL.USE_AFFINE:
                    mapping_to_pytorch.update({
                        '{}.non_local.affine.weight'.format(my_prefix): '{}.non_local.bn.weight'.format(pytorch_prefix),
                        '{}.non_local.affine.bias'.format(my_prefix): '{}.non_local.bn.bias'.format(pytorch_prefix)
                    })
        except:
            pass
        # deform conv weight mapping
        try:
            if getattr(module_ref[blk_id], 'conv1_2_offset'):
                mapping_to_detectron.update({
                    '{}.conv1_2_offset.weight'.format(my_prefix): '{}.conv1_2_offset.weight'.format(pytorch_prefix),
                    '{}.conv1_2_offset.bias'.format(my_prefix): '{}.conv1_2_offset.bias'.format(pytorch_prefix),
                    '{}.conv2_2_offset.weight'.format(my_prefix): '{}.conv2_2_offset.weight'.format(pytorch_prefix),
                    '{}.conv2_2_offset.bias'.format(my_prefix): '{}.conv2_2_offset.bias'.format(pytorch_prefix),
                    '{}.conv2_3_offset.weight'.format(my_prefix): '{}.conv2_3_offset.weight'.format(pytorch_prefix),
                    '{}.conv2_3_offset.bias'.format(my_prefix): '{}.conv2_3_offset.bias'.format(pytorch_prefix),
                })
        except:
            pass
        
    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
