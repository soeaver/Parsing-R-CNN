import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (DetAir32, DetAir59, DetAir110, DetAir161, ...)
# ---------------------------------------------------------------------------- #

def DetAir32_conv5_body():
    return DetAir_convX_body((2, 2, 2, 2))


def DetAir32_conv6_body():
    return DetAir_convX_body((2, 2, 2, 2, 2))


def DetAir59_conv5_body():
    return DetAir_convX_body((3, 4, 6, 3))


def DetAir59_conv6_body():
    return DetAir_convX_body((3, 4, 6, 3, 3))


def DetAir110_conv5_body():
    return DetAir_convX_body((3, 4, 23, 3))


def DetAir110_conv6_body():
    return DetAir_convX_body((3, 4, 23, 3, 3))


def DetAir161_conv5_body():
    return DetAir_convX_body((3, 8, 36, 3))


def DetAir161_conv6_body():
    return DetAir_convX_body((3, 8, 36, 3, 3))


def DetAir32_roi_conv6_head(dim_in, roi_xform_func, spatial_scale):
    return DetAir_roi_conv6_head(dim_in, roi_xform_func, spatial_scale, (2, 2, 2, 2))


def DetAir59_roi_conv6_head(dim_in, roi_xform_func, spatial_scale):
    return DetAir_roi_conv6_head(dim_in, roi_xform_func, spatial_scale, (3, 4, 6, 3))


def DetAir110_roi_conv6_head(dim_in, roi_xform_func, spatial_scale):
    return DetAir_roi_conv6_head(dim_in, roi_xform_func, spatial_scale, (3, 4, 23, 3))


def DetAir161_roi_conv6_head(dim_in, roi_xform_func, spatial_scale):
    return DetAir_roi_conv6_head(dim_in, roi_xform_func, spatial_scale, (3, 8, 36, 3))


# ---------------------------------------------------------------------------- #
# Generic DetAir components
# ---------------------------------------------------------------------------- #


class DetAir_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts  # 4 or 5
        self.convX = len(block_counts) + 1  # 5 or 6
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2  #

        self.detair1 = globals()[cfg.DETAIRS.STEM_FUNC]()
        dim_in = 64
        # cfg.DETAIRS.NUM_GROUPS: 1     cfg.DETAIRS.WIDTH_PER_GROUP: 64
        dim_bottleneck = cfg.DETAIRS.NUM_GROUPS * cfg.DETAIRS.WIDTH_PER_GROUP
        self.detair2, dim_in = add_stage(dim_in, cfg.DETAIRS.WIDTH_OUTPLANE, dim_bottleneck, block_counts[0],
                                         dilation=1, stride_init=1)
        self.detair3, dim_in = add_stage(dim_in, cfg.DETAIRS.WIDTH_OUTPLANE * 2, dim_bottleneck * 2, block_counts[1],
                                         dilation=1, stride_init=2)
        self.detair4, dim_in = add_stage(dim_in, cfg.DETAIRS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4, block_counts[2],
                                         dilation=1, stride_init=2)
        self.detair5, dim_in = add_stage(dim_in, cfg.DETAIRS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4, block_counts[3],
                                         dilation=2, stride_init=1)
        if len(block_counts) == 5:
            self.detair6, dim_in = add_stage(dim_in, cfg.DETAIRS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4,
                                             block_counts[4], dilation=2, stride_init=1)
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.DETAIRS.FREEZE_AT in [0, 2, 3, 4, 5, 6]  # cfg.DETAIRS.FREEZE_AT: 2
        assert cfg.DETAIRS.FREEZE_AT <= self.convX
        for i in range(1, cfg.DETAIRS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'detair%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.DETAIRS.USE_GN:
            mapping_to_detectron = {
                'detair1.conv1.weight': 'conv1_w',
                'detair1.gn1.weight': 'conv1_gn_s',
                'detair1.gn1.bias': 'conv1_gn_b',
                'detair1.conv2.weight': 'conv2_w',
                'detair1.gn2.weight': 'conv2_gn_s',
                'detair1.gn2.bias': 'conv2_gn_b',
                'detair1.conv3.weight': 'conv3_w',
                'detair1.gn3.weight': 'conv3_gn_s',
                'detair1.gn3.bias': 'conv3_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        elif cfg.DETAIRS.USE_SN:
            mapping_to_detectron = {
                'detair1.conv1.weight': 'conv1_w',
                'detair1.sn1.weight': 'conv1_sn_s',
                'detair1.sn1.bias': 'conv1_sn_b',
                'detair1.sn1.mean_weight': 'conv1_sn_mean_weight',
                'detair1.sn1.var_weight': 'conv1_sn_var_weight',
                'detair1.conv2.weight': 'conv2_w',
                'detair1.sn2.weight': 'conv2_sn_s',
                'detair1.sn2.bias': 'conv2_sn_b',
                'detair1.sn2.mean_weight': 'conv2_sn_mean_weight',
                'detair1.sn2.var_weight': 'conv2_sn_var_weight',
                'detair1.conv3.weight': 'conv3_w',
                'detair1.sn3.weight': 'conv3_sn_s',
                'detair1.sn3.bias': 'conv3_sn_b',
                'detair1.sn3.mean_weight': 'conv3_sn_mean_weight',
                'detair1.sn3.var_weight': 'conv3_sn_var_weight',
            }
            if cfg.DETAIRS.SN.USE_BN:
                mapping_to_detectron.update({
                    'detair1.sn1.running_mean': 'conv1_sn_rm',
                    'detair1.sn1.running_var': 'conv1_sn_riv',
                    'detair1.sn2.running_mean': 'conv2_sn_rm',
                    'detair1.sn2.running_var': 'conv2_sn_riv',
                    'detair1.sn3.running_mean': 'conv3_sn_rm',
                    'detair1.sn3.running_var': 'conv3_sn_riv',
                })
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'detair1.conv1.weight': 'conv1_w',
                'detair1.bn1.weight': 'conv1_bn_s',
                'detair1.bn1.bias': 'conv1_bn_b',
                'detair1.conv2.weight': 'conv2_w',
                'detair1.bn2.weight': 'conv2_bn_s',
                'detair1.bn2.bias': 'conv2_bn_b',
                'detair1.conv3.weight': 'conv3_w',
                'detair1.bn3.weight': 'conv3_bn_s',
                'detair1.bn3.bias': 'conv3_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'conv2_b', 'conv3_b', 'fc_w', 'fc_b']

        for detair_id in range(2, self.convX + 1):
            stage_name = 'detair%d' % detair_id  # detair_id = 2, 3, 4, 5, 6
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, detair_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        if cfg.DETAIRS.USE_GN:
            mapping_to_pytorch = {
                'detair1.conv1.weight': 'conv1.weight',
                'detair1.gn1.weight': 'bn1.weight',
                'detair1.gn1.bias': 'bn1.bias',
                'detair1.conv2.weight': 'conv2.weight',
                'detair1.gn2.weight': 'bn2.weight',
                'detair1.gn2.bias': 'bn2.bias',
                'detair1.conv3.weight': 'conv3.weight',
                'detair1.gn3.weight': 'bn3.weight',
                'detair1.gn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']
        elif cfg.DETAIRS.USE_SN:
            mapping_to_pytorch = {
                'detair1.conv1.weight': 'conv1.weight',
                'detair1.sn1.weight': 'sn1.weight',
                'detair1.sn1.bias': 'sn1.bias',
                'detair1.sn1.mean_weight': 'sn1.mean_weight',
                'detair1.sn1.var_weight': 'sn1.var_weight',
                'detair1.conv2.weight': 'conv2.weight',
                'detair1.sn2.weight': 'sn2.weight',
                'detair1.sn2.bias': 'sn2.bias',
                'detair1.sn2.mean_weight': 'sn2.mean_weight',
                'detair1.sn2.var_weight': 'sn2.var_weight',
                'detair1.conv3.weight': 'conv3.weight',
                'detair1.sn3.weight': 'sn3.weight',
                'detair1.sn3.bias': 'sn3.bias',
                'detair1.sn3.mean_weight': 'sn3.mean_weight',
                'detair1.sn3.var_weight': 'sn3.var_weight',
            }
            if cfg.DETAIRS.SN.USE_BN:
                mapping_to_pytorch.update({
                    'detair1.sn1.running_mean': 'sn1.running_mean',
                    'detair1.sn1.running_var': 'sn1.running_var',
                    'detair1.sn2.running_mean': 'sn2.running_mean',
                    'detair1.sn2.running_var': 'sn2.running_var',
                    'detair1.sn3.running_mean': 'sn3.running_mean',
                    'detair1.sn3.running_var': 'sn3.running_var',
                })
            orphan_in_pytorch = ['pred_w', 'pred_b']
        else:
            mapping_to_pytorch = {
                'detair1.conv1.weight': 'conv1.weight',
                'detair1.bn1.weight': 'bn1.weight',
                'detair1.bn1.bias': 'bn1.bias',
                'detair1.conv2.weight': 'conv2.weight',
                'detair1.bn2.weight': 'bn2.weight',
                'detair1.bn2.bias': 'bn2.bias',
                'detair1.conv3.weight': 'conv3.weight',
                'detair1.bn3.weight': 'bn3.weight',
                'detair1.bn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']

        for detair_id in range(2, self.convX + 1):
            stage_name = 'detair%d' % detair_id  # detair_id = 2, 3, 4, 5, 6
            mapping, orphans = residual_stage_pytorch_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, detair_id)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.DETAIRS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'detair%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'detair%d' % (i + 1))(x)
        return x


class DetAir_roi_conv6_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, block_counts):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.block_counts = block_counts

        dim_bottleneck = cfg.DETAIRS.NUM_GROUPS * cfg.DETAIRS.WIDTH_PER_GROUP
        # stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.detair6, self.dim_out = add_stage(dim_in, cfg.DETAIRS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4, 3,
                                               dilation=2, stride_init=1)
        assert self.dim_out == cfg.DETAIRS.WIDTH_OUTPLANE * 4
        self.avgpool = nn.AvgPool2d(14)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
            residual_stage_detectron_mapping(self.detair6, 'detair6', self.block_counts, 6)
        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch, orphan_in_pytorch = \
            residual_stage_pytorch_mapping(self.detair6, 'detair6', self.block_counts, 6)
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
        detair6_feat = self.detair6(x)
        x = self.avgpool(detair6_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, detair6_feat
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
    need_downsample = True
    for _ in range(nblocks):
        res_blocks.append(add_residual_block(
            inplanes, outplanes, innerplanes, dilation, stride, need_downsample=need_downsample
        ))
        need_downsample = False
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*res_blocks), outplanes


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride, need_downsample=True):
    """Return a residual block module, including residual connection, """
    if need_downsample and (stride != 1 or inplanes != outplanes or dilation != 1):
        shortcut_func = globals()[cfg.DETAIRS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.DETAIRS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride, dilation=dilation, group=cfg.DETAIRS.NUM_GROUPS, downsample=downsample
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


def basic_sn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)),
        ('sn1', mynn.SwitchNorm(32, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                using_bn=cfg.DETAIRS.SN.USE_BN)),
        ('relu', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)),
        ('sn2', mynn.SwitchNorm(32, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                using_bn=cfg.DETAIRS.SN.USE_BN)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)),
        ('sn3', mynn.SwitchNorm(64, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                using_bn=cfg.DETAIRS.SN.USE_BN)),
        ('relu3', nn.ReLU(inplace=True)),
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


def basic_sn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        mynn.SwitchNorm(outplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                        using_bn=cfg.DETAIRS.SN.USE_BN)
    )


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, downsample=None):
        super().__init__()

        group_2 = max(1, group // 2)
        self.conv1_1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.bn1_1 = mynn.AffineChannel2d(innerplanes)
        self.conv1_2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=stride,
                                 padding=1 * dilation, dilation=dilation, groups=group, bias=False)

        self.conv2_1 = nn.Conv2d(inplanes, innerplanes // 2, kernel_size=1, stride=1, bias=False)
        self.bn2_1 = mynn.AffineChannel2d(innerplanes // 2)
        self.conv2_2 = nn.Conv2d(innerplanes // 2, innerplanes // 2, kernel_size=3, stride=stride,
                                 padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)
        self.bn2_2 = mynn.AffineChannel2d(innerplanes // 2)
        self.conv2_3 = nn.Conv2d(innerplanes // 2, innerplanes // 2, kernel_size=3, stride=1,
                                 padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)

        self.cat_bn = mynn.AffineChannel2d(innerplanes + (innerplanes // 2))
        self.conv3 = nn.Conv2d(innerplanes + (innerplanes // 2), outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        branch1 = self.conv1_1(x)
        branch1 = self.bn1_1(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.conv1_2(branch1)

        branch2 = self.conv2_1(x)
        branch2 = self.bn2_1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_2(branch2)
        branch2 = self.bn2_2(branch2)
        branch2 = self.relu(branch2)
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


class bottleneck_sn_transformation(nn.Module):
    """ Bottleneck Residual Block With GN"""

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, downsample=None):
        super().__init__()

        group_2 = max(1, group // 2)
        self.conv1_1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.sn1_1 = mynn.SwitchNorm(innerplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                     using_bn=cfg.DETAIRS.SN.USE_BN)
        self.conv1_2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=stride,
                                 padding=1 * dilation, dilation=dilation, groups=group, bias=False)

        self.conv2_1 = nn.Conv2d(inplanes, innerplanes // 2, kernel_size=1, stride=1, bias=False)
        self.sn2_1 = mynn.SwitchNorm(innerplanes // 2, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                     using_bn=cfg.DETAIRS.SN.USE_BN)
        self.conv2_2 = nn.Conv2d(innerplanes // 2, innerplanes // 2, kernel_size=3, stride=stride,
                                 padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)
        self.sn2_2 = mynn.SwitchNorm(innerplanes // 2, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                     using_bn=cfg.DETAIRS.SN.USE_BN)
        self.conv2_3 = nn.Conv2d(innerplanes // 2, innerplanes // 2, kernel_size=3, stride=1,
                                 padding=1 * dilation, dilation=dilation, groups=group_2, bias=False)

        self.cat_sn = mynn.SwitchNorm(innerplanes + (innerplanes // 2),
                                      using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                      using_bn=cfg.DETAIRS.SN.USE_BN)
        self.conv3 = nn.Conv2d(innerplanes + (innerplanes // 2), outplanes, kernel_size=1, stride=1, bias=False)
        self.sn3 = mynn.SwitchNorm(outplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                   using_bn=cfg.DETAIRS.SN.USE_BN)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        branch1 = self.conv1_1(x)
        branch1 = self.sn1_1(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.conv1_2(branch1)

        branch2 = self.conv2_1(x)
        branch2 = self.sn2_1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_2(branch2)
        branch2 = self.sn2_2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_3(branch2)

        out = torch.cat((branch1, branch2), 1)
        out = self.cat_sn(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.sn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def residual_stage_detectron_mapping(module_ref, module_name, block_counts, detair_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `detair_id`
    """

    num_blocks = block_counts[detair_id - 2]
    if cfg.DETAIRS.USE_GN:
        norm_suffix = '_gn'
    elif cfg.DETAIRS.USE_SN:
        norm_suffix = '_sn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'detair%d' % (sum(block_counts[:detair_id - 2]) + blk_id + 1)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: detair2, detair3, detair4, detair5, detair6

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_match_conv'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'
            if cfg.DETAIRS.USE_SN:
                mapping_to_detectron[my_prefix
                                     + '.downsample.1.mean_weight'] = dtt_bp + norm_suffix + '_mean_weight'
                mapping_to_detectron[my_prefix
                                     + '.downsample.1.var_weight'] = dtt_bp + norm_suffix + '_var_weight'
                if cfg.DETAIRS.SN.USE_BN:
                    mapping_to_detectron[my_prefix
                                         + '.downsample.1.running_mean'] = dtt_bp + norm_suffix + '_rm'
                    mapping_to_detectron[my_prefix
                                         + '.downsample.1.running_var'] = dtt_bp + norm_suffix + '_riv'

        # conv branch1
        for i in range(1, 3):
            dtt_bp = detectron_prefix + '_conv1_' + str(i)
            mapping_to_detectron[my_prefix + '.conv1_%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            if i < 2:
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '1_%d.weight' % i] = dtt_bp + \
                                                                                              norm_suffix + '_s'
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '1_%d.bias' % i] = dtt_bp + norm_suffix + '_b'
                if cfg.DETAIRS.USE_SN:
                    mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '1_%d.mean_weight' % i] = \
                        dtt_bp + norm_suffix + '_mean_weight'
                    mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '1_%d.var_weight' % i] = \
                        dtt_bp + norm_suffix + '_var_weight'
                    if cfg.DETAIRS.SN.USE_BN:
                        mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '1_%d.running_mean' % i] = \
                            dtt_bp + norm_suffix + '_rm'
                        mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '1_%d.running_var' % i] = \
                            dtt_bp + norm_suffix + '_riv'

        # conv branch2
        for i in range(1, 4):
            dtt_bp = detectron_prefix + '_conv2_' + str(i)
            mapping_to_detectron[my_prefix + '.conv2_%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            if i < 3:
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '2_%d.weight' % i] = dtt_bp + \
                                                                                              norm_suffix + '_s'
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '2_%d.bias' % i] = dtt_bp + norm_suffix + '_b'
                if cfg.DETAIRS.USE_SN:
                    mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '2_%d.mean_weight' % i] = \
                        dtt_bp + norm_suffix + '_mean_weight'
                    mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '2_%d.var_weight' % i] = \
                        dtt_bp + norm_suffix + '_var_weight'
                    if cfg.DETAIRS.SN.USE_BN:
                        mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '2_%d.running_mean' % i] = \
                            dtt_bp + norm_suffix + '_rm'
                        mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '2_%d.running_var' % i] = \
                            dtt_bp + norm_suffix + '_riv'

        # cat_bn
        mapping_to_detectron[my_prefix + '.cat_' + norm_suffix[1:] + '.weight'] = \
            detectron_prefix + '_cat' + norm_suffix + '_s'
        mapping_to_detectron[my_prefix + '.cat_' + norm_suffix[1:] + '.bias'] = \
            detectron_prefix + '_cat' + norm_suffix + '_b'
        if cfg.DETAIRS.USE_SN:
            mapping_to_detectron[my_prefix + '.cat_' + norm_suffix[1:] + '.mean_weight'] = \
                detectron_prefix + '_cat' + norm_suffix + '_mean_weight'
            mapping_to_detectron[my_prefix + '.cat_' + norm_suffix[1:] + '.var_weight'] = \
                detectron_prefix + '_cat' + norm_suffix + '_var_weight'
            if cfg.DETAIRS.SN.USE_BN:
                mapping_to_detectron[my_prefix + '.cat_' + norm_suffix[1:] + '.running_mean'] = \
                    detectron_prefix + '_cat' + norm_suffix + '_rm'
                mapping_to_detectron[my_prefix + '.cat_' + norm_suffix[1:] + '.running_var'] = \
                    detectron_prefix + '_cat' + norm_suffix + '_riv'

        # conv3
        mapping_to_detectron[my_prefix + '.conv3.weight'] = detectron_prefix + '_conv3' + '_w'
        orphan_in_detectron.append(detectron_prefix + '_conv3' + '_b')
        mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '3.weight'] = detectron_prefix + '_conv3' + \
                                                                               norm_suffix + '_s'
        mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '3.bias'] = detectron_prefix + '_conv3' + \
                                                                             norm_suffix + '_b'
        if cfg.DETAIRS.USE_SN:
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '3.mean_weight'] = \
                detectron_prefix + '_conv3' + norm_suffix + '_mean_weight'
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '3.var_weight'] = \
                detectron_prefix + '_conv3' + norm_suffix + '_var_weight'
            if cfg.DETAIRS.SN.USE_BN:
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '3.running_mean'] = \
                    detectron_prefix + '_conv3' + norm_suffix + '_rm'
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '3.running_var'] = \
                    detectron_prefix + '_conv3' + norm_suffix + '_riv'

    return mapping_to_detectron, orphan_in_detectron


def residual_stage_pytorch_mapping(module_ref, module_name, block_counts, detair_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `detair_id`
    """

    num_blocks = block_counts[detair_id - 2]
    if cfg.DETAIRS.USE_GN:
        my_norm_suffix = '_gn'
        py_norm_suffix = '_bn'
    elif cfg.DETAIRS.USE_SN:
        my_norm_suffix = '_sn'
        py_norm_suffix = '_sn'
    else:
        my_norm_suffix = '_bn'
        py_norm_suffix = '_bn'
    mapping_to_pytorch = {}
    orphan_in_pytorch = []
    for blk_id in range(num_blocks):
        pytorch_prefix = 'layer{}.{}'.format(detair_id - 1, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: detair2, detair3, detair4, detair5, detair6

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = pytorch_prefix + '.downsample'  # short for "pytorch_branch_prefix"
            mapping_to_pytorch[my_prefix + '.downsample.0.weight'] = dtt_bp + '.0.weight'
            mapping_to_pytorch[my_prefix + '.downsample.1.weight'] = dtt_bp + '.1.weight'
            mapping_to_pytorch[my_prefix + '.downsample.1.bias'] = dtt_bp + '.1.bias'
            if cfg.DETAIRS.USE_SN:
                mapping_to_pytorch[my_prefix + '.downsample.1.mean_weight'] = dtt_bp + '.1.mean_weight'
                mapping_to_pytorch[my_prefix + '.downsample.1.var_weight'] = dtt_bp + '.1.var_weight'
                if cfg.DETAIRS.SN.USE_BN:
                    mapping_to_pytorch[my_prefix + '.downsample.1.running_mean'] = dtt_bp + '.1.running_mean'
                    mapping_to_pytorch[my_prefix + '.downsample.1.running_var'] = dtt_bp + '.1.running_var'

        # conv branch1
        for i in range(1, 3):
            dtt_bp = pytorch_prefix
            mapping_to_pytorch[my_prefix + '.conv1_%d.weight' % i] = dtt_bp + '.conv1_%d.weight' % i
            if i < 2:
                mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '1_%d.weight' % i] = \
                    dtt_bp + '.' + py_norm_suffix[1:] + '1_%d.weight' % i
                mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '1_%d.bias' % i] = \
                    dtt_bp + '.' + py_norm_suffix[1:] + '1_%d.bias' % i
                if cfg.DETAIRS.USE_SN:
                    mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '1_%d.mean_weight' % i] = \
                        dtt_bp + '.' + py_norm_suffix[1:] + '1_%d.mean_weight' % i
                    mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '1_%d.var_weight' % i] = \
                        dtt_bp + '.' + py_norm_suffix[1:] + '1_%d.var_weight' % i
                    if cfg.DETAIRS.SN.USE_BN:
                        mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '1_%d.running_mean' % i] = \
                            dtt_bp + '.' + py_norm_suffix[1:] + '1_%d.running_mean' % i
                        mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '1_%d.running_var' % i] = \
                            dtt_bp + '.' + py_norm_suffix[1:] + '1_%d.running_var' % i

        # conv branch2
        for i in range(1, 4):
            dtt_bp = pytorch_prefix
            mapping_to_pytorch[my_prefix + '.conv2_%d.weight' % i] = dtt_bp + '.conv2_%d.weight' % i
            if i < 3:
                mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '2_%d.weight' % i] = \
                    dtt_bp + '.' + py_norm_suffix[1:] + '2_%d.weight' % i
                mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '2_%d.bias' % i] = \
                    dtt_bp + '.' + py_norm_suffix[1:] + '2_%d.bias' % i
                if cfg.DETAIRS.USE_SN:
                    mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '2_%d.mean_weight' % i] = \
                        dtt_bp + '.' + py_norm_suffix[1:] + '2_%d.mean_weight' % i
                    mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '2_%d.var_weight' % i] = \
                        dtt_bp + '.' + py_norm_suffix[1:] + '2_%d.var_weight' % i
                    if cfg.DETAIRS.SN.USE_BN:
                        mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '2_%d.running_mean' % i] = \
                            dtt_bp + '.' + py_norm_suffix[1:] + '2_%d.running_mean' % i
                        mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '2_%d.running_var' % i] = \
                            dtt_bp + '.' + py_norm_suffix[1:] + '2_%d.running_var' % i

        # cat_bn
        mapping_to_pytorch[my_prefix + '.cat_' + my_norm_suffix[1:] + '.weight'] = \
            pytorch_prefix + '.' + py_norm_suffix[1:] + '_concat.weight'
        mapping_to_pytorch[my_prefix + '.cat_' + my_norm_suffix[1:] + '.bias'] = \
            pytorch_prefix + '.' + py_norm_suffix[1:] + '_concat.bias'
        if cfg.DETAIRS.USE_SN:
            mapping_to_pytorch[my_prefix + '.cat_' + my_norm_suffix[1:] + '.mean_weight'] = \
                pytorch_prefix + '.' + py_norm_suffix[1:] + '_concat.mean_weight'
            mapping_to_pytorch[my_prefix + '.cat_' + my_norm_suffix[1:] + '.var_weight'] = \
                pytorch_prefix + '.' + py_norm_suffix[1:] + '_concat.var_weight'
            if cfg.DETAIRS.SN.USE_BN:
                mapping_to_pytorch[my_prefix + '.cat_' + my_norm_suffix[1:] + '.running_mean'] = \
                    pytorch_prefix + '.' + py_norm_suffix[1:] + '_concat.running_mean'
                mapping_to_pytorch[my_prefix + '.cat_' + my_norm_suffix[1:] + '.running_var'] = \
                    pytorch_prefix + '.' + py_norm_suffix[1:] + '_concat.running_var'

        # conv3
        mapping_to_pytorch[my_prefix + '.conv3.weight'] = pytorch_prefix + '.conv.weight'
        mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '3.weight'] = \
            pytorch_prefix + '.' + py_norm_suffix[1:] + '.weight'
        mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '3.bias'] = \
            pytorch_prefix + '.' + py_norm_suffix[1:] + '.bias'
        if cfg.DETAIRS.USE_SN:
            mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '3.mean_weight'] = \
                pytorch_prefix + '.' + py_norm_suffix[1:] + '.mean_weight'
            mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '3.var_weight'] = \
                pytorch_prefix + '.' + py_norm_suffix[1:] + '.var_weight'
            if cfg.DETAIRS.SN.USE_BN:
                mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '3.running_mean'] = \
                    pytorch_prefix + '.' + py_norm_suffix[1:] + '.running_mean'
                mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '3.running_var'] = \
                    pytorch_prefix + '.' + py_norm_suffix[1:] + '.running_var'

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
