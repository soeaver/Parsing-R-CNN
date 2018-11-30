import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (DetNet59, DetNet110, ...)
# ---------------------------------------------------------------------------- #


def DetNet59_conv5_body():
    return DetNet_convX_body((3, 4, 6, 3))


def DetNet59_conv6_body():
    return DetNet_convX_body((3, 4, 6, 3, 3))


def DetNet110_conv5_body():
    return DetNet_convX_body((3, 4, 23, 3))


def DetNet110_conv6_body():
    return DetNet_convX_body((3, 4, 23, 3, 3))


def DetNet59_roi_conv6_head(dim_in, roi_xform_func, spatial_scale):
    return DetNet_roi_conv6_head(dim_in, roi_xform_func, spatial_scale, (3, 4, 6, 3))


# ---------------------------------------------------------------------------- #
# Generic DetNet59 components
# ---------------------------------------------------------------------------- #


class DetNet_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts  # 4 or 5
        self.convX = len(block_counts) + 1  # 5 or 6
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2  #

        self.detnet1 = globals()[cfg.DETNETS.STEM_FUNC]()
        dim_in = 64
        # cfg.DETNETS.NUM_GROUPS: 1     cfg.DETNETS.WIDTH_PER_GROUP: 64
        dim_bottleneck = cfg.DETNETS.NUM_GROUPS * cfg.DETNETS.WIDTH_PER_GROUP
        self.detnet2, dim_in = add_stage(dim_in, cfg.DETNETS.WIDTH_OUTPLANE, dim_bottleneck, block_counts[0],
                                         dilation=1, stride_init=1)
        self.detnet3, dim_in = add_stage(dim_in, cfg.DETNETS.WIDTH_OUTPLANE * 2, dim_bottleneck * 2, block_counts[1],
                                         dilation=1, stride_init=2)
        self.detnet4, dim_in = add_stage(dim_in, cfg.DETNETS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4, block_counts[2],
                                         dilation=1, stride_init=2)
        self.detnet5, dim_in = add_stage(dim_in, cfg.DETNETS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4, block_counts[3],
                                         dilation=2, stride_init=1)
        if len(block_counts) == 5:
            self.detnet6, dim_in = add_stage(dim_in, cfg.DETNETS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4,
                                             block_counts[4], dilation=2, stride_init=1)
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.DETNETS.FREEZE_AT in [0, 2, 3, 4, 5, 6]  # cfg.DETNETS.FREEZE_AT: 2
        assert cfg.DETNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.DETNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'detnet%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.DETNETS.USE_GN:
            mapping_to_detectron = {
                'detnet1.conv1.weight': 'conv1_w',
                'detnet1.gn1.weight': 'conv1_gn_s',
                'detnet1.gn1.bias': 'conv1_gn_b',
                'detnet1.conv2.weight': 'conv2_w',
                'detnet1.gn2.weight': 'conv2_gn_s',
                'detnet1.gn2.bias': 'conv2_gn_b',
                'detnet1.conv3.weight': 'conv3_w',
                'detnet1.gn3.weight': 'conv3_gn_s',
                'detnet1.gn3.bias': 'conv3_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        elif cfg.DETNETS.USE_SN:
            mapping_to_detectron = {
                'detnet1.conv1.weight': 'conv1_w',
                'detnet1.sn1.weight': 'conv1_sn_s',
                'detnet1.sn1.bias': 'conv1_sn_b',
                'detnet1.sn1.mean_weight': 'conv1_sn_mean_weight',
                'detnet1.sn1.var_weight': 'conv1_sn_var_weight',
                'detnet1.conv2.weight': 'conv2_w',
                'detnet1.sn2.weight': 'conv2_sn_s',
                'detnet1.sn2.bias': 'conv2_sn_b',
                'detnet1.sn2.mean_weight': 'conv2_sn_mean_weight',
                'detnet1.sn2.var_weight': 'conv2_sn_var_weight',
                'detnet1.conv3.weight': 'conv3_w',
                'detnet1.sn3.weight': 'conv3_sn_s',
                'detnet1.sn3.bias': 'conv3_sn_b',
                'detnet1.sn3.mean_weight': 'conv3_sn_mean_weight',
                'detnet1.sn3.var_weight': 'conv3_sn_var_weight',
            }
            if cfg.DETNETS.SN.USE_BN:
                mapping_to_detectron.update({
                    'detnet1.sn1.running_mean': 'conv1_sn_rm',
                    'detnet1.sn1.running_var': 'conv1_sn_riv',
                    'detnet1.sn2.running_mean': 'conv2_sn_rm',
                    'detnet1.sn2.running_var': 'conv2_sn_riv',
                    'detnet1.sn3.running_mean': 'conv3_sn_rm',
                    'detnet1.sn3.running_var': 'conv3_sn_riv',
                })
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'detnet1.conv1.weight': 'conv1_w',
                'detnet1.bn1.weight': 'conv1_bn_s',
                'detnet1.bn1.bias': 'conv1_bn_b',
                'detnet1.conv2.weight': 'conv2_w',
                'detnet1.bn2.weight': 'conv2_bn_s',
                'detnet1.bn2.bias': 'conv2_bn_b',
                'detnet1.conv3.weight': 'conv3_w',
                'detnet1.bn3.weight': 'conv3_bn_s',
                'detnet1.bn3.bias': 'conv3_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'conv2_b', 'conv3_b', 'fc_w', 'fc_b']

        for detnet_id in range(2, self.convX + 1):
            stage_name = 'detnet%d' % detnet_id  # detnet_id = 2, 3, 4, 5, 6
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, detnet_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        if cfg.DETNETS.USE_GN:
            mapping_to_pytorch = {
                'detnet1.conv1.weight': 'conv1.weight',
                'detnet1.gn1.weight': 'bn1.weight',
                'detnet1.gn1.bias': 'bn1.bias',
                'detnet1.conv2.weight': 'conv2.weight',
                'detnet1.gn2.weight': 'bn2.weight',
                'detnet1.gn2.bias': 'bn2.bias',
                'detnet1.conv3.weight': 'conv3.weight',
                'detnet1.gn3.weight': 'bn3.weight',
                'detnet1.gn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']
        elif cfg.DETNETS.USE_SN:
            mapping_to_pytorch = {
                'detnet1.conv1.weight': 'conv1.weight',
                'detnet1.sn1.weight': 'sn1.weight',
                'detnet1.sn1.bias': 'sn1.bias',
                'detnet1.sn1.mean_weight': 'sn1.mean_weight',
                'detnet1.sn1.var_weight': 'sn1.var_weight',
                'detnet1.conv2.weight': 'conv2.weight',
                'detnet1.sn2.weight': 'sn2.weight',
                'detnet1.sn2.bias': 'sn2.bias',
                'detnet1.sn2.mean_weight': 'sn2.mean_weight',
                'detnet1.sn2.var_weight': 'sn2.var_weight',
                'detnet1.conv3.weight': 'conv3.weight',
                'detnet1.sn3.weight': 'sn3.weight',
                'detnet1.sn3.bias': 'sn3.bias',
                'detnet1.sn3.mean_weight': 'sn3.mean_weight',
                'detnet1.sn3.var_weight': 'sn3.var_weight',
            }
            if cfg.DETNETS.SN.USE_BN:
                mapping_to_pytorch.update({
                    'detnet1.sn1.running_mean': 'sn1.running_mean',
                    'detnet1.sn1.running_var': 'sn1.running_var',
                    'detnet1.sn2.running_mean': 'sn2.running_mean',
                    'detnet1.sn2.running_var': 'sn2.running_var',
                    'detnet1.sn3.running_mean': 'sn3.running_mean',
                    'detnet1.sn3.running_var': 'sn3.running_var',
                })
            orphan_in_pytorch = ['pred_w', 'pred_b']
        else:
            mapping_to_pytorch = {
                'detnet1.conv1.weight': 'conv1.weight',
                'detnet1.bn1.weight': 'bn1.weight',
                'detnet1.bn1.bias': 'bn1.bias',
                'detnet1.conv2.weight': 'conv2.weight',
                'detnet1.bn2.weight': 'bn2.weight',
                'detnet1.bn2.bias': 'bn2.bias',
                'detnet1.conv3.weight': 'conv3.weight',
                'detnet1.bn3.weight': 'bn3.weight',
                'detnet1.bn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']

        for detnet_id in range(2, self.convX + 1):
            stage_name = 'detnet%d' % detnet_id  # detnet_id = 2, 3, 4, 5, 6
            mapping, orphans = residual_stage_pytorch_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, detnet_id)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.DETNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'detnet%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'detnet%d' % (i + 1))(x)
        return x


class DetNet_roi_conv6_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, block_counts):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.block_counts = block_counts

        dim_bottleneck = cfg.DETNETS.NUM_GROUPS * cfg.DETNETS.WIDTH_PER_GROUP
        # stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.detnet6, self.dim_out = add_stage(dim_in, cfg.DETNETS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4, 3,
                                               dilation=2, stride_init=1)
        assert self.dim_out == cfg.DETNETS.WIDTH_OUTPLANE * 4
        self.avgpool = nn.AvgPool2d(14)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
            residual_stage_detectron_mapping(self.detnet6, 'detnet6', self.block_counts, 6)
        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch, orphan_in_pytorch = \
            residual_stage_pytorch_mapping(self.detnet6, 'detnet6', self.block_counts, 6)
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
        detnet6_feat = self.detnet6(x)
        x = self.avgpool(detnet6_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, detnet6_feat
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
        shortcut_func = globals()[cfg.DETNETS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.DETNETS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride, dilation=dilation, group=cfg.DETNETS.NUM_GROUPS, downsample=downsample
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
                                using_bn=cfg.DETNETS.SN.USE_BN)),
        ('relu', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)),
        ('sn2', mynn.SwitchNorm(32, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                using_bn=cfg.DETNETS.SN.USE_BN)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)),
        ('sn3', mynn.SwitchNorm(64, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                using_bn=cfg.DETNETS.SN.USE_BN)),
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
                        using_bn=cfg.DETNETS.SN.USE_BN)
    )


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.bn1 = mynn.AffineChannel2d(innerplanes)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=stride,
                               padding=1 * dilation, dilation=dilation, groups=group, bias=False)
        self.bn2 = mynn.AffineChannel2d(innerplanes)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
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

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=stride,
                               padding=1 * dilation, dilation=dilation, groups=group, bias=False)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class bottleneck_sn_transformation(nn.Module):
    """ Bottleneck Residual Block With SN"""

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.sn1 = mynn.SwitchNorm(innerplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                   using_bn=cfg.DETNETS.SN.USE_BN)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=stride,
                               padding=1 * dilation, dilation=dilation, groups=group, bias=False)
        self.sn2 = mynn.SwitchNorm(innerplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                   using_bn=cfg.DETNETS.SN.USE_BN)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.sn3 = mynn.SwitchNorm(outplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                   using_bn=cfg.DETNETS.SN.USE_BN)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.sn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sn2(out)
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

def residual_stage_detectron_mapping(module_ref, module_name, block_counts, detnet_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `detnet_id`
    """

    num_blocks = block_counts[detnet_id - 2]
    if cfg.DETNETS.USE_GN:
        norm_suffix = '_gn'
    elif cfg.DETNETS.USE_SN:
        norm_suffix = '_sn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'detnet%d' % (sum(block_counts[:detnet_id - 2]) + blk_id + 1)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: detnet2, detnet3, detnet4, detnet5, detnet6

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_match_conv'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'
            if cfg.DETNETS.USE_SN:
                mapping_to_detectron[my_prefix
                                     + '.downsample.1.mean_weight'] = dtt_bp + norm_suffix + '_mean_weight'
                mapping_to_detectron[my_prefix
                                     + '.downsample.1.var_weight'] = dtt_bp + norm_suffix + '_var_weight'
                if cfg.DETNETS.SN.USE_BN:
                    mapping_to_detectron[my_prefix
                                         + '.downsample.1.running_mean'] = dtt_bp + norm_suffix + '_rm'
                    mapping_to_detectron[my_prefix
                                         + '.downsample.1.running_var'] = dtt_bp + norm_suffix + '_riv'
        # conv branch
        for i in range(1, 4):
            dtt_bp = detectron_prefix + '_conv' + str(i)
            mapping_to_detectron[my_prefix + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'
            if cfg.DETNETS.USE_SN:
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.mean_weight' % i] = \
                    dtt_bp + norm_suffix + '_mean_weight'
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.var_weight' % i] = \
                    dtt_bp + norm_suffix + '_var_weight'
                if cfg.DETNETS.SN.USE_BN:
                    mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.running_mean' % i] = \
                        dtt_bp + norm_suffix + '_rm'
                    mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.running_var' % i] = \
                        dtt_bp + norm_suffix + '_riv'

    return mapping_to_detectron, orphan_in_detectron


def residual_stage_pytorch_mapping(module_ref, module_name, block_counts, detnet_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `detnet_id`
    """

    num_blocks = block_counts[detnet_id - 2]
    if cfg.DETNETS.USE_GN:
        my_norm_suffix = '_gn'
        py_norm_suffix = '_bn'
    elif cfg.DETNETS.USE_SN:
        my_norm_suffix = '_sn'
        py_norm_suffix = '_sn'
    else:
        my_norm_suffix = '_bn'
        py_norm_suffix = '_bn'
    mapping_to_pytorch = {}
    orphan_in_pytorch = []
    for blk_id in range(num_blocks):
        pytorch_prefix = 'layer{}.{}'.format(detnet_id - 1, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: detnet2, detnet3, detnet4, detnet5, detnet6

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = pytorch_prefix + '.downsample'  # short for "pytorch_branch_prefix"
            mapping_to_pytorch[my_prefix + '.downsample.0.weight'] = dtt_bp + '.0.weight'
            mapping_to_pytorch[my_prefix + '.downsample.1.weight'] = dtt_bp + '.1.weight'
            mapping_to_pytorch[my_prefix + '.downsample.1.bias'] = dtt_bp + '.1.bias'
            if cfg.DETNETS.USE_SN:
                mapping_to_pytorch[my_prefix + '.downsample.1.mean_weight'] = dtt_bp + '.1.mean_weight'
                mapping_to_pytorch[my_prefix + '.downsample.1.var_weight'] = dtt_bp + '.1.var_weight'
                if cfg.DETNETS.SN.USE_BN:
                    mapping_to_pytorch[my_prefix + '.downsample.1.running_mean'] = dtt_bp + '.1.running_mean'
                    mapping_to_pytorch[my_prefix + '.downsample.1.running_var'] = dtt_bp + '.1.running_var'
        # conv branch
        for i in range(1, 4):
            mapping_to_pytorch[my_prefix + '.conv%d.weight' % i] = pytorch_prefix + '.conv%d.weight' % i
            mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '%d.weight' % i] = \
                pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.weight' % i
            mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '%d.bias' % i] = \
                pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.bias' % i
            if cfg.DETNETS.USE_SN:
                mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '%d.mean_weight' % i] = \
                    pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.mean_weight' % i
                mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '%d.var_weight' % i] = \
                    pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.var_weight' % i
                if cfg.DETNETS.SN.USE_BN:
                    mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '%d.running_mean' % i] = \
                        pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.running_mean' % i
                    mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '%d.running_var' % i] = \
                        pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.running_var' % i

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
