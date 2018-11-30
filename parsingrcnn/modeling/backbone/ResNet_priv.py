import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #


def ResNet26_conv4_body():
    return ResNet_priv_convX_body((2, 2, 2))


def ResNet26_conv5_body():
    return ResNet_priv_convX_body((2, 2, 2, 2))


def ResNet50_conv4_body():
    return ResNet_priv_convX_body((3, 4, 6))


def ResNet50_conv5_body():
    return ResNet_priv_convX_body((3, 4, 6, 3))


def ResNet101_conv4_body():
    return ResNet_priv_convX_body((3, 4, 23))


def ResNet101_conv5_body():
    return ResNet_priv_convX_body((3, 4, 23, 3))


def ResNet152_conv5_body():
    return ResNet_priv_convX_body((3, 8, 36, 3))


def ResNet26_roi_conv5_head(dim_in, roi_xform_func, spatial_scale):
    return ResNet_roi_conv5_head(dim_in, roi_xform_func, spatial_scale, (2, 2, 2, 2))


def ResNet26_roi_conv5_lighthead(dim_in, roi_xform_func, spatial_scale):
    return ResNet_roi_conv5_lighthead(dim_in, roi_xform_func, spatial_scale, (2, 2, 2, 2))


def ResNet50_roi_conv5_head(dim_in, roi_xform_func, spatial_scale):
    return ResNet_roi_conv5_head(dim_in, roi_xform_func, spatial_scale, (3, 4, 6, 3))


def ResNet50_roi_conv5_lighthead(dim_in, roi_xform_func, spatial_scale):
    return ResNet_roi_conv5_lighthead(dim_in, roi_xform_func, spatial_scale, (3, 4, 6, 3))


def ResNet101_roi_conv5_head(dim_in, roi_xform_func, spatial_scale):
    return ResNet_roi_conv5_head(dim_in, roi_xform_func, spatial_scale, (3, 4, 23, 3))


# ---------------------------------------------------------------------------- #
# Generic ResNet_priv components
# ---------------------------------------------------------------------------- #


class ResNet_priv_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts  # 3 or 4
        self.convX = len(block_counts) + 1  # 4 or 5
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2  #

        self.res1 = globals()[cfg.AIRS.STEM_FUNC]()
        dim_in = 64
        # cfg.AIRS.NUM_GROUPS: 1     cfg.AIRS.WIDTH_PER_GROUP: 64
        dim_bottleneck = cfg.AIRS.NUM_GROUPS * cfg.AIRS.WIDTH_PER_GROUP
        self.res2, dim_in = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE, dim_bottleneck, block_counts[0],
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE * 2, dim_bottleneck * 2, block_counts[1],
                                      dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE * 4, dim_bottleneck * 4, block_counts[2],
                                      dilation=1, stride_init=2)
        if len(block_counts) == 4:
            if cfg.AIRS.C5_DILATION != 1:
                stride = 1
            else:
                stride = 2
            self.res5, dim_in = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE * 8, dim_bottleneck * 8, block_counts[3],
                                          dilation=cfg.AIRS.C5_DILATION, stride_init=stride)
            self.spatial_scale = 1 / 32 * cfg.AIRS.C5_DILATION  # cfg.AIRS.C5_DILATION: 1
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.AIRS.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.AIRS.FREEZE_AT: 2
        assert cfg.AIRS.FREEZE_AT <= self.convX
        for i in range(1, cfg.AIRS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.AIRS.USE_GN:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.gn1.weight': 'conv1_gn_s',
                'res1.gn1.bias': 'conv1_gn_b',
                'res1.conv2.weight': 'conv2_w',
                'res1.gn2.weight': 'conv2_gn_s',
                'res1.gn2.bias': 'conv2_gn_b',
                'res1.conv3.weight': 'conv3_w',
                'res1.gn3.weight': 'conv3_gn_s',
                'res1.gn3.bias': 'conv3_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.bn1.weight': 'conv1_bn_s',
                'res1.bn1.bias': 'conv1_bn_b',
                'res1.conv2.weight': 'conv2_w',
                'res1.bn2.weight': 'conv2_bn_s',
                'res1.bn2.bias': 'conv2_bn_b',
                'res1.conv3.weight': 'conv3_w',
                'res1.bn3.weight': 'conv3_bn_s',
                'res1.bn3.bias': 'conv3_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'conv2_b', 'conv3_b', 'fc_w', 'fc_b']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id  # res_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        if cfg.AIRS.USE_GN:
            mapping_to_pytorch = {
                'res1.conv1.weight': 'conv1.weight',
                'res1.gn1.weight': 'bn1.weight',
                'res1.gn1.bias': 'bn1.bias',
                'res1.conv2.weight': 'conv2.weight',
                'res1.gn2.weight': 'bn2.weight',
                'res1.gn2.bias': 'bn2.bias',
                'res1.conv3.weight': 'conv3.weight',
                'res1.gn3.weight': 'bn3.weight',
                'res1.gn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']
        else:
            mapping_to_pytorch = {
                'res1.conv1.weight': 'conv1.weight',
                'res1.bn1.weight': 'bn1.weight',
                'res1.bn1.bias': 'bn1.bias',
                'res1.conv2.weight': 'conv2.weight',
                'res1.bn2.weight': 'bn2.weight',
                'res1.bn2.bias': 'bn2.bias',
                'res1.conv3.weight': 'conv3.weight',
                'res1.bn3.weight': 'bn3.weight',
                'res1.bn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id  # res_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_pytorch_mapping(
                getattr(self, stage_name), stage_name, self.block_counts, res_id)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.AIRS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x


class ResNet_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, block_counts):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.block_counts = block_counts

        dim_bottleneck = cfg.AIRS.NUM_GROUPS * cfg.AIRS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, self.dim_out = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE * 8, dim_bottleneck * 8, block_counts[3],
                                            dilation=1, stride_init=stride_init)
        assert self.dim_out == cfg.AIRS.WIDTH_OUTPLANE * 8
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
            residual_stage_detectron_mapping(self.res5, 'res5', self.block_counts, 5)
        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch, orphan_in_pytorch = \
            residual_stage_pytorch_mapping(self.res5, 'res5', self.block_counts, 5)
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
        res5_feat = self.res5(x)
        x = self.avgpool(res5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, res5_feat
        else:
            return x


class ResNet_roi_conv5_lighthead(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, block_counts):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.block_counts = block_counts

        dim_bottleneck = cfg.AIRS.NUM_GROUPS * cfg.AIRS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, dim_in = add_stage(dim_in, cfg.AIRS.WIDTH_OUTPLANE * 8, dim_bottleneck * 8, block_counts[3],
                                      dilation=1, stride_init=stride_init)

        # Light head setting
        large_kernel = cfg.FAST_RCNN.LIGHT_HEAD_KERNEL
        middle_dim = cfg.FAST_RCNN.LIGHT_HEAD_MIDDLE_DIM
        output_dim = cfg.FAST_RCNN.LIGHT_HEAD_OUTPUT_DIM
        fc_dim = cfg.FAST_RCNN.LIGHT_HEAD_FC_DIM
        pad_size = large_kernel // 2
        self.light_head_branch1 = nn.Sequential(
            nn.Conv2d(dim_in, middle_dim, kernel_size=(large_kernel, 1), stride=1, padding=(pad_size, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_dim, output_dim, kernel_size=(1, large_kernel), stride=1, padding=(0, pad_size)),
            nn.ReLU(inplace=True)
        )
        self.light_head_branch2 = nn.Sequential(
            nn.Conv2d(dim_in, middle_dim, kernel_size=(1, large_kernel), stride=1, padding=(0, pad_size)),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_dim, output_dim, kernel_size=(large_kernel, 1), stride=1, padding=(pad_size, 0)),
            nn.ReLU(inplace=True)
        )

        dim_in = output_dim
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc_tail = nn.Sequential(
            nn.Linear(dim_in * roi_size ** 2, fc_dim[0]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim[0], fc_dim[1]),
            nn.ReLU(inplace=True)
        )

        self.dim_out = fc_dim[1]

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)

        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
            residual_stage_detectron_mapping(self.res5, 'res5', self.block_counts, 5)

        mapping_to_detectron.update({
            'light_head_branch1.0.weight': 'light_head_branch1_1_w',
            'light_head_branch1.0.bias': 'light_head_branch1_1_b',
            'light_head_branch1.2.weight': 'light_head_branch1_2_w',
            'light_head_branch1.2.bias': 'light_head_branch1_2_b',
            'light_head_branch2.0.weight': 'light_head_branch2_1_w',
            'light_head_branch2.0.bias': 'light_head_branch2_1_b',
            'light_head_branch2.2.weight': 'light_head_branch2_2_w',
            'light_head_branch2.2.bias': 'light_head_branch2_2_b',
        })

        mapping_to_detectron.update({
            'fc_tail.0.weight': 'fc1_w',
            'fc_tail.0.bias': 'fc1_b',
            'fc_tail.2.weight': 'fc2_w',
            'fc_tail.2.bias': 'fc2_b',
        })
        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch, orphan_in_pytorch = \
            residual_stage_pytorch_mapping(self.res5, 'res5', self.block_counts, 5)
        return mapping_to_pytorch, orphan_in_pytorch

    def forward(self, x, rpn_ret):
        x = self.res5(x)

        x1 = self.light_head_branch1(x)
        x2 = self.light_head_branch2(x)
        x = x1 + x2

        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.fc_tail(x.view(batch_size, -1))

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
        shortcut_func = globals()[cfg.AIRS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.AIRS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride, dilation=dilation, group=cfg.AIRS.NUM_GROUPS, downsample=downsample
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


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def residual_stage_detectron_mapping(module_ref, module_name, block_counts, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """

    num_blocks = block_counts[res_id - 2]
    if cfg.AIRS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'res%d' % (sum(block_counts[:res_id - 2]) + blk_id + 1)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: res2, res3, res4, res5

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

    return mapping_to_detectron, orphan_in_detectron


def residual_stage_pytorch_mapping(module_ref, module_name, block_counts, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """

    num_blocks = block_counts[res_id - 2]
    if cfg.AIRS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_pytorch = {}
    orphan_in_pytorch = []
    for blk_id in range(num_blocks):
        pytorch_prefix = 'layer{}.{}'.format(res_id - 1, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)  # module_name: res2, res3, res4, res5

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

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
