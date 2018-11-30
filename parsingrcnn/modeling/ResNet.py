import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
from parsingrcnn.model.dcn.deform_conv import DeformConv2d
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils
import parsingrcnn.modeling.nonlocal_helper as nonlocal_helper


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #

def ResNet50_conv4_body():
    return ResNet_convX_body((3, 4, 6))


def ResNet50_conv5_body():
    return ResNet_convX_body((3, 4, 6, 3))


def ResNet101_conv4_body():
    return ResNet_convX_body((3, 4, 23))


def ResNet101_conv5_body():
    return ResNet_convX_body((3, 4, 23, 3))


def ResNet152_conv5_body():
    return ResNet_convX_body((3, 8, 36, 3))


# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #


class ResNet_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = globals()[cfg.RESNETS.STEM_FUNC]()
        dim_in = 64
        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        self.res2, dim_in = add_stage(dim_in, 256, dim_bottleneck, block_counts[0],
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 512, dim_bottleneck * 2, block_counts[1],
                                      dilation=1, stride_init=2, use_deform_stage=(True and cfg.FPN.FPN_ON))
        self.res4, dim_in = add_stage(dim_in, 1024, dim_bottleneck * 4, block_counts[2],
                                      dilation=1, stride_init=2, use_nonlocal_stage=True, 
                                      use_deform_stage=(True and cfg.FPN.FPN_ON))
        if len(block_counts) == 4:
            stride_init = 2 if cfg.RESNETS.RES5_DILATION == 1 else 1
            self.res5, dim_in = add_stage(dim_in, 2048, dim_bottleneck * 8, block_counts[3],
                                          cfg.RESNETS.RES5_DILATION, stride_init,
                                          use_deform_stage=True, all_deform=True)
            self.spatial_scale = 1 / 32 * cfg.RESNETS.RES5_DILATION
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.RESNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.RESNETS.USE_GN:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.gn1.weight': 'conv1_gn_s',
                'res1.gn1.bias': 'conv1_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        elif cfg.RESNETS.USE_SN:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.sn1.weight': 'conv1_sn_s',
                'res1.sn1.bias': 'conv1_sn_b',
                'res1.sn1.mean_weight': 'conv1_sn_mean_weight',
                'res1.sn1.var_weight': 'conv1_sn_var_weight',
            }
            if cfg.RESNETS.SN.USE_BN:
                mapping_to_detectron.update({
                    'res1.sn1.running_mean': 'conv1_sn_rm',
                    'res1.sn1.running_var': 'conv1_sn_riv',
                })
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.bn1.weight': 'res_conv1_bn_s',
                'res1.bn1.bias': 'res_conv1_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'fc1000_w', 'fc1000_b']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name,
                self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        if cfg.RESNETS.USE_GN:
            mapping_to_pytorch = {
                'res1.conv1.weight': 'conv1.weight',
                'res1.gn1.weight': 'bn1.weight',
                'res1.gn1.bias': 'bn1.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']
        elif cfg.RESNETS.USE_SN:
            mapping_to_pytorch = {
                'res1.conv1.weight': 'conv1.weight',
                'res1.sn1.weight': 'sn1.weight',
                'res1.sn1.bias': 'sn1.bias',
                'res1.sn1.mean_weight': 'sn1.mean_weight',
                'res1.sn1.var_weight': 'sn1.var_weight',
            }
            if cfg.RESNETS.SN.USE_BN:
                mapping_to_pytorch.update({
                    'res1.sn1.running_mean': 'sn1.running_mean',
                    'res1.sn1.running_var': 'sn1.running_var',
                })
            orphan_in_pytorch = ['pred_w', 'pred_b']
        else:
            mapping_to_pytorch = {
                'res1.conv1.weight': 'conv1.weight',
                'res1.bn1.weight': 'bn1.weight',
                'res1.bn1.bias': 'bn1.bias',
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

        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x


class ResNet_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, self.dim_out = add_stage(dim_in, 2048, dim_bottleneck * 8, 3,
                                            dilation=1, stride_init=stride_init)
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
            residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch, orphan_in_pytorch = \
            residual_stage_pytorch_mapping(self.res5, 'res5', 3, 5)
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
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, dim_in = add_stage(dim_in, 2048, dim_bottleneck * 8, 3,
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
            residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)

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
            residual_stage_pytorch_mapping(self.res5, 'res5', 3, 5)
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
        if cfg.RESNETS.USE_NONLOCAL and use_nonlocal_stage and _ == nblocks - 2:
            use_nonlocal = True
        else:
            use_nonlocal = False
            
        if cfg.RESNETS.USE_DEFORM and use_deform_stage and (_ == nblocks - 1 or all_deform):
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
        shortcut_func = globals()[cfg.RESNETS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.RESNETS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride, dilation=dilation, group=cfg.RESNETS.NUM_GROUPS, 
        downsample=downsample, use_nonlocal=use_nonlocal, use_deform=use_deform
    )

    return res_block


# ------------------------------------------------------------------------------
# various downsample shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        mynn.AffineChannel2d(outplanes),
    )


def basic_gn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                     eps=cfg.GROUP_NORM.EPSILON)
    )


def basic_sn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        mynn.SwitchNorm(outplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                        using_bn=cfg.RESNETS.SN.USE_BN)
    )


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('bn1', mynn.AffineChannel2d(64)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


def basic_gn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('gn1', nn.GroupNorm(net_utils.get_group_gn(64), 64,
                             eps=cfg.GROUP_NORM.EPSILON)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


def basic_sn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('sn1', mynn.SwitchNorm(64, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                using_bn=cfg.RESNETS.SN.USE_BN)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None, use_nonlocal=False, use_deform=False):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride
        self.use_nonlocal = use_nonlocal
        self.use_deform = use_deform

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = mynn.AffineChannel2d(innerplanes)

        if self.use_deform:
            self.conv2_offset = nn.Conv2d(
                innerplanes, 72, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv2 = DeformConv2d(
                innerplanes, innerplanes, kernel_size=3, stride=str3x3,
                padding=1 * dilation, dilation=dilation, num_deformable_groups=4)
        else:
            self.conv2 = nn.Conv2d(
                innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
                padding=1 * dilation, dilation=dilation, groups=group)
        self.bn2 = mynn.AffineChannel2d(innerplanes)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        if self.use_nonlocal:
            self.non_local = nonlocal_helper.SpaceNonLocal(outplanes, outplanes // 2, outplanes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.use_deform:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        else:
            out = self.conv2(out)
        out = self.bn2(out)
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
    expansion = 4

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None, use_nonlocal=False, use_deform=False):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride
        self.use_nonlocal = use_nonlocal
        self.use_deform = use_deform

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        if self.use_deform:
            self.conv2_offset = nn.Conv2d(
                innerplanes, 72, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv2 = DeformConv2d(
                innerplanes, innerplanes, kernel_size=3, stride=str3x3,
                padding=1 * dilation, dilation=dilation, num_deformable_groups=4)
        else:
            self.conv2 = nn.Conv2d(
                innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
                padding=1 * dilation, dilation=dilation, groups=group)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
        if self.use_nonlocal:
            self.non_local = nonlocal_helper.SpaceNonLocal(outplanes, outplanes // 2, outplanes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        if self.use_deform:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        else:
            out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.use_nonlocal:
            out = self.non_local(out)

        return out


class bottleneck_sn_transformation(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None, use_nonlocal=False, use_deform=False):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride
        self.use_nonlocal = use_nonlocal
        self.use_deform = use_deform

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.sn1 = mynn.SwitchNorm(innerplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                   using_bn=cfg.RESNETS.SN.USE_BN)

        if self.use_deform:
            self.conv2_offset = nn.Conv2d(
                innerplanes, 72, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv2 = DeformConv2d(
                innerplanes, innerplanes, kernel_size=3, stride=str3x3,
                padding=1 * dilation, dilation=dilation, num_deformable_groups=4)
        else:
            self.conv2 = nn.Conv2d(
                innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
                padding=1 * dilation, dilation=dilation, groups=group)
        self.sn2 = mynn.SwitchNorm(innerplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                   using_bn=cfg.RESNETS.SN.USE_BN)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.sn3 = mynn.SwitchNorm(outplanes, using_moving_average=(not cfg.TEST.USE_BATCH_AVG),
                                   using_bn=cfg.RESNETS.SN.USE_BN)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        if self.use_nonlocal:
            self.non_local = nonlocal_helper.SpaceNonLocal(outplanes, outplanes // 2, outplanes)
            
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.sn1(out)
        out = self.relu(out)

        if self.use_deform:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        else:
            out = self.conv2(out)
        out = self.sn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.sn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.use_nonlocal:
            out = self.non_local(out)

        return out


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def residual_stage_detectron_mapping(module_ref, module_name, num_blocks, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    if cfg.RESNETS.USE_GN:
        norm_suffix = '_gn'
    elif cfg.RESNETS.USE_SN:
        norm_suffix = '_sn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'res%d_%d' % (res_id, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'
            if cfg.RESNETS.USE_SN:
                mapping_to_detectron[my_prefix
                                     + '.downsample.1.mean_weight'] = dtt_bp + norm_suffix + '_mean_weight'
                mapping_to_detectron[my_prefix
                                     + '.downsample.1.var_weight'] = dtt_bp + norm_suffix + '_var_weight'
                if cfg.RESNETS.SN.USE_BN:
                    mapping_to_detectron[my_prefix
                                         + '.downsample.1.running_mean'] = dtt_bp + norm_suffix + '_rm'
                    mapping_to_detectron[my_prefix
                                         + '.downsample.1.running_var'] = dtt_bp + norm_suffix + '_riv'
        # conv branch
        for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
            dtt_bp = detectron_prefix + '_branch2' + c
            mapping_to_detectron[my_prefix + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'
            if cfg.RESNETS.USE_SN:
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.mean_weight' % i] = \
                    dtt_bp + norm_suffix + '_mean_weight'
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.var_weight' % i] = \
                    dtt_bp + norm_suffix + '_var_weight'
                if cfg.RESNETS.SN.USE_BN:
                    mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.running_mean' % i] = \
                        dtt_bp + norm_suffix + '_rm'
                    mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.running_var' % i] = \
                        dtt_bp + norm_suffix + '_riv'
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
            if getattr(module_ref[blk_id], 'conv2_offset'):
                mapping_to_detectron.update({
                    '{}.conv2_offset.weight'.format(my_prefix): '{}_branch2b_offset_w'.format(detectron_prefix),
                    '{}.conv2_offset.bias'.format(my_prefix): '{}_branch2b_offset_b'.format(detectron_prefix),
                })
        except:
            pass
            

    return mapping_to_detectron, orphan_in_detectron


def residual_stage_pytorch_mapping(module_ref, module_name, num_blocks, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """

    if cfg.RESNETS.USE_GN:
        my_norm_suffix = '_gn'
        py_norm_suffix = '_bn'
    elif cfg.RESNETS.USE_SN:
        my_norm_suffix = '_sn'
        py_norm_suffix = '_sn'
    else:
        my_norm_suffix = '_bn'
        py_norm_suffix = '_bn'
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
            if cfg.RESNETS.USE_SN:
                mapping_to_pytorch[my_prefix + '.downsample.1.mean_weight'] = dtt_bp + '.1.mean_weight'
                mapping_to_pytorch[my_prefix + '.downsample.1.var_weight'] = dtt_bp + '.1.var_weight'
                if cfg.RESNETS.SN.USE_BN:
                    mapping_to_pytorch[my_prefix + '.downsample.1.running_mean'] = dtt_bp + '.1.running_mean'
                    mapping_to_pytorch[my_prefix + '.downsample.1.running_var'] = dtt_bp + '.1.running_var'
        # conv branch
        for i in range(1, 4):
            mapping_to_pytorch[my_prefix + '.conv%d.weight' % i] = pytorch_prefix + '.conv%d.weight' % i
            mapping_to_pytorch[
                my_prefix + '.' + my_norm_suffix[1:] + '%d.weight' % i] = \
                pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.weight' % i
            mapping_to_pytorch[
                my_prefix + '.' + my_norm_suffix[1:] + '%d.bias' % i] = \
                pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.bias' % i
            if cfg.RESNETS.USE_SN:
                mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '%d.mean_weight' % i] = \
                    pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.mean_weight' % i
                mapping_to_pytorch[
                    my_prefix + '.' + my_norm_suffix[1:] + '%d.var_weight' % i] = \
                    pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.var_weight' % i
                if cfg.RESNETS.SN.USE_BN:
                    mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '%d.running_mean' % i] = \
                        pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.running_mean' % i
                    mapping_to_pytorch[my_prefix + '.' + my_norm_suffix[1:] + '%d.running_var' % i] = \
                        pytorch_prefix + '.' + py_norm_suffix[1:] + '%d.running_var' % i
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
            if getattr(module_ref[blk_id], 'conv2_offset'):
                mapping_to_detectron.update({
                    '{}.conv2_offset.weight'.format(my_prefix): '{}.conv2_offset.weight'.format(pytorch_prefix),
                    '{}.conv2_offset.bias'.format(my_prefix): '{}.conv2_offset.bias'.format(pytorch_prefix),
                })
        except:
            pass

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
