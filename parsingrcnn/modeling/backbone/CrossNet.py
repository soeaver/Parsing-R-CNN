import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (CrossNet58K5_B70G2, CrossNet58K5_B40G2, ...)
# ---------------------------------------------------------------------------- #

def CrossNet_C4_body():
    return CrossNet_CX_body(cfg.CROSSNET.BLOCK_COUNTS[:3])


def CrossNet_C5_body():
    return CrossNet_CX_body(cfg.CROSSNET.BLOCK_COUNTS)


def CrossNet_roi_C5_head(dim_in, roi_xform_func, spatial_scale):
    return CrossNet_roi_C5_head_base(dim_in, roi_xform_func, spatial_scale, cfg.CROSSNET.BLOCK_COUNTS)


# ---------------------------------------------------------------------------- #
# Generic CrossNet components (Not support GN)
# ---------------------------------------------------------------------------- #

class CrossNet_CX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts  # 3 or 4
        self.convX = len(block_counts) + 1  # 4 or 5
        basic_width = cfg.CROSSNET.BASE_WIDTH
        expansion = cfg.CROSSNET.EXPANSION

        head_dim = 24
        if cfg.CROSSNET.BASE_WIDTH == 80:
            head_dim = 32
        elif cfg.CROSSNET.BASE_WIDTH == 90 or cfg.CROSSNET.BASE_WIDTH == 100:
            head_dim = 48

        self.cross1 = globals()[cfg.CROSSNET.STEM_FUNC](head_dim)
        dim_in = head_dim
        self.cross2, dim_in = add_stage(dim_in, int(basic_width * expansion), basic_width, block_counts[0],
                                        dilation=1, stride_init=1)
        self.cross3, dim_in = add_stage(dim_in, int(basic_width * 2 * expansion), basic_width * 2, block_counts[1],
                                        dilation=1, stride_init=2)
        self.cross4, dim_in = add_stage(dim_in, int(basic_width * 4 * expansion), basic_width * 4, block_counts[2],
                                        dilation=1, stride_init=2)
        if len(block_counts) == 4:
            if cfg.CROSSNET.C5_DILATION != 1:
                stride = 1
            else:
                stride = 2
            self.cross5, dim_in = add_stage(dim_in, int(basic_width * 8 * expansion), basic_width * 8, block_counts[3],
                                            dilation=cfg.CROSSNET.C5_DILATION, stride_init=stride)
            self.spatial_scale = 1 / 32 * cfg.CROSSNET.C5_DILATION  # cfg.CROSSNET.C5_DILATION: 1
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.CROSSNET.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.CROSSNET.FREEZE_AT: 2
        assert cfg.CROSSNET.FREEZE_AT <= self.convX
        for i in range(1, cfg.CROSSNET.FREEZE_AT + 1):
            freeze_params(getattr(self, 'cross%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.CROSSNET.USE_GN:
            mapping_to_detectron = {
                'cross1.conv1.weight': 'conv1_w',
                'cross1.gn1.weight': 'conv1_gn_s',
                'cross1.gn1.bias': 'conv1_gn_b',
                'cross1.conv2.weight': 'conv2_w',
                'cross1.gn2.weight': 'conv2_gn_s',
                'cross1.gn2.bias': 'conv2_gn_b',
                'cross1.conv3.weight': 'conv3_w',
                'cross1.gn3.weight': 'conv3_gn_s',
                'cross1.gn3.bias': 'conv3_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'cross1.conv1.weight': 'conv1_w',
                'cross1.bn1.weight': 'conv1_bn_s',
                'cross1.bn1.bias': 'conv1_bn_b',
                'cross1.conv2.weight': 'conv2_w',
                'cross1.bn2.weight': 'conv2_bn_s',
                'cross1.bn2.bias': 'conv2_bn_b',
                'cross1.conv3.weight': 'conv3_w',
                'cross1.bn3.weight': 'conv3_bn_s',
                'cross1.bn3.bias': 'conv3_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'conv2_b', 'conv3_b', 'fc_w', 'fc_b']

        for cross_id in range(2, self.convX + 1):
            stage_name = 'cross%d' % cross_id  # cross_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_detectron_mapping(stage_name, self.block_counts, cross_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        if cfg.CROSSNET.USE_GN:
            mapping_to_pytorch = {
                'cross1.conv1.weight': 'conv1.weight',
                'cross1.gn1.weight': 'bn1.weight',
                'cross1.gn1.bias': 'bn1.bias',
                'cross1.conv2.weight': 'conv2.weight',
                'cross1.gn2.weight': 'bn2.weight',
                'cross1.gn2.bias': 'bn2.bias',
                'cross1.conv3.weight': 'conv3.weight',
                'cross1.gn3.weight': 'bn3.weight',
                'cross1.gn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']
        else:
            mapping_to_pytorch = {
                'cross1.conv1.weight': 'conv1.weight',
                'cross1.bn1.weight': 'bn1.weight',
                'cross1.bn1.bias': 'bn1.bias',
                'cross1.conv2.weight': 'conv2.weight',
                'cross1.bn2.weight': 'bn2.weight',
                'cross1.bn2.bias': 'bn2.bias',
                'cross1.conv3.weight': 'conv3.weight',
                'cross1.bn3.weight': 'bn3.weight',
                'cross1.bn3.bias': 'bn3.bias',
            }
            orphan_in_pytorch = ['fc.weight', 'fc.bias']

        for cross_id in range(2, self.convX + 1):
            stage_name = 'cross%d' % cross_id  # cross_id = 2, 3, 4, 5
            mapping, orphans = residual_stage_pytorch_mapping(stage_name, self.block_counts, cross_id)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.CROSSNET.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'cross%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'cross%d' % (i + 1))(x)
        return x


class CrossNet_roi_C5_head_base(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, block_counts):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.block_counts = block_counts

        basic_width = cfg.CROSSNET.BASE_WIDTH
        expansion = cfg.CROSSNET.EXPANSION

        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.cross5, self.dim_out = add_stage(dim_in, int(basic_width * 8 * expansion), basic_width * 8,
                                              block_counts[3], dilation=1, stride_init=stride_init)
        assert self.dim_out == int(basic_width * 8 * expansion)
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
            residual_stage_detectron_mapping('cross5', self.block_counts, 5)
        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch, orphan_in_pytorch = \
            residual_stage_pytorch_mapping('cross5', self.block_counts, 5)
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
        cross5_feat = self.cross5(x)
        x = self.avgpool(cross5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, cross5_feat
        else:
            return x


def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2):
    stage_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        stage_blocks.append(add_pod(inplanes, outplanes, innerplanes, dilation, stride))
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*stage_blocks), outplanes


class add_pod(nn.Module):
    """Return a cross pod, including residual and dense connection, """

    def __init__(self, inplanes, outplanes, innerplanes, dilation=1, stride=1):
        super().__init__()

        init_trans_func = globals()[cfg.CROSSNET.INIT_TRANS_FUNC]
        cross_trans_func = globals()[cfg.CROSSNET.CROSS_TRANS_FUNC]

        self.crs1 = init_trans_func(inplanes, outplanes, innerplanes, stride=stride, dilation=dilation,
                                    kernel=cfg.CROSSNET.KERNEL_SIZE, groups=cfg.CROSSNET.LAST1X1_GROUP)
        if cfg.CROSSNET.POD_DEPTH > 1:
            self.crs2 = cross_trans_func(outplanes, outplanes, innerplanes, xid=1, stride=1, dilation=dilation,
                                         kernel=cfg.CROSSNET.KERNEL_SIZE, groups=cfg.CROSSNET.LAST1X1_GROUP)
        if cfg.CROSSNET.POD_DEPTH > 2:
            self.crs3 = cross_trans_func(outplanes, outplanes, innerplanes, xid=2, stride=1, dilation=dilation,
                                         kernel=cfg.CROSSNET.KERNEL_SIZE, groups=cfg.CROSSNET.LAST1X1_GROUP)
        if cfg.CROSSNET.POD_DEPTH > 3:
            self.crs4 = cross_trans_func(outplanes, outplanes, innerplanes, xid=3, stride=1, dilation=dilation,
                                         kernel=cfg.CROSSNET.KERNEL_SIZE, groups=cfg.CROSSNET.LAST1X1_GROUP)

    def forward(self, x):
        x1, x2 = self.crs1.forward(x)
        if cfg.CROSSNET.POD_DEPTH > 1:
            x1, x2 = self.crs2.forward(x1, x2)
        if cfg.CROSSNET.POD_DEPTH > 2:
            x1, x2 = self.crs3.forward(x1, x2)
        if cfg.CROSSNET.POD_DEPTH > 3:
            x1, x2 = self.crs4.forward(x1, x2)

        return x1


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_stem(head_dim):
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, head_dim, kernel_size=3, stride=2, padding=1, bias=False)),
        ('bn1', mynn.AffineChannel2d(head_dim)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(head_dim, head_dim, kernel_size=1, stride=1, padding=0, bias=False)),
        ('bn2', mynn.AffineChannel2d(head_dim)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(head_dim, head_dim, kernel_size=cfg.CROSSNET.KERNEL_SIZE, stride=2,
                            padding=cfg.CROSSNET.KERNEL_SIZE // 2, groups=head_dim, bias=False)),
        ('bn3', mynn.AffineChannel2d(head_dim)),
        ('relu3', nn.ReLU(inplace=True))])
    )


def basic_gn_stem(head_dim):
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, head_dim, kernel_size=3, stride=2, padding=1, bias=False)),
        ('gn1', nn.GroupNorm(net_utils.get_group_gn(head_dim), head_dim, eps=cfg.GROUP_NORM.EPSILON)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(head_dim, head_dim, kernel_size=1, stride=1, padding=0, bias=False)),
        ('gn2', nn.GroupNorm(net_utils.get_group_gn(head_dim), head_dim, eps=cfg.GROUP_NORM.EPSILON)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(head_dim, head_dim, kernel_size=cfg.CROSSNET.KERNEL_SIZE, stride=2,
                            padding=cfg.CROSSNET.KERNEL_SIZE // 2, groups=head_dim, bias=False)),
        ('gn3', nn.GroupNorm(net_utils.get_group_gn(head_dim), head_dim, eps=cfg.GROUP_NORM.EPSILON)),
        ('relu3', nn.ReLU(inplace=True))])
    )


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


class linear_bottleneck_transformation(nn.Module):
    """ Linear Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, kernel=3, groups=2):
        super().__init__()
        self.stride = stride
        self.inplanes, self.outplanes = int(inplanes), int(outplanes)

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn1 = mynn.AffineChannel2d(innerplanes)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel, stride=stride, dilation=dilation,
                               padding=(dilation * kernel - dilation) // 2, groups=innerplanes, bias=False)
        self.bn2 = mynn.AffineChannel2d(innerplanes)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, padding=0, groups=groups, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 1 and self.inplanes == self.outplanes:
            residual = x
        else:
            residual = None

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out = self.conv3(out1)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual
        else:
            pass

        return out, out1


class linear_bottleneck_gn_transformation(nn.Module):
    """ Linear Bottleneck Residual Block With GN"""

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, kernel=3, groups=2):
        super().__init__()
        self.stride = stride
        self.inplanes, self.outplanes = int(inplanes), int(outplanes)

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, padding=0, stride=1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel, stride=stride, dilation=dilation,
                               padding=(dilation * kernel - dilation) // 2, groups=innerplanes, bias=False)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, padding=0, groups=groups, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 1 and self.inplanes == self.outplanes:
            residual = x
        else:
            residual = None

        out1 = self.conv1(x)
        out1 = self.gn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.gn2(out1)
        out1 = self.relu(out1)

        out = self.conv3(out1)
        out = self.gn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual
        else:
            pass

        return out, out1


class cross_bottleneck_transformation(nn.Module):
    """ Cross Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, xid=1, stride=1, dilation=1, kernel=3, groups=2):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.bn1 = mynn.AffineChannel2d(innerplanes)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel, stride=stride, dilation=dilation,
                               padding=(dilation * kernel - dilation) // 2, groups=innerplanes, bias=False)
        self.bn2 = mynn.AffineChannel2d(innerplanes)
        self.conv3 = nn.Conv2d(innerplanes * (xid + 1), outplanes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        residual = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out1 = torch.cat((out1, x2), 1)

        out = self.conv3(out1)
        out = self.bn3(out)

        out += residual

        return out, out1


class cross_bottleneck_gn_transformation(nn.Module):
    """ Cross Bottleneck Residual Block With GN """

    def __init__(self, inplanes, outplanes, innerplanes, xid=1, stride=1, dilation=1, kernel=3, groups=2):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel, stride=stride, dilation=dilation,
                               padding=(dilation * kernel - dilation) // 2, groups=innerplanes, bias=False)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv3 = nn.Conv2d(innerplanes * (xid + 1), outplanes, kernel_size=1, groups=groups, bias=False)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        residual = x1

        out1 = self.conv1(x1)
        out1 = self.gn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.gn2(out1)
        out1 = self.relu(out1)

        out1 = torch.cat((out1, x2), 1)

        out = self.conv3(out1)
        out = self.gn3(out)

        out += residual

        return out, out1


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #


def residual_stage_detectron_mapping(module_name, block_counts, cross_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `cross_id`
    """

    num_blocks = block_counts[cross_id - 2]

    if cfg.CROSSNET.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for pod_id in range(num_blocks):
        for blk_id in range(cfg.CROSSNET.POD_DEPTH):
            detectron_prefix = 'pod{}crs{}'.format(sum(block_counts[:cross_id - 2]) + pod_id + 1, blk_id + 1)
            # module_name: cross2, cross3, cross4, cross5
            my_prefix = '{}.{}.crs{}'.format(module_name, pod_id, blk_id + 1)

            # conv branch
            for i in range(1, 4):
                dtt_bp = detectron_prefix + '_conv' + str(i)
                mapping_to_detectron[my_prefix + '.conv%d.weight' % i] = dtt_bp + '_w'
                orphan_in_detectron.append(dtt_bp + '_b')
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
                mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'

    return mapping_to_detectron, orphan_in_detectron


def residual_stage_pytorch_mapping(module_name, block_counts, cross_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `cross_id`
    """

    num_blocks = block_counts[cross_id - 2]

    if cfg.CROSSNET.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_pytorch = {}
    orphan_in_pytorch = []

    for pod_id in range(num_blocks):
        for blk_id in range(cfg.CROSSNET.POD_DEPTH):
            if blk_id == 0:
                pytorch_prefix = 'layer{}.{}.init'.format(cross_id - 1, pod_id)
            else:
                pytorch_prefix = 'layer{}.{}.cross{}'.format(cross_id - 1, pod_id, blk_id)
            # module_name: cross2, cross3, cross4, cross5
            my_prefix = '{}.{}.crs{}'.format(module_name, pod_id, blk_id + 1)

            # conv branch
            for i in range(1, 4):
                mapping_to_pytorch[my_prefix + '.conv%d.weight' % i] = pytorch_prefix + '.conv%d.weight' % i
                mapping_to_pytorch[
                    my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = pytorch_prefix + '.bn%d.weight' % i
                mapping_to_pytorch[
                    my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = pytorch_prefix + '.bn%d.bias' % i

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
