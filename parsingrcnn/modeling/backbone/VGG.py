import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (VGG16, VGG19, ...)
# ---------------------------------------------------------------------------- #


def VGG13_conv5_body():
    return VGG_convX_body((2, 2, 2, 2, 2))


def VGG16_conv5_body():
    return VGG_convX_body((2, 2, 3, 3, 3))


# ---------------------------------------------------------------------------- #
# Generic VGG components
# ---------------------------------------------------------------------------- #

class VGG_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts  # 4 or 5
        self.convX = len(block_counts)  # 4 or 5
        self.num_layers = sum(block_counts) + 2

        vgg_block_func = globals()[cfg.VGGS.BLOCK_FUNC]

        dim_in = 3
        self.vgg1, dim_in = vgg_block_func(dim_in, 64, block_counts[0])  # 1/2
        self.vgg2, dim_in = vgg_block_func(dim_in, 128, block_counts[1])  # 1/4
        self.vgg3, dim_in = vgg_block_func(dim_in, 256, block_counts[2])  # 1/8
        self.vgg4, dim_in = vgg_block_func(dim_in, 512, block_counts[3])  # 1/16

        if len(block_counts) == 5:
            self.vgg5, dim_in = vgg_block_func(dim_in, 512, block_counts[4], downsample=cfg.VGGS.USE_C5_POOL)  # 1/16
            self.spatial_scale = 1 / 16 * (1 if not cfg.VGGS.USE_C5_POOL else 2)
        else:
            self.spatial_scale = 1 / 16

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.VGGS.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.VGGS.FREEZE_AT: 2
        assert cfg.VGGS.FREEZE_AT <= self.convX
        for i in range(1, cfg.VGGS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'vgg%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        orphan_in_detectron = []
        for vgg_id in range(1, self.convX + 1):
            stage_name = 'vgg%d' % vgg_id  # vgg_id = 1, 2, 3, 4, 5
            mapping, orphans = vgg_block_detectron_mapping(stage_name, self.block_counts, vgg_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch = {}
        orphan_in_pytorch = []
        for vgg_id in range(1, self.convX + 1):
            stage_name = 'vgg%d' % vgg_id  # vgg_id = 1, 2, 3, 4, 5
            mapping, orphans = vgg_block_pytorch_mapping(stage_name, self.block_counts, vgg_id)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.VGGS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'vgg%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'vgg%d' % (i + 1))(x)
        return x


class VGG_roi_fc_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale  # 1/16.
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM  # 4096

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION  # 7
        self.fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def pytorch_weight_mapping(self):
        mapping_to_pytorch = {
            'fc1.weight': 'classifier.0.weight',
            'fc1.bias': 'classifier.0.bias',
            'fc2.weight': 'classifier.3.weight',
            'fc2.bias': 'classifier.3.bias'
        }
        return mapping_to_pytorch, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,  # 7
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def vgg_block(inplanes, outplanes, nblocks, dilation=1, downsample=True):
    dim_in = inplanes
    module_list = []
    for _ in range(nblocks):
        module_list.append(nn.Conv2d(dim_in, outplanes, kernel_size=3, padding=1 * dilation,
                                     dilation=dilation, stride=1))
        module_list.append(nn.ReLU(inplace=True))
        dim_in = outplanes

    if downsample:
        module_list.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    return nn.Sequential(*module_list), dim_in


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def vgg_block_detectron_mapping(module_name, block_counts, vgg_id):
    """Construct weight mapping relation for a vgg stage with `num_blocks` of
    vgg blocks given the stage id: `vgg_id`
    """

    num_blocks = block_counts[vgg_id - 1]
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for conv_id in range(num_blocks):
        mapping_to_detectron[module_name + '.{}.weight'.format(conv_id * 2)] = \
            'conv_{}_{}_w'.format(vgg_id, conv_id + 1)
        mapping_to_detectron[module_name + '.{}.bias'.format(conv_id * 2)] = \
            'conv_{}_{}_b'.format(vgg_id, conv_id + 1)

    return mapping_to_detectron, orphan_in_detectron


def vgg_block_pytorch_mapping(module_name, block_counts, vgg_id):
    """Construct weight mapping relation for a vgg stage with `num_blocks` of
    vgg blocks given the stage id: `vgg_id`
    """

    num_blocks = block_counts[vgg_id - 1]
    mapping_to_pytorch = {}
    orphan_in_pytorch = []

    py_id = sum(block_counts[:vgg_id - 1]) * 2 + (vgg_id - 1)
    for conv_id in range(num_blocks):
        mapping_to_pytorch[module_name + '.{}.weight'.format(conv_id * 2)] = \
            'features.{}.weight'.format(py_id + conv_id * 2)
        mapping_to_pytorch[module_name + '.{}.bias'.format(conv_id * 2)] = \
            'features.{}.bias'.format(py_id + conv_id * 2)

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
