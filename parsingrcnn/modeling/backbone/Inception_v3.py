import os
import numbers
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (Inception_v3_C4_body, Inception_v3_C5_body, ...)
# ---------------------------------------------------------------------------- #

def Inception_v3_C4_body():
    return Inception_v3_CX_body(block_counts=3)


def Inception_v3_C5_body():
    return Inception_v3_CX_body(block_counts=4)


# ---------------------------------------------------------------------------- #
# Generic Inception_v3 components
# ---------------------------------------------------------------------------- #


class Inception_v3_CX_body(nn.Module):
    def __init__(self, block_counts=4):
        super().__init__()
        self.block_counts = block_counts  # 3 or 4
        self.convX = self.block_counts + 1  # 4 or 5

        self.incepv3_1 = stem()

        self.incepv3_2 = add_stage2()

        self.incepv3_3 = add_stage3()

        self.incepv3_4 = add_stage4()

        if block_counts == 4:
            self.incepv3_5 = add_stage5(stride_init=cfg.INCEPTIONV3.C5_STRIDE)
            self.spatial_scale = 1 / 16 / cfg.INCEPTIONV3.C5_STRIDE  # cfg.INCEPTIONV3.C5_STRIDE: 2
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = 2048

        self._init_modules()

    def _init_modules(self):
        assert cfg.INCEPTIONV3.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.INCEPTIONV3.FREEZE_AT: 2
        assert cfg.INCEPTIONV3.FREEZE_AT <= self.convX
        for i in range(1, cfg.INCEPTIONV3.FREEZE_AT + 1):
            freeze_params(getattr(self, 'incepv3_%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {
            'incepv3_1.conv1.weight': 'conv1_w',
            'incepv3_1.bn1.weight': 'conv1_bn_s',
            'incepv3_1.bn1.bias': 'conv1_bn_b',
            'incepv3_1.conv2.weight': 'conv2_w',
            'incepv3_1.bn2.weight': 'conv2_bn_s',
            'incepv3_1.bn2.bias': 'conv2_bn_b',
            'incepv3_1.conv3.weight': 'conv3_w',
            'incepv3_1.bn3.weight': 'conv3_bn_s',
            'incepv3_1.bn3.bias': 'conv3_bn_b',
            'incepv3_2.conv4.weight': 'conv4_w',
            'incepv3_2.bn4.weight': 'conv4_bn_s',
            'incepv3_2.bn4.bias': 'conv4_bn_b',
            'incepv3_2.conv5.weight': 'conv5_w',
            'incepv3_2.bn5.weight': 'conv5_bn_s',
            'incepv3_2.bn5.bias': 'conv5_bn_b',
        }
        orphan_in_detectron = ['pred_w', 'pred_b']

        for incepv3_id in range(3, self.convX + 1):
            stage_name = 'incepv3_%d' % incepv3_id  # incepv3_id = 3, 4, 5
            mapping, orphans = inception_v3_block_detectron_mapping(getattr(self, stage_name), stage_name)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def pytorch_weight_mapping(self):
        mapping_to_pytorch = {
            'incepv3_1.conv1.weight': 'Conv2d_1a_3x3.conv.weight',
            'incepv3_1.bn1.weight': 'Conv2d_1a_3x3.bn.weight',
            'incepv3_1.bn1.bias': 'Conv2d_1a_3x3.bn.bias',
            'incepv3_1.conv2.weight': 'Conv2d_2a_3x3.conv.weight',
            'incepv3_1.bn2.weight': 'Conv2d_2a_3x3.bn.weight',
            'incepv3_1.bn2.bias': 'Conv2d_2a_3x3.bn.bias',
            'incepv3_1.conv3.weight': 'Conv2d_2b_3x3.conv.weight',
            'incepv3_1.bn3.weight': 'Conv2d_2b_3x3.bn.weight',
            'incepv3_1.bn3.bias': 'Conv2d_2b_3x3.bn.bias',
            'incepv3_2.conv4.weight': 'Conv2d_3b_1x1.conv.weight',
            'incepv3_2.bn4.weight': 'Conv2d_3b_1x1.bn.weight',
            'incepv3_2.bn4.bias': 'Conv2d_3b_1x1.bn.bias',
            'incepv3_2.conv5.weight': 'Conv2d_4a_3x3.conv.weight',
            'incepv3_2.bn5.weight': 'Conv2d_4a_3x3.bn.weight',
            'incepv3_2.bn5.bias': 'Conv2d_4a_3x3.bn.bias',
        }
        orphan_in_pytorch = ['fc.weight', 'fc.bias']

        for incepv3_id in range(3, self.convX + 1):
            stage_name = 'incepv3_%d' % incepv3_id  # incepv3_id = 3, 4, 5
            mapping, orphans = inception_v3_block_pytorch_mapping(getattr(self, stage_name), stage_name)
            mapping_to_pytorch.update(mapping)
            orphan_in_pytorch.extend(orphans)

        return mapping_to_pytorch, orphan_in_pytorch

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.INCEPTIONV3.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'incepv3_%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'incepv3_%d' % (i + 1))(x)
        return x


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=int(cfg.INCEPTIONV3.ALIGN), bias=False)),
        ('bn1', mynn.AffineChannel2d(32)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=int(cfg.INCEPTIONV3.ALIGN), bias=False)),
        ('bn2', mynn.AffineChannel2d(32)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)),
        ('bn3', mynn.AffineChannel2d(64)),
        ('relu3', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=int(cfg.INCEPTIONV3.ALIGN)))])
    )


def add_stage2():
    return nn.Sequential(OrderedDict([
        ('conv4', nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0, bias=False)),
        ('bn4', mynn.AffineChannel2d(80)),
        ('relu4', nn.ReLU(inplace=True)),
        ('conv5', nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=int(cfg.INCEPTIONV3.ALIGN), bias=False)),
        ('bn5', mynn.AffineChannel2d(192)),
        ('relu5', nn.ReLU(inplace=True))])
    )


def add_stage3():
    return nn.Sequential(OrderedDict([
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=int(cfg.INCEPTIONV3.ALIGN))),
        ('Mixed_5b', InceptionA(192, pool_features=32)),
        ('Mixed_5c', InceptionA(256, pool_features=64)),
        ('Mixed_5d', InceptionA(288, pool_features=64))])
    )


def add_stage4():
    return nn.Sequential(OrderedDict([
        ('Mixed_6a', InceptionB(288)),
        ('Mixed_6b', InceptionC(768, channels_7x7=128)),
        ('Mixed_6c', InceptionC(768, channels_7x7=160)),
        ('Mixed_6d', InceptionC(768, channels_7x7=160)),
        ('Mixed_6e', InceptionC(768, channels_7x7=192))])
    )


def add_stage5(stride_init=2):
    return nn.Sequential(OrderedDict([
        ('Mixed_7a', InceptionD(768, stride=stride_init)),
        ('Mixed_7b', InceptionE(1280)),
        ('Mixed_7c', InceptionE(2048))])
    )

# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


class BasicConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, bias=False, **kwargs)
        self.bn = mynn.AffineChannel2d(outplanes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, padding=int(cfg.INCEPTIONV3.ALIGN))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2, padding=int(cfg.INCEPTIONV3.ALIGN))

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=int(cfg.INCEPTIONV3.ALIGN))

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels, stride=2):
        super(InceptionD, self).__init__()
        self.stride = stride

        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=stride, padding=int(cfg.INCEPTIONV3.ALIGN))

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=stride, padding=int(cfg.INCEPTIONV3.ALIGN))

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=int(cfg.INCEPTIONV3.ALIGN))
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #


def inception_v3_block_detectron_mapping(module_ref, module_name):
    """inception_v3_block_detectron_mapping. module_name: incepv3_3, incepv3_4, incepv3_5
    """

    mapping_to_detectron = {}
    orphan_in_detectron = []
    for m in module_ref.state_dict():
        mapping_to_detectron[module_name + '.' + m] = m

    return mapping_to_detectron, orphan_in_detectron


def inception_v3_block_pytorch_mapping(module_ref, module_name):
    """inception_v3_block_pytorch_mapping. module_name: incepv3_3, incepv3_4, incepv3_5
    """

    mapping_to_pytorch = {}
    orphan_in_pytorch = []
    for m in module_ref.state_dict():
        mapping_to_pytorch[module_name + '.' + m] = m

    return mapping_to_pytorch, orphan_in_pytorch


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
