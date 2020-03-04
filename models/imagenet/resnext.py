"""
Creates a ResNeXt Model as defined in:
Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He. (2017 CVPR). 
Aggregated residual transformations for deep neural networks. 
Copyright (c) Yang Lu, 2017
"""
import math

import torch
import torch.nn as nn

import models.ops as ops
from utils.net import make_norm


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, base_width, cardinality, stride=1, dilation=1, norm='bn', conv='normal',
                 context='none', ctx_ratio=0.0625, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            base_width: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality

        if conv == 'normal':
            conv_op = nn.Conv2d
        elif conv == 'deform':
            conv_op = ops.DeformConvPack
        elif conv == 'deformv2':
            conv_op = ops.ModulatedDeformConvPack
        else:
            raise ValueError('{} type conv operation is not supported.'.format(conv))

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = make_norm(D * C, norm=norm)
        self.conv2 = conv_op(D * C, D * C, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                             groups=C, bias=False)
        self.bn2 = make_norm(D * C, norm=norm)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = make_norm(planes * 4, norm=norm)

        if context == 'none':
            self.ctx = None
        elif context == 'se':
            self.ctx = ops.SeConv2d(planes * 4, int(planes * 4 * ctx_ratio))
        elif context == 'gcb':
            self.ctx = ops.GlobalContextBlock(planes * 4, int(planes * 4 * ctx_ratio))
        else:
            raise ValueError('{} type context operation is not supported.'.format(context))

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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
        if self.ctx is not None:
            out = self.ctx(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AlignedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, base_width, cardinality, stride=1, dilation=1, norm='bn', conv='normal',
                 context='none', ctx_ratio=0.0625, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            base_width: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(AlignedBottleneck, self).__init__()

        D = int(math.floor(planes * (base_width / 64.0)))  # when planes=64, C=32, base_width=4, then D=4
        C = cardinality

        if conv == 'normal':
            conv_op = nn.Conv2d
        elif conv == 'deform':
            conv_op = ops.DeformConvPack
        elif conv == 'deformv2':
            conv_op = ops.ModulatedDeformConvPack
        else:
            raise ValueError('{} type conv operation is not supported.'.format(conv))

        self.conv1_1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1_1 = make_norm(D * C, norm=norm)
        self.conv1_2 = conv_op(D * C, D * C, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               groups=C, bias=False)
        self.conv2_1 = nn.Conv2d(inplanes, D * C // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2_1 = make_norm(D * C // 2, norm=norm)
        self.conv2_2 = conv_op(D * C // 2, D * C // 2, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, groups=C // 2, bias=False)
        self.bn2_2 = make_norm(D * C // 2, norm=norm)
        self.conv2_3 = conv_op(D * C // 2, D * C // 2, kernel_size=3, stride=1, dilation=dilation, padding=dilation,
                               groups=C // 2, bias=False)
        self.bn_concat = make_norm(D * C + D * C // 2, norm=norm)

        self.conv = nn.Conv2d(D * C + D * C // 2, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = make_norm(planes * 4, norm=norm)

        if context == 'none':
            self.ctx = None
        elif context == 'se':
            self.ctx = ops.SeConv2d(planes * 4, int(planes * 4 * ctx_ratio))
        elif context == 'gcb':
            self.ctx = ops.GlobalContextBlock(planes * 4, int(planes * 4 * ctx_ratio))
        else:
            raise ValueError('{} type context operation is not supported.'.format(context))

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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
        out = self.bn_concat(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.bn(out)
        if self.ctx is not None:
            out = self.ctx(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, aligned=False, use_3x3x3stem=False, avg_down=False, base_width=4, cardinality=32,
                 layers=(3, 4, 23, 3), norm='bn',
                 stage_with_conv=('normal', 'normal', 'normal', 'normal'),
                 stage_with_context=('none', 'none', 'none', 'none'), ctx_ratio=16,
                 num_classes=1000):
        """ Constructor
        Args:
            base_width: base_width for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., (3, 4, 23, 3)
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()
        if aligned:
            block = AlignedBottleneck
        else:
            block = Bottleneck
        self.expansion = block.expansion
        self.avg_down = avg_down
        self.norm = norm
        self.cardinality = cardinality
        self.base_width = base_width
        self.ctx_ratio = ctx_ratio

        self.inplanes = 64
        self.use_3x3x3stem = use_3x3x3stem
        if not self.use_3x3x3stem:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
            self.bn1 = make_norm(self.inplanes, norm=self.norm)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes // 2, 3, 2, 1, bias=False)
            self.bn1 = make_norm(self.inplanes // 2, norm=self.norm)
            self.conv2 = nn.Conv2d(self.inplanes // 2, self.inplanes // 2, 3, 1, 1, bias=False)
            self.bn2 = make_norm(self.inplanes // 2, norm=self.norm)
            self.conv3 = nn.Conv2d(self.inplanes // 2, self.inplanes, 3, 1, 1, bias=False)
            self.bn3 = make_norm(self.inplanes, norm=self.norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1, conv=stage_with_conv[0], context=stage_with_context[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2, conv=stage_with_conv[1], context=stage_with_context[1])
        self.layer3 = self._make_layer(block, 256, layers[2], 2, conv=stage_with_conv[2], context=stage_with_context[2])
        self.layer4 = self._make_layer(block, 512, layers[3], 2, conv=stage_with_conv[3], context=stage_with_context[3])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)

        self._init_weights()

    @property
    def stage_out_dim(self):
        return [64, 64 * self.expansion, 128 * self.expansion, 256 * self.expansion, 512 * self.expansion]

    @property
    def stage_out_spatial(self):
        return [1 / 2., 1 / 4., 1 / 8., 1 / 16., 1 / 32.]

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.0001)
                nn.init.constant_(m.bias, 0)
        # zero init deform conv offset
        for m in self.modules():
            if isinstance(m, ops.DeformConvPack):
                nn.init.constant_(m.conv_offset.weight, 0)
                nn.init.constant_(m.conv_offset.bias, 0)
            if isinstance(m, ops.ModulatedDeformConvPack):
                nn.init.constant_(m.conv_offset_mask.weight, 0)
                nn.init.constant_(m.conv_offset_mask.bias, 0)
        # zero gamma for last bn of each block
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, AlignedBottleneck):
                nn.init.constant_(m.bn.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, conv='normal', context='none'):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    make_norm(planes * block.expansion, norm=self.norm),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    make_norm(planes * block.expansion, norm=self.norm),
                )

        layers = []
        layers.append(block(self.inplanes, planes, self.base_width, self.cardinality, stride, dilation, self.norm,
                            conv, context, self.ctx_ratio, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.base_width, self.cardinality, 1, dilation, self.norm,
                                conv, context, self.ctx_ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.use_3x3x3stem:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
