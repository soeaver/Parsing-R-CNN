"""
Creates a ResNet Model as defined in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2015 CVPR). 
Deep Residual Learning for Image Recognition. 
Copyright (c) Yang Lu, 2017
"""
import torch
import torch.nn as nn

import models.ops as ops
from utils.net import make_norm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, base_width=64, stride=1, dilation=1, norm='bn', conv='normal', context='none',
                 ctx_ratio=0.0625, stride_3x3=False, downsample=None):
        super(BasicBlock, self).__init__()
        if conv == 'normal':
            conv_op = nn.Conv2d
        elif conv == 'deform':
            conv_op = ops.DeformConvPack
        elif conv == 'deformv2':
            conv_op = ops.ModulatedDeformConvPack
        else:
            raise ValueError('{} type conv operation is not supported.'.format(conv))
        assert context in ['none', 'se', 'gcb']
        width = int(planes * (base_width / 64.))

        self.conv1 = conv_op(inplanes, width, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                             bias=False)
        self.bn1 = make_norm(width, norm=norm, an_k=10 if planes < 256 else 20)
        self.conv2 = conv_op(width, width, kernel_size=3, stride=1, dilation=dilation, padding=dilation,
                             bias=False)
        self.bn2 = make_norm(width, norm=norm, an_k=10 if planes < 256 else 20)

        if context == 'none':
            self.ctx = None
        elif context == 'se':
            self.ctx = ops.SeConv2d(width, int(width * ctx_ratio))
        elif context == 'gcb':
            self.ctx = ops.GlobalContextBlock(width, int(width * ctx_ratio))
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
        if self.ctx is not None:
            out = self.ctx(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, base_width=64, stride=1, dilation=1, norm='bn', conv='normal', context='none',
                 ctx_ratio=0.0625, stride_3x3=False, downsample=None):
        super(Bottleneck, self).__init__()
        if conv == 'normal':
            conv_op = nn.Conv2d
        elif conv == 'deform':
            conv_op = ops.DeformConvPack
        elif conv == 'deformv2':
            conv_op = ops.ModulatedDeformConvPack
        else:
            raise ValueError('{} type conv operation is not supported.'.format(conv))
        (str1x1, str3x3) = (1, stride) if stride_3x3 else (stride, 1)
        width = int(planes * (base_width / 64.))

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = make_norm(width, norm=norm.split('_')[-1])
        self.conv2 = conv_op(width, width, kernel_size=3, stride=str3x3, dilation=dilation, padding=dilation,
                             bias=False)
        self.bn2 = make_norm(width, norm=norm, an_k=10 if planes < 256 else 20)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = make_norm(planes * self.expansion, norm=norm.split('_')[-1])

        if context == 'none':
            self.ctx = None
        elif context == 'se':
            self.ctx = ops.SeConv2d(planes * self.expansion, int(planes * self.expansion * ctx_ratio))
        elif context == 'gcb':
            self.ctx = ops.GlobalContextBlock(planes * self.expansion, int(planes * self.expansion * ctx_ratio))
        elif context == 'nonlocal':
            self.ctx = ops.NonLocal2d(planes * self.expansion, int(planes * self.expansion * ctx_ratio), 
                                      planes * self.expansion, use_gn=True)
        elif context == 'msa':
            self.ctx = ops.MS_NonLocal2d(planes * self.expansion, int(planes * self.expansion * ctx_ratio), 
                                         planes * self.expansion, use_gn=True)            
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

    def __init__(self, inplanes, planes, base_width=64, stride=1, dilation=1, norm='bn', conv='normal', context='none',
                 ctx_ratio=0.0625, stride_3x3=False, downsample=None):
        super(AlignedBottleneck, self).__init__()
        if conv == 'normal':
            conv_op = nn.Conv2d
        elif conv == 'deform':
            conv_op = ops.DeformConvPack
        elif conv == 'deformv2':
            conv_op = ops.ModulatedDeformConvPack
        else:
            raise ValueError('{} type conv operation is not supported.'.format(conv))
        width = int(planes * (base_width / 64.))

        self.conv1_1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1_1 = make_norm(width, norm=norm.split('_')[-1])
        self.conv1_2 = conv_op(width, width, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False)
        self.conv2_1 = nn.Conv2d(inplanes, width // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2_1 = make_norm(width // 2, norm=norm.split('_')[-1])
        self.conv2_2 = conv_op(width // 2, width // 2, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2_2 = make_norm(width // 2, norm=norm, an_k=10 if planes < 256 else 20)
        self.conv2_3 = conv_op(width // 2, width // 2, kernel_size=3, stride=1, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn_concat = make_norm(width + (width // 2), norm=norm, an_k=10 if planes < 256 else 20)

        self.conv = nn.Conv2d(width + (width // 2), planes * self.expansion, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.bn = make_norm(planes * self.expansion, norm=norm.split('_')[-1])

        if context == 'none':
            self.ctx = None
        elif context == 'se':
            self.ctx = ops.SeConv2d(planes * self.expansion, int(planes * self.expansion * ctx_ratio))
        elif context == 'gcb':
            self.ctx = ops.GlobalContextBlock(planes * self.expansion, int(planes * self.expansion * ctx_ratio))
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


class ResNet(nn.Module):
    def __init__(self, bottleneck=True, aligned=False, use_3x3x3stem=False, stride_3x3=False, avg_down=False,
                 base_width=64, layers=(3, 4, 6, 3), norm='bn',
                 stage_with_conv=('normal', 'normal', 'normal', 'normal'),
                 stage_with_context=('none', 'none', 'none', 'none'), ctx_ratio=16,
                 num_classes=1000):
        """ Constructor
        Args:
            layers: config of layers, e.g., (3, 4, 23, 3)
            num_classes: number of classes
        """
        super(ResNet, self).__init__()
        if aligned:
            block = AlignedBottleneck
        else:
            if bottleneck:
                block = Bottleneck
            else:
                block = BasicBlock
        self.expansion = block.expansion
        self.stride_3x3 = stride_3x3
        self.avg_down = avg_down
        self.base_width = base_width
        self.norm = norm
        self.ctx_ratio = ctx_ratio

        self.inplanes = 64
        self.use_3x3x3stem = use_3x3x3stem
        if not self.use_3x3x3stem:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
            self.bn1 = make_norm(self.inplanes, norm=self.norm.split('_')[-1])
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes // 2, 3, 2, 1, bias=False)
            self.bn1 = make_norm(self.inplanes // 2, norm=self.norm.split('_')[-1])
            self.conv2 = nn.Conv2d(self.inplanes // 2, self.inplanes // 2, 3, 1, 1, bias=False)
            self.bn2 = make_norm(self.inplanes // 2, norm=self.norm.split('_')[-1])
            self.conv3 = nn.Conv2d(self.inplanes // 2, self.inplanes, 3, 1, 1, bias=False)
            self.bn3 = make_norm(self.inplanes, norm=self.norm.split('_')[-1])
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
                if not isinstance(m, (ops.MixtureBatchNorm2d, ops.MixtureGroupNorm)):
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
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, AlignedBottleneck):
                nn.init.constant_(m.bn.weight, 0)
                    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, conv='normal', context='none'):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNet
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
                    make_norm(planes * block.expansion, norm=self.norm.split('_')[-1]),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    make_norm(planes * block.expansion, norm=self.norm.split('_')[-1]),
                )

        layers = []
        layers.append(block(self.inplanes, planes, self.base_width, stride, dilation, self.norm, conv, context,
                            self.ctx_ratio, self.stride_3x3, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.base_width, 1, dilation, self.norm, conv, context,
                                self.ctx_ratio, self.stride_3x3))

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
