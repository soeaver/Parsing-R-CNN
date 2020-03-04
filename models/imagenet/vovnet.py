"""
Creates a ResNet Model as defined in:
Youngwan Lee, Joong-won Hwang, Sangrok Lee, Yuseok Bae. (2019 arxiv).
An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection.
Copyright (c) Yang Lu, 2019
"""
import torch
import torch.nn as nn

import models.ops as ops
from utils.net import make_norm


class OSABlock(nn.Module):
    def __init__(self, inplanes, planes, outplanes, num_conv=5, dilation=1, norm='bn', conv='normal', identity=False):
        super(OSABlock, self).__init__()
        if conv == 'normal':
            conv_op = nn.Conv2d
        elif conv == 'deform':
            conv_op = ops.DeformConvPack
        elif conv == 'deformv2':
            conv_op = ops.ModulatedDeformConvPack
        else:
            raise ValueError('{} type conv operation is not supported.'.format(conv))

        self.identity = identity
        self.layers = nn.ModuleList()
        dim_in = inplanes
        for i in range(num_conv):
            self.layers.append(
                nn.Sequential(
                    conv_op(dim_in, planes, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=False),
                    make_norm(planes, norm=norm),
                    nn.ReLU(inplace=True)
                )
            )
            dim_in = planes

        # feature aggregation
        dim_in = inplanes + num_conv * planes
        self.concat = nn.Sequential(
            conv_op(dim_in, outplanes, kernel_size=1, stride=1, bias=False),
            make_norm(outplanes, norm=norm),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity_feat = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class VoVNet(nn.Module):
    def __init__(self, base_width=64, stage_dims=(128, 160, 192, 224), concat_dims=(256, 512, 768, 1024),
                 layers=(1, 1, 2, 2), num_conv=5, norm='bn', stage_with_conv=('normal', 'normal', 'normal', 'normal'),
                 num_classes=1000):
        """ Constructor
        Args:
            layers: config of layers, e.g., (1, 1, 2, 2)
            num_classes: number of classes
        """
        super(VoVNet, self).__init__()
        block = OSABlock
        self.num_conv = num_conv
        self.norm = norm
        self.channels = [base_width] + list(concat_dims)

        self.inplanes = base_width
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, 2, 1, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=self.norm)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, 3, 1, 1, bias=False)
        self.bn2 = make_norm(self.inplanes, norm=self.norm)
        self.conv3 = nn.Conv2d(self.inplanes, self.inplanes * 2, 3, 2, 1, bias=False)
        self.bn3 = make_norm(self.inplanes * 2, norm=self.norm)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = self.inplanes * 2

        self.layer1 = self._make_layer(block, stage_dims[0], concat_dims[0], layers[0], 1, conv=stage_with_conv[0])
        self.layer2 = self._make_layer(block, stage_dims[1], concat_dims[1], layers[1], 2, conv=stage_with_conv[1])
        self.layer3 = self._make_layer(block, stage_dims[2], concat_dims[2], layers[2], 2, conv=stage_with_conv[2])
        self.layer4 = self._make_layer(block, stage_dims[3], concat_dims[3], layers[3], 2, conv=stage_with_conv[3])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inplanes, num_classes)

        self._init_weights()

    @property
    def stage_out_dim(self):
        return self.channels

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
                
    def _make_layer(self, block, planes, outplanes, blocks, stride=1, dilation=1, conv='normal'):
        layers = []
        if stride != 1:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        layers.append(block(self.inplanes, planes, outplanes, self.num_conv, dilation, self.norm, conv))
        self.inplanes = outplanes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, outplanes, self.num_conv, dilation, self.norm, conv, True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
