"""
Creates a MobileNetV1 Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017 CVPR). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
Copyright (c) Yang Lu, 2017
"""
import torch.nn as nn
import torch.nn.functional as F

import models.ops as ops
from models.imagenet.utils import make_divisible
from utils.net import make_norm


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, kernel=3, norm='bn', use_se=False, activation=nn.ReLU):
        super(BasicBlock, self).__init__()
        padding = (dilation * kernel - dilation) // 2
        self.inplanes, self.planes = int(inplanes), int(planes)

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=kernel, padding=padding, stride=stride,
                               dilation=dilation, groups=inplanes, bias=False)
        self.bn1 = make_norm(inplanes, norm=norm)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = make_norm(planes, norm=norm)
        self.se = ops.Se2d(planes, reduction=4) if use_se else None
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
           
        if self.se is not None:
            out = self.se(out)

        return out


class MobileNetV1(nn.Module):
    def __init__(self, use_se=False, widen_factor=1.0, kernel=3, layers=(2, 2, 6, 2), norm='bn',  
                 activation=nn.ReLU, drop_rate=0.0, num_classes=1000):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNetV1, self).__init__()
        block = BasicBlock
        self.use_se = use_se
        self.norm = norm
        self.drop_rate = drop_rate
        self.activation_type = activation
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()

        num_of_channels = [32, 64, 128, 256, 512, 1024]
        channels = [make_divisible(ch * widen_factor, 8) for ch in num_of_channels]
        self.channels = channels

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = make_norm(channels[0], norm=self.norm)
        self.conv2 = nn.Conv2d(channels[0], channels[0], kernel_size=kernel, stride=1, padding=kernel // 2,
                               groups=channels[0], bias=False)
        self.bn2 = make_norm(channels[0], norm=self.norm)
        self.conv3 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = make_norm(channels[1], norm=self.norm)
        self.inplanes = channels[1]

        self.layer1 = self._make_layer(block, channels[2], layers[0], stride=2, dilation=1, kernel=kernel)
        self.layer2 = self._make_layer(block, channels[3], layers[1], stride=2, dilation=1, kernel=kernel)
        self.layer3 = self._make_layer(block, channels[4], layers[2], stride=2, dilation=1, kernel=kernel)
        self.layer4 = self._make_layer(block, channels[5], layers[3], stride=2, dilation=1, kernel=kernel)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[5], num_classes)

        self._init_weights()

    @property
    def stage_out_dim(self):
        return self.channels[1:]

    @property
    def stage_out_spatial(self):
        return [1 / 2., 1 / 4., 1 / 8., 1 / 16., 1 / 32.]

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, outplanes, blocks, stride=1, dilation=1, kernel=3):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNet
            outplanes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        layers = []
        layers.append(block(self.inplanes, outplanes, stride, dilation=dilation, kernel=kernel, norm=self.norm,
                            use_se=self.use_se, activation=self.activation_type))
        self.inplanes = outplanes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, outplanes, stride=1, dilation=dilation, kernel=kernel, norm=self.norm,
                                use_se=self.use_se, activation=self.activation_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
