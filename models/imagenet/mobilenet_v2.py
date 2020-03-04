"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen, et.al. (2018 CVPR). 
Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentation. 
Copyright (c) Yang Lu, 2018
"""
import copy

import torch.nn as nn
import torch.nn.functional as F

import models.ops as ops
from models.imagenet.utils import make_divisible
from utils.net import make_norm


MV2_CFG = {
    # 0,      1,           2,        3,   4,      5,      6,      7
    # kernel, out_channel, se_ratio, act, stride, group1, group2, t
    'A': [
        [[3, 32, 0, 0, 2, 0, 0, 0]],  # stem (conv1)
        [[3, 16, 0, 0, 1, 1, 1, 1]],  # layer0
        [[3, 24, 0, 0, 2, 1, 1, 6],  # layer1
         [3, 24, 0, 0, 1, 1, 1, 6]],
        [[3, 32, 0, 0, 2, 1, 1, 6],  # layer2
         [3, 32, 0, 0, 1, 1, 1, 6],
         [3, 32, 0, 0, 1, 1, 1, 6]],
        [[3, 64, 0, 0, 2, 1, 1, 6],  # layer3
         [3, 64, 0, 0, 1, 1, 1, 6],
         [3, 64, 0, 0, 1, 1, 1, 6],
         [3, 64, 0, 0, 1, 1, 1, 6],
         [3, 96, 0, 0, 1, 1, 1, 6],
         [3, 96, 0, 0, 1, 1, 1, 6],
         [3, 96, 0, 0, 1, 1, 1, 6]],
        [[3, 160, 0, 0, 2, 1, 1, 6],  # layer4
         [3, 160, 0, 0, 1, 1, 1, 6],
         [3, 160, 0, 0, 1, 1, 1, 6],
         [3, 320, 0, 0, 1, 1, 1, 6]],
        [[1, 1280, 0, 0, 1, 0, 0, 0]]  # head (conv_out)
    ]
}


def model_se(ls_cfg, se_ratio=0.25):
    new_ls_cfg = copy.deepcopy(ls_cfg)
    for l_cfg in new_ls_cfg:
        for l in l_cfg:
            l[2] = se_ratio
    return new_ls_cfg


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, dilation=1, kernel=3, groups=(1, 1), t=6, norm='bn', se_ratio=0,
                 activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        padding = (dilation * kernel - dilation) // 2
        self.stride = stride
        self.inplanes, self.outplanes, innerplanes = int(inplanes), int(outplanes), int(inplanes * abs(t))
        self.t = t
        if self.t != 1:
            self.conv1 = nn.Conv2d(self.inplanes, innerplanes, kernel_size=1, padding=0, stride=1, groups=groups[0],
                                   bias=False)
            self.bn1 = make_norm(innerplanes, norm=norm)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel, padding=padding, stride=stride,
                               dilation=dilation, groups=innerplanes, bias=False)
        self.bn2 = make_norm(innerplanes, norm=norm)
        self.se = ops.SeConv2d(innerplanes, int(self.inplanes * se_ratio), activation) if se_ratio else None
        self.conv3 = nn.Conv2d(innerplanes, self.outplanes, kernel_size=1, padding=0, stride=1, groups=groups[1],
                               bias=False)
        self.bn3 = make_norm(self.outplanes, norm=norm)
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()

    def forward(self, x):
        if self.stride == 1 and self.inplanes == self.outplanes and self.t != 1:
            residual = x
        else:
            residual = None

        if self.t != 1:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)
        else:
            out = x
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        if self.se is not None:
            out = self.se(out)
            
        out = self.conv3(out)
        out = self.bn3(out)

        out = out if residual is None else out + residual

        return out


class MobileNetV2(nn.Module):
    def __init__(self, use_se=False, widen_factor=1.0, norm='bn', activation=nn.ReLU6, drop_rate=0.0, num_classes=1000):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNetV2, self).__init__()
        block = LinearBottleneck
        self.use_se = use_se
        self.widen_factor = widen_factor
        self.norm = norm
        self.drop_rate = drop_rate
        self.activation_type = activation
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()

        layers_cfg = model_se(MV2_CFG['A']) if self.use_se else MV2_CFG['A']
        num_of_channels = [lc[-1][1] for lc in layers_cfg[1:-1]]
        self.channels = [make_divisible(ch * self.widen_factor, 8) for ch in num_of_channels]

        self.inplanes = make_divisible(layers_cfg[0][0][1] * self.widen_factor, 8)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=layers_cfg[0][0][0], stride=layers_cfg[0][0][4],
                               padding=layers_cfg[0][0][0] // 2, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=self.norm)

        self.layer0 = self._make_layer(block, layers_cfg[1], dilation=1)
        self.layer1 = self._make_layer(block, layers_cfg[2], dilation=1)
        self.layer2 = self._make_layer(block, layers_cfg[3], dilation=1)
        self.layer3 = self._make_layer(block, layers_cfg[4], dilation=1)
        self.layer4 = self._make_layer(block, layers_cfg[5], dilation=1)

        out_ch = layers_cfg[-1][-1][1]
        self.conv_out = nn.Conv2d(self.inplanes, out_ch, kernel_size=layers_cfg[-1][-1][0],
                                  stride=layers_cfg[-1][-1][4], padding=layers_cfg[-1][-1][0] // 2, bias=False)
        self.bn_out = make_norm(out_ch, norm=self.norm)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_ch, num_classes)

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
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, lc, dilation=1):
        layers = []
        for i in range(0, len(lc)):
            layers.append(block(self.inplanes, make_divisible(lc[i][1] * self.widen_factor, 8),
                                stride=lc[i][4], dilation=dilation, kernel=lc[i][0], groups=(lc[i][5], lc[i][6]),
                                t=lc[i][7], norm=self.norm, se_ratio=lc[i][2],
                                activation=self.activation_type if lc[i][3] else nn.ReLU6))
            self.inplanes = make_divisible(lc[i][1] * self.widen_factor, 8)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.activation(x)

        x = self.avgpool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
