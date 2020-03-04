"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, 
Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam, et.al. (2019 ArXiv). 
Searching for MobileNetV3. 
Copyright (c) Yang Lu, 2019
"""
import torch.nn as nn
import torch.nn.functional as F

import models.ops as ops
from models.imagenet.utils import make_divisible
from utils.net import make_norm


MV3_CFG = {
    # 0,      1,           2,        3,   4,      5,      6,      7
    # kernel, out_channel, se_ratio, act, stride, group1, group2, t
    'large': [
        [[3, 16, 0, 1, 2, 0, 0, 0]],  # stem (conv1)
        [[3, 16, 0, 0, 1, 1, 1, 1]],  # layer0
        [[3, 24, 0, 0, 2, 1, 1, 4],  # layer1
         [3, 24, 0, 0, 1, 1, 1, 3]],
        [[5, 40, 0.25, 0, 2, 1, 1, 3],  # layer2
         [5, 40, 0.25, 0, 1, 1, 1, 3],
         [5, 40, 0.25, 0, 1, 1, 1, 3]],
        [[3, 80, 0, 1, 2, 1, 1, 6],  # layer3
         [3, 80, 0, 1, 1, 1, 1, 2.5],
         [3, 80, 0, 1, 1, 1, 1, 2.3],
         [3, 80, 0, 1, 1, 1, 1, 2.3],
         [3, 112, 0.25, 1, 1, 1, 1, 6],
         [3, 112, 0.25, 1, 1, 1, 1, 6]],
        [[5, 160, 0.25, 1, 2, 1, 1, 6],  # layer4
         [5, 160, 0.25, 1, 1, 1, 1, 6],
         [5, 160, 0.25, 1, 1, 1, 1, 6]],
        [[1, 960, 0, 1, 1, 0, 0, 0],
         [1, 1280, 0, 1, 1, 0, 0, 0]]  # head (conv_out)
    ],
    'large_minimal': [
        [[3, 16, 0, 0, 2, 0, 0, 0]],  # stem (conv1)
        [[3, 16, 0, 0, 1, 1, 1, 1]],  # layer0
        [[3, 24, 0, 0, 2, 1, 1, 4],  # layer1
         [3, 24, 0, 0, 1, 1, 1, 3]],
        [[3, 40, 0, 0, 2, 1, 1, 3],  # layer2
         [3, 40, 0, 0, 1, 1, 1, 3],
         [3, 40, 0, 0, 1, 1, 1, 3]],
        [[3, 80, 0, 0, 2, 1, 1, 6],  # layer3
         [3, 80, 0, 0, 1, 1, 1, 2.5],
         [3, 80, 0, 0, 1, 1, 1, 2.3],
         [3, 80, 0, 0, 1, 1, 1, 2.3],
         [3, 112, 0, 0, 1, 1, 1, 6],
         [3, 112, 0, 0, 1, 1, 1, 6]],
        [[3, 160, 0, 0, 2, 1, 1, 6],  # layer4
         [3, 160, 0, 0, 1, 1, 1, 6],
         [3, 160, 0, 0, 1, 1, 1, 6]],
        [[1, 960, 0, 0, 1, 0, 0, 0],
         [1, 1280, 0, 0, 1, 0, 0, 0]]  # head (conv_out)
    ],
    'small': [
        [[3, 16, 0, 1, 2, 0, 0, 0]],  # stem (conv1)
        [[0, 16, 0, 0, 0, 0, 0, 0]],  # layer0
        [[3, 16, 0.25, 0, 2, 1, 1, 1]],  # layer1
        [[3, 24, 0, 0, 2, 1, 1, 4.5],  # layer2
         [3, 24, 0, 0, 1, 1, 1, 3.67]],
        [[5, 40, 0.25, 1, 2, 1, 1, 4],  # layer3
         [5, 40, 0.25, 1, 1, 1, 1, 6],
         [5, 40, 0.25, 1, 1, 1, 1, 6],
         [5, 48, 0.25, 1, 1, 1, 1, 3],
         [5, 48, 0.25, 1, 1, 1, 1, 3]],
        [[5, 96, 0.25, 1, 2, 1, 1, 6],  # layer4
         [5, 96, 0.25, 1, 1, 1, 1, 6],
         [5, 96, 0.25, 1, 1, 1, 1, 6]],
        [[1, 576, 0.25, 1, 1, 0, 0, 0],
         [1, 1280, 0, 1, 1, 0, 0, 0]]  # head (conv_out)
    ],
    'small_official': [
        [[3, 16, 0, 1, 2, 0, 0, 0]],  # stem (conv1)
        [[0, 16, 0, 0, 0, 0, 0, 0]],  # layer0
        [[3, 16, 0.25, 0, 2, 1, 1, 1]],  # layer1
        [[3, 24, 0, 0, 2, 1, 1, 4.5],  # layer2
         [3, 24, 0, 0, 1, 1, 1, 3.67]],
        [[5, 40, 0.25, 1, 2, 1, 1, 4],  # layer3
         [5, 40, 0.25, 1, 1, 1, 1, 6],
         [5, 40, 0.25, 1, 1, 1, 1, 6],
         [5, 48, 0.25, 1, 1, 1, 1, 3],
         [5, 48, 0.25, 1, 1, 1, 1, 3]],
        [[5, 96, 0.25, 1, 2, 1, 1, 6],  # layer4
         [5, 96, 0.25, 1, 1, 1, 1, 6],
         [5, 96, 0.25, 1, 1, 1, 1, 6]],
        [[1, 576, 0, 1, 1, 0, 0, 0],
         [1, 1024, 0, 1, 1, 0, 0, 0]]  # head (conv_out)
    ],
    'small_minimal_official': [
        [[3, 16, 0, 0, 2, 0, 0, 0]],  # stem (conv1)
        [[0, 16, 0, 0, 0, 0, 0, 0]],  # layer0
        [[3, 16, 0, 0, 2, 1, 1, 1]],  # layer1
        [[3, 24, 0, 0, 2, 1, 1, 4.5],  # layer2
         [3, 24, 0, 0, 1, 1, 1, 3.67]],
        [[3, 40, 0, 0, 2, 1, 1, 4],  # layer3
         [3, 40, 0, 0, 1, 1, 1, 6],
         [3, 40, 0, 0, 1, 1, 1, 6],
         [3, 48, 0, 0, 1, 1, 1, 3],
         [3, 48, 0, 0, 1, 1, 1, 3]],
        [[3, 96, 0, 0, 2, 1, 1, 6],  # layer4
         [3, 96, 0, 0, 1, 1, 1, 6],
         [3, 96, 0, 0, 1, 1, 1, 6]],
        [[1, 576, 0, 0, 1, 0, 0, 0],
         [1, 1024, 0, 0, 1, 0, 0, 0]]  # head (conv_out)
    ],
    'moga_a': [
        [[3, 16, 0, 1, 2, 0, 0, 0]],  # stem (conv1)
        [[3, 16, 0, 0, 1, 1, 1, 1]],  # layer0
        [[5, 24, 0, 0, 2, 1, 1, 6],  # layer1
         [7, 24, 0, 0, 1, 1, 1, 6]],
        [[3, 40, 0, 0, 2, 1, 1, 6],  # layer2
         [3, 40, 0.25, 0, 1, 1, 1, 6],
         [3, 40, 0.25, 0, 1, 1, 1, 3]],
        [[3, 80, 0.25, 1, 2, 1, 1, 6],  # layer3
         [3, 80, 0, 1, 1, 1, 1, 6],
         [7, 80, 0, 1, 1, 1, 1, 6],
         [7, 80, 0.25, 1, 1, 1, 1, 3],
         [7, 112, 0, 1, 1, 1, 1, 6],
         [3, 112, 0, 1, 1, 1, 1, 6]],
        [[3, 160, 0, 1, 2, 1, 1, 6],  # layer4
         [5, 160, 0.25, 1, 1, 1, 1, 6],
         [5, 160, 0.25, 1, 1, 1, 1, 6]],
        [[1, 960, 0, 1, 1, 0, 0, 0],
         [1, 1280, 0, 1, 1, 0, 0, 0]]  # head (conv_out)
    ],
    'moga_b': [
        [[3, 16, 0, 1, 2, 0, 0, 0]],  # stem (conv1)
        [[3, 16, 0, 0, 1, 1, 1, 1]],  # layer0
        [[3, 24, 0, 0, 2, 1, 1, 3],  # layer1
         [3, 24, 0, 0, 1, 1, 1, 3]],
        [[7, 40, 0, 0, 2, 1, 1, 6],  # layer2
         [3, 40, 0, 0, 1, 1, 1, 3],
         [5, 40, 0, 0, 1, 1, 1, 6]],
        [[3, 80, 0.25, 1, 2, 1, 1, 6],  # layer3
         [5, 80, 0.25, 1, 1, 1, 1, 6],
         [3, 80, 0, 1, 1, 1, 1, 3],
         [7, 80, 0.25, 1, 1, 1, 1, 6],
         [7, 112, 0, 1, 1, 1, 1, 6],
         [5, 112, 0, 1, 1, 1, 1, 3]],
        [[7, 160, 0.25, 1, 2, 1, 1, 6],  # layer4
         [7, 160, 0.25, 1, 1, 1, 1, 6],
         [3, 160, 0.25, 1, 1, 1, 1, 6]],
        [[1, 960, 0, 1, 1, 0, 0, 0],
         [1, 1280, 0, 1, 1, 0, 0, 0]]  # head (conv_out)
    ],
    'moga_c': [
        [[3, 16, 0, 1, 2, 0, 0, 0]],  # stem (conv1)
        [[3, 16, 0, 0, 1, 1, 1, 1]],  # layer0
        [[5, 24, 0, 0, 2, 1, 1, 3],  # layer1
         [3, 24, 0, 0, 1, 1, 1, 3]],
        [[5, 40, 0, 0, 2, 1, 1, 3],  # layer2
         [3, 40, 0, 0, 1, 1, 1, 3],
         [5, 40, 0, 0, 1, 1, 1, 3]],
        [[5, 80, 0, 1, 2, 1, 1, 3],  # layer3
         [5, 80, 0.25, 1, 1, 1, 1, 6],
         [5, 80, 0, 1, 1, 1, 1, 3],
         [5, 80, 0, 1, 1, 1, 1, 3],
         [3, 112, 0, 1, 1, 1, 1, 6],
         [3, 112, 0.25, 1, 1, 1, 1, 6]],
        [[3, 160, 0.25, 1, 2, 1, 1, 6],  # layer4
         [3, 160, 0.25, 1, 1, 1, 1, 6],
         [3, 160, 0.25, 1, 1, 1, 1, 6]],
        [[1, 960, 0, 1, 1, 0, 0, 0],
         [1, 1280, 0, 1, 1, 0, 0, 0]]  # head (conv_out)
    ]
}


class H_Swish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class H_Sigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeConv2d(nn.Module):
    def __init__(self, planse, innerplanse, activation=nn.ReLU):
        super(SeConv2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(planse, innerplanse, kernel_size=1),
            activation(),
            nn.Conv2d(innerplanse, planse, kernel_size=1),
            H_Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, dilation=1, kernel=3, groups=(1, 1), t=6, norm='bn', bn_eps=1e-5,
                 se_ratio=0, activation=nn.ReLU6, se_reduce_mid=False, se_divisible=False, force_residual=False,
                 sync_se_act=True):
        super(LinearBottleneck, self).__init__()
        padding = (dilation * kernel - dilation) // 2
        self.stride = stride
        self.inplanes, self.outplanes, innerplanes = int(inplanes), int(outplanes), make_divisible(inplanes * t, 8)
        self.t = t
        self.force_residual = force_residual
        if self.t != 1:
            self.conv1 = nn.Conv2d(self.inplanes, innerplanes, kernel_size=1, padding=0, stride=1, groups=groups[0],
                                   bias=False)
            self.bn1 = make_norm(innerplanes, norm=norm, eps=bn_eps)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=kernel, padding=padding, stride=stride,
                               dilation=dilation, groups=innerplanes, bias=False)
        self.bn2 = make_norm(innerplanes, norm=norm, eps=bn_eps)
        se_base_chs = innerplanes if se_reduce_mid else self.inplanes
        se_innerplanse = make_divisible(se_base_chs * se_ratio, 8) if se_divisible else int(se_base_chs * se_ratio)
        self.se = SeConv2d(innerplanes, se_innerplanse, activation if sync_se_act else nn.ReLU) if se_ratio else None
        self.conv3 = nn.Conv2d(innerplanes, self.outplanes, kernel_size=1, padding=0, stride=1, groups=groups[1],
                               bias=False)
        self.bn3 = make_norm(self.outplanes, norm=norm, eps=bn_eps)
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()

    def forward(self, x):
        if self.stride == 1 and self.inplanes == self.outplanes and (self.t != 1 or self.force_residual):
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


class MobileNetV3(nn.Module):
    def __init__(self, setting='large', widen_factor=1.0, norm='bn', activation=H_Swish, drop_rate=0.0,
                 num_classes=1000):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNetV3, self).__init__()
        block = LinearBottleneck
        self.widen_factor = widen_factor
        self.norm = norm
        self.se_reduce_mid = True
        self.se_divisible = False
        self.head_use_bias = False
        self.force_residual = False
        self.sync_se_act = True
        self.bn_eps = 1e-5
        self.drop_rate = drop_rate
        self.activation_type = activation

        layers_cfg = MV3_CFG[setting]
        num_of_channels = [lc[-1][1] for lc in layers_cfg[1:-1]]
        self.channels = [make_divisible(ch * self.widen_factor, 8) for ch in num_of_channels]
        self.activation = activation() if layers_cfg[0][0][3] else nn.ReLU(inplace=True)

        self.inplanes = make_divisible(layers_cfg[0][0][1] * self.widen_factor, 8)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=layers_cfg[0][0][0], stride=layers_cfg[0][0][4],
                               padding=layers_cfg[0][0][0] // 2, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=self.norm, eps=self.bn_eps)

        self.layer0 = self._make_layer(block, layers_cfg[1], dilation=1) if layers_cfg[1][0][0] else None
        self.layer1 = self._make_layer(block, layers_cfg[2], dilation=1)
        self.layer2 = self._make_layer(block, layers_cfg[3], dilation=1)
        self.layer3 = self._make_layer(block, layers_cfg[4], dilation=1)
        self.layer4 = self._make_layer(block, layers_cfg[5], dilation=1)

        last_ch = make_divisible(layers_cfg[-1][0][1] * self.widen_factor, 8)
        self.last_stage = nn.Sequential(
            nn.Conv2d(self.inplanes, last_ch, kernel_size=layers_cfg[-1][0][0], stride=layers_cfg[-1][0][4],
                      padding=layers_cfg[-1][0][0] // 2, bias=False),
            make_norm(last_ch, norm=self.norm, eps=self.bn_eps),
            activation() if layers_cfg[-1][0][3] else nn.ReLU(inplace=True),
            SeConv2d(last_ch, int(last_ch * layers_cfg[-1][0][2]), activation) if layers_cfg[-1][0][2] else nn.Identity()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(last_ch, layers_cfg[-1][1][1], kernel_size=layers_cfg[-1][1][0], stride=layers_cfg[-1][1][4],
                      padding=layers_cfg[-1][1][0] // 2, bias=self.head_use_bias),
            activation() if layers_cfg[-1][1][3] else nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(layers_cfg[-1][1][1], num_classes)

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
                                t=lc[i][7], norm=self.norm, bn_eps=self.bn_eps, se_ratio=lc[i][2],
                                activation=self.activation_type if lc[i][3] else nn.ReLU,
                                se_reduce_mid=self.se_reduce_mid, se_divisible=self.se_divisible,
                                force_residual=self.force_residual, sync_se_act=self.sync_se_act))
            self.inplanes = make_divisible(lc[i][1] * self.widen_factor, 8)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        if self.layer0 is not None:
            x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.last_stage(x)
        x = self.avgpool(x)
        x = self.conv_out(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
