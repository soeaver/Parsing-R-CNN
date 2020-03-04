"""
Creates a HRNet Model as defined in:
Ke Sun, Bin Xiao, Dong Liu and Jingdong Wang. (2019 CVPR).
Deep High-Resolution Representation Learning for Human Pose Estimation.
Copyright (c) Yang Lu, 2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.ops as ops
from utils.net import make_norm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm='bn', conv='normal', use_se=False,
                 stride_3x3=False, downsample=None):
        super(BasicBlock, self).__init__()
        if conv == 'normal':
            conv_op = nn.Conv2d
        elif conv == 'deform':
            conv_op = ops.DeformConvPack
        elif conv == 'deformv2':
            conv_op = ops.ModulatedDeformConvPack
        else:
            raise ValueError('{} type conv operation is not supported.'.format(conv))

        self.conv1 = conv_op(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                             bias=False)
        self.bn1 = make_norm(planes, norm=norm)
        self.conv2 = conv_op(planes, planes, kernel_size=3, stride=1, dilation=dilation, padding=dilation,
                             bias=False)
        self.bn2 = make_norm(planes, norm=norm)
        self.se = ops.Se2d(planes, reduction=16) if use_se else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AlignedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm='bn', conv='normal', use_se=False,
                 stride_3x3=False, downsample=None):
        super(AlignedBasicBlock, self).__init__()
        if conv == 'normal':
            conv_op = nn.Conv2d
        elif conv == 'deform':
            conv_op = ops.DeformConvPack
        elif conv == 'deformv2':
            conv_op = ops.ModulatedDeformConvPack
        else:
            raise ValueError('{} type conv operation is not supported.'.format(conv))

        self.conv1_1 = conv_op(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False)
        self.conv2_1 = conv_op(inplanes, planes // 2, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False)
        self.bn2_1 = make_norm(planes // 2, norm=norm)
        self.conv2_2 = conv_op(planes // 2, planes // 2, kernel_size=3, stride=1, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn_concat = make_norm(planes + (planes // 2), norm=norm)

        self.conv = nn.Conv2d(planes + (planes // 2), planes, kernel_size=3, stride=1, dilation=dilation,
                              padding=dilation, bias=False)
        self.bn = make_norm(planes, norm=norm)

        self.se = ops.Se2d(planes, reduction=16) if use_se else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        branch1 = self.conv1_1(x)

        branch2 = self.conv2_1(x)
        branch2 = self.bn2_1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_2(branch2)

        out = torch.cat((branch1, branch2), 1)
        out = self.bn_concat(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.bn(out)
        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm='bn', conv='normal', use_se=False,
                 stride_3x3=False, downsample=None):
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

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = make_norm(planes, norm=norm)
        self.conv2 = conv_op(planes, planes, kernel_size=3, stride=str3x3, dilation=dilation, padding=dilation,
                             bias=False)
        self.bn2 = make_norm(planes, norm=norm)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = make_norm(planes * 4, norm=norm)
        self.se = ops.Se2d(planes * 4, reduction=16) if use_se else None
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
        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AlignedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm='bn', conv='normal', use_se=False,
                 stride_3x3=False, downsample=None):
        super(AlignedBottleneck, self).__init__()
        if conv == 'normal':
            conv_op = nn.Conv2d
        elif conv == 'deform':
            conv_op = ops.DeformConvPack
        elif conv == 'deformv2':
            conv_op = ops.ModulatedDeformConvPack
        else:
            raise ValueError('{} type conv operation is not supported.'.format(conv))

        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1_1 = make_norm(planes, norm=norm)
        self.conv1_2 = conv_op(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False)
        self.conv2_1 = nn.Conv2d(inplanes, planes // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2_1 = make_norm(planes // 2, norm=norm)
        self.conv2_2 = conv_op(planes // 2, planes // 2, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2_2 = make_norm(planes // 2, norm=norm)
        self.conv2_3 = conv_op(planes // 2, planes // 2, kernel_size=3, stride=1, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn_concat = make_norm(planes + (planes // 2), norm=norm)

        self.conv = nn.Conv2d(planes + (planes // 2), planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = make_norm(planes * 4, norm=norm)

        self.se = ops.Se2d(planes * 4, reduction=16) if use_se else None
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
        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, block, planes, norm='bn', conv='normal', use_se=False, use_global=False, stage=2,
                 output_branches=2):
        super(StageModule, self).__init__()
        self.use_global = use_global

        self.branches = nn.ModuleList()
        for i in range(stage):
            w = planes * (2 ** i)
            branch = nn.Sequential(
                block(w, w, 1, 1, norm, conv, use_se, True),
                block(w, w, 1, 1, norm, conv, use_se, True),
                block(w, w, 1, 1, norm, conv, use_se, True),
                block(w, w, 1, 1, norm, conv, use_se, True),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        if self.use_global:
            self.global_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(planes * (2 ** j), planes * (2 ** i), 1, 1, 0, bias=False),
                        make_norm(planes * (2 ** i), norm=norm),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(planes * (2 ** j), planes * (2 ** j), 3, 2, 1, bias=False),
                            make_norm(planes * (2 ** j), norm=norm),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(planes * (2 ** j), planes * (2 ** i), 3, 2, 1, bias=False),
                        make_norm(planes * (2 ** i), norm=norm),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
            if self.use_global:
                sum_planes = sum([planes * (2 ** k) for k in range(stage)])
                self.global_layers.append(
                    nn.Sequential(
                        nn.Conv2d(sum_planes, planes * (2 ** i), 1, 1, 0, bias=False),
                        make_norm(planes * (2 ** i), norm=norm),
                        nn.Sigmoid()
                    )
                )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        if self.use_global:
            x_global = [F.adaptive_avg_pool2d(b, 1) for b in x]
            x_global = torch.cat(tuple(x_global), 1)

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])
            if self.use_global:
                x_fused[i] = x_fused[i] * self.global_layers[i](x_global)

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class HRNet(nn.Module):
    def __init__(self, aligned=False, use_se=False, use_global=False, avg_down=False, base_width=32, norm='bn',
                 stage_with_conv=('normal', 'normal', 'normal', 'normal'), num_classes=1000):
        """ Constructor
        Args:
            layers: config of layers, e.g., (3, 4, 23, 3)
            num_classes: number of classes
        """
        super(HRNet, self).__init__()
        block_1 = AlignedBottleneck if aligned else Bottleneck
        block_2 = AlignedBasicBlock if aligned else BasicBlock

        self.use_se = use_se
        self.avg_down = avg_down
        self.base_width = base_width
        self.norm = norm
        self.head_dim = (32, 64, 128, 256)

        self.inplanes = 64  # default 64
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = make_norm(64, norm=self.norm)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = make_norm(64, norm=self.norm)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block_1, 64, 4, 1, conv=stage_with_conv[0])  # 4 blocks without down sample
        self.transition1 = self._make_transition(index=1, stride=2)  # Fusion layer 1: create full and 1/2 resolution

        self.stage2 = nn.Sequential(
            StageModule(block_2, base_width, norm, stage_with_conv[1], use_se, False, stage=2, output_branches=2),
        )  # Stage 2 with 1 group of block modules, which has 2 branches
        self.transition2 = self._make_transition(index=2, stride=2)  # Fusion layer 2: create 1/4 resolution

        self.stage3 = nn.Sequential(
            StageModule(block_2, base_width, norm, stage_with_conv[2], use_se, use_global, stage=3, output_branches=3),
            StageModule(block_2, base_width, norm, stage_with_conv[2], use_se, use_global, stage=3, output_branches=3),
            StageModule(block_2, base_width, norm, stage_with_conv[2], use_se, use_global, stage=3, output_branches=3),
            StageModule(block_2, base_width, norm, stage_with_conv[2], use_se, use_global, stage=3, output_branches=3),
        )  # Stage 3 with 4 groups of block modules, which has 3 branches
        self.transition3 = self._make_transition(index=3, stride=2)  # Fusion layer 3: create 1/8 resolution

        self.stage4 = nn.Sequential(
            StageModule(block_2, base_width, norm, stage_with_conv[3], use_se, use_global, stage=4, output_branches=4),
            StageModule(block_2, base_width, norm, stage_with_conv[3], use_se, use_global, stage=4, output_branches=4),
            StageModule(block_2, base_width, norm, stage_with_conv[3], use_se, use_global, stage=4, output_branches=4),
        )  # Stage 4 with 3 groups of block modules, which has 4 branches

        pre_stage_channels = [base_width, base_width * 2, base_width * 4, base_width * 8]
        self.incre_modules, self.downsamp_modules, self.final_layer = \
            self._make_head(block_1, pre_stage_channels, outplanes=2048, conv=stage_with_conv[3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

        self._init_weights()

    @property
    def stage_out_dim(self):
        return [64, self.base_width, self.base_width * 2, self.base_width * 4, self.base_width * 8]

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
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, AlignedBottleneck):
                nn.init.constant_(m.bn.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, conv='normal'):
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
        layers.append(block(self.inplanes, planes, stride, dilation, self.norm, conv, self.use_se, True,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, dilation, self.norm, conv, self.use_se, True))

        return nn.Sequential(*layers)

    def _make_transition(self, index=1, stride=1):
        transition = nn.ModuleList()
        if index == 1:
            transition.append(nn.Sequential(
                nn.Conv2d(self.inplanes, self.base_width, kernel_size=3, stride=1, padding=1, bias=False),
                make_norm(self.base_width, norm=self.norm),
                nn.ReLU(inplace=True),
            ))
        else:
            transition.extend([nn.Sequential() for _ in range(index)])
        transition.append(nn.Sequential(
            nn.Sequential(  # Double Sequential to fit with official pre-trained weights
                nn.Conv2d(self.inplanes if index == 1 else self.base_width * (2 ** (index - 1)),
                          self.base_width * (2 ** index), kernel_size=3, stride=stride, padding=1, bias=False),
                make_norm(self.base_width * (2 ** index), norm=self.norm),
                nn.ReLU(inplace=True),
            )
        ))

        return transition

    def _make_head(self, block, pre_stage_channels, outplanes=2048, conv='normal'):
        # Increasing the #channels on each resolution, from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            self.inplanes = channels
            incre_module = self._make_layer(block, self.head_dim[i], 1, stride=1, dilation=1, conv=conv)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = self.head_dim[i] * block.expansion
            out_channels = self.head_dim[i + 1] * block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),  # official implementation forgets bias=False
                make_norm(out_channels, norm=self.norm),
                nn.ReLU(inplace=True)
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(self.head_dim[3] * block.expansion, outplanes, 1, 1, 0),
            make_norm(outplanes, norm=self.norm),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        y = self.incre_modules[0](x[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](x[i + 1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)

        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)

        return y
