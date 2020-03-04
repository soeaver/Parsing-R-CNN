import math

import torch.nn as nn
import torch.nn.functional as F

import models.imagenet.resnet as res
import models.ops as ops
from models.imagenet.utils import convert_conv2convws_model
from utils.net import freeze_params, make_norm
from rcnn.utils.poolers import Pooler
from rcnn.modeling import registry
from rcnn.core.config import cfg


def get_norm():
    norm = 'bn'
    if cfg.BACKBONE.RESNET.USE_GN:
        norm = 'gn'
    if cfg.BACKBONE.RESNET.USE_AN:
        norm = 'an_' + norm  # an_bn or an_gn
    return norm


class ResNet(res.ResNet):
    def __init__(self, norm='bn', stride=32):
        """ Constructor
        """
        super(ResNet, self).__init__()
        if cfg.BACKBONE.RESNET.USE_ALIGN:
            block = res.AlignedBottleneck
        else:
            if cfg.BACKBONE.RESNET.BOTTLENECK:
                block = res.Bottleneck  # not use the original Bottleneck module
            else:
                block = res.BasicBlock
        self.expansion = block.expansion
        self.stride_3x3 = cfg.BACKBONE.RESNET.STRIDE_3X3
        self.avg_down = cfg.BACKBONE.RESNET.AVG_DOWN
        self.norm = norm
        self.stride = stride
        
        layers = cfg.BACKBONE.RESNET.LAYERS[:int(math.log(stride, 2)) - 1]
        self.layers = layers
        self.base_width = cfg.BACKBONE.RESNET.WIDTH
        stage_with_context = cfg.BACKBONE.RESNET.STAGE_WITH_CONTEXT
        self.ctx_ratio = cfg.BACKBONE.RESNET.CTX_RATIO
        stage_with_conv = cfg.BACKBONE.RESNET.STAGE_WITH_CONV
        c5_dilation = cfg.BACKBONE.RESNET.C5_DILATION

        self.inplanes = 64
        self.use_3x3x3stem = cfg.BACKBONE.RESNET.USE_3x3x3HEAD
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

        if len(layers) == 4:
            if c5_dilation != 1:
                c5_stride = 1
            else:
                c5_stride = 2
            self.layer4 = self._make_layer(block, 512, layers[3], c5_stride, dilation=c5_dilation,
                                           conv=stage_with_conv[3], context=stage_with_context[3])
            self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32. * c5_dilation]
        else:
            del self.layer4
            self.spatial_scale = [1 / 4., 1 / 8., 1 / 16.]

        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

        del self.avgpool
        del self.fc
        self._init_weights()
        self._init_modules()

    def _init_modules(self):
        assert cfg.BACKBONE.RESNET.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.BACKBONE.RESNET.FREEZE_AT: 2
        assert cfg.BACKBONE.RESNET.FREEZE_AT <= len(self.layers) + 1
        if cfg.BACKBONE.RESNET.FREEZE_AT > 0:
            freeze_params(getattr(self, 'conv1'))
            freeze_params(getattr(self, 'bn1'))
            if self.use_3x3x3stem:
                freeze_params(getattr(self, 'conv2'))
                freeze_params(getattr(self, 'bn2'))
                freeze_params(getattr(self, 'conv3'))
                freeze_params(getattr(self, 'bn3'))
        for i in range(1, cfg.BACKBONE.RESNET.FREEZE_AT):
            freeze_params(getattr(self, 'layer%d' % i))
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, ops.AffineChannel2d) else None)

    def train(self, mode=True):
        # Override train mode
        self.training = mode
        if cfg.BACKBONE.RESNET.FREEZE_AT < 1:
            getattr(self, 'conv1').train(mode)
            getattr(self, 'bn1').train(mode)
            if self.use_3x3x3stem:
                getattr(self, 'conv2').train(mode)
                getattr(self, 'bn2').train(mode)
                getattr(self, 'conv3').train(mode)
                getattr(self, 'bn3').train(mode)
        for i in range(cfg.BACKBONE.RESNET.FREEZE_AT, len(self.layers) + 1):
            if i == 0:
                continue
            getattr(self, 'layer%d' % i).train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                
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
        
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        if len(self.layers) == 4:
            x5 = self.layer4(x4)
            return [x2, x3, x4, x5]
        else:
            return [x2, x3, x4]


class ResNet_C5_Head(res.ResNet):
    def __init__(self, dim_in, spatial_scale, norm='bn'):
        super().__init__()
        self.dim_in = dim_in[-1]

        if cfg.BACKBONE.RESNET.USE_ALIGN:
            block = res.AlignedBottleneck
        else:
            if cfg.BACKBONE.RESNET.BOTTLENECK:
                block = res.Bottleneck  # not use the original Bottleneck module
            else:
                block = res.BasicBlock
        self.expansion = block.expansion
        self.stride_3x3 = cfg.BACKBONE.RESNET.STRIDE_3X3
        self.avg_down = cfg.BACKBONE.RESNET.AVG_DOWN
        self.norm = norm

        layers = cfg.BACKBONE.RESNET.LAYERS
        self.base_width = cfg.BACKBONE.RESNET.WIDTH
        stage_with_context = cfg.BACKBONE.RESNET.STAGE_WITH_CONTEXT
        self.ctx_ratio = cfg.BACKBONE.RESNET.CTX_RATIO
        stage_with_conv = cfg.BACKBONE.RESNET.STAGE_WITH_CONV
        c5_dilation = cfg.BACKBONE.RESNET.C5_DILATION

        method = cfg.FAST_RCNN.ROI_XFORM_METHOD
        resolution = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        self.inplanes = self.dim_in
        c5_stride = min(resolution) // 7
        self.layer4 = self._make_layer(block, 512, layers[3], c5_stride, dilation=c5_dilation,
                                       conv=stage_with_conv[3], context=stage_with_context[3])
        self.dim_out = self.stage_out_dim[-1]

        del self.conv1
        del self.bn1
        del self.relu
        del self.maxpool
        del self.layer1
        del self.layer2
        del self.layer3
        del self.avgpool
        del self.fc
        self._init_weights()

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.layer4(x)

        return x


class ResNet_2mlp_Head(res.ResNet):
    def __init__(self, dim_in, spatial_scale, norm='bn'):
        super().__init__()
        self.dim_in = dim_in[-1]

        if cfg.BACKBONE.RESNET.USE_ALIGN:
            block = res.AlignedBottleneck
        else:
            if cfg.BACKBONE.RESNET.BOTTLENECK:
                block = res.Bottleneck  # not use the original Bottleneck module
            else:
                block = res.BasicBlock
        self.expansion = block.expansion
        self.stride_3x3 = cfg.BACKBONE.RESNET.STRIDE_3X3
        self.avg_down = cfg.BACKBONE.RESNET.AVG_DOWN
        self.norm = norm

        layers = cfg.BACKBONE.RESNET.LAYERS
        self.base_width = cfg.BACKBONE.RESNET.WIDTH
        stage_with_context = cfg.BACKBONE.RESNET.STAGE_WITH_CONTEXT
        self.ctx_ratio = cfg.BACKBONE.RESNET.CTX_RATIO
        stage_with_conv = cfg.BACKBONE.RESNET.STAGE_WITH_CONV
        c5_dilation = cfg.BACKBONE.RESNET.C5_DILATION

        self.inplanes = self.dim_in
        c5_stride = 2 if c5_dilation == 1 else 1
        self.layer4 = self._make_layer(block, 512, layers[3], c5_stride, dilation=c5_dilation,
                                       conv=stage_with_conv[3], context=stage_with_context[3])
        self.conv_new = nn.Sequential(
            nn.Conv2d(512 * self.expansion, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.dim_in = 256
        method = cfg.FAST_RCNN.ROI_XFORM_METHOD
        resolution = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        input_size = self.dim_in * resolution[0] * resolution[1]
        mlp_dim = cfg.FAST_RCNN.MLP_HEAD.MLP_DIM
        self.fc1 = nn.Linear(input_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.dim_out = mlp_dim

        del self.conv1
        del self.bn1
        del self.relu
        del self.maxpool
        del self.layer1
        del self.layer2
        del self.layer3
        del self.avgpool
        del self.fc
        self._init_weights()

    def forward(self, x, proposals):
        x = self.layer4(x[0])
        x = self.conv_new(x)

        x = self.pooler([x], proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


# ---------------------------------------------------------------------------- #
# ResNet Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("resnet")
def resnet():
    model = ResNet(norm=get_norm())
    if cfg.BACKBONE.RESNET.USE_WS:
        model = convert_conv2convws_model(model)
    return model


@registry.BACKBONES.register("resnet_c4")
def resnet():
    model = ResNet(norm=get_norm(), stride=16)
    if cfg.BACKBONE.RESNET.USE_WS:
        model = convert_conv2convws_model(model)
    return model


# ---------------------------------------------------------------------------- #
# ResNet C5 Head
# ---------------------------------------------------------------------------- #
@registry.ROI_BOX_HEADS.register("resnet_c5_head")
def resnet_c5_head(dim_in, spatial_scale):
    model = ResNet_C5_Head(dim_in, spatial_scale, norm=get_norm())
    if cfg.BACKBONE.RESNET.USE_WS:
        model = convert_conv2convws_model(model)
    return model


# ---------------------------------------------------------------------------- #
# ResNet 2mlp Head
# ---------------------------------------------------------------------------- #
@registry.ROI_BOX_HEADS.register("resnet_2mlp_head")
def resnet_2mlp_head(dim_in, spatial_scale):
    model = ResNet_2mlp_Head(dim_in, spatial_scale, norm=get_norm())
    if cfg.BACKBONE.RESNET.USE_WS:
        model = convert_conv2convws_model(model)
    return model
