import math

import torch.nn as nn

import models.imagenet.resnext as resx
import models.ops as ops
from utils.net import freeze_params, make_norm
from rcnn.modeling import registry
from rcnn.core.config import cfg


def get_norm():
    norm = 'bn'
    if cfg.BACKBONE.RESNEXT.USE_GN:
        norm = 'gn'
    return norm


class ResNeXt(resx.ResNeXt):
    def __init__(self, norm='bn', stride=32):
        """ Constructor
        """
        super(ResNeXt, self).__init__()
        if cfg.BACKBONE.RESNEXT.USE_ALIGN:
            block = resx.AlignedBottleneck
        else:
            block = resx.Bottleneck
        self.expansion = block.expansion
        self.avg_down = cfg.BACKBONE.RESNEXT.AVG_DOWN
        self.norm = norm
        self.cardinality = cfg.BACKBONE.RESNEXT.C
        self.base_width = cfg.BACKBONE.RESNEXT.WIDTH
        self.stride = stride

        layers = cfg.BACKBONE.RESNEXT.LAYERS
        self.layers = layers
        stage_with_context = cfg.BACKBONE.RESNEXT.STAGE_WITH_CONTEXT
        self.ctx_ratio = cfg.BACKBONE.RESNEXT.CTX_RATIO
        stage_with_conv = cfg.BACKBONE.RESNEXT.STAGE_WITH_CONV
        c5_dilation = cfg.BACKBONE.RESNEXT.C5_DILATION

        self.inplanes = 64  # default 64
        self.use_3x3x3stem = cfg.BACKBONE.RESNEXT.USE_3x3x3HEAD
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
        self.layer4 = self._make_layer(block, 512, layers[3], 1 if c5_dilation != 1 else 2, dilation=c5_dilation,
                                       conv=stage_with_conv[3], context=stage_with_context[3])

        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32. * c5_dilation]
        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

        del self.avgpool
        del self.fc
        self._init_weights()
        self._init_modules()

    def _init_modules(self):
        assert cfg.BACKBONE.RESNEXT.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.BACKBONE.RESNEXT.FREEZE_AT: 2
        assert cfg.BACKBONE.RESNEXT.FREEZE_AT <= len(self.layers) + 1
        if cfg.BACKBONE.RESNEXT.FREEZE_AT > 0:
            freeze_params(getattr(self, 'conv1'))
            freeze_params(getattr(self, 'bn1'))
            if self.use_3x3x3stem:
                freeze_params(getattr(self, 'conv2'))
                freeze_params(getattr(self, 'bn2'))
                freeze_params(getattr(self, 'conv3'))
                freeze_params(getattr(self, 'bn3'))
        for i in range(1, cfg.BACKBONE.RESNEXT.FREEZE_AT):
            freeze_params(getattr(self, 'layer%d' % i))
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, ops.AffineChannel2d) else None)

    def train(self, mode=True):
        # Override train mode
        self.training = mode
        if cfg.BACKBONE.RESNEXT.FREEZE_AT < 1:
            getattr(self, 'conv1').train(mode)
            getattr(self, 'bn1').train(mode)
            if self.use_3x3x3stem:
                getattr(self, 'conv2').train(mode)
                getattr(self, 'bn2').train(mode)
                getattr(self, 'conv3').train(mode)
                getattr(self, 'bn3').train(mode)
        for i in range(cfg.BACKBONE.RESNEXT.FREEZE_AT, len(self.layers) + 1):
            if i == 0:
                continue
            getattr(self, 'layer%d' % i).train(mode)

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
        x5 = self.layer4(x4)

        return [x2, x3, x4, x5]


# ---------------------------------------------------------------------------- #
# ResNeXt Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("resnext")
def resnext():
    model = ResNeXt(norm=get_norm())
    return model
