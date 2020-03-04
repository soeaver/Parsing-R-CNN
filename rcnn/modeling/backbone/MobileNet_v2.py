import math

import torch.nn as nn

import models.imagenet.mobilenet_v2 as mv2
import models.ops as ops
from models.imagenet.utils import make_divisible
from utils.net import freeze_params, make_norm
from rcnn.modeling import registry
from rcnn.core.config import cfg


class MobileNetV2(mv2.MobileNetV2):
    def __init__(self, norm='bn', activation=nn.ReLU6, stride=32):
        """ Constructor
        """
        super(MobileNetV2, self).__init__()
        block = mv2.LinearBottleneck
        self.use_se = cfg.BACKBONE.MV2.USE_SE
        self.widen_factor = cfg.BACKBONE.MV2.WIDEN_FACTOR
        self.norm = norm
        self.activation_type = activation
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()
        self.stride = stride

        layers_cfg = mv2.model_se(mv2.MV2_CFG['A']) if self.use_se else mv2.MV2_CFG['A']
        num_of_channels = [lc[-1][1] for lc in layers_cfg[1:-1]]
        self.channels = [make_divisible(ch * self.widen_factor, 8) for ch in num_of_channels]
        self.layers = [len(lc) for lc in layers_cfg[2:-1]]

        self.inplanes = make_divisible(layers_cfg[0][0][1] * self.widen_factor, 8)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=layers_cfg[0][0][0], stride=layers_cfg[0][0][4],
                               padding=layers_cfg[0][0][0] // 2, bias=False)
        self.bn1 = make_norm(self.inplanes, norm=self.norm)

        self.layer0 = self._make_layer(block, layers_cfg[1], dilation=1)
        self.layer1 = self._make_layer(block, layers_cfg[2], dilation=1)
        self.layer2 = self._make_layer(block, layers_cfg[3], dilation=1)
        self.layer3 = self._make_layer(block, layers_cfg[4], dilation=1)
        self.layer4 = self._make_layer(block, layers_cfg[5], dilation=1)

        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32.]
        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

        del self.conv_out
        del self.bn_out
        del self.avgpool
        del self.fc
        self._init_weights()
        self._init_modules()

    def _init_modules(self):
        assert cfg.BACKBONE.MV2.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.BACKBONE.MV2.FREEZE_AT: 2
        assert cfg.BACKBONE.MV2.FREEZE_AT <= len(self.layers) + 1
        if cfg.BACKBONE.MV2.FREEZE_AT > 0:
            freeze_params(getattr(self, 'conv1'))
            freeze_params(getattr(self, 'bn1'))
        for i in range(0, cfg.BACKBONE.MV2.FREEZE_AT):
            freeze_params(getattr(self, 'layer%d' % i))
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, ops.AffineChannel2d) else None)

    def train(self, mode=True):
        # Override train mode
        self.training = mode
        if cfg.BACKBONE.MV2.FREEZE_AT < 1:
            getattr(self, 'conv1').train(mode)
            getattr(self, 'bn1').train(mode)
        for i in range(cfg.BACKBONE.MV2.FREEZE_AT, len(self.layers) + 1):
            getattr(self, 'layer%d' % i).train(mode)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer0(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return [x2, x3, x4, x5]


# ---------------------------------------------------------------------------- #
# MobileNetV2 Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("mobilenet_v2")
def mobilenet_v2():
    model = MobileNetV2()
    return model
