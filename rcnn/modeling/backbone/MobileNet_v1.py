import math

import torch.nn as nn

import models.imagenet.mobilenet_v1 as mv1
import models.ops as ops
from models.imagenet.utils import make_divisible
from utils.net import freeze_params, make_norm
from rcnn.modeling import registry
from rcnn.core.config import cfg


class MobileNetV1(mv1.MobileNetV1):
    def __init__(self, norm='bn', activation=nn.ReLU, stride=32):
        """ Constructor
        """
        super(MobileNetV1, self).__init__()
        block = mv1.BasicBlock
        self.use_se = cfg.BACKBONE.MV1.USE_SE
        self.norm = norm
        self.activation_type = activation
        try:
            self.activation = activation(inplace=True)
        except:
            self.activation = activation()
        self.stride = stride

        layers = cfg.BACKBONE.MV1.LAYERS[:int(math.log(stride, 2)) - 1]
        self.layers = layers
        kernel = cfg.BACKBONE.MV1.KERNEL
        c5_dilation = cfg.BACKBONE.MV1.C5_DILATION
        num_of_channels = cfg.BACKBONE.MV1.NUM_CHANNELS
        channels = [make_divisible(ch * cfg.BACKBONE.MV1.WIDEN_FACTOR, 8) for ch in num_of_channels]
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
        self.layer4 = self._make_layer(block, channels[5], layers[3], stride=1 if c5_dilation != 1 else 2,
                                       dilation=c5_dilation, kernel=kernel)

        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32. * c5_dilation]
        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

        self.dropblock = ops.DropBlock2D(keep_prob=0.9, block_size=7) if cfg.BACKBONE.MV1.USE_DP else None

        del self.avgpool
        del self.fc
        self._init_weights()
        self._init_modules()

    def _init_modules(self):
        assert cfg.BACKBONE.MV1.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.BACKBONE.MV1.FREEZE_AT: 2
        assert cfg.BACKBONE.MV1.FREEZE_AT <= len(self.layers) + 1
        if cfg.BACKBONE.MV1.FREEZE_AT > 0:
            freeze_params(getattr(self, 'conv1'))
            freeze_params(getattr(self, 'bn1'))
            freeze_params(getattr(self, 'conv2'))
            freeze_params(getattr(self, 'bn2'))
            freeze_params(getattr(self, 'conv3'))
            freeze_params(getattr(self, 'bn3'))
        for i in range(1, cfg.BACKBONE.MV1.FREEZE_AT):
            freeze_params(getattr(self, 'layer%d' % i))
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, ops.AffineChannel2d) else None)

    def train(self, mode=True):
        # Override train mode
        self.training = mode
        if cfg.BACKBONE.MV1.FREEZE_AT < 1:
            getattr(self, 'conv1').train(mode)
            getattr(self, 'bn1').train(mode)
            getattr(self, 'conv2').train(mode)
            getattr(self, 'bn2').train(mode)
            getattr(self, 'conv3').train(mode)
            getattr(self, 'bn3').train(mode)
        for i in range(cfg.BACKBONE.MV1.FREEZE_AT, len(self.layers) + 1):
            if i == 0:
                continue
            getattr(self, 'layer%d' % i).train(mode)

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

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        if self.dropblock is not None:
            x4 = self.dropblock(x4)
        x5 = self.layer4(x4)
        if self.dropblock is not None:
            x5 = self.dropblock(x5)
            
        return [x2, x3, x4, x5]


# ---------------------------------------------------------------------------- #
# MobileNetV1 Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("mobilenet_v1")
def mobilenet_v1():
    model = MobileNetV1()
    return model
