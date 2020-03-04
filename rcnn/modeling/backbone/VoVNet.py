import math

import torch.nn as nn

import models.imagenet.vovnet as vov
import models.ops as ops
from utils.net import freeze_params, make_norm
from rcnn.modeling import registry
from rcnn.core.config import cfg


def get_norm():
    norm = 'bn'
    if cfg.BACKBONE.VOV.USE_GN:
        norm = 'gn'
    return norm


class VoVNet(vov.VoVNet):
    def __init__(self, norm='bn', stride=32):
        """ Constructor
        """
        super(VoVNet, self).__init__()
        block = vov.OSABlock
        self.num_conv = cfg.BACKBONE.VOV.NUM_CONV   # 5
        self.norm = norm
        self.stride = stride

        base_width = cfg.BACKBONE.VOV.WIDTH    # 64
        stage_dims = cfg.BACKBONE.VOV.STAGE_DIMS
        concat_dims = cfg.BACKBONE.VOV.CONCAT_DIMS
        layers = cfg.BACKBONE.VOV.LAYERS
        self.layers = layers
        stage_with_conv = cfg.BACKBONE.VOV.STAGE_WITH_CONV
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

        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32.]
        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

        del self.avgpool
        del self.fc
        self._init_weights()
        self._init_modules()

    def _init_modules(self):
        assert cfg.BACKBONE.VOV.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.BACKBONE.VOV.FREEZE_AT: 2
        assert cfg.BACKBONE.VOV.FREEZE_AT <= len(self.layers) + 1
        if cfg.BACKBONE.VOV.FREEZE_AT > 0:
            freeze_params(getattr(self, 'conv1'))
            freeze_params(getattr(self, 'bn1'))
            freeze_params(getattr(self, 'conv2'))
            freeze_params(getattr(self, 'bn2'))
            freeze_params(getattr(self, 'conv3'))
            freeze_params(getattr(self, 'bn3'))
        for i in range(1, cfg.BACKBONE.VOV.FREEZE_AT):
            freeze_params(getattr(self, 'layer%d' % i))
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, ops.AffineChannel2d) else None)

    def train(self, mode=True):
        # Override train mode
        self.training = mode
        if cfg.BACKBONE.VOV.FREEZE_AT < 1:
            getattr(self, 'conv1').train(mode)
            getattr(self, 'bn1').train(mode)
            getattr(self, 'conv2').train(mode)
            getattr(self, 'bn2').train(mode)
            getattr(self, 'conv3').train(mode)
            getattr(self, 'bn3').train(mode)
        for i in range(cfg.BACKBONE.VOV.FREEZE_AT, len(self.layers) + 1):
            if i == 0:
                continue
            getattr(self, 'layer%d' % i).train(mode)

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

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return [x2, x3, x4, x5]

# ---------------------------------------------------------------------------- #
# VoVNet Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("vovnet")
def vovnet():
    model = VoVNet(norm=get_norm())
    return model
