import math

import torch.nn as nn

import models.imagenet.hrnet as hr
import models.ops as ops
from utils.net import freeze_params, make_norm
from rcnn.modeling import registry
from rcnn.core.config import cfg


def get_norm():
    norm = 'bn'
    if cfg.BACKBONE.HRNET.USE_GN:
        norm = 'gn'
    return norm


class HRNet(hr.HRNet):
    def __init__(self, norm='bn', stride=32):
        """ Constructor
        """
        super(HRNet, self).__init__()
        block_1 = hr.AlignedBottleneck if cfg.BACKBONE.HRNET.USE_ALIGN else hr.Bottleneck
        block_2 = hr.AlignedBasicBlock if cfg.BACKBONE.HRNET.USE_ALIGN else hr.BasicBlock

        use_se = cfg.BACKBONE.HRNET.USE_SE
        self.use_se = use_se
        self.use_global = cfg.BACKBONE.HRNET.USE_GLOBAL
        self.avg_down = cfg.BACKBONE.HRNET.AVG_DOWN
        base_width = cfg.BACKBONE.HRNET.WIDTH
        self.base_width = base_width
        self.norm = norm
        self.stride = stride

        stage_with_conv = cfg.BACKBONE.HRNET.STAGE_WITH_CONV

        self.inplanes = 64  # default 64
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = make_norm(64, norm=self.norm)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = make_norm(64, norm=self.norm)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block_1, 64, 4, 1, conv=stage_with_conv[0])  # 4 blocks without down sample
        self.transition1 = self._make_transition(index=1, stride=2)  # Fusion layer 1: create full and 1/2 resolution

        self.stage2 = nn.Sequential(
            hr.StageModule(block_2, base_width, norm, stage_with_conv[1], use_se, False, 2, 2),
        )  # Stage 2 with 1 group of block modules, which has 2 branches
        self.transition2 = self._make_transition(index=2, stride=2)  # Fusion layer 2: create 1/4 resolution

        self.stage3 = nn.Sequential(
            hr.StageModule(block_2, base_width, norm, stage_with_conv[2], use_se, self.use_global, 3, 3),
            hr.StageModule(block_2, base_width, norm, stage_with_conv[2], use_se, self.use_global, 3, 3),
            hr.StageModule(block_2, base_width, norm, stage_with_conv[2], use_se, self.use_global, 3, 3),
            hr.StageModule(block_2, base_width, norm, stage_with_conv[2], use_se, self.use_global, 3, 3),
        )  # Stage 3 with 4 groups of block modules, which has 3 branches
        self.transition3 = self._make_transition(index=3, stride=2)  # Fusion layer 3: create 1/8 resolution

        self.stage4 = nn.Sequential(
            hr.StageModule(block_2, base_width, norm, stage_with_conv[3], use_se, self.use_global, 4, 4),
            hr.StageModule(block_2, base_width, norm, stage_with_conv[3], use_se, self.use_global, 4, 4),
            hr.StageModule(block_2, base_width, norm, stage_with_conv[3], use_se, self.use_global, 4, 4),   # multi out
        )  # Stage 4 with 3 groups of block modules, which has 4 branches

        self.spatial_scale = [1 / 4., 1 / 8., 1 / 16., 1 / 32.]
        self.dim_out = self.stage_out_dim[1:int(math.log(self.stride, 2))]

        del self.incre_modules
        del self.downsamp_modules
        del self.final_layer
        del self.avgpool
        del self.classifier
        self._init_weights()
        self._init_modules()

    def _init_modules(self):
        assert cfg.BACKBONE.HRNET.FREEZE_AT in [0, 2, 3, 4, 5]  # cfg.BACKBONE.HRNET.FREEZE_AT: 2
        assert cfg.BACKBONE.HRNET.FREEZE_AT <= 5
        if cfg.BACKBONE.HRNET.FREEZE_AT > 0:
            freeze_params(getattr(self, 'conv1'))
            freeze_params(getattr(self, 'bn1'))
            freeze_params(getattr(self, 'conv2'))
            freeze_params(getattr(self, 'bn2'))
            freeze_params(getattr(self, 'layer1'))
        for i in range(2, cfg.BACKBONE.HRNET.FREEZE_AT):
            freeze_params(getattr(self, 'transition%d' % (i - 1)))
            freeze_params(getattr(self, 'stage%d' % i))
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, ops.AffineChannel2d) else None)

    def train(self, mode=True):
        # Override train mode
        self.training = mode
        if cfg.BACKBONE.HRNET.FREEZE_AT < 1:
            getattr(self, 'conv1').train(mode)
            getattr(self, 'bn1').train(mode)
            getattr(self, 'conv2').train(mode)
            getattr(self, 'bn2').train(mode)
            getattr(self, 'layer1').train(mode)
        for i in range(cfg.BACKBONE.HRNET.FREEZE_AT, 5):
            if i == 0:
                continue
            getattr(self, 'transition%d' % (i - 1)).train(mode)
            getattr(self, 'stage%d' % i).train(mode)

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

        return x    # x is a list


# ---------------------------------------------------------------------------- #
# HRNet Conv Body
# ---------------------------------------------------------------------------- #
@registry.BACKBONES.register("hrnet")
def hrnet_s4():
    model = HRNet(norm=get_norm())
    return model
