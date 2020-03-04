import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.net import make_conv
from rcnn.core.config import cfg
from rcnn.modeling import registry


# ---------------------------------------------------------------------------- #
# Functions for bolting HRFPN onto a backbone architectures
# ---------------------------------------------------------------------------- #
@registry.FPN_BODY.register("hrfpn")
class hrfpn(nn.Module):
    # dim_in = [w, w * 2, w * 4, w * 8]
    # spatial_scale = [1/4, 1/8, 1/16, 1/32]
    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = sum(dim_in)
        self.spatial_scale = spatial_scale

        hrfpn_dim = cfg.FPN.HRFPN.DIM  # 256
        use_lite = cfg.FPN.HRFPN.USE_LITE
        use_bn = cfg.FPN.HRFPN.USE_BN
        use_gn = cfg.FPN.HRFPN.USE_GN
        if cfg.FPN.HRFPN.POOLING_TYPE == 'AVG':
            self.pooling = F.avg_pool2d
        else:
            self.pooling = F.max_pool2d
        self.num_extra_pooling = cfg.FPN.HRFPN.NUM_EXTRA_POOLING    # 1
        self.num_output = len(dim_in) + self.num_extra_pooling  # 5

        self.reduction_conv = make_conv(self.dim_in, hrfpn_dim, kernel=1, use_bn=use_bn, use_gn=use_gn)
        self.dim_in = hrfpn_dim

        self.fpn_conv = nn.ModuleList()
        for i in range(self.num_output):
            self.fpn_conv.append(
                make_conv(self.dim_in, hrfpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn, use_gn=use_gn,
                          suffix_1x1=use_lite)
            )
            self.dim_in = hrfpn_dim

        if self.num_extra_pooling:
            self.spatial_scale.append(self.spatial_scale[-1] * 0.5)
        self.dim_out = [self.dim_in for _ in range(self.num_output)]
        self._init_weights()

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = [x[0]]
        for i in range(1, len(x)):
            outs.append(F.interpolate(x[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        out = self.reduction_conv(out)

        outs = [out]
        for i in range(1, self.num_output):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        fpn_output_blobs = []
        for i in range(self.num_output):
            fpn_output_blobs.append(self.fpn_conv[i](outs[i]))

        # use all levels
        return fpn_output_blobs  # [P2 - P6]
