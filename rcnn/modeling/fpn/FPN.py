import torch
import torch.nn as nn
import torch.nn.functional as F

from models.imagenet.utils import convert_conv2convws_model
from utils.net import make_conv
from rcnn.core.config import cfg
from rcnn.modeling import registry


# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #
@registry.FPN_BODY.register("fpn")
class fpn(nn.Module):
    # dim_in = [256, 512, 1024, 2048]
    # spatial_scale = [1/4, 1/8, 1/16, 1/32]
    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in[-1]  # 2048
        self.spatial_scale = spatial_scale

        fpn_dim = cfg.FPN.DIM  # 256
        use_lite = cfg.FPN.USE_LITE
        use_bn = cfg.FPN.USE_BN
        use_gn = cfg.FPN.USE_GN
        min_level, max_level = get_min_max_levels()  # 2, 6
        self.num_backbone_stages = len(dim_in) - (
                    min_level - cfg.FPN.LOWEST_BACKBONE_LVL)  # 4 (cfg.FPN.LOWEST_BACKBONE_LVL=2)

        # P5 in
        self.p5_in = make_conv(self.dim_in, fpn_dim, kernel=1, use_bn=use_bn, use_gn=use_gn)

        # P5 out
        self.p5_out = make_conv(fpn_dim, fpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn, use_gn=use_gn,
                                suffix_1x1=use_lite)

        # fpn module
        self.fpn_in = []
        self.fpn_out = []
        for i in range(self.num_backbone_stages - 1):  # skip the top layer
            px_in = make_conv(dim_in[-i - 2], fpn_dim, kernel=1, use_bn=use_bn, use_gn=use_gn)    # from P4 to P2
            px_out = make_conv(fpn_dim, fpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn, use_gn=use_gn,
                               suffix_1x1=use_lite)
            self.fpn_in.append(px_in)
            self.fpn_out.append(px_out)
        self.fpn_in = nn.ModuleList(self.fpn_in)  # [P4, P3, P2]
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.dim_in = fpn_dim

        # P6. Original FPN P6 level implementation from CVPR'17 FPN paper.
        if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == cfg.FPN.HIGHEST_BACKBONE_LVL + 1:
            self.maxpool_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
            self.spatial_scale.append(self.spatial_scale[-1] * 0.5)

        # Coarser FPN levels introduced for RetinaNet
        if cfg.FPN.EXTRA_CONV_LEVELS and max_level > cfg.FPN.HIGHEST_BACKBONE_LVL:
            self.extra_pyramid_modules = nn.ModuleList()
            if cfg.FPN.USE_C5:
                self.dim_in = dim_in[-1]
            for i in range(cfg.FPN.HIGHEST_BACKBONE_LVL + 1, max_level + 1):
                self.extra_pyramid_modules.append(
                    make_conv(self.dim_in, fpn_dim, kernel=3, stride=2, use_dwconv=use_lite, use_bn=use_bn,
                              use_gn=use_gn, suffix_1x1=use_lite)
                )
                self.dim_in = fpn_dim
                self.spatial_scale.append(self.spatial_scale[-1] * 0.5)

        # self.spatial_scale.reverse()  # [1/64, 1/32, 1/16, 1/8, 1/4]
        # self.dim_out = [self.dim_in]
        num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
        # Retain only the spatial scales that will be used for RoI heads. `self.spatial_scale`
        # may include extra scales that are used for RPN proposals, but not for RoI heads.
        self.spatial_scale = self.spatial_scale[:num_roi_levels]
        self.dim_out = [self.dim_in for _ in range(num_roi_levels)]
        
        if cfg.FPN.USE_WS:
            self = convert_conv2convws_model(self)
        
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c5_out = x[-1]
        px = self.p5_in(c5_out)
        fpn_output_blobs = [self.p5_out(px)]  # [P5]
        for i in range(self.num_backbone_stages - 1):  # [P5 - P2]
            cx_out = x[-i - 2]  # C4, C3, C2
            cx_out = self.fpn_in[i](cx_out)  # lateral branch
            if cx_out.size()[2:] != px.size()[2:]:
                px = F.interpolate(px, scale_factor=2, mode='nearest')
            px = cx_out + px
            fpn_output_blobs.insert(0, self.fpn_out[i](px))  # [P2 - P5]

        if hasattr(self, 'maxpool_p6'):
            fpn_output_blobs.append(self.maxpool_p6(fpn_output_blobs[-1]))  # [P2 - P6]

        if hasattr(self, 'extra_pyramid_modules'):
            if cfg.FPN.USE_C5:
                p6_in = c5_out
            else:
                p6_in = fpn_output_blobs[-1]
            fpn_output_blobs.append(self.extra_pyramid_modules[0](p6_in))
            for module in self.extra_pyramid_modules[1:]:
                fpn_output_blobs.append(module(F.relu(fpn_output_blobs[-1])))  # [P2 - P6, P7]

        # use all levels
        return fpn_output_blobs  # [P2 - P6]


def get_min_max_levels():
    """The min and max FPN levels required for supporting RPN and/or RoI
    transform operations on multiple FPN levels.
    """
    min_level = cfg.FPN.LOWEST_BACKBONE_LVL
    max_level = cfg.FPN.HIGHEST_BACKBONE_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level
