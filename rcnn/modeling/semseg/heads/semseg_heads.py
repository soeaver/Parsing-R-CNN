import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.net import make_conv
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_SEMSEG_HEADS.register("fused_head")
class fused_head(nn.Module):

    def __init__(self, dim_in, spatial_scale):
        super(fused_head, self).__init__()

        self.dim_in = dim_in[-1]
        self.fusion_level = cfg.SEMSEG.SEMSEG_HEAD.FUSION_LEVEL
        self.num_convs = cfg.SEMSEG.SEMSEG_HEAD.NUM_CONVS

        num_ins = cfg.SEMSEG.SEMSEG_HEAD.NUM_IN_STAGE
        conv_dim = cfg.SEMSEG.SEMSEG_HEAD.CONV_DIM
        use_bn = cfg.SEMSEG.SEMSEG_HEAD.USE_BN
        use_gn = cfg.SEMSEG.SEMSEG_HEAD.USE_GN

        lateral_convs = []
        for layer_idx in range(num_ins):
            lateral_convs.append(
                make_conv(
                    self.dim_in, conv_dim, kernel=1, stride=1, use_bn=use_bn, use_gn=use_gn, use_relu=True,
                    inplace=False
                )
            )
        self.add_module('lateral_convs', nn.Sequential(*lateral_convs))
        self.dim_in = conv_dim

        convs = []
        for layer_idx in range(self.num_convs):
            convs.append(
                make_conv(self.dim_in, conv_dim, kernel=3, stride=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
            )
            self.dim_in = conv_dim
        self.add_module('convs', nn.Sequential(*convs))

        self.conv_embedding = make_conv(
            self.dim_in, dim_in[-1], kernel=3, stride=1, use_bn=use_bn, use_gn=use_gn, use_relu=True
        )

        self.dim_out = self.dim_in

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def enhance_feature(self, feats, semseg_feature):
        feats = feats[:4]
        enhance_feats = []
        up_scale = 0.5 ** (np.array(range(4)) - (self.fusion_level - 1))
        for i, conv in enumerate(feats):
            resize_semseg_feats = F.interpolate(
                semseg_feature, scale_factor=up_scale[i], mode="bilinear", align_corners=False
            )
            enhance_feats.append(conv + resize_semseg_feats)
        return enhance_feats

    def forward(self, feats):
        x = self.lateral_convs[self.fusion_level - 1](feats[self.fusion_level - 1])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level - 1:
                feat = F.interpolate(feat, size=fused_size, mode='bilinear', align_corners=True)
                x += self.lateral_convs[i](feat)

        for layer_idx in range(self.num_convs):
            x = self.convs[layer_idx](x)

        semseg_feature = self.conv_embedding(x)
        semseg_feature = self.enhance_feature(feats, semseg_feature)

        return x, semseg_feature
