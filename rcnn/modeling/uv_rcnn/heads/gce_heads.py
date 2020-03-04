import torch
from torch import nn

from pet.models.ops import interpolate, NonLocal2d
from pet.rcnn.core.config import cfg
from pet.rcnn.modeling import registry
from pet.rcnn.utils.poolers import Pooler
from pet.utils.net import make_conv


@registry.ROI_UV_HEADS.register("roi_gce_head")
class roi_gce_head(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(roi_gce_head, self).__init__()
        self.dim_in = dim_in[-1]

        method = cfg.UVRCNN.ROI_XFORM_METHOD
        resolution = cfg.UVRCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.UVRCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_nl = cfg.UVRCNN.GCE_HEAD.USE_NL
        use_bn = cfg.UVRCNN.GCE_HEAD.USE_BN
        use_gn = cfg.UVRCNN.GCE_HEAD.USE_GN
        conv_dim = cfg.UVRCNN.GCE_HEAD.CONV_DIM
        asppv3_dim = cfg.UVRCNN.GCE_HEAD.ASPPV3_DIM
        num_convs_before_asppv3 = cfg.UVRCNN.GCE_HEAD.NUM_CONVS_BEFORE_ASPPV3
        asppv3_dilation = cfg.UVRCNN.GCE_HEAD.ASPPV3_DILATION
        num_convs_after_asppv3 = cfg.UVRCNN.GCE_HEAD.NUM_CONVS_AFTER_ASPPV3

        # convx before asppv3 module
        before_asppv3_list = []
        for _ in range(num_convs_before_asppv3):
            before_asppv3_list.append(
                make_conv(self.dim_in, conv_dim, kernel=3, stride=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
            )
            self.dim_in = conv_dim
        self.conv_before_asppv3 = nn.Sequential(*before_asppv3_list) if len(before_asppv3_list) else None

        # asppv3 module
        self.asppv3 = []
        self.asppv3.append(
            make_conv(self.dim_in, asppv3_dim, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
        )
        for dilation in asppv3_dilation:
            self.asppv3.append(
                make_conv(self.dim_in, asppv3_dim, kernel=3, dilation=dilation, use_bn=use_bn, use_gn=use_gn,
                          use_relu=True)
            )
        self.asppv3 = nn.ModuleList(self.asppv3)
        self.im_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            make_conv(self.dim_in, asppv3_dim, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
        )
        self.dim_in = (len(asppv3_dilation) + 2) * asppv3_dim

        feat_list = []
        feat_list.append(
            make_conv(self.dim_in, conv_dim, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
        )
        if use_nl:
            feat_list.append(
                NonLocal2d(conv_dim, int(conv_dim * cfg.UVRCNN.GCE_HEAD.NL_RATIO), conv_dim, use_gn=True)
            )
        self.feat = nn.Sequential(*feat_list)
        self.dim_in = conv_dim

        # convx after asppv3 module
        assert num_convs_after_asppv3 >= 1
        after_asppv3_list = []
        for _ in range(num_convs_after_asppv3):
            after_asppv3_list.append(
                make_conv(self.dim_in, conv_dim, kernel=3, use_bn=use_bn, use_gn=use_gn, use_relu=True)
            )
            self.dim_in = conv_dim
        self.conv_after_asppv3 = nn.Sequential(*after_asppv3_list) if len(after_asppv3_list) else None
        self.dim_out = self.dim_in

    def forward(self, x, proposals):
        resolution = cfg.UVRCNN.ROI_XFORM_RESOLUTION
        x = self.pooler(x, proposals)

        if self.conv_before_asppv3 is not None:
            x = self.conv_before_asppv3(x)

        asppv3_out = [interpolate(self.im_pool(x), scale_factor=resolution, 
                                  mode="bilinear", align_corners=False)]
        for i in range(len(self.asppv3)):
            asppv3_out.append(self.asppv3[i](x))
        asppv3_out = torch.cat(asppv3_out, 1)
        asppv3_out = self.feat(asppv3_out)

        if self.conv_after_asppv3 is not None:
            x = self.conv_after_asppv3(asppv3_out)
        return x
