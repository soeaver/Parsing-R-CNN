import torch.nn as nn
import torch.nn.functional as F

from models.imagenet.utils import convert_conv2convws_model
from utils.net import make_conv, make_fc
from rcnn.utils.poolers import Pooler
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_CASCADE_HEADS.register("roi_xconv1fc_head")
class roi_xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head"""

    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in[-1]

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

        use_lite = cfg.FAST_RCNN.CONVFC_HEAD.USE_LITE
        use_bn = cfg.FAST_RCNN.CONVFC_HEAD.USE_BN
        use_gn = cfg.FAST_RCNN.CONVFC_HEAD.USE_GN
        conv_dim = cfg.FAST_RCNN.CONVFC_HEAD.CONV_DIM
        num_stacked_convs = cfg.FAST_RCNN.CONVFC_HEAD.NUM_STACKED_CONVS
        dilation = cfg.FAST_RCNN.CONVFC_HEAD.DILATION
        
        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                make_conv(self.dim_in, conv_dim, kernel=3, stride=1, dilation=dilation, use_dwconv=use_lite,
                          use_bn=use_bn, use_gn=use_gn, suffix_1x1=use_lite, use_relu=True)
            )
            self.dim_in = conv_dim
        self.add_module("xconvs", nn.Sequential(*xconvs))
        
        input_size = self.dim_in * resolution[0] * resolution[1]
        mlp_dim = cfg.FAST_RCNN.CONVFC_HEAD.MLP_DIM
        self.fc6 = make_fc(input_size, mlp_dim, use_bn=False, use_gn=False)
        self.dim_out = mlp_dim

        if cfg.FAST_RCNN.CONVFC_HEAD.USE_WS:
            self = convert_conv2convws_model(self)
            
    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x), inplace=True)
        
        return x
