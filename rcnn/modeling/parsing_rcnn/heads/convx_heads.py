from torch import nn
from torch.nn import functional as F

from utils.net import make_conv
from rcnn.utils.poolers import Pooler
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_PARSING_HEADS.register("roi_convx_head")
class roi_convx_head(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(roi_convx_head, self).__init__()
        self.dim_in = dim_in[-1]

        method = cfg.PRCNN.ROI_XFORM_METHOD
        resolution = cfg.PRCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.PRCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_lite = cfg.PRCNN.CONVX_HEAD.USE_LITE
        use_bn = cfg.PRCNN.CONVX_HEAD.USE_BN
        use_gn = cfg.PRCNN.CONVX_HEAD.USE_GN
        conv_dim = cfg.PRCNN.CONVX_HEAD.CONV_DIM
        num_stacked_convs = cfg.PRCNN.CONVX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.PRCNN.CONVX_HEAD.DILATION

        self.blocks = []
        for layer_idx in range(num_stacked_convs):
            layer_name = "parsing_fcn{}".format(layer_idx + 1)
            module = make_conv(self.dim_in, conv_dim, kernel=3, stride=1, dilation=dilation, use_dwconv=use_lite,
                               use_bn=use_bn, use_gn=use_gn, suffix_1x1=use_lite)
            self.add_module(layer_name, module)
            self.dim_in = conv_dim
            self.blocks.append(layer_name)
        self.dim_out = self.dim_in

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        roi_feature = x
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x, roi_feature
