from torch import nn
from torch.nn import functional as F

from utils.net import make_conv
from rcnn.utils.poolers import Pooler
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_KEYPOINT_HEADS.register("roi_convx_head")
class roi_convx_head(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(roi_convx_head, self).__init__()
        self.dim_in = dim_in[-1]

        method = cfg.KRCNN.ROI_XFORM_METHOD
        resolution = cfg.KRCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_lite = cfg.KRCNN.CONVX_HEAD.USE_LITE
        use_bn = cfg.KRCNN.CONVX_HEAD.USE_BN
        use_gn = cfg.KRCNN.CONVX_HEAD.USE_GN
        conv_dim = cfg.KRCNN.CONVX_HEAD.CONV_DIM
        num_stacked_convs = cfg.KRCNN.CONVX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.KRCNN.CONVX_HEAD.DILATION

        self.blocks = []
        for layer_idx in range(num_stacked_convs):
            layer_name = "keypoint_fcn{}".format(layer_idx + 1)
            module = make_conv(self.dim_in, conv_dim, kernel=3, stride=1, dilation=dilation, use_dwconv=use_lite,
                               use_bn=use_bn, use_gn=use_gn, suffix_1x1=use_lite)
            self.add_module(layer_name, module)
            self.dim_in = conv_dim
            self.blocks.append(layer_name)
        self.dim_out = self.dim_in

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x
