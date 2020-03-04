import torch
import torch.nn as nn
import torch.nn.functional as F

from models.imagenet.utils import convert_conv2convws_model
from utils.net import make_fc
from rcnn.utils.poolers import Pooler
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_CASCADE_HEADS.register("roi_2mlp_head")
class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""

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
        input_size = self.dim_in * resolution[0] * resolution[1]
        mlp_dim = cfg.FAST_RCNN.MLP_HEAD.MLP_DIM
        use_bn = cfg.FAST_RCNN.MLP_HEAD.USE_BN
        use_gn = cfg.FAST_RCNN.MLP_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, mlp_dim, use_bn, use_gn)
        self.fc7 = make_fc(mlp_dim, mlp_dim, use_bn, use_gn)
        self.dim_out = mlp_dim
        
        if cfg.FAST_RCNN.MLP_HEAD.USE_WS:
            self = convert_conv2convws_model(self)
            
    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x), inplace=True)
        x = F.relu(self.fc7(x), inplace=True)

        return x
