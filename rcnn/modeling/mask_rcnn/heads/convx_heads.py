from torch import nn
from torch.nn import functional as F

from models.imagenet.utils import convert_conv2convws_model
from utils.net import make_conv
from rcnn.utils.poolers import Pooler
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_MASK_HEADS.register("roi_convx_head")
class roi_convx_head(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, dim_in, spatial_scale):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(roi_convx_head, self).__init__()
        self.dim_in = dim_in[-1]

        method = cfg.MRCNN.ROI_XFORM_METHOD
        resolution = cfg.MRCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_lite = cfg.MRCNN.CONVX_HEAD.USE_LITE
        use_bn = cfg.MRCNN.CONVX_HEAD.USE_BN
        use_gn = cfg.MRCNN.CONVX_HEAD.USE_GN
        conv_dim = cfg.MRCNN.CONVX_HEAD.CONV_DIM
        num_stacked_convs = cfg.MRCNN.CONVX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MRCNN.CONVX_HEAD.DILATION

        self.blocks = []
        for layer_idx in range(num_stacked_convs):
            layer_name = "mask_fcn{}".format(layer_idx + 1)
            module = make_conv(self.dim_in, conv_dim, kernel=3, stride=1, dilation=dilation, use_dwconv=use_lite,
                               use_bn=use_bn, use_gn=use_gn, suffix_1x1=use_lite)
            self.add_module(layer_name, module)
            self.dim_in = conv_dim
            self.blocks.append(layer_name)
        self.dim_out = self.dim_in
        
        if cfg.MRCNN.CONVX_HEAD.USE_WS:
            self = convert_conv2convws_model(self)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        roi_feature = x
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x, roi_feature

