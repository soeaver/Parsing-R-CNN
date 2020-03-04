from torch import nn
from torch.nn import functional as F

from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_MASK_OUTPUTS.register("mask_deconv_output")
class Mask_deconv_output(nn.Module):
    def __init__(self, dim_in):
        super(Mask_deconv_output, self).__init__()
        num_classes = cfg.MODEL.NUM_CLASSES

        self.mask_deconv = nn.ConvTranspose2d(dim_in, dim_in, 2, 2, 0)
        self.mask_fcn_logits = nn.Conv2d(dim_in, num_classes, 1, 1, 0)

        # init
        nn.init.kaiming_normal_(self.mask_deconv.weight, mode='fan_out', nonlinearity="relu")
        if self.mask_deconv.bias is not None:
            nn.init.zeros_(self.mask_deconv.bias)
        nn.init.normal_(self.mask_fcn_logits.weight, std=0.001)
        if self.mask_fcn_logits.bias is not None:
            nn.init.constant_(self.mask_fcn_logits.bias, 0)

    def forward(self, x):
        x = F.relu(self.mask_deconv(x))
        return self.mask_fcn_logits(x)


@registry.ROI_MASK_OUTPUTS.register("mask_logits_output")
class Mask_logits_output(nn.Module):
    def __init__(self, dim_in):
        super(Mask_logits_output, self).__init__()
        num_classes = cfg.MODEL.NUM_CLASSES

        self.mask_fcn_logits = nn.Conv2d(dim_in, num_classes, 1, 1, 0)

        # init
        nn.init.normal_(self.mask_fcn_logits.weight, std=0.001)
        if self.mask_fcn_logits.bias is not None:
            nn.init.constant_(self.mask_fcn_logits.bias, 0)

    def forward(self, x):
        return self.mask_fcn_logits(x)
