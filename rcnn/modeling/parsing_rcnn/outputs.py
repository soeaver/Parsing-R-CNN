from torch import nn
from torch.nn import functional as F

from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_PARSING_OUTPUTS.register("parsing_output")
class Parsing_output(nn.Module):
    def __init__(self, dim_in):
        super(Parsing_output, self).__init__()
        num_parsing = cfg.PRCNN.NUM_PARSING
        assert cfg.PRCNN.RESOLUTION[0] // cfg.PRCNN.ROI_XFORM_RESOLUTION[0] == \
               cfg.PRCNN.RESOLUTION[1] // cfg.PRCNN.ROI_XFORM_RESOLUTION[1]
        self.up_scale = cfg.PRCNN.RESOLUTION[0] // (cfg.PRCNN.ROI_XFORM_RESOLUTION[0] * 2)

        deconv_kernel = 4
        self.parsing_score_lowres = nn.ConvTranspose2d(
            dim_in,
            num_parsing,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )

        nn.init.kaiming_normal_(self.parsing_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.parsing_score_lowres.bias, 0)

        self.dim_out = num_parsing

    def forward(self, x):
        x = self.parsing_score_lowres(x)
        if self.up_scale > 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)

        return x
