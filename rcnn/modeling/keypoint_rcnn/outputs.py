from torch import nn
from torch.nn import functional as F

from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_KEYPOINT_OUTPUTS.register("keypoint_output")
class Keypoint_output(nn.Module):
    def __init__(self, dim_in):
        super(Keypoint_output, self).__init__()
        num_keypoints = cfg.KRCNN.NUM_CLASSES
        assert cfg.KRCNN.RESOLUTION[0] // cfg.KRCNN.ROI_XFORM_RESOLUTION[0] == \
               cfg.KRCNN.RESOLUTION[1] // cfg.KRCNN.ROI_XFORM_RESOLUTION[1]
        self.up_scale = cfg.KRCNN.RESOLUTION[0] // (cfg.KRCNN.ROI_XFORM_RESOLUTION[0] * 2)

        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(
            dim_in,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )

        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.kps_score_lowres.bias, 0)

        self.dim_out = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        if self.up_scale > 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)

        return x
