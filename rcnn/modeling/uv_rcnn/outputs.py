from torch import nn
from torch.nn import functional as F

from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.ROI_UV_OUTPUTS.register("uv_output")
class UV_output(nn.Module):
    def __init__(self, dim_in):
        super(UV_output, self).__init__()
        num_patches = cfg.UVRCNN.NUM_PATCHES
        deconv_kernel = 4
        assert cfg.UVRCNN.RESOLUTION[0] // cfg.UVRCNN.ROI_XFORM_RESOLUTION[0] == \
               cfg.UVRCNN.RESOLUTION[1] // cfg.UVRCNN.ROI_XFORM_RESOLUTION[1]
        self.up_scale = cfg.UVRCNN.RESOLUTION[0] // (cfg.UVRCNN.ROI_XFORM_RESOLUTION[0] * 2)

        self.deconv_Ann = nn.ConvTranspose2d(dim_in, 15, deconv_kernel, 2, padding=deconv_kernel // 2 - 1)
        self.deconv_Index = nn.ConvTranspose2d(dim_in, num_patches + 1, deconv_kernel, 2,
                                               padding=deconv_kernel // 2 - 1)
        self.deconv_U = nn.ConvTranspose2d(dim_in, num_patches + 1, deconv_kernel, 2, padding=deconv_kernel // 2 - 1)
        self.deconv_V = nn.ConvTranspose2d(dim_in, num_patches + 1, deconv_kernel, 2, padding=deconv_kernel // 2 - 1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_Ann = self.deconv_Ann(x)
        x_Index = self.deconv_Index(x)
        x_U = self.deconv_U(x)
        x_V = self.deconv_V(x)

        if self.up_scale > 1:
            x_Ann = F.interpolate(x_Ann, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
            x_Index = F.interpolate(x_Index, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
            x_U = F.interpolate(x_U, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
            x_V = F.interpolate(x_V, scale_factor=self.up_scale, mode="bilinear", align_corners=False)

        return [x_Ann, x_Index, x_U, x_V]
