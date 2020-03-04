import torch
from torch import nn
from torch.nn import functional as F

from models.ops import Conv2d
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.MASKIOU_HEADS.register("convx_head")
class MaskIoU_head(nn.Module):
    """
    MaskIou head feature extractor.
    """
    def __init__(self, dim_in):
        super(MaskIoU_head, self).__init__()

        self.dim_in = dim_in[-1] + 1
        conv_dim = cfg.MRCNN.MASKIOU.CONV_DIM
        mlp_dim = cfg.MRCNN.MASKIOU.MLP_DIM
        resolution = cfg.MRCNN.ROI_XFORM_RESOLUTION

        self.maskiou_fcn1 = Conv2d(self.dim_in, conv_dim, 3, 1, 1)
        self.maskiou_fcn2 = Conv2d(conv_dim, conv_dim, 3, 1, 1)
        self.maskiou_fcn3 = Conv2d(conv_dim, conv_dim, 3, 1, 1)
        self.maskiou_fcn4 = Conv2d(conv_dim, conv_dim, 3, 2, 1)
        self.maskiou_fc1 = nn.Linear(conv_dim * (resolution[0] // 2) * (resolution[1] // 2), mlp_dim)
        self.maskiou_fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.dim_out = mlp_dim

        for l in [self.maskiou_fcn1, self.maskiou_fcn2, self.maskiou_fcn3, self.maskiou_fcn4]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        for l in [self.maskiou_fc1, self.maskiou_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        x = F.relu(self.maskiou_fcn1(x))
        x = F.relu(self.maskiou_fcn2(x))
        x = F.relu(self.maskiou_fcn3(x))
        x = F.relu(self.maskiou_fcn4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))

        return x
