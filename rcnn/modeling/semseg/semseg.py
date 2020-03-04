import torch.nn as nn
from torch.nn import functional as F

from rcnn.modeling.semseg import heads
from rcnn.modeling.semseg import outputs
from rcnn.modeling.semseg.loss import semseg_loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class SemSeg(nn.Module):
    """
    Generic SemSeg Head class.
    """

    def __init__(self, dim_in, spatial_scale):
        super(SemSeg, self).__init__()

        self.Head = registry.ROI_SEMSEG_HEADS[cfg.SEMSEG.ROI_SEMSEG_HEAD](dim_in, spatial_scale)
        self.Output = registry.ROI_SEMSEG_OUTPUTS[cfg.SEMSEG.ROI_SEMSEG_OUTPUT](self.Head.dim_out)
        self.loss_evaluator = semseg_loss_evaluator()

    def forward(self, features, targets=None):
        x, semseg_feature = self.Head(features)
        semseg_pred = self.Output(x)
        if not self.training:
            semseg_pred = F.softmax(semseg_pred, dim=1)
            return semseg_pred, semseg_feature, {}

        loss_seg = self.loss_evaluator(semseg_pred, targets)
        return semseg_pred, semseg_feature, dict(loss_seg=loss_seg)
