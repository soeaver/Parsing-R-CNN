import torch.nn as nn
import torch.nn.init as init

from rcnn.modeling import registry
from rcnn.core.config import cfg


# ---------------------------------------------------------------------------- #
# R-CNN bbox branch outputs
# ---------------------------------------------------------------------------- #
@registry.ROI_CASCADE_OUTPUTS.register("box_output")
class Box_output(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in

        self.cls_score = nn.Linear(self.dim_in, cfg.MODEL.NUM_CLASSES)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        if cfg.FAST_RCNN.CLS_AGNOSTIC_BBOX_REG:  # bg and fg
            self.bbox_pred = nn.Linear(self.dim_in, 4 * 2)
        else:
            raise NotImplementedError
            # self.bbox_pred = nn.Linear(self.dim_in, 4 * cfg.MODEL.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            # x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred
