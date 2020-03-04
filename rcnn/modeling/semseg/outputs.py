import torch.nn as nn
import torch.nn.init as init

from rcnn.modeling import registry
from rcnn.core.config import cfg


# ---------------------------------------------------------------------------- #
# R-CNN semseg branch outputs
# ---------------------------------------------------------------------------- #
@registry.ROI_SEMSEG_OUTPUTS.register("semseg_output")
class SemSeg_output(nn.Module):
    def __init__(self, dim_in):
        super(SemSeg_output, self).__init__()
        num_classes = cfg.SEMSEG.SEMSEG_NUM_CLASSES
        self.conv_logits = nn.Conv2d(dim_in, num_classes, 1)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.conv_logits(x)
