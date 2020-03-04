from torch import nn

from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.MASKIOU_OUTPUTS.register("linear_output")
class MaskIoU_output(nn.Module):
    def __init__(self, dim_in):
        super(MaskIoU_output, self).__init__()
        num_classes = cfg.MODEL.NUM_CLASSES

        self.maskiou = nn.Linear(dim_in, num_classes)

        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)

    def forward(self, x):
        maskiou = self.maskiou(x)
        return maskiou
