from torch import nn

from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.PARSINGIOU_OUTPUTS.register("linear_output")
class ParsingIoU_output(nn.Module):
    def __init__(self, dim_in):
        super(ParsingIoU_output, self).__init__()
        num_classes = 1

        self.parsingiou = nn.Linear(dim_in, num_classes)

        nn.init.normal_(self.parsingiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.parsingiou.bias, 0)

    def forward(self, x):
        parsingiou = self.parsingiou(x)
        return parsingiou
