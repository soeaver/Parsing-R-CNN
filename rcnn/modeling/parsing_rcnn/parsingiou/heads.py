import torch
from torch import nn
from torch.nn import functional as F

from utils.net import make_conv, make_fc
from models.ops import Conv2d
from rcnn.modeling import registry
from rcnn.core.config import cfg


@registry.PARSINGIOU_HEADS.register("convx_head")
class convx_head(nn.Module):
    """
    ParsingIoU convx_head feature extractor.
    """
    def __init__(self, dim_in):
        super(convx_head, self).__init__()

        self.dim_in = dim_in + cfg.PRCNN.NUM_PARSING
        num_stacked_convs = cfg.PRCNN.PARSINGIOU.NUM_STACKED_CONVS  # default = 2
        conv_dim = cfg.PRCNN.PARSINGIOU.CONV_DIM
        mlp_dim = cfg.PRCNN.PARSINGIOU.MLP_DIM
        use_bn = cfg.PRCNN.PARSINGIOU.USE_BN
        use_gn = cfg.PRCNN.PARSINGIOU.USE_GN

        convx = []
        for _ in range(num_stacked_convs):
            layer_stride = 1 if _ < num_stacked_convs - 1 else 2
            convx.append(
                make_conv(
                    self.dim_in, conv_dim, kernel=3, stride=layer_stride, use_bn=use_bn, use_gn=use_gn, use_relu=True
                )
            )
            self.dim_in = conv_dim
        self.convx = nn.Sequential(*convx)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.parsingiou_fc1 = make_fc(self.dim_in, mlp_dim, use_bn=False, use_gn=False)
        self.parsingiou_fc2 = make_fc(mlp_dim, mlp_dim, use_bn=False, use_gn=False)
        self.dim_out = mlp_dim

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, parsing_logits):
        parsing_pool = F.max_pool2d(parsing_logits, kernel_size=4, stride=4)
        x = torch.cat((x, parsing_pool), 1)
        x = self.convx(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.parsingiou_fc1(x))
        x = F.relu(self.parsingiou_fc2(x))

        return x
