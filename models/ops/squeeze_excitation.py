from torch import nn


class SeConv2d(nn.Module):
    def __init__(self, inplanes, innerplanse, activation=nn.ReLU):
        super(SeConv2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, innerplanse, kernel_size=1),
            activation(),
            nn.Conv2d(innerplanse, inplanes, kernel_size=1),
            nn.Sigmoid()
        )
        self.reset_parameters()

    def reset_parameters(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

    
class GDWSe2d(nn.Module):
    def __init__(self, inplanes, kernel=3, reduction=16, with_padding=False):
        super(GDWSe2d, self).__init__()
        if with_padding:
            padding = kernel // 2
        else:
            padding = 0
         
        self.globle_dw = nn.Conv2d(inplanes, inplanes, kernel_size=kernel, padding=padding, stride=1,
                                   groups=inplanes, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inplanes, inplanes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes // reduction, inplanes),
            nn.Sigmoid()
        )
        
        self._init_weights()

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.globle_dw(x)
        y = self.bn(y)
        y = self.relu(y)
        
        n, c, _, _ = x.size()
        y = self.avg_pool(y).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)
