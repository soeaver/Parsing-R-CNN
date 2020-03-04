import torch
import torch.nn as nn
import torch.nn.functional as F


class H_Sigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


def make_norm(c, norm='bn', group=1, eps=1e-5):
    if norm == 'bn':
        return nn.BatchNorm2d(c, eps=eps)
    elif norm == 'gn':
        assert c % group == 0
        return nn.GroupNorm(group, c, eps=eps)
    elif norm == 'none':
        return None
    else:
        return nn.BatchNorm2d(c, eps=eps)


class AttentionWeights(nn.Module):
    expansion = 2

    def __init__(self, num_channels, k, norm=None, groups=1, use_hsig=True):
        super(AttentionWeights, self).__init__()
        # num_channels *= 2
        self.k = k
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(num_channels, k, 1, bias=False),
            make_norm(k, norm, groups),
            H_Sigmoid() if use_hsig else nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x)  # .view(b, c)
        var = torch.var(x, dim=(2, 3)).view(b, c, 1, 1)
        y *= (var + 1e-3).rsqrt()
        # y = torch.cat((y, var), dim=1)
        return self.attention(y).view(b, self.k)


# TODO: keep it to use FP32 always, need to figure out how to set it using apex ?
class MixtureBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_channels, k, eps=1e-5, momentum=0.1, track_running_stats=True):
        super(MixtureBatchNorm2d, self).__init__(
            num_channels, eps=eps, momentum=momentum, affine=False, track_running_stats=track_running_stats)
        self.k = k
        self.weight_ = nn.Parameter(torch.Tensor(k, num_channels))
        self.bias_ = nn.Parameter(torch.Tensor(k, num_channels))

        self.attention_weights = AttentionWeights(num_channels, k, norm='bn')

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.weight_, 1, 0.1)
        nn.init.normal_(self.bias_, 0, 0.1)

    def forward(self, x):
        output = super(MixtureBatchNorm2d, self).forward(x)
        size = output.size()
        y = self.attention_weights(x)  # bxk # or use output as attention input

        weight = y @ self.weight_  # bxc
        bias = y @ self.bias_  # bxc
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias


# Modified on top of nn.GroupNorm
# TODO: keep it to use FP32 always, need to figure out how to set it using apex ?
class MixtureGroupNorm(nn.Module):
    __constants__ = ['num_groups', 'num_channels', 'k', 'eps', 'weight', 'bias']

    def __init__(self, num_channels, num_groups, k, eps=1e-5):
        super(MixtureGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.affine = True
        self.weight_ = nn.Parameter(torch.Tensor(k, num_channels))
        self.bias_ = nn.Parameter(torch.Tensor(k, num_channels))
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

        self.attention_weights = AttentionWeights(num_channels, k, norm='gn', groups=1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_, 1, 0.1)
        nn.init.normal_(self.bias_, 0, 0.1)

    def forward(self, x):
        output = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        size = output.size()

        y = self.attention_weights(x)  # TODO: use output as attention input

        weight = y @ self.weight_
        bias = y @ self.bias_

        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)
