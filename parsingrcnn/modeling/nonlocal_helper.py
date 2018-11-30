import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn


# 2d space nonlocal (v1: spatial downsample)
class SpaceNonLocal(nn.Module):
    def __init__(self, dim_in, dim_inner, dim_out, max_pool_stride=2):
        super().__init__()
        self.dim_inner = dim_inner

        self.theta = nn.Conv2d(dim_in, dim_inner, 1, stride=1, padding=0, bias=not cfg.NONLOCAL.NO_BIAS)

        # phi and g: half spatial size
        # e.g., (N, 1024, 14, 14) => (N, 1024, 7, 7)
        if cfg.NONLOCAL.USE_MAXPOOL is True:
            self.pool = nn.MaxPool2d(kernel_size=max_pool_stride, stride=max_pool_stride, padding=0)

        self.phi = nn.Conv2d(dim_in, dim_inner, 1, stride=1, padding=0, bias=not cfg.NONLOCAL.NO_BIAS)
        self.g = nn.Conv2d(dim_in, dim_inner, 1, stride=1, padding=0, bias=not cfg.NONLOCAL.NO_BIAS)

        self.out = nn.Conv2d(dim_inner, dim_out, 1, stride=1, padding=0, bias=not cfg.NONLOCAL.NO_BIAS)
        if cfg.NONLOCAL.USE_BN is True:
            self.bn = nn.BatchNorm2d(dim_out)
        if cfg.NONLOCAL.USE_AFFINE is True:
            self.affine = mynn.AffineChannel2d(dim_out)

        self.apply(self._init_modules)

    def _init_modules(self, m):
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight, std=cfg.NONLOCAL.CONV_INIT_STD)
            if not cfg.NONLOCAL.NO_BIAS:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        # theta_x=>(n, c, h, w)->(n, c, hw)->(n, hw, c)
        theta_x = self.theta(x).view(batch_size, self.dim_inner, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if cfg.NONLOCAL.USE_MAXPOOL is True:
            pool_x = self.pool(x)
        else:
            pool_x = x
        # phi_x=>(n, c, h/2, w/2)->(n, c, hw/4); g_x=>(n, c, h/2, w/2)->(n, c, hw/4)
        phi_x = self.phi(pool_x).view(batch_size, self.dim_inner, -1)
        g_x = self.g(pool_x).view(batch_size, self.dim_inner, -1)

        # theta_phi=>(n, hw, c) * (n, c, hw/4)->(n, hw, hw/4)
        theta_phi = torch.matmul(theta_x, phi_x)
        if cfg.NONLOCAL.USE_SCALE is True:
            theta_phi_sc = theta_phi * (self.dim_inner ** -.5)
        else:
            theta_phi_sc = theta_phi
        p_x = F.softmax(theta_phi_sc, dim=-1)

        # y=>(n, c, hw/4) * (n, hw/4, hw)->(n, c, hw)->(n, c, h, w)
        p_x = p_x.permute(0, 2, 1)
        t_x = torch.matmul(g_x, p_x)
        t_x = t_x.view(batch_size, self.dim_inner, *x.size()[2:])

        y = self.out(t_x)
        if cfg.NONLOCAL.USE_BN is True:
            y = self.bn(y)
        if cfg.NONLOCAL.USE_AFFINE is True:
            y = self.affine(y)

        return y + x

