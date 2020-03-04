import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

from .affine import AffineChannel2d
from .misc import Conv2d


class MS_NonLocal2d(nn.Module):
    def __init__(self, dim_in, dim_inner, dim_out, use_gn=False, use_scale=True):
        super().__init__()
        self.dim_inner = dim_inner
        self.use_gn = use_gn
        self.use_scale = use_scale

        self.theta_scale1 = Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)
        self.theta_scale2 = Conv2d(dim_in, dim_inner * 4, 1, stride=2, padding=0)
        self.theta_scale3 = Conv2d(dim_in, dim_inner * 16, 1, stride=4, padding=0)

        self.phi = Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)
        self.g = Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)

        self.out = Conv2d(dim_inner, dim_out, 1, stride=1, padding=0)
        if self.use_gn:
            self.gn = nn.GroupNorm(32, dim_out, eps=1e-5)

        self.apply(self._init_modules)

    def _init_modules(self, m):
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        # theta_s1=>(n, c, h, w)->(n, c, hw)->(n, hw, c)
        theta_s1 = self.theta_scale1(x).view(batch_size, self.dim_inner, -1)
        theta_s1 = theta_s1.permute(0, 2, 1)

        # theta_s2=>(n, 4c, h/2, w/2)->(n, c, hw)->(n, hw, c)
        theta_s2 = self.theta_scale2(x).view(batch_size, self.dim_inner, -1)
        theta_s2 = theta_s2.permute(0, 2, 1)

        # theta_s3=>(n, 16c, h/4, w/4)->(n, c, hw)->(n, hw, c)
        theta_s3 = self.theta_scale3(x).view(batch_size, self.dim_inner, -1)
        theta_s3 = theta_s3.permute(0, 2, 1)

        # phi_x=>(n, c, h, w)->(n, c, hw);
        phi_x = self.phi(x).view(batch_size, self.dim_inner, -1)

        # theta_phi_s1=>(n, hw, c) * (n, c, hw)->(n, hw, hw)
        theta_phi_s1 = torch.matmul(theta_s1, phi_x)

        # theta_phi_s2=>(n, hw, c) * (n, c, hw)->(n, hw, hw)
        theta_phi_s2 = torch.matmul(theta_s2, phi_x)

        # theta_phi_s3=>(n, hw, c) * (n, c, hw)->(n, hw, hw)
        theta_phi_s3 = torch.matmul(theta_s3, phi_x)

        # theta_phi=>(n, hw, hw)
        theta_phi = theta_phi_s1 + theta_phi_s2 + theta_phi_s3
        if self.use_scale:
            theta_phi_sc = theta_phi * (self.dim_inner ** -.5)
        else:
            theta_phi_sc = theta_phi

        # p_x=>(n, hw, hw)
        p_x = F.softmax(theta_phi_sc, dim=-1)

        # p_x=>(n, hw, hw)
        p_x = p_x.permute(0, 2, 1)

        # g_x=>(n, c, h, w)->(n, c, hw)
        g_x = self.g(x).view(batch_size, self.dim_inner, -1)

        # t_x=>(n, c, hw) * (n, hw, hw)->(n, c, hw)->(n, c, h, w)
        t_x = torch.matmul(g_x, p_x)
        t_x = t_x.view(batch_size, self.dim_inner, *x.size()[2:])

        y = self.out(t_x)
        if self.use_gn:
            y = self.gn(y)

        return y + x
    

# 2d space nonlocal (v1: spatial downsample)
class NonLocal2d(nn.Module):
    def __init__(self, dim_in, dim_inner, dim_out, max_pool_stride=2, 
                 use_maxpool=True, use_gn=False, use_scale=True, use_affine=False):
        super().__init__()
        self.dim_inner = dim_inner
        self.use_maxpool = use_maxpool
        self.use_gn = use_gn
        self.use_scale = use_scale
        self.use_affine = use_affine

        self.theta = Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)

        # phi and g: half spatial size
        # e.g., (N, 1024, 14, 14) => (N, 1024, 7, 7)
        if self.use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=max_pool_stride, stride=max_pool_stride, padding=0)

        self.phi = Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)
        self.g = Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)

        self.out = Conv2d(dim_inner, dim_out, 1, stride=1, padding=0)
        if self.use_gn:
            self.gn = nn.GroupNorm(32, dim_out, eps=1e-5)
        if self.use_affine:
            self.affine = AffineChannel2d(dim_out)

        self.apply(self._init_modules)

    def _init_modules(self, m):
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        # theta_x=>(n, c, h, w)->(n, c, hw)->(n, hw, c)
        theta_x = self.theta(x).view(batch_size, self.dim_inner, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.use_maxpool:
            pool_x = self.pool(x)
        else:
            pool_x = x
        # phi_x=>(n, c, h/2, w/2)->(n, c, hw/4); g_x=>(n, c, h/2, w/2)->(n, c, hw/4)
        phi_x = self.phi(pool_x).view(batch_size, self.dim_inner, -1)
        g_x = self.g(pool_x).view(batch_size, self.dim_inner, -1)

        # theta_phi=>(n, hw, c) * (n, c, hw/4)->(n, hw, hw/4)
        theta_phi = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            theta_phi_sc = theta_phi * (self.dim_inner ** -.5)
        else:
            theta_phi_sc = theta_phi
        p_x = F.softmax(theta_phi_sc, dim=-1)

        # y=>(n, c, hw/4) * (n, hw/4, hw)->(n, c, hw)->(n, c, h, w)
        p_x = p_x.permute(0, 2, 1)
        t_x = torch.matmul(g_x, p_x)
        t_x = t_x.view(batch_size, self.dim_inner, *x.size()[2:])

        y = self.out(t_x)
        if self.use_gn:
            y = self.gn(y)
        if self.use_affine:
            y = self.affine(y)

        return y + x

