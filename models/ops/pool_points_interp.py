import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from models.ops import _C

from apex import amp


class _PoolPointsInterp(Function):
    @staticmethod
    def forward(ctx, input, roi, spatial_scale):
        ctx.save_for_backward(roi)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        output = _C.pool_points_interp_forward(
            input, roi, spatial_scale)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.pool_points_interp_backward(
            grad_output,
            rois,
            spatial_scale,
            bs,
            ch,
            h,
            w,
        )
        return grad_input, None, None


pool_points_interp = _PoolPointsInterp.apply


class PoolPointsInterp(nn.Module):
    def __init__(self, spatial_scale=1.0):
        super(PoolPointsInterp, self).__init__()
        self.spatial_scale = spatial_scale

    @amp.float_function
    def forward(self, input, rois):
        return pool_points_interp(input, rois, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr
