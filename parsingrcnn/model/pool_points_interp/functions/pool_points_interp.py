import torch
from torch.autograd import Function
from .._ext import pool_points_interp


# TODO use save_for_backward instead
class PoolpointsinterpFunction(Function):
    def __init__(self, spatial_scale):
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels).zero_()
        if features.is_cuda:
            pool_points_interp.pool_points_interp_forward_cuda(
                self.spatial_scale, features, rois, output
            )
        else:
            raise NotImplementedError

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = self.rois.new(batch_size, num_channels,
                                   data_height, data_width).zero_()
        pool_points_interp.pool_points_interp_backward_cuda(
            self.spatial_scale, grad_output, self.rois, grad_input
        )

        # print grad_input

        return grad_input, None
