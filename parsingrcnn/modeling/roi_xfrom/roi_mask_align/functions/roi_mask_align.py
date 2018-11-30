import torch
from torch.autograd import Function
from .._ext import roi_mask_align

import logging
logger = logging.getLogger(__name__)

# TODO use save_for_backward instead
class RoIMaskAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio, spatial_shift, half_part, roi_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.spatial_shift = float(spatial_shift)
        self.half_part = int(half_part)
        self.roi_scale = float(roi_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            roi_mask_align.roi_mask_align_forward_cuda(self.aligned_height,
                                                       self.aligned_width,
                                                       self.spatial_scale, 
                                                       self.sampling_ratio, 
                                                       self.spatial_shift,
                                                       self.half_part,
                                                       self.roi_scale,
                                                       features,
                                                       rois, 
                                                       output)
        else:
            raise NotImplementedError

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = self.rois.new(batch_size, num_channels, data_height, data_width).zero_()
        roi_mask_align.roi_mask_align_backward_cuda(self.aligned_height,
                                                    self.aligned_width,
                                                    self.spatial_scale, 
                                                    self.sampling_ratio, 
                                                    self.spatial_shift,
                                                    self.half_part,
                                                    self.roi_scale,
                                                    grad_output,
                                                    self.rois, 
                                                    grad_input)
        return grad_input, None
