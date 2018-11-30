from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
from ..functions.roi_mask_align import RoIMaskAlignFunction
 

class RoIMaskAlign(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio, spatial_shift, half_part, roi_scale):
        super(RoIMaskAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.spatial_shift = float(spatial_shift)
        self.half_part = int(half_part)
        self.roi_scale = float(roi_scale)

    def forward(self, features, rois):
        return RoIMaskAlignFunction(self.aligned_height, self.aligned_width,
                                self.spatial_scale, self.sampling_ratio,
                                self.spatial_shift, self.half_part,
                                self.roi_scale)(features, rois)

class RoIMaskAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio, spatial_shift, half_part, roi_scale):
        super(RoIMaskAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.spatial_shift = float(spatial_shift)
        self.half_part = int(half_part)
        self.roi_scale = float(roi_scale)

    def forward(self, features, rois):
        x =  RoIMaskAlignFunction(self.aligned_height+1, self.aligned_width+1,
                                  self.spatial_scale, self.sampling_ratio,
                                  self.spatial_shift, self.half_part,
                                  self.roi_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)

class RoIMaskAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio, spatial_shift, half_part, roi_scale):
        super(RoIMaskAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.spatial_shift = float(spatial_shift)
        self.half_part = int(half_part)
        self.roi_scale = float(roi_scale)

    def forward(self, features, rois):
        x =  RoIMaskAlignFunction(self.aligned_height+1, self.aligned_width+1,
                                  self.spatial_scale, self.sampling_ratio,
                                  self.spatial_shift, self.half_part,
                                  self.roi_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)
