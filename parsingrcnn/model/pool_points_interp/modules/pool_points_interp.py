from torch.nn.modules.module import Module
from ..functions.pool_points_interp import PoolpointsinterpFunction


class Poolpointsinterp(Module):
    def __init__(self, spatial_scale=1.0):
        super(Poolpointsinterp, self).__init__()

        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return PoolpointsinterpFunction(self.spatial_scale)(features, rois)
