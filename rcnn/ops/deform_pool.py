import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from models.ops import _C


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None,
                sample_per_part=4, trans_std=.0):
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError

        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        _C.deform_psroi_pooling_forward(
            data,
            rois,
            offset,
            output,
            output_count,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.out_channels,
            ctx.group_size,
            ctx.out_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std
        )

        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)

        _C.deform_psroi_pooling_backward(
            grad_output,
            data,
            rois,
            offset,
            output_count,
            grad_input,
            grad_offset,
            ctx.no_trans,
            ctx.spatial_scale,
            ctx.out_channels,
            ctx.group_size,
            ctx.out_size,
            ctx.part_size,
            ctx.sample_per_part,
            ctx.trans_std
        )
        return (grad_input, grad_rois, grad_offset, None, None, None, None, None, None, None, None)


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None,
                 sample_per_part=4, trans_std=.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(
            data, rois, offset, self.spatial_scale, self.out_size,
            self.out_channels, self.no_trans, self.group_size, self.part_size,
            self.sample_per_part, self.trans_std)


class DeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_channels=1024):
        super(DeformRoIPoolingPack,
              self).__init__(spatial_scale, out_size, out_channels, no_trans,
                             group_size, part_size, sample_per_part, trans_std)

        self.deform_fc_channels = deform_fc_channels

        if not no_trans:
            self.offset_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                   self.out_size, self.out_channels, True,
                                   self.group_size, self.part_size,
                                   self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None,
                 sample_per_part=4, trans_std=.0, deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(
            spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std
        )

        self.deform_fc_channels = deform_fc_channels

        if not no_trans:
            self.offset_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels, self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels, self.out_size * self.out_size * 2)
            )

            self.mask_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels, self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels, self.out_size * self.out_size * 1),
                nn.Sigmoid()
            )

            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            self.mask_fc[2].weight.data.zero_()
            self.mask_fc[2].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans,
                self.group_size, self.part_size, self.sample_per_part, self.trans_std
            )
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, True,
                self.group_size, self.part_size, self.sample_per_part, self.trans_std
            )
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.out_size, self.out_size)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans,
                self.group_size, self.part_size, self.sample_per_part, self.trans_std
            ) * mask
