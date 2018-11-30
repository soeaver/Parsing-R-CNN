import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from parsingrcnn.core.config import cfg
from parsingrcnn.model.pool_points_interp.modules.pool_points_interp import Poolpointsinterp
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils
import parsingrcnn.modeling.nonlocal_helper as nonlocal_helper


# ---------------------------------------------------------------------------- #
# uv_rois R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

class uv_rcnn_outputs(nn.Module):
    """Mask R-CNN uv specific outputs: uv heatmaps."""
    def __init__(self, dim_in):
        super().__init__()
        self.deconv_Ann = nn.ConvTranspose2d(
            dim_in, 15, cfg.UVRCNN.DECONV_KERNEL,
            2, padding=int(cfg.UVRCNN.DECONV_KERNEL / 2) - 1
        )
        self.deconv_Index = nn.ConvTranspose2d(
            dim_in, cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.DECONV_KERNEL,
            2, padding=int(cfg.UVRCNN.DECONV_KERNEL / 2) - 1
        )
        self.deconv_U = nn.ConvTranspose2d(
            dim_in, cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.DECONV_KERNEL,
            2, padding=int(cfg.UVRCNN.DECONV_KERNEL / 2) - 1
        )
        self.deconv_V = nn.ConvTranspose2d(
            dim_in, cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.DECONV_KERNEL,
            2, padding=int(cfg.UVRCNN.DECONV_KERNEL / 2) - 1
        )
        
        self.upsample_Ann = mynn.BilinearInterpolation2d(
            cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.UP_SCALE
        )
        self.upsample_Index = mynn.BilinearInterpolation2d(
            cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.UP_SCALE
        )
        self.upsample_U = mynn.BilinearInterpolation2d(
            cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.UP_SCALE
        )
        self.upsample_V = mynn.BilinearInterpolation2d(
            cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.NUM_PATCHES+1, cfg.UVRCNN.UP_SCALE
        )

        self._init_weights()

    def _init_weights(self):
        if cfg.UVRCNN.CONV_INIT == 'GaussianFill':
            init.normal_(self.deconv_Ann.weight, std=0.001)
            init.normal_(self.deconv_Index.weight, std=0.001)
            init.normal_(self.deconv_U.weight, std=0.001)
            init.normal_(self.deconv_V.weight, std=0.001)
        elif cfg.UVRCNN.CONV_INIT == 'MSRAFill':
            mynn.init.MSRAFill(self.deconv_Ann.weight)
            mynn.init.MSRAFill(self.deconv_Index.weight)
            mynn.init.MSRAFill(self.deconv_U.weight)
            mynn.init.MSRAFill(self.deconv_V.weight)
        else:
            raise ValueError(cfg.UVRCNN.CONV_INIT)
        init.constant_(self.deconv_Ann.bias, 0)
        init.constant_(self.deconv_Index.bias, 0)
        init.constant_(self.deconv_U.bias, 0)
        init.constant_(self.deconv_V.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        detectron_weight_mapping.update({
            'deconv_Ann.weight': 'AnnIndex_lowres_w',
            'deconv_Ann.bias': 'AnnIndex_lowres_b'
        })
        detectron_weight_mapping.update({
            'deconv_Index.weight': 'Index_UV_lowres_w',
            'deconv_Index.bias': 'Index_UV_lowres_b'
        })
        detectron_weight_mapping.update({
            'deconv_U.weight': 'U_lowres_w',
            'deconv_U.bias': 'U_lowres_b'
        })
        detectron_weight_mapping.update({
            'deconv_V.weight': 'V_lowres_w',
            'deconv_V.bias': 'V_lowres_b'
        })

        detectron_weight_mapping.update({
            'upsample.upconv.weight': None,  # 0: don't load from or save to checkpoint
            'upsample.upconv.bias': None
        })
        detectron_weight_mapping.update({
            'upsample_Ann.upconv.weight': None,  # 0: don't load from or save to checkpoint
            'upsample_Ann.upconv.bias': None
        })
        detectron_weight_mapping.update({
            'upsample_Index.upconv.weight': None,  # 0: don't load from or save to checkpoint
            'upsample_Index.upconv.bias': None
        })
        detectron_weight_mapping.update({
            'upsample_U.upconv.weight': None,  # 0: don't load from or save to checkpoint
            'upsample_U.upconv.bias': None
        })
        detectron_weight_mapping.update({
            'upsample_V.upconv.weight': None,  # 0: don't load from or save to checkpoint
            'upsample_V.upconv.bias': None
        })


        return detectron_weight_mapping, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x):
        x_Ann = self.deconv_Ann(x)
        x_Index = self.deconv_Index(x)
        x_U = self.deconv_U(x)
        x_V = self.deconv_V(x)

        device_id = x_Ann.get_device()
        x_Ann_zero = torch.zeros(x_Ann.size()[0], 10, x_Ann.size()[2], x_Ann.size()[3]).cuda(device_id)
        x_Ann = torch.cat((x_Ann, x_Ann_zero), dim=1)
        x_Ann = self.upsample_Ann(x_Ann)
        x_Index = self.upsample_Index(x_Index)
        x_U = self.upsample_U(x_U)
        x_V = self.upsample_V(x_V)

        return x_Ann, x_Index, x_U, x_V


def uv_losses(
        UV_pred_Ann, UV_pred_Index, UV_pred_U, UV_pred_V,
        uv_ann_labels, uv_ann_weights, uv_X_points,
        uv_Y_points, uv_Ind_points, uv_I_points,
        uv_U_points, uv_V_points, uv_point_weights
    ):

    device_id = UV_pred_Ann.get_device()

    uv_X_points = uv_X_points.reshape((-1 ,1))
    uv_Y_points = uv_Y_points.reshape((-1 ,1))
    uv_Ind_points = uv_Ind_points.reshape((-1 ,1))
    uv_I_points = uv_I_points.reshape(-1).astype('int64')
    uv_I_points = Variable(torch.from_numpy(uv_I_points)).cuda(device_id)

    Coordinate_Shapes = np.concatenate((uv_Ind_points, uv_X_points, uv_Y_points), axis = 1)
    Coordinate_Shapes = Variable(torch.from_numpy(Coordinate_Shapes)).cuda(device_id)

    uv_U_points = uv_U_points.reshape((-1, cfg.UVRCNN.NUM_PATCHES+1, 196))
    uv_U_points = uv_U_points.transpose((0,2,1))
    uv_U_points = uv_U_points.reshape((1, 1, -1, cfg.UVRCNN.NUM_PATCHES+1))
    uv_U_points = Variable(torch.from_numpy(uv_U_points)).cuda(device_id)

    uv_V_points = uv_V_points.reshape((-1, cfg.UVRCNN.NUM_PATCHES+1, 196))
    uv_V_points = uv_V_points.transpose((0,2,1))
    uv_V_points = uv_V_points.reshape((1, 1, -1, cfg.UVRCNN.NUM_PATCHES+1))
    uv_V_points = Variable(torch.from_numpy(uv_V_points)).cuda(device_id)

    uv_point_weights = uv_point_weights.reshape((-1, cfg.UVRCNN.NUM_PATCHES+1, 196))
    uv_point_weights = uv_point_weights.transpose((0,2,1))
    uv_point_weights = uv_point_weights.reshape((1, 1, -1, cfg.UVRCNN.NUM_PATCHES+1))
    uv_point_weights = Variable(torch.from_numpy(uv_point_weights)).cuda(device_id)

    PPI_op = Poolpointsinterp()
    interp_U = PPI_op(UV_pred_U, Coordinate_Shapes)
    interp_V = PPI_op(UV_pred_V, Coordinate_Shapes)
    interp_Index_UV = PPI_op(UV_pred_Index, Coordinate_Shapes)

    UV_pred_Ann = torch.transpose(UV_pred_Ann, 1, 3)
    UV_pred_Ann = torch.transpose(UV_pred_Ann, 1, 2)

    uv_ann_labels = uv_ann_labels.reshape(-1).astype('int64')
    uv_ann_weights = uv_ann_weights.reshape(-1).squeeze()

    uv_ann_labels = Variable(torch.from_numpy(uv_ann_labels)).cuda(device_id)
    uv_ann_weights = Variable(torch.from_numpy(uv_ann_weights)).cuda(device_id)
    UV_pred_Ann = UV_pred_Ann.contiguous().view(-1, cfg.UVRCNN.NUM_PATCHES+1)

    loss_seg_AnnIndex = F.cross_entropy(UV_pred_Ann, uv_ann_labels, reduce=False)
    if torch.sum(uv_ann_weights):
        loss_seg_AnnIndex = torch.sum(loss_seg_AnnIndex * uv_ann_weights) / torch.sum(uv_ann_weights)
    else:
        loss_seg_AnnIndex = torch.sum(loss_seg_AnnIndex * uv_ann_weights)
    loss_seg_AnnIndex *= cfg.UVRCNN.INDEX_WEIGHTS

    loss_IndexUVPoints = F.cross_entropy(interp_Index_UV, uv_I_points)
    loss_IndexUVPoints *= cfg.UVRCNN.PART_WEIGHTS

    loss_Upoints = net_utils.smooth_l1_loss(interp_U, uv_U_points, 
        uv_point_weights, uv_point_weights)
    loss_Upoints *= cfg.UVRCNN.POINT_REGRESSION_WEIGHTS

    loss_Vpoints = net_utils.smooth_l1_loss(interp_V, uv_V_points, 
        uv_point_weights, uv_point_weights)
    loss_Vpoints *= cfg.UVRCNN.POINT_REGRESSION_WEIGHTS

    loss = [loss_Upoints, loss_Vpoints, loss_seg_AnnIndex, loss_IndexUVPoints]

    return loss


# ---------------------------------------------------------------------------- #
# uv_rois heads
# ---------------------------------------------------------------------------- #

class roi_uv_head_v1convX(nn.Module):
    """Mask R-CNN uv head. v1convX design: X * (conv)."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.UVRCNN.CONV_HEAD_DIM
        kernel_size = cfg.UVRCNN.CONV_HEAD_KERNEL
        pad_size = kernel_size // 2
        module_list = []
        for _ in range(cfg.UVRCNN.NUM_STACKED_CONVS):
            module_list.append(nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size))
            module_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        self.body_conv_fcn = nn.Sequential(*module_list)
        self.dim_out = hidden_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.UVRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.01)
            elif cfg.UVRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                ValueError('Unexpected cfg.UVRCNN.CONV_INIT: {}'.format(cfg.UVRCNN.CONV_INIT))
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        for i in range(cfg.UVRCNN.NUM_STACKED_CONVS):
            detectron_weight_mapping['body_conv_fcn.%d.weight' % (2*i)] = 'body_conv_fcn%d_w' % (i+1)
            detectron_weight_mapping['body_conv_fcn.%d.bias' % (2*i)] = 'body_conv_fcn%d_b' % (i+1)

        return detectron_weight_mapping, orphan_in_detectron

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='uv_rois',
            method=cfg.UVRCNN.ROI_XFORM_METHOD,
            resolution=cfg.UVRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.UVRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.body_conv_fcn(x)
        return x


class roi_uv_head_aspp_convx(nn.Module):
    """Mask R-CNN uv_rois head. v1convX design: X * (conv)."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.UVRCNN.CONV_HEAD_DIM
        kernel_size = cfg.UVRCNN.CONV_HEAD_KERNEL
        aspp_dim = cfg.UVRCNN.ASPP_DIM
        d1, d2, d3 = cfg.UVRCNN.ASPP_DILATION
        pad_size = kernel_size // 2
        module_list = []
        before_aspp_list = []
        after_aspp_list = []
        for _ in range(cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP):
            before_aspp_list.append(
                nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size)
            )
            before_aspp_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        if cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP > 0:
            self.conv_before_aspp = nn.Sequential(*before_aspp_list)

        aspp1_list = []
        aspp2_list = []
        aspp3_list = []
        aspp4_list = []
        aspp5_list = []
        feat_list = []

        aspp1_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 1, 1), 
            nn.ReLU(inplace=True)
        ])

        aspp2_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 3, 1, d1, dilation=d1), 
            nn.ReLU(inplace=True)
        ])

        aspp3_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 3, 1, d2, dilation=d2),
            nn.ReLU(inplace=True)
        ])

        aspp4_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 3, 1, d3, dilation=d3),
            nn.ReLU(inplace=True)
        ])

        aspp5_list.extend([
            nn.AvgPool2d(cfg.UVRCNN.ROI_XFORM_RESOLUTION, 1, 0),
            nn.Conv2d(dim_in, aspp_dim, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=cfg.UVRCNN.ROI_XFORM_RESOLUTION, mode='nearest')
        ])

        feat_list.extend([
            nn.Conv2d(aspp_dim * 5, hidden_dim, 1, 1), 
            nn.ReLU(inplace=True)
        ])


        self.aspp1 = nn.Sequential(*aspp1_list)
        self.aspp2 = nn.Sequential(*aspp2_list)
        self.aspp3 = nn.Sequential(*aspp3_list)
        self.aspp4 = nn.Sequential(*aspp4_list)
        self.aspp5 = nn.Sequential(*aspp5_list)
        self.feat = nn.Sequential(*feat_list)

        for _ in range(cfg.UVRCNN.NUM_CONVS_AFTER_ASPP):
            after_aspp_list.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, pad_size)
            )
            after_aspp_list.append(nn.ReLU(inplace=True))
        if cfg.UVRCNN.NUM_CONVS_AFTER_ASPP > 0:
            self.conv_after_aspp = nn.Sequential(*after_aspp_list)

        self.dim_out = hidden_dim
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.UVRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.01)
            elif cfg.UVRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                ValueError('Unexpected cfg.UVRCNN.CONV_INIT: {}'.format(cfg.UVRCNN.CONV_INIT))
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        d1, d2, d3 = cfg.UVRCNN.ASPP_DILATION
        for i in range(cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP):
            detectron_weight_mapping['conv_before_aspp.%d.weight' % (2*i)] = \
                'aspp_conv%d_w' % (i+1)
            detectron_weight_mapping['conv_before_aspp.%d.bias' % (2*i)] = \
                'aspp_conv%d_b' % (i+1)

        detectron_weight_mapping.update({
            'aspp1.0.weight': 'aspp_conv1_1_w',
            'aspp1.0.bias': 'aspp_conv1_1_b',
            'aspp2.0.weight': 'aspp_conv3_{}d_w'.format(str(d1)),
            'aspp2.0.bias': 'aspp_conv3_{}d_b'.format(str(d1)),
            'aspp3.0.weight': 'aspp_conv3_{}d_w'.format(str(d2)),
            'aspp3.0.bias': 'aspp_conv3_{}d_b'.format(str(d2)),
            'aspp4.0.weight': 'aspp_conv3_{}d_w'.format(str(d3)),
            'aspp4.0.bias': 'aspp_conv3_{}d_b'.format(str(d3)),
            'aspp5.1.weight': 'image_pool_conv_w',
            'aspp5.1.bias': 'image_pool_conv_b',
            'feat.0.weight': 'feat_w',
            'feat.0.bias': 'feat_b',
        })

        for i in range(cfg.UVRCNN.NUM_CONVS_AFTER_ASPP):
            detectron_weight_mapping['conv_after_aspp.%d.weight' % (2*i)] = \
                'aspp_conv%d_w' % (i + 1 + cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP)
            detectron_weight_mapping['conv_after_aspp.%d.bias' % (2*i)] = \
                'aspp_conv%d_b' % (i + 1 + cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP)

        return detectron_weight_mapping, orphan_in_detectron

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='uv_rois',
            method=cfg.UVRCNN.ROI_XFORM_METHOD,
            resolution=cfg.UVRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.UVRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        if cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP > 0:
            x = self.conv_before_aspp(x)
        x = torch.cat(
            (self.aspp1(x), self.aspp2(x), self.aspp3(x),
            self.aspp4(x), self.aspp5(x)), 1
        )
        x = self.feat(x)
        if cfg.UVRCNN.NUM_CONVS_AFTER_ASPP > 0:
            x = self.conv_after_aspp(x)
        return x


class roi_uv_head_aspp_convx_gn(nn.Module):
    """Mask R-CNN uv_rois head. v1convX design: X * (conv)."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.UVRCNN.CONV_HEAD_DIM
        kernel_size = cfg.UVRCNN.CONV_HEAD_KERNEL
        aspp_dim = cfg.UVRCNN.ASPP_DIM
        d1, d2, d3 = cfg.UVRCNN.ASPP_DILATION
        pad_size = kernel_size // 2
        module_list = []
        before_aspp_list = []
        after_aspp_list = []
        for _ in range(cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP):
            before_aspp_list.append(
                nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size, bias=False)
            )
            before_aspp_list.append(
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), 
                    hidden_dim, eps=cfg.GROUP_NORM.EPSILON)
            )
            before_aspp_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        if cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP > 0:
            self.conv_before_aspp = nn.Sequential(*before_aspp_list)

        aspp1_list = []
        aspp2_list = []
        aspp3_list = []
        aspp4_list = []
        aspp5_list = []
        feat_list = []

        aspp1_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 1, 1, bias=False),
            nn.GroupNorm(
                net_utils.get_group_gn(aspp_dim), 
                aspp_dim, eps=cfg.GROUP_NORM.EPSILON
            ), 
            nn.ReLU(inplace=True)
        ])

        aspp2_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 3, 1, d1, dilation=d1, bias=False),
            nn.GroupNorm(
                net_utils.get_group_gn(aspp_dim), 
                aspp_dim, eps=cfg.GROUP_NORM.EPSILON
            ), 
            nn.ReLU(inplace=True)
        ])

        aspp3_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 3, 1, d2, dilation=d2, bias=False),
            nn.GroupNorm(
                net_utils.get_group_gn(aspp_dim), 
                aspp_dim, eps=cfg.GROUP_NORM.EPSILON
            ), 
            nn.ReLU(inplace=True)
        ])

        aspp4_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 3, 1, d3, dilation=d3, bias=False),
            nn.GroupNorm(
                net_utils.get_group_gn(aspp_dim), 
                aspp_dim, eps=cfg.GROUP_NORM.EPSILON
            ), 
            nn.ReLU(inplace=True)
        ])

        aspp5_list.extend([
            nn.AvgPool2d(cfg.UVRCNN.ROI_XFORM_RESOLUTION, 1, 0),
            nn.Conv2d(dim_in, aspp_dim, 1, 1, bias=False),
            nn.GroupNorm(
                net_utils.get_group_gn(aspp_dim), 
                aspp_dim, eps=cfg.GROUP_NORM.EPSILON
            ), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=cfg.UVRCNN.ROI_XFORM_RESOLUTION, mode='nearest')
        ])

        feat_list.extend([
            nn.Conv2d(aspp_dim * 5, hidden_dim, 1, 1, bias=False),
            nn.GroupNorm(
                net_utils.get_group_gn(hidden_dim), 
                hidden_dim, eps=cfg.GROUP_NORM.EPSILON
            ),
            nn.ReLU(inplace=True)
        ])

        self.aspp1 = nn.Sequential(*aspp1_list)
        self.aspp2 = nn.Sequential(*aspp2_list)
        self.aspp3 = nn.Sequential(*aspp3_list)
        self.aspp4 = nn.Sequential(*aspp4_list)
        self.aspp5 = nn.Sequential(*aspp5_list)
        self.feat = nn.Sequential(*feat_list)

        for _ in range(cfg.UVRCNN.NUM_CONVS_AFTER_ASPP):
            after_aspp_list.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, pad_size, bias=False)
            )
            after_aspp_list.append(
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), 
                    hidden_dim, eps=cfg.GROUP_NORM.EPSILON)
            )
            after_aspp_list.append(nn.ReLU(inplace=True))
        if cfg.UVRCNN.NUM_CONVS_AFTER_ASPP > 0:
            self.conv_after_aspp = nn.Sequential(*after_aspp_list)

        self.dim_out = hidden_dim
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.UVRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.01)
            elif cfg.UVRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                ValueError('Unexpected cfg.UVRCNN.CONV_INIT: {}'.format(cfg.UVRCNN.CONV_INIT))
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        d1, d2, d3 = cfg.UVRCNN.ASPP_DILATION
        for i in range(cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP):
            detectron_weight_mapping['conv_before_aspp.%d.weight' % (3*i)] = \
                'aspp_conv%d_w' % (i+1)
            detectron_weight_mapping['conv_before_aspp.%d.weight' % (3*i+1)] = \
                'aspp_conv%d_gn_s' % (i+1)
            detectron_weight_mapping['conv_before_aspp.%d.bias' % (3*i+1)] = \
                'aspp_conv%d_gn_b' % (i+1)

        detectron_weight_mapping.update({
            'aspp1.0.weight': 'aspp_conv1_1_w',
            'aspp1.1.weight': 'aspp_conv1_1_gn_s',
            'aspp1.1.bias': 'aspp_conv1_1_gn_b',
            'aspp2.0.weight': 'aspp_conv3_{}d_w'.format(str(d1)),
            'aspp2.1.weight': 'aspp_conv3_{}d_gn_s'.format(str(d1)),
            'aspp2.1.bias': 'aspp_conv3_{}d_gn_b'.format(str(d1)),
            'aspp3.0.weight': 'aspp_conv3_{}d_w'.format(str(d2)),
            'aspp3.1.weight': 'aspp_conv3_{}d_gn_s'.format(str(d2)),
            'aspp3.1.bias': 'aspp_conv3_{}d_gn_b'.format(str(d2)),
            'aspp4.0.weight': 'aspp_conv3_{}d_w'.format(str(d3)),
            'aspp4.1.weight': 'aspp_conv3_{}d_gn_s'.format(str(d3)),
            'aspp4.1.bias': 'aspp_conv3_{}d_gn_b'.format(str(d3)),
            'aspp5.1.weight': 'image_pool_conv_w',
            'aspp5.2.weight': 'image_pool_conv_gn_s',
            'aspp5.2.bias': 'image_pool_conv_gn_b',
            'feat.0.weight': 'feat_w',
            'feat.1.weight': 'feat_gn_s',
            'feat.1.bias': 'feat_gn_b',
        })

        for i in range(cfg.UVRCNN.NUM_CONVS_AFTER_ASPP):
            detectron_weight_mapping['conv_after_aspp.%d.weight' % (3*i)] = \
                'aspp_conv%d_w' % (i + 1 + cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP)
            detectron_weight_mapping['conv_after_aspp.%d.weight' % (3*i+1)] = \
                'aspp_conv%d_gn_s' % (i + 1 + cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP)
            detectron_weight_mapping['conv_after_aspp.%d.bias' % (3*i+1)] = \
                'aspp_conv%d_gn_b' % (i + 1 + cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP)

        return detectron_weight_mapping, orphan_in_detectron

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='uv_rois',
            method=cfg.UVRCNN.ROI_XFORM_METHOD,
            resolution=cfg.UVRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.UVRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        if cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP > 0:
            x = self.conv_before_aspp(x)
        x = torch.cat(
            (self.aspp1(x), self.aspp2(x), self.aspp3(x),
            self.aspp4(x), self.aspp5(x)), 1
        )
        x = self.feat(x)
        if cfg.UVRCNN.NUM_CONVS_AFTER_ASPP > 0:
            x = self.conv_after_aspp(x)
        return x

   
class roi_uv_head_aspp_convx_nonlocal(nn.Module):
    """Mask R-CNN uv head. v1convX design: X * (conv) with nonlocal."""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.UVRCNN.CONV_HEAD_DIM
        kernel_size = cfg.UVRCNN.CONV_HEAD_KERNEL
        aspp_dim = cfg.UVRCNN.ASPP_DIM
        d1, d2, d3 = cfg.UVRCNN.ASPP_DILATION
        pad_size = kernel_size // 2
        before_aspp_list = []
        after_aspp_list = []
        for _ in range(cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP):
            before_aspp_list.append(
                nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size)
            )
            before_aspp_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        if cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP > 0:
            self.conv_before_aspp = nn.Sequential(*before_aspp_list)

        aspp1_list = []
        aspp2_list = []
        aspp3_list = []
        aspp4_list = []
        aspp5_list = []
        feat_list = []

        aspp1_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 1, 1),
            nn.ReLU(inplace=True)
        ])

        aspp2_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 3, 1, d1, dilation=d1),
            nn.ReLU(inplace=True)
        ])

        aspp3_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 3, 1, d2, dilation=d2),
            nn.ReLU(inplace=True)
        ])

        aspp4_list.extend([
            nn.Conv2d(dim_in, aspp_dim, 3, 1, d3, dilation=d3),
            nn.ReLU(inplace=True)
        ])

        aspp5_list.extend([
            nn.AvgPool2d(cfg.UVRCNN.ROI_XFORM_RESOLUTION, 1, 0),
            nn.Conv2d(dim_in, aspp_dim, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=cfg.UVRCNN.ROI_XFORM_RESOLUTION, mode='nearest')
        ])

        feat_list.extend([
            nn.Conv2d(aspp_dim * 5, hidden_dim, 1, 1),
            nn.ReLU(inplace=True),
            nonlocal_helper.SpaceNonLocal(hidden_dim, int(hidden_dim * cfg.NONLOCAL.REDUCTION_RATIO), 
                                          hidden_dim)
        ])

        self.aspp1 = nn.Sequential(*aspp1_list)
        self.aspp2 = nn.Sequential(*aspp2_list)
        self.aspp3 = nn.Sequential(*aspp3_list)
        self.aspp4 = nn.Sequential(*aspp4_list)
        self.aspp5 = nn.Sequential(*aspp5_list)
        self.feat = nn.Sequential(*feat_list)

        for _ in range(cfg.UVRCNN.NUM_CONVS_AFTER_ASPP):
            after_aspp_list.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, pad_size)
            )
            after_aspp_list.append(nn.ReLU(inplace=True))
        if cfg.UVRCNN.NUM_CONVS_AFTER_ASPP > 0:
            self.conv_after_aspp = nn.Sequential(*after_aspp_list)

        self.dim_out = hidden_dim
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.UVRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.01)
            elif cfg.UVRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                ValueError('Unexpected cfg.UVRCNN.CONV_INIT: {}'.format(cfg.UVRCNN.CONV_INIT))
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        d1, d2, d3 = cfg.UVRCNN.ASPP_DILATION
        for i in range(cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP):
            detectron_weight_mapping['conv_before_aspp.%d.weight' % (2 * i)] = \
                'aspp_conv%d_w' % (i + 1)
            detectron_weight_mapping['conv_before_aspp.%d.bias' % (2 * i)] = \
                'aspp_conv%d_b' % (i + 1)

        detectron_weight_mapping.update({
            'aspp1.0.weight': 'aspp_conv1_1_w',
            'aspp1.0.bias': 'aspp_conv1_1_b',
            'aspp2.0.weight': 'aspp_conv3_{}d_w'.format(str(d1)),
            'aspp2.0.bias': 'aspp_conv3_{}d_b'.format(str(d1)),
            'aspp3.0.weight': 'aspp_conv3_{}d_w'.format(str(d2)),
            'aspp3.0.bias': 'aspp_conv3_{}d_b'.format(str(d2)),
            'aspp4.0.weight': 'aspp_conv3_{}d_w'.format(str(d3)),
            'aspp4.0.bias': 'aspp_conv3_{}d_b'.format(str(d3)),
            'aspp5.1.weight': 'image_pool_conv_w',
            'aspp5.1.bias': 'image_pool_conv_b',
            'feat.0.weight': 'feat_w',
            'feat.0.bias': 'feat_b',
            'feat.2.theta.weight': 'feat_theta_w',
            'feat.2.theta.bias': 'feat_theta_b',
            'feat.2.phi.weight': 'feat_phi_w',
            'feat.2.phi.bias': 'feat_phi_b',
            'feat.2.g.weight': 'feat_g_w',
            'feat.2.g.bias': 'feat_g_b',
            'feat.2.out.weight': 'feat_out_w',
            'feat.2.out.bias': 'feat_out_b',

        })
        if cfg.NONLOCAL.USE_BN:
            detectron_weight_mapping.update({
                'feat.2.bn.weight': 'feat_bn_s',
                'feat.2.bn.bias': 'feat_bn_b',
                'feat.2.bn.running_mean': 'feat_bn_running_mean',
                'feat.2.bn.running_var': 'feat_bn_running_var',
            })
        if cfg.NONLOCAL.USE_AFFINE:
            detectron_weight_mapping.update({
                'feat.2.affine.weight': 'feat_bn_s',
                'feat.2.affine.bias': 'feat_bn_b',
            })

        for i in range(cfg.UVRCNN.NUM_CONVS_AFTER_ASPP):
            detectron_weight_mapping['conv_after_aspp.%d.weight' % (2 * i)] = \
                'aspp_conv%d_w' % (i + 1 + cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP)
            detectron_weight_mapping['conv_after_aspp.%d.bias' % (2 * i)] = \
                'aspp_conv%d_b' % (i + 1 + cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP)

        return detectron_weight_mapping, orphan_in_detectron

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='uv_rois',
            method=cfg.UVRCNN.ROI_XFORM_METHOD,
            resolution=cfg.UVRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.UVRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        if cfg.UVRCNN.NUM_CONVS_BEFORE_ASPP > 0:
            x = self.conv_before_aspp(x)
        x = torch.cat(
            (self.aspp1(x), self.aspp2(x), self.aspp3(x),
             self.aspp4(x), self.aspp5(x)), 1
        )
        x = self.feat(x)
        if cfg.UVRCNN.NUM_CONVS_AFTER_ASPP > 0:
            x = self.conv_after_aspp(x)
        return x
