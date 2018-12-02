import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils
import parsingrcnn.modeling.nonlocal_helper as nonlocal_helper


# ---------------------------------------------------------------------------- #
# Keypoint R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

class keypoint_outputs(nn.Module):
    """Mask R-CNN keypoint specific outputs: keypoint heatmaps."""
    def __init__(self, dim_in):
        super().__init__()
        self.upsample_heatmap = (cfg.KRCNN.UP_SCALE > 1)

        if cfg.KRCNN.USE_DECONV:
            # Apply ConvTranspose to the feature representation; results in 2x # upsampling
            self.deconv = nn.ConvTranspose2d(
                dim_in, cfg.KRCNN.DECONV_DIM, cfg.KRCNN.DECONV_KERNEL,
                2, padding=int(cfg.KRCNN.DECONV_KERNEL / 2) - 1)
            dim_in = cfg.KRCNN.DECONV_DIM

        if cfg.KRCNN.USE_DECONV_OUTPUT:
            # Use ConvTranspose to predict heatmaps; results in 2x upsampling
            self.classify = nn.ConvTranspose2d(
                dim_in, cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.DECONV_KERNEL,
                2, padding=int(cfg.KRCNN.DECONV_KERNEL / 2 - 1))
        else:
            # Use Conv to predict heatmaps; does no upsampling
            self.classify = nn.Conv2d(dim_in, cfg.KRCNN.NUM_KEYPOINTS, 1, 1, padding=0)

        if self.upsample_heatmap:
            # self.upsample = nn.UpsamplingBilinear2d(scale_factor=cfg.KRCNN.UP_SCALE)
            self.upsample = mynn.BilinearInterpolation2d(
                cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.UP_SCALE)

        self._init_weights()

    def _init_weights(self):
        if cfg.KRCNN.USE_DECONV:
            init.normal_(self.deconv.weight, std=0.01)
            init.constant_(self.deconv.bias, 0)

        if cfg.KRCNN.CONV_INIT == 'GaussianFill':
            init.normal_(self.classify.weight, std=0.001)
        elif cfg.KRCNN.CONV_INIT == 'MSRAFill':
            mynn.init.MSRAFill(self.classify.weight)
        else:
            raise ValueError(cfg.KRCNN.CONV_INIT)
        init.constant_(self.classify.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        if cfg.KRCNN.USE_DECONV:
            detectron_weight_mapping.update({
                'deconv.weight': 'kps_deconv_w',
                'deconv.bias': 'kps_deconv_b'
            })

        if self.upsample_heatmap:
            blob_name = 'kps_score_lowres'
            detectron_weight_mapping.update({
                'upsample.upconv.weight': None,  # 0: don't load from or save to checkpoint
                'upsample.upconv.bias': None
            })
        else:
            blob_name = 'kps_score'
        detectron_weight_mapping.update({
            'classify.weight': blob_name + '_w',
            'classify.bias': blob_name + '_b'
        })

        return detectron_weight_mapping, []

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x):
        if cfg.KRCNN.USE_DECONV:
            x = F.relu(self.deconv(x), inplace=True)
        x = self.classify(x)
        if self.upsample_heatmap:
            x = self.upsample(x)
        return x


def keypoint_losses(kps_pred, keypoint_locations_int32, keypoint_weights,
                    keypoint_loss_normalizer=None):
    if cfg.KRCNN.GAUSS_HEATMAP:
        return keypoint_gauss_heatmap_losses(
            kps_pred, keypoint_locations_int32, 
            keypoint_weights
        )
    else:
        return keypoint_softmax_losses(
            kps_pred, keypoint_locations_int32, 
            keypoint_weights, keypoint_loss_normalizer=keypoint_loss_normalizer
        )

def keypoint_softmax_losses(kps_pred, keypoint_locations_int32, keypoint_weights,
                             keypoint_loss_normalizer=None):
    """Mask R-CNN keypoint specific losses."""
    device_id = kps_pred.get_device()
    kps_target = Variable(torch.from_numpy(
        keypoint_locations_int32.squeeze().astype('int64'))).cuda(device_id)
    keypoint_weights = Variable(torch.from_numpy(keypoint_weights.squeeze())).cuda(device_id)
    # Softmax across **space** (woahh....space!)
    # Note: this is not what is commonly called "spatial softmax"
    # (i.e., softmax applied along the channel dimension at each spatial
    # location); This is softmax applied over a set of spatial locations (i.e.,
    # each spatial location is a "class").
    loss = F.cross_entropy(
        kps_pred.view(-1, cfg.KRCNN.HEATMAP_SIZE**2), kps_target, reduce=False)
    if torch.sum(keypoint_weights):
        loss = torch.sum(loss * keypoint_weights) / torch.sum(keypoint_weights)
    else:
        loss = torch.sum(loss * keypoint_weights)
    loss *= cfg.KRCNN.LOSS_WEIGHT

    if not cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
        # Discussion: the softmax loss above will average the loss by the sum of
        # keypoint_weights, i.e. the total number of visible keypoints. Since
        # the number of visible keypoints can vary significantly between
        # minibatches, this has the effect of up-weighting the importance of
        # minibatches with few visible keypoints. (Imagine the extreme case of
        # only one visible keypoint versus N: in the case of N, each one
        # contributes 1/N to the gradient compared to the single keypoint
        # determining the gradient direction). Instead, we can normalize the
        # loss by the total number of keypoints, if it were the case that all
        # keypoints were visible in a full minibatch. (Returning to the example,
        # this means that the one visible keypoint contributes as much as each
        # of the N keypoints.)
        loss *= keypoint_loss_normalizer.item() # np.float32 to float
    return loss


def keypoint_gauss_heatmap_losses(kps_pred, keypoint_locations_int32, keypoint_weights):
    """Mask R-CNN keypoint specific losses."""
    device_id = kps_pred.get_device()
    kps_target = Variable(torch.from_numpy(
        keypoint_locations_int32)).cuda(device_id)
    keypoint_weights = Variable(
        torch.from_numpy(keypoint_weights)).cuda(device_id)
    loss_fn = nn.MSELoss(reduce=False).cuda(device_id)
    shape = (
            kps_pred.shape[0] * kps_pred.shape[1], 
            kps_pred.shape[2], kps_pred.shape[3]
        )
    loss = loss_fn(kps_pred.view(shape), kps_target)
    loss = loss.mean(dim=2).mean(dim=1)
    loss = torch.sum(loss * keypoint_weights) / torch.sum(keypoint_weights)
    loss *= (cfg.KRCNN.LOSS_WEIGHT / 20.0)

    return loss


# ---------------------------------------------------------------------------- #
# Keypoint heads
# ---------------------------------------------------------------------------- #

class roi_pose_head_v1convX(nn.Module):
    """Mask R-CNN keypoint head. v1convX design: X * (conv)."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.KRCNN.CONV_HEAD_DIM
        kernel_size = cfg.KRCNN.CONV_HEAD_KERNEL
        pad_size = kernel_size // 2
        module_list = []
        for _ in range(cfg.KRCNN.NUM_STACKED_CONVS):
            module_list.append(nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size))
            module_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        self.conv_fcn = nn.Sequential(*module_list)
        self.dim_out = hidden_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.KRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.01)
            elif cfg.KRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                ValueError('Unexpected cfg.KRCNN.CONV_INIT: {}'.format(cfg.KRCNN.CONV_INIT))
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        for i in range(cfg.KRCNN.NUM_STACKED_CONVS):
            detectron_weight_mapping['conv_fcn.%d.weight' % (2*i)] = 'conv_fcn%d_w' % (i+1)
            detectron_weight_mapping['conv_fcn.%d.bias' % (2*i)] = 'conv_fcn%d_b' % (i+1)

        return detectron_weight_mapping, orphan_in_detectron

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='keypoint_rois',
            method=cfg.KRCNN.ROI_XFORM_METHOD,
            resolution=cfg.KRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.conv_fcn(x)
        return x


class roi_pose_head_v1convX_adp(nn.Module):
    """Mask R-CNN keypoint head. v1convX design: X * (conv) with adp."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.KRCNN.CONV_HEAD_DIM
        kernel_size = cfg.KRCNN.CONV_HEAD_KERNEL
        pad_size = kernel_size // 2

        self.pose_conv1 = nn.ModuleList()
        self.num_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
        for i in range(self.num_levels):
            self.pose_conv1.append(nn.Sequential(
                nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size),
                nn.ReLU(inplace=True)
            ))
        dim_in = hidden_dim

        module_list = []
        for _ in range(cfg.KRCNN.NUM_STACKED_CONVS - 1):
            module_list.append(nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size))
            module_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        self.conv_fcn = nn.Sequential(*module_list)
        self.dim_out = hidden_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.KRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.01)
            elif cfg.KRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                ValueError('Unexpected cfg.KRCNN.CONV_INIT: {}'.format(cfg.KRCNN.CONV_INIT))
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        for i in range(self.num_levels):
            detectron_weight_mapping.update({
                'pose_conv1.{}.0.weight'.format(i): 'panet{}_pose_w'.format(i+1),
                'pose_conv1.{}.0.bias'.format(i): 'panet{}_pose_b'.format(i + 1)
            })

        for i in range(cfg.KRCNN.NUM_STACKED_CONVS - 1):
            detectron_weight_mapping['conv_fcn.%d.weight' % (2*i)] = 'conv_fcn%d_w' % (i+1)
            detectron_weight_mapping['conv_fcn.%d.bias' % (2*i)] = 'conv_fcn%d_b' % (i+1)

        return detectron_weight_mapping, orphan_in_detectron

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='keypoint_rois',
            method=cfg.KRCNN.ROI_XFORM_METHOD,
            resolution=cfg.KRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO,
            panet=True
        )
        for i in range(len(x)):
            x[i] = self.pose_conv1[i](x[i])
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]
        x = self.conv_fcn(x)
        return x


class roi_pose_head_gce_convX(nn.Module):
    """Mask R-CNN pose head. v1convX design: X * (conv) with gce module."""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.KRCNN.CONV_HEAD_DIM
        kernel_size = cfg.KRCNN.CONV_HEAD_KERNEL
        aspp_dim = cfg.KRCNN.ASPP_DIM
        d1, d2, d3 = cfg.KRCNN.ASPP_DILATION
        pad_size = kernel_size // 2
        before_aspp_list = []
        after_aspp_list = []
        for _ in range(cfg.KRCNN.NUM_CONVS_BEFORE_ASPP):
            before_aspp_list.append(
                nn.Conv2d(dim_in, hidden_dim, kernel_size, 1, pad_size)
            )
            before_aspp_list.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        if cfg.KRCNN.NUM_CONVS_BEFORE_ASPP > 0:
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
            nn.AvgPool2d(cfg.KRCNN.ROI_XFORM_RESOLUTION, 1, 0),
            nn.Conv2d(dim_in, aspp_dim, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=cfg.KRCNN.ROI_XFORM_RESOLUTION, mode='nearest')
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

        for _ in range(cfg.KRCNN.NUM_CONVS_AFTER_ASPP):
            after_aspp_list.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, pad_size)
            )
            after_aspp_list.append(nn.ReLU(inplace=True))
        if cfg.KRCNN.NUM_CONVS_AFTER_ASPP > 0:
            self.conv_after_aspp = nn.Sequential(*after_aspp_list)

        self.dim_out = hidden_dim
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if cfg.KRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.01)
            elif cfg.KRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                ValueError('Unexpected cfg.KRCNN.CONV_INIT: {}'.format(cfg.KRCNN.CONV_INIT))
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        orphan_in_detectron = []
        d1, d2, d3 = cfg.KRCNN.ASPP_DILATION
        for i in range(cfg.KRCNN.NUM_CONVS_BEFORE_ASPP):
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

        for i in range(cfg.KRCNN.NUM_CONVS_AFTER_ASPP):
            detectron_weight_mapping['conv_after_aspp.%d.weight' % (2 * i)] = \
                'aspp_conv%d_w' % (i + 1 + cfg.KRCNN.NUM_CONVS_BEFORE_ASPP)
            detectron_weight_mapping['conv_after_aspp.%d.bias' % (2 * i)] = \
                'aspp_conv%d_b' % (i + 1 + cfg.KRCNN.NUM_CONVS_BEFORE_ASPP)

        return detectron_weight_mapping, orphan_in_detectron

    def pytorch_weight_mapping(self):
        return {}, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='keypoint_rois',
            method=cfg.KRCNN.ROI_XFORM_METHOD,
            resolution=cfg.KRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        if cfg.KRCNN.NUM_CONVS_BEFORE_ASPP > 0:
            x = self.conv_before_aspp(x)
        x = torch.cat(
            (self.aspp1(x), self.aspp2(x), self.aspp3(x),
             self.aspp4(x), self.aspp5(x)), 1
        )
        x = self.feat(x)
        if cfg.KRCNN.NUM_CONVS_AFTER_ASPP > 0:
            x = self.conv_after_aspp(x)
        return x
    
