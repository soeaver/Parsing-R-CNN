from functools import wraps
import importlib
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from parsingrcnn.core.config import cfg
from parsingrcnn.model.roi_pooling.functions.roi_pool import RoIPoolFunction
from parsingrcnn.model.roi_crop.functions.roi_crop import RoICropFunction
from parsingrcnn.modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
from parsingrcnn.modeling.roi_xfrom.roi_mask_align.functions.roi_mask_align import RoIMaskAlignFunction
import parsingrcnn.modeling.retinanet_heads as retinanet_heads
import parsingrcnn.modeling.rpn_heads as rpn_heads
import parsingrcnn.modeling.fast_rcnn_heads as fast_rcnn_heads
import parsingrcnn.modeling.mask_rcnn_heads as mask_rcnn_heads
import parsingrcnn.modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import parsingrcnn.modeling.parsing_rcnn_heads as parsing_rcnn_heads
import parsingrcnn.modeling.uv_rcnn_heads as uv_rcnn_heads
import parsingrcnn.utils.blob as blob_utils
import parsingrcnn.utils.net as net_utils

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'parsingrcnn.modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.mapping_to_pytorch = None
        self.orphans_in_pytorch = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        # Mask Branch
        if cfg.MODEL.MASK_ON:
            if cfg.MRCNN.FINEST_LEVEL_ROI:
                self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(
                    self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale[-1])
            else:
                self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(
                    self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        # Keypoints Branch
        if cfg.MODEL.KEYPOINTS_ON:
            if cfg.KRCNN.FINEST_LEVEL_ROI:
                self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                    self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale[-1])
            else:
                self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                    self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)               
            self.Keypoint_Outs = keypoint_rcnn_heads.keypoint_outputs(self.Keypoint_Head.dim_out)

        # Parsing Branch
        if cfg.MODEL.PARSING_ON:
            if cfg.PRCNN.FINEST_LEVEL_ROI:
                self.Parsing_Head = get_func(cfg.PRCNN.ROI_PARSING_HEAD)(
                    self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale[-1])
            else:
                self.Parsing_Head = get_func(cfg.PRCNN.ROI_PARSING_HEAD)(
                    self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Parsing_Outs = parsing_rcnn_heads.parsing_rcnn_outputs(self.Parsing_Head.dim_out)

        # UV Branch
        if cfg.MODEL.UV_ON:
            if cfg.UVRCNN.FINEST_LEVEL_ROI:
                self.UV_Head = get_func(cfg.UVRCNN.ROI_HEAD)(
                    self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale[-1])
            else:
                self.UV_Head = get_func(cfg.UVRCNN.ROI_HEAD)(
                    self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.UV_Outs = uv_rcnn_heads.uv_rcnn_outputs(self.UV_Head.dim_out)

        self._init_modules()

    def _init_modules(self):
        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, im_info, roidb=None, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, roidb, **rpn_kwargs)

    def _forward(self, data, im_info, roidb=None, **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)

        rpn_ret = self.RPN(blob_conv, im_info, roidb)

        # if self.training:
        #     # can be used to infer fg/bg ratio
        #     return_dict['rois_label'] = rpn_ret['labels_int32']

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret)
            else:
                box_feat = self.Box_Head(blob_conv, rpn_ret)
            cls_score, bbox_pred = self.Box_Outs(box_feat)
        else:
            # TODO: complete the returns for RPN only situation
            pass

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss
            rpn_kwargs.update(dict(
                (k, rpn_ret[k]) for k in rpn_ret.keys()
                if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
            ))
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
            if cfg.FPN.FPN_ON:
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                    return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
            else:
                return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

            # bbox loss
            loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
            return_dict['losses']['loss_cls'] = loss_cls
            return_dict['losses']['loss_bbox'] = loss_bbox
            return_dict['metrics']['accuracy_cls'] = accuracy_cls

            if cfg.MODEL.MASK_ON:
                if cfg.MRCNN.FINEST_LEVEL_ROI:
                    mask_feat = self.Mask_Head(blob_conv[-1], rpn_ret)
                else:
                    mask_feat = self.Mask_Head(blob_conv, rpn_ret)
                if getattr(self.Mask_Head, 'SHARE_RES5', False):
                    mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                               roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
                mask_pred = self.Mask_Outs(mask_feat)
                # return_dict['mask_pred'] = mask_pred
                # mask loss
                loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
                return_dict['losses']['loss_mask'] = loss_mask

            if cfg.MODEL.KEYPOINTS_ON:
                if cfg.KRCNN.FINEST_LEVEL_ROI:
                    kps_feat = self.Keypoint_Head(blob_conv[-1], rpn_ret)
                else:
                    kps_feat = self.Keypoint_Head(blob_conv, rpn_ret)
                if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                    # No corresponding keypoint head implemented yet (Neither in Detectron)
                    # Also, rpn need to generate the label 'roi_has_keypoints_int32'
                    kps_feat = self.Keypoint_Head(res5_feat, rpn_ret,
                                                  roi_has_keypoints_int32=rpn_ret['roi_has_keypoint_int32'])
                kps_pred = self.Keypoint_Outs(kps_feat)
                # return_dict['keypoints_pred'] = kps_pred
                # keypoints loss
                if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'])
                else:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'],
                        rpn_ret['keypoint_loss_normalizer'])
                return_dict['losses']['loss_kps'] = loss_keypoints

            if cfg.MODEL.PARSING_ON:
                if cfg.PRCNN.FINEST_LEVEL_ROI:
                    parsing_feat = self.Parsing_Head(blob_conv[-1], rpn_ret)
                else:
                    parsing_feat = self.Parsing_Head(blob_conv, rpn_ret)
                parsing_pred = self.Parsing_Outs(parsing_feat)
                # return_dict['parsing_pred'] = parsing_pred
                # parsing loss
                loss_parsing = parsing_rcnn_heads.parsing_rcnn_losses(
                    parsing_pred, rpn_ret['parsing_int32'], rpn_ret['parsing_weights'])
                return_dict['losses']['loss_parsing'] = loss_parsing

            if cfg.MODEL.UV_ON:
                if cfg.UVRCNN.FINEST_LEVEL_ROI:
                    UV_feat = self.UV_Head(blob_conv[-1], rpn_ret)
                else:
                    UV_feat = self.UV_Head(blob_conv, rpn_ret)
                UV_pred_Ann, UV_pred_Index, UV_pred_U, UV_pred_V = self.UV_Outs(UV_feat)
                # return_dict['UV_pred'] = UV_pred
                # UV loss
                loss_UV = uv_rcnn_heads.uv_losses(
                    UV_pred_Ann, UV_pred_Index, UV_pred_U, UV_pred_V, 
                    rpn_ret['uv_ann_labels'], rpn_ret['uv_ann_weights'], rpn_ret['uv_X_points'],
                    rpn_ret['uv_Y_points'], rpn_ret['uv_Ind_points'], rpn_ret['uv_I_points'],
                    rpn_ret['uv_U_points'], rpn_ret['uv_V_points'], rpn_ret['uv_point_weights']
                    )
                return_dict['losses']['loss_Upoints'] = loss_UV[0]
                return_dict['losses']['loss_Vpoints'] = loss_UV[1]
                return_dict['losses']['loss_seg_AnnIndex'] = loss_UV[2]
                return_dict['losses']['loss_IndexUVPoints'] = loss_UV[3]

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rpn_ret['rois']
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0, panet=False):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign', 'RoIMaskAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                if not panet:
                    bl_rois = blob_rois + '_fpn' + str(lvl)
                else:
                    bl_rois = blob_rois
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    elif method == 'RoIMaskAlign':
                        xform_out = RoIMaskAlignFunction(
                            resolution, resolution, sc, sampling_ratio,
                            cfg.FAST_RCNN.ROI_XFORM_SPATIAL_SHIFT,
                            cfg.FAST_RCNN.ROI_XFORM_HALF_PART,
                            cfg.FAST_RCNN.ROI_XFORM_ROI_SCALE)(bl_in, rois)
                    bl_out_list.append(xform_out)
            if not panet:
                # The pooled features from all levels are concatenated along the
                # batch dimension into a single 4D tensor.
                xform_shuffled = torch.cat(bl_out_list, dim=0)

                # Unshuffle to match rois from dataloader
                device_id = xform_shuffled.get_device()
                restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
                restore_bl = Variable(
                    torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
                xform_out = xform_shuffled[restore_bl]
            else:
                return bl_out_list
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)
            elif method == 'RoIMaskAlign':
                xform_out = RoIMaskAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio,
                    cfg.FAST_RCNN.ROI_XFORM_SPATIAL_SHIFT,
                    cfg.FAST_RCNN.ROI_XFORM_HALF_PART,
                    cfg.FAST_RCNN.ROI_XFORM_ROI_SCALE)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        if cfg.MRCNN.FINEST_LEVEL_ROI:
            mask_feat = self.Mask_Head(blob_conv[-1], rpn_blob)
        else:
            mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @check_inference
    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        if cfg.KRCNN.FINEST_LEVEL_ROI:
            kps_feat = self.Keypoint_Head(blob_conv[-1], rpn_blob)
        else:
            kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return kps_pred

    @check_inference
    def parsing_net(self, blob_conv, rpn_blob):
        """For inference"""
        if cfg.PRCNN.FINEST_LEVEL_ROI:
            parsing_feat = self.Parsing_Head(blob_conv[-1], rpn_blob)
        else:
            parsing_feat = self.Parsing_Head(blob_conv, rpn_blob)
        parsing_pred = self.Parsing_Outs(parsing_feat)
        return parsing_pred

    @check_inference
    def UV_net(self, blob_conv, rpn_blob):
        """For inference"""
        if cfg.UVRCNN.FINEST_LEVEL_ROI:
            UV_feat = self.UV_Head(blob_conv[-1], rpn_blob)
        else:
            UV_feat = self.UV_Head(blob_conv, rpn_blob)
        UV_pred_Ann, UV_pred_Index, UV_pred_U, UV_pred_V = self.UV_Outs(UV_feat)
        return UV_pred_Ann, UV_pred_Index, UV_pred_U, UV_pred_V

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    @property
    def pytorch_weight_mapping(self):
        if self.mapping_to_pytorch is None:
            d_wmap = {}  # pytorch_weight_mapping
            d_orphan = []  # pytorch orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.pytorch_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_pytorch = d_wmap
            self.orphans_in_pytorch = d_orphan

        return self.mapping_to_pytorch, self.orphans_in_pytorch

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value

      
class RetinaNet(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.mapping_to_pytorch = None
        self.orphans_in_pytorch = None

        self.training = False

        # Backbone for feature extraction
        '''
        FPN.fpn_ResNet50_conv5_body for example
        get_func will get the function by name
        here is where FPN_ResNet lay functions are called
        It returns a ResNet that has been associated with FPN
        '''
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()
        self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
        self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]
        self.Box_Head = retinanet_heads.retinanet_outputs(self.Conv_Body.dim_out)

        self._init_modules()

    def _init_modules(self):
        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, im_info, roidb=None, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, roidb, **rpn_kwargs)

    def _forward(self, data, im_info, roidb=None, **rpn_kwargs):
        # im_data = data
        # usqueeze is needed because this is a single image, but input will require a batch size,
        # unsqueeze will add the leading 1 dimension, saying batch_size = 1

        if type(data) is np.ndarray:
            im_data = torch.from_numpy(data).type(torch.cuda.FloatTensor).unsqueeze_(0)
        else:
            im_data = data

        return_dict = {}  # A dict to collect return variables
        blob_conv = self.Conv_Body(im_data)
        if not self.training:
            return_dict['blob_conv'] = blob_conv
        cls_score, bbox_pred = self.Box_Head(blob_conv, self.training)

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}

            loss_bbox, loss_cls, accuracy_cls = retinanet_heads.fpn_retinanet_losses(
                rpn_kwargs, cls_score, bbox_pred)

            if cfg.FPN.FPN_ON:
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    suffix = 'fpn{}'.format(lvl)
                    return_dict['losses']['loss_cls_fpn%d' % lvl] = loss_cls['fl_' + suffix]
                    return_dict['losses']['loss_bbox_fpn%d' % lvl] = loss_bbox['retnet_loss_bbox_' + suffix]
            else:
                return_dict['losses']['loss_cls'] = loss_cls
                return_dict['losses']['loss_bbox'] = loss_bbox

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
        else:
            # Testing
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred

        return return_dict

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    @property
    def pytorch_weight_mapping(self):
        if self.mapping_to_pytorch is None:
            d_wmap = {}  # pytorch_weight_mapping
            d_orphan = []  # pytorch orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.pytorch_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_pytorch = d_wmap
            self.orphans_in_pytorch = d_orphan

        return self.mapping_to_pytorch, self.orphans_in_pytorch
    
    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
