import numpy as np

import torch
import torch.nn as nn

from utils.data.structures.image_list import to_image_list
import models.ops as ops
import rcnn.modeling.backbone
import rcnn.modeling.fpn
from rcnn.modeling.rpn.rpn import build_rpn
from rcnn.modeling.fast_rcnn.fast_rcnn import FastRCNN
from rcnn.modeling.cascade_rcnn.cascade_rcnn import CascadeRCNN
from rcnn.modeling.mask_rcnn.mask_rcnn import MaskRCNN
from rcnn.modeling.keypoint_rcnn.keypoint_rcnn import KeypointRCNN
from rcnn.modeling.parsing_rcnn.parsing_rcnn import ParsingRCNN
from rcnn.modeling.uv_rcnn.uv_rcnn import UVRCNN
from rcnn.modeling import registry
from rcnn.core.config import cfg


class Generalized_RCNN(nn.Module):
    def __init__(self, is_train=True):
        super().__init__()

        # Normalization
        if not is_train:
            self.Norm = ops.AffineChannel2d(3)
            self.Norm.weight.data = torch.from_numpy(1. / np.array(cfg.PIXEL_STDS)).float()
            self.Norm.bias.data = torch.from_numpy(-1. * np.array(cfg.PIXEL_MEANS) / np.array(cfg.PIXEL_STDS)).float()

        # Backbone
        conv_body = registry.BACKBONES[cfg.BACKBONE.CONV_BODY]
        self.Conv_Body = conv_body()
        self.dim_in = self.Conv_Body.dim_out
        self.spatial_scale = self.Conv_Body.spatial_scale

        # Feature Pyramid Network
        if cfg.MODEL.FPN_ON:
            fpn_body = registry.FPN_BODY[cfg.FPN.BODY]
            self.Conv_Body_FPN = fpn_body(self.dim_in, self.spatial_scale)
            self.dim_in = self.Conv_Body_FPN.dim_out
            self.spatial_scale = self.Conv_Body_FPN.spatial_scale
        else:
            self.dim_in = self.dim_in[-1:]
            self.spatial_scale = self.spatial_scale[-1:]
            
        # Region Proposal Network
        if cfg.MODEL.RPN_ON:
            self.RPN = build_rpn(self.dim_in)

        # RoI Head
        if cfg.MODEL.FASTER_ON:
            if cfg.MODEL.CASCADE_ON:
                self.Cascade_RCNN = CascadeRCNN(self.dim_in, self.spatial_scale)
            else:
                self.Fast_RCNN = FastRCNN(self.dim_in, self.spatial_scale)

        if cfg.MODEL.MASK_ON:
            self.Mask_RCNN = MaskRCNN(self.dim_in, self.spatial_scale)

        if cfg.MODEL.KEYPOINT_ON:
            self.Keypoint_RCNN = KeypointRCNN(self.dim_in, self.spatial_scale)

        if cfg.MODEL.PARSING_ON:
            self.Parsing_RCNN = ParsingRCNN(self.dim_in, self.spatial_scale)

        if cfg.MODEL.UV_ON:
            self.UV_RCNN = UVRCNN(self.dim_in, self.spatial_scale)

        self._init_modules()

    def _init_modules(self):
        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False
            if cfg.MODEL.FPN_ON:
                for p in self.Conv_Body_FPN.parameters():
                    p.requires_grad = False

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        # Backbone
        conv_features = self.Conv_Body(images.tensors)

        # FPN
        if cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]

        # RPN
        proposal_losses = {}
        if cfg.MODEL.RPN_ON:
            proposals, loss_rpn = self.RPN(images, conv_features, targets)
            proposal_losses.update(loss_rpn)
        else:
            proposals = None

        # RoI Head
        roi_losses = {}
        if cfg.MODEL.FASTER_ON:
            if cfg.MODEL.CASCADE_ON:
                box_features, result, loss_box = self.Cascade_RCNN(conv_features, proposals, targets)
            else:
                box_features, result, loss_box = self.Fast_RCNN(conv_features, proposals, targets)
            roi_losses.update(loss_box)
        else:
            result = proposals

        if cfg.MODEL.MASK_ON:
            x, result, loss_mask = self.Mask_RCNN(conv_features, result, targets)
            roi_losses.update(loss_mask)

        if cfg.MODEL.KEYPOINT_ON:
            x, result, loss_keypoint = self.Keypoint_RCNN(conv_features, result, targets)
            roi_losses.update(loss_keypoint)

        if cfg.MODEL.PARSING_ON:
            x, result, loss_parsing = self.Parsing_RCNN(conv_features, result, targets)
            roi_losses.update(loss_parsing)

        if cfg.MODEL.UV_ON:
            x, result, loss_uv = self.UV_RCNN(conv_features, result, targets)
            roi_losses.update(loss_uv)

        if self.training:
            outputs = {'metrics': {}, 'losses': {}}
            outputs['losses'].update(proposal_losses)
            outputs['losses'].update(roi_losses)
            return outputs

        return result

    def box_net(self, images, targets=None):
        images = to_image_list(images, cfg.TEST.SIZE_DIVISIBILITY)
        images_norm = self.Norm(images.tensors)
        conv_features = self.Conv_Body(images_norm)

        if cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]

        if cfg.MODEL.RPN_ON:
            proposals, proposal_losses = self.RPN(images, conv_features, targets)
        else:
            proposals = None

        if cfg.MODEL.FASTER_ON:
            if cfg.MODEL.CASCADE_ON:
                box_features, result, loss_box = self.Cascade_RCNN(conv_features, proposals, targets)
            else:
                box_features, result, loss_box = self.Fast_RCNN(conv_features, proposals, targets)
        else:
            result = proposals

        return conv_features, result

    def mask_net(self, conv_features, result, targets=None):
        if len(result[0]) == 0:
            return {}
        with torch.no_grad():
            x, result, loss_mask = self.Mask_RCNN(conv_features, result, targets)

        return result

    def keypoint_net(self, conv_features, result, targets=None):
        if len(result[0]) == 0:
            return {}
        with torch.no_grad():
            x, result, loss_keypoint = self.Keypoint_RCNN(conv_features, result, targets)

        return result

    def parsing_net(self, conv_features, result, targets=None):
        if len(result[0]) == 0:
            return result
        with torch.no_grad():
            x, result, loss_parsing = self.Parsing_RCNN(conv_features, result, targets)

        return result

    def uv_net(self, conv_features, result, targets=None):
        if len(result[0]) == 0:
            return {}
        with torch.no_grad():
            x, result, loss_uv = self.UV_RCNN(conv_features, result, targets)

        return result

