# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""RetinaNet model heads and losses. See: https://arxiv.org/abs/1708.02002."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils
import parsingrcnn.utils.blob as blob_utils


class retinanet_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        dim_out = dim_in
        num_classes = cfg.MODEL.NUM_CLASSES
        k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
        k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
        A = len(cfg.RETINANET.ASPECT_RATIOS) * cfg.RETINANET.SCALES_PER_OCTAVE

        # compute init for bias
        # bias_init = get_retinanet_bias_init(model)
        cls_pred_dim = (
            num_classes if cfg.RETINANET.SOFTMAX else (num_classes - 1)
        )
        # unpacked bbox feature and add prediction layers
        bbox_regr_dim = (
            4 * (model.num_classes - 1) if cfg.RETINANET.CLASS_SPECIFIC_BBOX else 4
        )

        # ==========================================================================
        # classification tower with logits and prob prediction
        # ==========================================================================

        self.cls_pred_modules = nn.ModuleList()
        lvl = k_max

        x = self.cls_pred_modules
        for nconv in range(cfg.RETINANET.NUM_CONVS):
            dim_in, dim_out = dim_in, dim_in
            self.cls_pred_modules.append(nn.Conv2d(dim_in, dim_out, 3, 1, 1))

        # cls tower stack convolution ends. Add the logits layer now

        self.cls_pred_logit = nn.Conv2d(dim_in, cls_pred_dim * A, 3, 1, 1)

        # ==========================================================================
        # bbox tower if not sharing features with the classification tower with
        # logits and prob prediction
        # ==========================================================================
        if not cfg.RETINANET.SHARE_CLS_BBOX_TOWER:
            # bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
            self.bbox_pred_modules = nn.ModuleList()
            for nconv in range(cfg.RETINANET.NUM_CONVS):
                dim_in, dim_out = dim_in, dim_in
                self.bbox_pred_modules.append(nn.Conv2d(dim_in, dim_out, 3, 1, 1))

        # Depending on the features [shared/separate] for bbox, add prediction layer
        self.bbox_pred_logit = nn.Conv2d(dim_in, bbox_regr_dim * A, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        prior_prob = cfg.RETINANET.PRIOR_PROB
        for i in range(len(self.cls_pred_modules)):
            init.normal_(self.cls_pred_modules[i].weight, std=0.01)
            init.constant_(self.cls_pred_modules[i].bias, 0)
            if not cfg.RETINANET.SHARE_CLS_BBOX_TOWER:
                init.normal_(self.bbox_pred_modules[i].weight, std=0.01)
                init.constant_(self.bbox_pred_modules[i].bias, 0)
        init.normal_(self.cls_pred_logit.weight, std=0.01)
        init.constant_(self.cls_pred_logit.bias, -np.log((1 - prior_prob) / prior_prob))
        init.normal_(self.bbox_pred_logit.weight, std=0.01)
        init.constant_(self.bbox_pred_logit.bias, 0)

    def forward(self, blobs_in, training=False):
        #         cls_score_dict = dict()
        #         bbox_score_dict = dict()

        k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
        k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
        cls_score_dict = []
        bbox_score_dict = []
        for lvl in range(k_min, k_max + 1):
            x = blobs_in[k_max - lvl]
            for i in range(len(self.cls_pred_modules)):
                x = self.cls_pred_modules[i](x)
                x = F.relu(x, inplace=True)

            cls_score = self.cls_pred_logit(x)
            # FIXME: if cfg.RETINANET.SOFTMAX, then we need a softmax, haven't implement yet
            if not training:
                cls_score = F.sigmoid(cls_score)
            cls_score_dict.append(cls_score)

            if not cfg.RETINANET.SHARE_CLS_BBOX_TOWER:  # not cfg.RETINANET.SHARE_CLS_BBOX_TOWER:
                x = blobs_in[k_max - lvl]
                for i in range(len(self.bbox_pred_modules)):
                    x = self.bbox_pred_modules[i](x)
                    x = F.relu(x, inplace=True)

            bbox_score = self.bbox_pred_logit(x)
            bbox_score_dict.append(bbox_score)
        return cls_score_dict, bbox_score_dict

    def detectron_weight_mapping(self):
        k_min = cfg.FPN.RPN_MIN_LEVEL
        mapping_to_detectron = {
            'cls_pred_modules.0.weight': 'retnet_cls_conv_n{}_fpn{}_w'.format(0, k_min),
            'cls_pred_modules.0.bias': 'retnet_cls_conv_n{}_fpn{}_b'.format(0, k_min),
            'cls_pred_modules.1.weight': 'retnet_cls_conv_n{}_fpn{}_w'.format(1, k_min),
            'cls_pred_modules.1.bias': 'retnet_cls_conv_n{}_fpn{}_b'.format(1, k_min),
            'cls_pred_modules.2.weight': 'retnet_cls_conv_n{}_fpn{}_w'.format(2, k_min),
            'cls_pred_modules.2.bias': 'retnet_cls_conv_n{}_fpn{}_b'.format(2, k_min),
            'cls_pred_modules.3.weight': 'retnet_cls_conv_n{}_fpn{}_w'.format(3, k_min),
            'cls_pred_modules.3.bias': 'retnet_cls_conv_n{}_fpn{}_b'.format(3, k_min),
            'cls_pred_logit.weight': 'retnet_cls_pred_fpn{}_w'.format(k_min),
            'cls_pred_logit.bias': 'retnet_cls_pred_fpn{}_b'.format(k_min),
            'bbox_pred_logit.weight': 'retnet_bbox_pred_fpn{}_w'.format(k_min),
            'bbox_pred_logit.bias': 'retnet_bbox_pred_fpn{}_b'.format(k_min)

        }
        if not cfg.RETINANET.SHARE_CLS_BBOX_TOWER:
            mapping_to_detectron['bbox_pred_modules.0.weight'] = 'retnet_bbox_conv_n{}_fpn{}_w'.format(0, k_min)
            mapping_to_detectron['bbox_pred_modules.0.bias'] = 'retnet_bbox_conv_n{}_fpn{}_b'.format(0, k_min)
            mapping_to_detectron['bbox_pred_modules.1.weight'] = 'retnet_bbox_conv_n{}_fpn{}_w'.format(1, k_min)
            mapping_to_detectron['bbox_pred_modules.1.bias'] = 'retnet_bbox_conv_n{}_fpn{}_b'.format(1, k_min)
            mapping_to_detectron['bbox_pred_modules.2.weight'] = 'retnet_bbox_conv_n{}_fpn{}_w'.format(2, k_min)
            mapping_to_detectron['bbox_pred_modules.2.bias'] = 'retnet_bbox_conv_n{}_fpn{}_b'.format(2, k_min)
            mapping_to_detectron['bbox_pred_modules.3.weight'] = 'retnet_bbox_conv_n{}_fpn{}_w'.format(3, k_min)
            mapping_to_detectron['bbox_pred_modules.3.bias'] = 'retnet_bbox_conv_n{}_fpn{}_b'.format(3, k_min)

        return mapping_to_detectron, []

    def pytorch_weight_mapping(self):
        return {}, []
    
    
def fpn_retinanet_losses(blobs, cls_score, bbox_pred):
    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
    bbox_loss, cls_focal_loss = {}, {}

    # model.AddMetrics(['retnet_fg_num', 'retnet_bg_num'])
    # ==========================================================================
    # bbox regression loss - SelectSmoothL1Loss for multiple anchors at a location
    # ==========================================================================

    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        # bbox_locs = blobs['retnet_roi_fg_bbox_locs_' + suffix]
        # debug_indx = min(lvl, blobs['retnet_roi_bbox_targets_' + suffix].shape[0]-1)
        # blobs['retnet_roi_bbox_targets_' + suffix][debug_indx,:] = 0
        # blobs['retnet_roi_fg_bbox_locs_' + suffix][debug_indx,:] = 0
        bbox_loss['retnet_loss_bbox_' + suffix] = net_utils.retinanet_smooth_l1_loss(
            bbox_pred=bbox_pred[lvl - k_min],
            bbox_targets=blobs['retnet_roi_bbox_targets_' + suffix],
            fg_bbox_locs=blobs['retnet_roi_fg_bbox_locs_' + suffix],
            fg_num=blobs['retnet_fg_num'],
            beta=cfg.RETINANET.BBOX_REG_BETA,
            scale=1.0 / cfg.NUM_GPUS * cfg.RETINANET.BBOX_REG_WEIGHT
        )

    # ==========================================================================
    # cls loss - depends on softmax/sigmoid outputs
    # ==========================================================================

    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        cls_focal_loss['fl_' + suffix] = net_utils.sigmoid_focal_loss(
            cls_pred=cls_score[lvl - k_min],
            cls_targets=blobs['retnet_cls_labels_' + suffix],
            fg_num=blobs['retnet_fg_num'],
            alpha=cfg.RETINANET.LOSS_ALPHA,
            scale=1.0 / cfg.NUM_GPUS,
            num_classes=cfg.MODEL.NUM_CLASSES - 1,
            gamma=cfg.RETINANET.LOSS_GAMMA,
            lvl=lvl
        )

    # cls_preds = cls_score.max(dim=1)[1]
    # accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)
    return bbox_loss, cls_focal_loss, None  # accuracy_cls
