import numpy as np

import torch
from torch.nn import functional as F

from models.ops import PoolPointsInterp, smooth_l1_loss_LW
from utils.data.structures.boxlist_ops import boxlist_iou
from utils.data.structures.boxlist_ops import cat_boxlist
from utils.data.structures.densepose_uv import uv_on_boxes
from rcnn.utils.matcher import Matcher
from rcnn.utils.misc import cat, keep_only_positive_boxes, across_sample
from rcnn.core.config import cfg


def project_uv_on_boxes(proposals, resolution):
    uv_anns = [[] for _ in range(8)]
    for proposals_per_image in proposals:
        if len(proposals_per_image) == 0:
            continue
        targets = proposals_per_image.get_field("uv_target")
        targets = targets.convert("xyxy")
        proposals_per_image = proposals_per_image.convert("xyxy")
        assert targets.size == proposals_per_image.size, "{}, {}".format(
            targets, proposals_per_image
        )
        uv_ann_per_image = uv_on_boxes(targets, proposals_per_image.bbox, resolution)
        for i in range(8):
            uv_anns[i].append(uv_ann_per_image[i])

    return uv_anns


def uv_cat(uv_anns, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    output = []
    for i, uv_ann in enumerate(uv_anns):
        assert isinstance(uv_ann, (list, tuple))
        if len(uv_ann) == 1:
            output.append(uv_ann[0])
            continue
        _uv_ann = np.concatenate(uv_ann, dim)
        output.append(_uv_ann)
    return output


class UVRCNNLossComputation(object):
    def __init__(self, proposal_matcher, resolution):
        """
        Arguments:
            proposal_matcher (Matcher)
            resolution (tuple)
        """
        self.proposal_matcher = proposal_matcher
        self.resolution = resolution

        self.across_sample = cfg.UVRCNN.ACROSS_SAMPLE
        self.roi_size_per_img = cfg.UVRCNN.ROI_SIZE_PER_IMG

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        target = target.copy_with_fields(["labels", "uv"])

        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        all_positive_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            uvs_per_image = matched_targets.get_field("uv")
            with_uv = torch.from_numpy(np.array([(len(uv) > 0) for uv in uvs_per_image.dp_uvs])).byte()

            labels_per_image[~with_uv] = -1

            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            positive_proposals = proposals_per_image[positive_inds]
            uv_targets_per_image = matched_targets[positive_inds]
            positive_proposals.add_field("uv_target", uv_targets_per_image)
            all_positive_proposals.append(positive_proposals)
        return all_positive_proposals

    def resample(self, proposals, targets):
        # get all positive proposals (for single image on per GPU)
        positive_proposals = keep_only_positive_boxes(proposals)
        # resample for getting targets or matching new IoU
        positive_proposals = self.prepare_targets(positive_proposals, targets)
        # apply across-sample strategy (for a batch of images on per GPU)
        positive_proposals = across_sample(
            positive_proposals, roi_size_per_img=self.roi_size_per_img, across_sample=self.across_sample
        )

        self.positive_proposals = positive_proposals

        all_num_positive_proposals = 0
        for positive_proposals_per_image in positive_proposals:
            all_num_positive_proposals += len(positive_proposals_per_image)
        if all_num_positive_proposals == 0:
            positive_proposals = [proposals[0][:1]]
        return positive_proposals

    def __call__(self, uv_logits):
        uv_anns = project_uv_on_boxes(self.positive_proposals, self.resolution)

        UV_pred_Ann, UV_pred_Index, UV_pred_U, UV_pred_V = uv_logits
        if len(uv_anns[0]) == 0:
            return UV_pred_U.sum() * 0, UV_pred_V.sum() * 0, UV_pred_Ann.sum() * 0, UV_pred_Index.sum() * 0
        else:
            uv_anns = uv_cat(uv_anns)
            uv_ann_labels, uv_X_points, uv_Y_points, uv_Ind_points, \
            uv_I_points, uv_U_points, uv_V_points, uv_point_weights = uv_anns

            device_id = UV_pred_Ann.get_device()

            uv_X_points = uv_X_points.reshape((-1, 1))
            uv_Y_points = uv_Y_points.reshape((-1, 1))
            uv_Ind_points = uv_Ind_points.reshape((-1, 1))
            uv_I_points = uv_I_points.reshape(-1).astype('int64')
            uv_I_points = torch.from_numpy(uv_I_points).cuda(device_id)

            Coordinate_Shapes = np.concatenate((uv_Ind_points, uv_X_points, uv_Y_points), axis=1)
            Coordinate_Shapes = torch.from_numpy(Coordinate_Shapes).cuda(device_id).float()

            uv_U_points = uv_U_points.reshape((-1, cfg.UVRCNN.NUM_PATCHES + 1, 196))
            uv_U_points = uv_U_points.transpose((0, 2, 1))
            uv_U_points = uv_U_points.reshape((1, 1, -1, cfg.UVRCNN.NUM_PATCHES + 1))
            uv_U_points = torch.from_numpy(uv_U_points).cuda(device_id)

            uv_V_points = uv_V_points.reshape((-1, cfg.UVRCNN.NUM_PATCHES + 1, 196))
            uv_V_points = uv_V_points.transpose((0, 2, 1))
            uv_V_points = uv_V_points.reshape((1, 1, -1, cfg.UVRCNN.NUM_PATCHES + 1))
            uv_V_points = torch.from_numpy(uv_V_points).cuda(device_id)

            uv_point_weights = uv_point_weights.reshape((-1, cfg.UVRCNN.NUM_PATCHES + 1, 196))
            uv_point_weights = uv_point_weights.transpose((0, 2, 1))
            uv_point_weights = uv_point_weights.reshape((1, 1, -1, cfg.UVRCNN.NUM_PATCHES + 1))
            uv_point_weights = torch.from_numpy(uv_point_weights).cuda(device_id)

            PPI_op = PoolPointsInterp()
            interp_U = PPI_op(UV_pred_U, Coordinate_Shapes)
            interp_V = PPI_op(UV_pred_V, Coordinate_Shapes)
            interp_Index_UV = PPI_op(UV_pred_Index, Coordinate_Shapes)

            uv_ann_labels = torch.from_numpy(uv_ann_labels.astype('int64')).cuda(device_id)

            loss_seg_AnnIndex = F.cross_entropy(UV_pred_Ann, uv_ann_labels)
            loss_seg_AnnIndex *= cfg.UVRCNN.INDEX_WEIGHTS

            loss_IndexUVPoints = F.cross_entropy(interp_Index_UV, uv_I_points)
            loss_IndexUVPoints *= cfg.UVRCNN.PART_WEIGHTS

            loss_Upoints = smooth_l1_loss_LW(interp_U, uv_U_points,
                                             uv_point_weights, uv_point_weights)
            loss_Upoints *= cfg.UVRCNN.POINT_REGRESSION_WEIGHTS

            loss_Vpoints = smooth_l1_loss_LW(interp_V, uv_V_points,
                                             uv_point_weights, uv_point_weights)
            loss_Vpoints *= cfg.UVRCNN.POINT_REGRESSION_WEIGHTS

        return loss_Upoints, loss_Vpoints, loss_seg_AnnIndex, loss_IndexUVPoints


def uv_loss_evaluator():
    matcher = Matcher(
        cfg.UVRCNN.FG_IOU_THRESHOLD,
        cfg.UVRCNN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    resolution = cfg.UVRCNN.RESOLUTION
    loss_evaluator = UVRCNNLossComputation(matcher, resolution)
    return loss_evaluator
