import torch
import torch.nn.functional as F
from torch import nn

from utils.data.structures.bounding_box import BoxList
from utils.data.structures.boxlist_ops import boxlist_nms, boxlist_soft_nms, boxlist_box_voting
from utils.data.structures.boxlist_ops import cat_boxlist
from rcnn.utils.box_coder import BoxCoder
from rcnn.core.config import cfg

import numpy as np


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100, box_coder=None, cls_agnostic_bbox_reg=False,
                 is_repeat=False):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.is_repeat = is_repeat

    def forward(self, x, boxes, targets=None):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image
            targets (list[BoxList])

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        assert self.cls_agnostic_bbox_reg, 'Use a class agnostic bounding box regressor in Cascade R-CNN'

        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(box_regression.view(sum(boxes_per_image), -1), concat_boxes)

        proposals[proposals < 0] = 0

        if self.is_repeat:
            proposals = proposals.repeat(1, class_prob.shape[1])
        else:
            proposals = proposals.split(boxes_per_image, dim=0)
            refine_proposals = self.refine(boxes, targets, proposals)
            return refine_proposals

        num_classes = class_prob.shape[1]
        proposals = proposals.split(boxes_per_image, dim=0)

        class_prob = class_prob.split(boxes_per_image, dim=0)
        results = []
        for prob, boxes_per_img, image_shape in zip(class_prob, proposals, image_shapes):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            # boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def refine(self, boxes, targets, proposals):
        """Refine every stage box prediction and return the result"""
        refine_proposals = []
        if targets is not None:
            for box, targets_per_image, proposals_per_image in zip(boxes, targets, proposals):
                # remove mal-boxes with non-positive width or height and ground
                # truth boxes during training
                keep = self._filter_boxes(proposals_per_image, box, targets_per_image)
                for field, value in box.extra_fields.items():
                    box.add_field(field, value[keep])
                box.bbox = proposals_per_image[keep]
                refine_proposals.append(box)
            refine_proposals = self.add_gt_proposals(refine_proposals, targets)
        else:
            for box, proposals_per_image in zip(boxes, proposals):
                box.bbox = proposals_per_image
                refine_proposals.append(box)
        return refine_proposals

    def _filter_boxes(self, bbox, last, gt):
        """Only keep boxes with positive height and width, and not-gt.
        """
        last_bbox = last.bbox
        gt_bbox = gt.bbox
        ws = bbox[:, 2] - bbox[:, 0] + 1
        hs = bbox[:, 3] - bbox[:, 1] + 1
        for i in range(gt_bbox.shape[0]):
            last_bbox = torch.where(last_bbox == gt_bbox[i], torch.full_like(last_bbox, -1), last_bbox)
        s = sum([last_bbox[:, 0], last_bbox[:, 1], last_bbox[:, 2], last_bbox[:, 3]])
        keep = np.where((ws.cpu() > 0) & (hs.cpu() > 0) & (s.cpu() > 0))[0]
        return keep

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields(['labels']) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        # print(proposals[0].get_field("regression_targets").shape, len(gt_boxes[0]))
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
            gt_box.add_field("regression_targets", torch.zeros((len(gt_box), 4), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist


def box_post_processor(idx, is_train=True):
    bbox_reg_weights = cfg.CASCADE_RCNN.BBOX_REG_WEIGHTS[idx]
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.FAST_RCNN.SCORE_THRESH
    nms_thresh = cfg.FAST_RCNN.NMS
    detections_per_img = cfg.FAST_RCNN.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.FAST_RCNN.CLS_AGNOSTIC_BBOX_REG

    final_test_stage = (idx == cfg.CASCADE_RCNN.TEST_STAGE - 1)
    final_train_stage = (idx == cfg.CASCADE_RCNN.NUM_STAGE - 1)
    is_repeat = (is_train and final_train_stage) or (not is_train and final_test_stage)

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        is_repeat,
    )
    return postprocessor
