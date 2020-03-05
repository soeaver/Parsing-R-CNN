import numpy as np
import pycocotools.mask as mask_util

import torch
from torch.nn import functional as F

from models.ops import smooth_l1_loss
from utils.data.structures.boxlist_ops import boxlist_iou
from rcnn.utils.matcher import Matcher
from rcnn.utils.misc import cat, keep_only_positive_boxes, across_sample
from rcnn.core.config import cfg


def project_masks_on_boxes(segmentation_masks, proposals, resolution):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    h, w = resolution
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((w, h))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)

    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, resolution):
        """
        Arguments:
            proposal_matcher (Matcher)
            resolution (tuple)
        """
        self.proposal_matcher = proposal_matcher
        self.resolution = resolution

        self.across_sample = cfg.MRCNN.ACROSS_SAMPLE
        self.roi_size_per_img = cfg.MRCNN.ROI_SIZE_PER_IMG

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
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

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.resolution
            )

            positive_proposals.add_field("labels", labels_per_image)
            positive_proposals.add_field("mask_targets", masks_per_image)

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

    def __call__(self, mask_logits):
        """
        Arguments:
            mask_logits (Tensor)

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
            If we use maskiou head, we will return extra feature for maskiou head.
        """
        labels = [proposals_per_img.get_field("labels") for proposals_per_img in self.positive_proposals]
        mask_targets = [proposals_per_img.get_field("mask_targets") for proposals_per_img in self.positive_proposals]
        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )
        mask_loss *= cfg.MRCNN.LOSS_WEIGHT

        return mask_loss


def mask_loss_evaluator():
    matcher = Matcher(
        cfg.FAST_RCNN.FG_IOU_THRESHOLD,
        cfg.FAST_RCNN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MRCNN.RESOLUTION
    )

    return loss_evaluator
