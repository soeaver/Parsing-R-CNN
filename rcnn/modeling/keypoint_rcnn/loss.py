import torch
from torch.nn import functional as F

from models.ops import smooth_l1_loss
from utils.data.structures.boxlist_ops import boxlist_iou
from utils.data.structures.boxlist_ops import cat_boxlist
from utils.data.structures.keypoint import keypoints_to_heat_map
from rcnn.utils.matcher import Matcher
from rcnn.utils.misc import cat, keep_only_positive_boxes, across_sample
from rcnn.core.config import cfg


def project_keypoints_to_heatmap(keypoints, proposals, resolution):
    proposals = proposals.convert("xyxy")
    return keypoints_to_heat_map(keypoints.keypoints, proposals.bbox, resolution)


def _within_box(points, boxes):
    """Validate which keypoints are contained inside a given box.
    points: NxKx2
    boxes: Nx4
    output: NxK
    """
    x_within = (points[..., 0] >= boxes[:, 0, None]) & (points[..., 0] <= boxes[:, 2, None])
    y_within = (points[..., 1] >= boxes[:, 1, None]) & (points[..., 1] <= boxes[:, 3, None])
    return x_within & y_within


class KeypointRCNNLossComputation(object):
    def __init__(self, proposal_matcher, resolution):
        """
        Arguments:
            proposal_matcher (Matcher)
            resolution (tuple)
        """
        self.proposal_matcher = proposal_matcher
        self.resolution = resolution

        self.across_sample = cfg.KRCNN.ACROSS_SAMPLE
        self.roi_size_per_img = cfg.KRCNN.ROI_SIZE_PER_IMG

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Keypoint RCNN needs "labels" and "keypoints "fields for creating the targets
        target = target.copy_with_fields(["labels", "keypoints"])
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
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            keypoints_per_image = matched_targets.get_field("keypoints")
            within_box = _within_box(keypoints_per_image.keypoints, matched_targets.bbox)
            vis_kp = keypoints_per_image.keypoints[..., 2] > 0
            is_visible = (within_box & vis_kp).sum(1) > 0

            labels_per_image[~is_visible] = -1

            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            positive_proposals = proposals_per_image[positive_inds]
            keypoints_per_image = keypoints_per_image[positive_inds]
            positive_proposals.add_field("keypoints_target", keypoints_per_image)
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

    def __call__(self, keypoint_logits):
        heatmaps = []
        valid = []
        for proposals_per_image in self.positive_proposals:
            kp = proposals_per_image.get_field("keypoints_target")
            heatmaps_per_image, valid_per_image = project_keypoints_to_heatmap(
                kp, proposals_per_image, self.resolution
            )
            heatmaps.append(heatmaps_per_image.view(-1))
            valid.append(valid_per_image.view(-1))

        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

        # torch.mean (in binary_cross_entropy_with_logits) does'nt
        # accept empty tensors, so handle it sepaartely
        if keypoint_targets.numel() == 0 or len(valid) == 0:
            return keypoint_logits.sum() * 0

        N, K, H, W = keypoint_logits.shape
        keypoint_logits = keypoint_logits.view(N * K, H * W)

        keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
        keypoint_loss *= cfg.KRCNN.LOSS_WEIGHT
        return keypoint_loss


def keypoint_loss_evaluator():
    matcher = Matcher(
        cfg.FAST_RCNN.FG_IOU_THRESHOLD,
        cfg.FAST_RCNN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    resolution = cfg.KRCNN.RESOLUTION
    
    loss_evaluator = KeypointRCNNLossComputation(matcher, resolution)
    return loss_evaluator
