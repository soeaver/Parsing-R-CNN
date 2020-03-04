import torch
import torch.nn.functional as F
from torch import nn

from utils.data.structures.bounding_box import BoxList
from utils.data.structures.boxlist_ops import boxlist_nms, boxlist_soft_nms, boxlist_box_voting
from utils.data.structures.boxlist_ops import cat_boxlist
from rcnn.utils.box_coder import BoxCoder
from rcnn.core.config import cfg


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100, box_coder=None, cls_agnostic_bbox_reg=False):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
            cls_agnostic_bbox_reg (bool)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if box_regression is not None:
            if self.cls_agnostic_bbox_reg:
                box_regression = box_regression[:, -4:]
            proposals = self.box_coder.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes
            )
        else:
            proposals = concat_boxes.repeat(1, class_prob.shape[1])

        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])
        proposals = proposals.split(boxes_per_image, dim=0)

        num_classes = class_prob.shape[1]
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
                class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            # boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

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

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4: (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class_old = boxlist_for_class
            if cfg.TEST.SOFT_NMS.ENABLED:
                boxlist_for_class = boxlist_soft_nms(
                    boxlist_for_class,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=self.nms,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
            else:
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms
                )
            # Refine the post-NMS boxes using bounding-box voting
            if cfg.TEST.BBOX_VOTE.ENABLED and boxes_j.shape[0] > 0:
                boxlist_for_class = boxlist_box_voting(
                    boxlist_for_class,
                    boxlist_for_class_old,
                    cfg.TEST.BBOX_VOTE.VOTE_TH,
                    scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def box_post_processor():
    bbox_reg_weights = cfg.FAST_RCNN.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.FAST_RCNN.SCORE_THRESH
    nms_thresh = cfg.FAST_RCNN.NMS
    detections_per_img = cfg.FAST_RCNN.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.FAST_RCNN.CLS_AGNOSTIC_BBOX_REG

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg
    )
    return postprocessor
