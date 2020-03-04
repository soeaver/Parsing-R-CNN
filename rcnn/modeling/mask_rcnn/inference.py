import cv2
import numpy as np
import pycocotools.mask as mask_util

import torch
from torch import nn

from utils.data.structures.bounding_box import BoxList
from models.ops.misc import interpolate
from rcnn.core.config import cfg


# TODO check if want to return a single BoxList or a composite
# object
class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self):
        super(MaskPostProcessor, self).__init__()

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()

        # select masks coresponding to the predicted classes
        num_masks = x.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]

        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            if cfg.MRCNN.MASKIOU_ON:
                bbox.add_field("mask", prob)
            else:
                bbox_scores = bbox.get_field("scores")
                bbox.add_field("mask", prob.cpu().numpy())
                bbox.add_field("mask_scores", bbox_scores.cpu().numpy())
            results.append(bbox)

        return results


def expand_boxes(boxes, h, w):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    h_scale = (h + 2.0) / h
    w_scale = (w + 2.0) / w
    w_half *= w_scale
    h_half *= h_scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def mask_results(masks, boxes):
    im_w, im_h = boxes.size
    boxes = boxes.bbox.numpy()
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    H, W = masks.shape[2:]
    ref_boxes = expand_boxes(boxes, H, W)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((H + 2, W + 2), dtype=np.float32)

    rels = []
    for _ in range(ref_boxes.shape[0]):
        padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

        ref_box = ref_boxes[mask_ind, :]
        w = (ref_box[2] - ref_box[0] + 1)
        h = (ref_box[3] - ref_box[1] + 1)
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)

        mask = cv2.resize(padded_mask, (w, h))
        mask = np.array(mask > 0.5, dtype=np.uint8)
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

        x_0 = max(ref_box[0], 0)
        x_1 = min(ref_box[2] + 1, im_w)
        y_0 = max(ref_box[1], 0)
        y_1 = min(ref_box[3] + 1, im_h)

        im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]

        # Get RLE encoding used by the COCO evaluation API
        rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
        # For dumping to json, need to decode the byte string.
        # https://github.com/cocodataset/cocoapi/issues/70
        rle['counts'] = rle['counts'].decode('ascii')
        rels.append(rle)
        mask_ind += 1

    assert mask_ind == masks.shape[0]
    return rels


def mask_post_processor():
    mask_post_processor = MaskPostProcessor()
    return mask_post_processor
