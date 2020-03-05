import cv2
import numpy as np

from torch import nn
from torch.nn import functional as F

from utils.data.structures.bounding_box import BoxList
from models.ops.misc import interpolate
from rcnn.core.config import cfg


# TODO check if want to return a single BoxList or a composite
# object
class ParsingPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.
    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self):
        super(ParsingPostProcessor, self).__init__()

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
        parsing_prob = x
        parsing_prob = F.softmax(parsing_prob, dim=1)

        boxes_per_image = [len(box) for box in boxes]
        parsing_prob = parsing_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, box in zip(parsing_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")

            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox_scores = bbox.get_field("scores")
            bbox.add_field("parsing", prob.cpu().numpy())
            bbox.add_field("parsing_scores", bbox_scores.cpu().numpy())
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


def parsing_results(parsings, boxes, semseg=None):
    im_w, im_h = boxes.size
    parsings = parsings.transpose((0, 2, 3, 1))
    boxes = boxes.bbox.numpy()
    H, W = parsings.shape[1:3]
    N = parsings.shape[3]

    boxes = expand_boxes(boxes, H, W)
    boxes = boxes.astype(np.int32)
    padded_parsing = np.zeros((H + 2, W + 2, N), dtype=np.float32)

    if semseg is not None:
        semseg = cv2.resize(semseg, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
    else:
        semseg = np.zeros((im_h, im_w, N), dtype=np.float32)

    parsing_results = []
    for i in range(boxes.shape[0]):
        padded_parsing[1:-1, 1:-1] = parsings[i]
        box = boxes[i, :]
        w = box[2] - box[0] + 1
        h = box[3] - box[1] + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)
        parsing = cv2.resize(padded_parsing, (w, h), interpolation=cv2.INTER_LINEAR)
        parsing_idx = np.argmax(parsing, axis=2)
        im_parsing = np.zeros((im_h, im_w), dtype=np.uint8)
        x_0 = max(box[0], 0)
        x_1 = min(box[2] + 1, im_w)
        y_0 = max(box[1], 0)
        y_1 = min(box[3] + 1, im_h)

        mask = np.where(parsing_idx >= 1, 1, 0)
        mask = mask[:, :, np.newaxis].repeat(N, axis=2)
        cropped_semseg = semseg[y_0:y_1, x_0:x_1] * mask[(y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])]

        parsing[(y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])] += \
            cropped_semseg * cfg.PRCNN.SEMSEG_FUSE_WEIGHT
        parsing = np.argmax(parsing, axis=2)

        im_parsing[y_0:y_1, x_0:x_1] = parsing[(y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])]
        parsing_results.append(im_parsing)
    return parsing_results


def parsing_post_processor():
    parsing_post_processor = ParsingPostProcessor()
    return parsing_post_processor
