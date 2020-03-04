import torch
import numpy as np

from utils.data.structures.bounding_box import BoxList
from models.ops import nms as _box_nms
from models.ops import ml_nms as _box_ml_nms
from models.ops import boxes as box_utils


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def boxlist_ml_nms(boxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)
    labels = boxlist.get_field(label_field)
    keep = _box_ml_nms(boxes, scores, labels.float(), nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def boxlist_soft_nms(boxlist, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear',
                     score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if overlap_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox.cpu()
    score = boxlist.get_field(score_field).cpu()
    dets = np.hstack((boxes, score[:, np.newaxis])).astype(np.float32, copy=False)
    dets, _ = box_utils.soft_nms(dets, sigma, overlap_thresh, score_thresh, method)
    boxlist = BoxList(torch.from_numpy(dets[:, :4]).cuda(), boxlist.size, mode="xyxy")
    boxlist.add_field("scores", torch.from_numpy(dets[:, -1]).cuda())
    return boxlist.convert(mode)


def boxlist_box_voting(top_boxlist, all_boxlist, thresh, scoring_method='ID', beta=1.0, score_field="scores"):
    if thresh <= 0:
        return top_boxlist
    mode = top_boxlist.mode
    top_boxes = top_boxlist.convert("xyxy").bbox.cpu()
    all_boxes = all_boxlist.convert("xyxy").bbox.cpu()
    top_score = top_boxlist.get_field(score_field).cpu()
    all_score = all_boxlist.get_field(score_field).cpu()
    top_dets = np.hstack((top_boxes, top_score[:, np.newaxis])).astype(np.float32, copy=False)
    all_dets = np.hstack((all_boxes, all_score[:, np.newaxis])).astype(np.float32, copy=False)
    dets = box_utils.box_voting(top_dets, all_dets, thresh, scoring_method, beta)
    boxlist = BoxList(torch.from_numpy(dets[:, :4]).cuda(), all_boxlist.size, mode="xyxy")
    boxlist.add_field("scores", torch.from_numpy(dets[:, -1]).cuda())
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


def remove_boxes_by_center(boxlist, crop_region):
    xyxy_boxes = boxlist.convert("xyxy").bbox
    left, up, right, bottom = crop_region
    # center of boxes should inside the crop img
    centers = (xyxy_boxes[:, :2] + xyxy_boxes[:, 2:]) / 2
    keep = (
            (centers[:, 0] > left) & (centers[:, 1] > up) &
            (centers[:, 0] < right) & (centers[:, 1] < bottom)
    ).nonzero().squeeze(1)
    return boxlist[keep]


def remove_boxes_by_overlap(ori_targets, crop_targets, iou_th):
    ori_targets.size = crop_targets.size
    iou_matrix = boxlist_iou(ori_targets, crop_targets)
    iou_list = torch.diag(iou_matrix, diagonal=0)
    keep = (iou_list >= iou_th).nonzero().squeeze(1)
    return crop_targets[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_partly_overlap(boxlist1, boxlist2):
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    overlap = iou > 0
    not_complete_overlap = (inter - area1[:, None]) * (inter - area2[None, :]) != 0
    partly_overlap = overlap * not_complete_overlap

    return partly_overlap


def boxlist_overlap(boxlist1, boxlist2):
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    overlap = iou > 0

    return overlap


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def boxes_to_masks(boxes, h, w, padding=0.0):
    n = boxes.shape[0]
    boxes = boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    b_w = x2 - x1
    b_h = y2 - y1
    x1 = torch.clamp(x1 - 1 - b_w * padding, min=0)
    x2 = torch.clamp(x2 + 1 + b_w * padding, max=w)
    y1 = torch.clamp(y1 - 1 - b_h * padding, min=0)
    y2 = torch.clamp(y2 + 1 + b_h * padding, max=h)

    rows = torch.arange(w, device=boxes.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
    cols = torch.arange(h, device=boxes.device, dtype=x1.dtype).view(1, -1, 1).expand(n, h, w)

    masks_left = rows >= x1.view(-1, 1, 1)
    masks_right = rows < x2.view(-1, 1, 1)
    masks_up = cols >= y1.view(-1, 1, 1)
    masks_down = cols < y2.view(-1, 1, 1)

    masks = masks_left * masks_right * masks_up * masks_down

    return masks


def crop_by_box(masks, box, padding=0.0):
    n, h, w = masks.size()

    b_w = box[2] - box[0]
    b_h = box[3] - box[1]
    x1 = torch.clamp(box[0:1] - b_w * padding - 1, min=0)
    x2 = torch.clamp(box[2:3] + b_w * padding + 1, max=w - 1)
    y1 = torch.clamp(box[1:2] - b_h * padding - 1, min=0)
    y2 = torch.clamp(box[3:4] + b_h * padding + 1, max=h - 1)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(n, h, w)

    masks_left = rows >= x1.expand(n, 1, 1)
    masks_right = rows < x2.expand(n, 1, 1)
    masks_up = cols >= y1.expand(n, 1, 1)
    masks_down = cols < y2.expand(n, 1, 1)

    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float(), crop_mask
