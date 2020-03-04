import os

import torch

from utils.data.structures.bounding_box import BoxList


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        pos_inds = labels > 0
        inds = pos_inds.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
    return positive_boxes


def across_sample(boxes, roi_size_per_img=-1, across_sample=False):
    if roi_size_per_img < 0:
        return boxes
    positive_boxes = []
    if across_sample and len(boxes) == 2:
        # TODO：Support input boxes of different batch size
        # assert len(boxes) == 2, 'only support 2 images on one gpu, but get {}'.format(boxes)
        roi_batch_size = len(boxes) * roi_size_per_img
        batch_pos_inds = []
        for boxes_per_image in boxes:
            inds = torch.arange(boxes_per_image.bbox.shape[0])
            batch_pos_inds.append(inds)
        _batch_pos_inds = torch.cat(batch_pos_inds)
        split_num = batch_pos_inds[0].shape[0]
        if _batch_pos_inds.shape[0] > roi_batch_size:
            ind = torch.LongTensor(sorted(torch.randperm(_batch_pos_inds.shape[0])[:roi_batch_size]))
            inds = [_batch_pos_inds[ind[ind < split_num]], _batch_pos_inds[ind[ind >= split_num]]]
        else:
            inds = batch_pos_inds
        for boxes_per_image, ind in zip(boxes, inds):
            positive_boxes.append(boxes_per_image[ind])
    else:
        for boxes_per_image in boxes:
            if roi_size_per_img < boxes_per_image.bbox.shape[0]:
                inds = torch.randperm(boxes_per_image.bbox.shape[0])[:roi_size_per_img]
                boxes_per_image = boxes_per_image[inds]
            positive_boxes.append(boxes_per_image)
    return positive_boxes


def random_jitter(proposals, amplitude=0.15):
    """Ramdom jitter positive proposals for training."""
    for proposal_per_img in proposals:
        bboxes = proposal_per_img.bbox
        random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(
            -amplitude, amplitude)
        # before jittering
        cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
        wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
        # after jittering
        new_cxcy = cxcy + wh * random_offsets[:, :2]
        new_wh = wh * (1 + random_offsets[:, 2:])
        # xywh to xyxy
        new_x1y1 = (new_cxcy - new_wh / 2)
        new_x2y2 = (new_cxcy + new_wh / 2)
        new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
        # clip bboxes
        max_shape = proposal_per_img.size
        if max_shape is not None:
            new_bboxes[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
            new_bboxes[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)

        proposal_per_img.bbox = new_bboxes
    return proposals
