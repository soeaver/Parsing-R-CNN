import torch
from torch import nn

from utils.data.structures.bounding_box import BoxList


# TODO get the predicted maskiou and mask score.
class MaskIoUPostProcessor(nn.Module):
    """
    Getting the maskiou according to the targeted label, and computing the mask score according to maskiou.
    """
    def __init__(self):
        super(MaskIoUPostProcessor, self).__init__()

    def forward(self, boxes, pred_maskiou, labels):
        num_masks = pred_maskiou.shape[0]
        index = torch.arange(num_masks, device=labels.device)
        maskious = pred_maskiou[index, labels]
        maskious = [maskious]
        results = []
        for maskiou, box in zip(maskious, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox_scores = bbox.get_field("scores")
            mask_scores = bbox_scores * maskiou
            bbox.add_field("mask_scores", mask_scores.cpu().numpy())
            prob = bbox.get_field("mask")
            bbox.add_field("mask", prob.cpu().numpy())
            results.append(bbox)

        return results


def maskiou_post_processor():
    maskiou_post_processor = MaskIoUPostProcessor()
    return maskiou_post_processor
