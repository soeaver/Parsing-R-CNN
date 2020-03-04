import torch

from models.ops import l2_loss
from rcnn.core.config import cfg


class MaskIoULossComputation(object):
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, labels, pred_maskiou, gt_maskiou):
        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]
        if labels_pos.numel() == 0:
            return pred_maskiou.sum() * 0
        gt_maskiou = gt_maskiou.detach()
        maskiou_loss = l2_loss(pred_maskiou[positive_inds, labels_pos], gt_maskiou)
        maskiou_loss = self.loss_weight * maskiou_loss

        return maskiou_loss


def maskiou_loss_evaluator():
    loss_weight = cfg.MRCNN.MASKIOU.LOSS_WEIGHT
    loss_evaluator = MaskIoULossComputation(loss_weight)

    return loss_evaluator
