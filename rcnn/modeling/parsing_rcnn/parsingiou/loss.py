import torch

from models.ops import l2_loss
from rcnn.core.config import cfg


class ParsingIoULossComputation(object):
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, pred_parsingiou, gt_parsingiou):
        if gt_parsingiou.numel() == 0:
            return pred_parsingiou.sum() * 0
        gt_parsingiou = gt_parsingiou.detach()
        parsingiou_loss = l2_loss(pred_parsingiou[:, 0], gt_parsingiou)
        parsingiou_loss = self.loss_weight * parsingiou_loss

        return parsingiou_loss


def parsingiou_loss_evaluator():
    loss_weight = cfg.PRCNN.PARSINGIOU.LOSS_WEIGHT
    
    loss_evaluator = ParsingIoULossComputation(loss_weight)
    return loss_evaluator
