import torch
from torch.nn import functional as F

from utils.data.structures.semantic_segmentation import semseg_batch_resize
from rcnn.utils.misc import cat
from rcnn.core.config import cfg


class SemSegLossComputation(object):
    def __init__(self, ignore_label, loss_weight, scale):
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.scale = scale

    def semseg_batch_resize(self, targets):
        labels = [target.get_field("semsegs").semseg for target in targets]
        labels = semseg_batch_resize(labels, size_divisible=cfg.TRAIN.SIZE_DIVISIBILITY, scale=self.scale)
        return labels

    def __call__(self, semantic_pred, targets):
        labels = self.semseg_batch_resize(targets)
        labels = cat([label for label in labels], dim=0).long()
        assert len(labels.shape) == 3

        loss_semseg = F.cross_entropy(semantic_pred, labels, ignore_index=self.ignore_label)
        loss_semseg *= self.loss_weight

        return loss_semseg


def semseg_loss_evaluator():
    ignore_label = cfg.SEMSEG.SEMSEG_IGNORE_LABEL
    loss_weight = cfg.SEMSEG.SEMSEG_LOSS_WEIGHT
    scale = 0.5 ** (cfg.SEMSEG.SEMSEG_HEAD.FUSION_LEVEL + 1)

    loss_evaluator = SemSegLossComputation(
        ignore_label,
        loss_weight,
        scale,
    )
    return loss_evaluator
