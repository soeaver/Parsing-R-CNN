import torch
from torch import nn

from utils.data.structures.bounding_box import BoxList
from rcnn.modeling.mask_rcnn.maskiou import heads
from rcnn.modeling.mask_rcnn.maskiou import outputs
from rcnn.modeling.mask_rcnn.maskiou.inference import maskiou_post_processor
from rcnn.modeling.mask_rcnn.maskiou.loss import maskiou_loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class MaskIoU(torch.nn.Module):
    def __init__(self, dim_in):
        super(MaskIoU, self).__init__()
        head = registry.MASKIOU_HEADS[cfg.MRCNN.MASKIOU.MASKIOU_HEAD]
        self.Head = head(dim_in)
        output = registry.MASKIOU_OUTPUTS[cfg.MRCNN.MASKIOU.MASKIOU_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.post_processor = maskiou_post_processor()
        self.loss_evaluator = maskiou_loss_evaluator()

    def forward(self, features, proposals, selected_mask, labels, maskiou_targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            selected_mask (list[Tensor]): targeted mask
            labels (list[Tensor]): class label of mask
            maskiou_targets (list[Tensor], optional): the ground-truth maskiou targets.

        Returns:
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
            results (list[BoxList]): during training, returns None. During testing, the predicted boxlists are returned.
                with the `mask` field set
        """
        if features.shape[0] == 0 and not self.training:
            return {}, proposals

        x = self.Head(features, selected_mask)
        pred_maskiou = self.Output(x)

        if self.training:
            return self._forward_train(pred_maskiou, labels, maskiou_targets)
        else:
            return self._forward_test(proposals, pred_maskiou, labels)

    def _forward_train(self, pred_maskiou, labels, maskiou_targets=None):
        loss_maskiou = self.loss_evaluator(labels, pred_maskiou, maskiou_targets)
        return loss_maskiou, None

    def _forward_test(self, proposals, pred_maskiou, labels):
        result = self.post_processor(proposals, pred_maskiou, labels)
        return {}, result
