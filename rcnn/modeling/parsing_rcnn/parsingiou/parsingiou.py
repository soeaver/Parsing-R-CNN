import torch
from torch import nn

from utils.data.structures.bounding_box import BoxList
from rcnn.modeling.parsing_rcnn.parsingiou import heads
from rcnn.modeling.parsing_rcnn.parsingiou import outputs
from rcnn.modeling.parsing_rcnn.parsingiou.inference import parsingiou_post_processor
from rcnn.modeling.parsing_rcnn.parsingiou.loss import parsingiou_loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class ParsingIoU(torch.nn.Module):
    def __init__(self, dim_in):
        super(ParsingIoU, self).__init__()
        head = registry.PARSINGIOU_HEADS[cfg.PRCNN.PARSINGIOU.PARSINGIOU_HEAD]
        self.Head = head(dim_in)
        output = registry.PARSINGIOU_OUTPUTS[cfg.PRCNN.PARSINGIOU.PARSINGIOU_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.post_processor = parsingiou_post_processor()
        self.loss_evaluator = parsingiou_loss_evaluator()

    def forward(self, features, proposals, parsing_logits, parsingiou_targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            parsing_logits (list[Tensor]): targeted parsing
            parsingiou_targets (list[Tensor], optional): the ground-truth parsingiou targets.

        Returns:
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
            results (list[BoxList]): during training, returns None. During testing, the predicted boxlists are returned.
                with the `parsing` field set
        """
        if features.shape[0] == 0 and not self.training:
            return {}, proposals

        x = self.Head(features, parsing_logits)
        pred_parsingiou = self.Output(x)

        if self.training:
            return self._forward_train(pred_parsingiou, parsingiou_targets)
        else:
            return self._forward_test(proposals, pred_parsingiou)

    def _forward_train(self, pred_parsingiou, parsingiou_targets=None):
        loss_parsingiou = self.loss_evaluator(pred_parsingiou, parsingiou_targets)
        return loss_parsingiou, None

    def _forward_test(self, proposals, pred_parsingiou):
        result = self.post_processor(proposals, pred_parsingiou)
        return {}, result
