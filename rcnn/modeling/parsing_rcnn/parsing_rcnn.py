import torch

from rcnn.modeling.parsing_rcnn import heads
from rcnn.modeling.parsing_rcnn import outputs
from rcnn.modeling.parsing_rcnn.inference import parsing_post_processor
from rcnn.modeling.parsing_rcnn.loss import parsing_loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class ParsingRCNN(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(ParsingRCNN, self).__init__()
        if len(cfg.PRCNN.ROI_STRIDES) == 0:
            self.spatial_scale = spatial_scale
        else:
            self.spatial_scale = [1. / stride for stride in cfg.PRCNN.ROI_STRIDES]

        head = registry.ROI_PARSING_HEADS[cfg.PRCNN.ROI_PARSING_HEAD]
        self.Head = head(dim_in, self.spatial_scale)
        output = registry.ROI_PARSING_OUTPUTS[cfg.PRCNN.ROI_PARSING_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.post_processor = parsing_post_processor()
        self.loss_evaluator = parsing_loss_evaluator()

    def forward(self, conv_features, proposals, targets=None):
        """
        Arguments:
            conv_features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.
        Returns:
            x (Tensor): the result of the feature extractor
            all_proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `parsing` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            return self._forward_train(conv_features, proposals, targets)
        else:
            return self._forward_test(conv_features, proposals)

    def _forward_train(self, conv_features, proposals, targets=None):
        all_proposals = proposals
        with torch.no_grad():
            proposals = self.loss_evaluator.resample(proposals, targets)

        x, roi_feature = self.Head(conv_features, proposals)
        parsing_logits = self.Output(x)

        loss_parsing = self.loss_evaluator(parsing_logits)
        return x, all_proposals, dict(loss_parsing=loss_parsing)

    def _forward_test(self, conv_features, proposals):
        x, roi_feature = self.Head(conv_features, proposals)
        parsing_logits = self.Output(x)

        result = self.post_processor(parsing_logits, proposals)
        return x, result, {}
