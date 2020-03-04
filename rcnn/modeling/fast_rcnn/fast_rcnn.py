import torch
from torch import nn

from rcnn.modeling.fast_rcnn import heads
from rcnn.modeling.fast_rcnn import outputs
from rcnn.modeling.fast_rcnn.inference import box_post_processor
from rcnn.modeling.fast_rcnn.loss import box_loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class FastRCNN(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, dim_in, spatial_scale):
        super(FastRCNN, self).__init__()
        head = registry.ROI_BOX_HEADS[cfg.FAST_RCNN.ROI_BOX_HEAD]
        self.Head = head(dim_in, spatial_scale)
        output = registry.ROI_BOX_OUTPUTS[cfg.FAST_RCNN.ROI_BOX_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.post_processor = box_post_processor()
        self.loss_evaluator = box_loss_evaluator()

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            return self._forward_train(features, proposals, targets)
        else:
            return self._forward_test(features, proposals)

    def _forward_train(self, features, proposals, targets=None):
        # Faster R-CNN subsamples during training the proposals with a fixed
        # positive / negative ratio
        with torch.no_grad():
            proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.Head(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.Output(x)

        losses = self.loss_evaluator([class_logits], [box_regression])
        return x, proposals, losses

    def _forward_test(self, features, proposals):
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.Head(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.Output(x)

        result = self.post_processor((class_logits, box_regression), proposals)
        return x, result, {}
