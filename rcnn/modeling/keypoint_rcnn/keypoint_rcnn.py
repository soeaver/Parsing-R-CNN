import torch

from rcnn.modeling.keypoint_rcnn import heads
from rcnn.modeling.keypoint_rcnn import outputs
from rcnn.modeling.keypoint_rcnn.inference import keypoint_post_processor
from rcnn.modeling.keypoint_rcnn.loss import keypoint_loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class KeypointRCNN(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(KeypointRCNN, self).__init__()
        if len(cfg.KRCNN.ROI_STRIDES) == 0:
            self.spatial_scale = spatial_scale
        else:
            self.spatial_scale = [1. / stride for stride in cfg.KRCNN.ROI_STRIDES]
            
        head = registry.ROI_KEYPOINT_HEADS[cfg.KRCNN.ROI_KEYPOINT_HEAD]
        self.Head = head(dim_in, self.spatial_scale)
        output = registry.ROI_KEYPOINT_OUTPUTS[cfg.KRCNN.ROI_KEYPOINT_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.post_processor = keypoint_post_processor()
        self.loss_evaluator = keypoint_loss_evaluator()

    def forward(self, conv_features, proposals, targets=None):
        if self.training:
            return self._forward_train(conv_features, proposals, targets)
        else:
            return self._forward_test(conv_features, proposals)

    def _forward_train(self, conv_features, proposals, targets=None):
        all_proposals = proposals
        with torch.no_grad():
            proposals = self.loss_evaluator.resample(proposals, targets)

        x = self.Head(conv_features, proposals)
        kp_logits = self.Output(x)

        loss_kp = self.loss_evaluator(kp_logits)

        return x, all_proposals, dict(loss_kp=loss_kp)

    def _forward_test(self, conv_features, proposals):
        x = self.Head(conv_features, proposals)
        kp_logits = self.Output(x)

        result = self.post_processor(kp_logits, proposals)
        return x, result, {}
