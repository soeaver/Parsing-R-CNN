import torch
from torch import nn

from rcnn.modeling.uv_rcnn import heads
from rcnn.modeling.uv_rcnn import outputs
from rcnn.modeling.uv_rcnn.inference import uv_post_processor
from rcnn.modeling.uv_rcnn.loss import uv_loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class UVRCNN(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(UVRCNN, self).__init__()
        if len(cfg.UVRCNN.ROI_STRIDES) == 0:
            self.spatial_scale = spatial_scale
        else:
            self.spatial_scale = [1. / stride for stride in cfg.UVRCNN.ROI_STRIDES]
        # self.roi_batch_size = cfg.UVRCNN.ROI_BATCH_SIZE   # TODO

        head = registry.ROI_UV_HEADS[cfg.UVRCNN.ROI_UV_HEAD]
        self.Head = head(dim_in, self.spatial_scale)
        output = registry.ROI_UV_OUTPUTS[cfg.UVRCNN.ROI_UV_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.post_processor = uv_post_processor()
        self.loss_evaluator = uv_loss_evaluator()

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
        uv_logits = self.Output(x)

        loss_Upoints, loss_Vpoints, loss_seg_AnnIndex, loss_IndexUVPoints = self.loss_evaluator(uv_logits)
        loss_dict = dict(loss_Upoints=loss_Upoints, loss_Vpoints=loss_Vpoints,
                         loss_seg_Ann=loss_seg_AnnIndex, loss_IPoints=loss_IndexUVPoints)

        return x, all_proposals, loss_dict

    def _forward_test(self, conv_features, proposals):
        x = self.Head(conv_features, proposals)
        uv_logits = self.Output(x)

        result = self.post_processor(uv_logits, proposals)
        return x, result, {}
