import torch
from torch import nn

from rcnn.modeling.cascade_rcnn import heads
from rcnn.modeling.cascade_rcnn import outputs
from rcnn.modeling.cascade_rcnn.inference import box_post_processor
from rcnn.modeling.cascade_rcnn.loss import box_loss_evaluator
from rcnn.modeling import registry
from rcnn.core.config import cfg


class CascadeRCNN(torch.nn.Module):
    """
    Generic Box Head class.
    """
    def __init__(self, dim_in, spatial_scale):
        super(CascadeRCNN, self).__init__()
        self.num_stage = cfg.CASCADE_RCNN.NUM_STAGE
        self.test_stage = cfg.CASCADE_RCNN.TEST_STAGE
        self.stage_loss_weights = cfg.CASCADE_RCNN.STAGE_WEIGHTS
        self.test_ensemble = cfg.CASCADE_RCNN.TEST_ENSEMBLE

        head = registry.ROI_CASCADE_HEADS[cfg.CASCADE_RCNN.ROI_BOX_HEAD]
        output = registry.ROI_CASCADE_OUTPUTS[cfg.CASCADE_RCNN.ROI_BOX_OUTPUT]

        for stage in range(1, self.num_stage + 1):
            stage_name = '_{}'.format(stage)
            setattr(self, 'Box_Head' + stage_name, head(dim_in, spatial_scale))
            setattr(self, 'Output' + stage_name, output(getattr(self, 'Box_Head' + stage_name).dim_out))

    def forward(self, features, proposals, targets=None):
        if self.training:
            return self._forward_train(features, proposals, targets)
        else:
            return self._forward_test(features, proposals)

    def _forward_train(self, features, proposals, targets=None):
        all_loss = dict()
        for i in range(self.num_stage):
            head = getattr(self, 'Box_Head_{}'.format(i + 1))
            output = getattr(self, 'Output_{}'.format(i + 1))
            loss_evaluator = box_loss_evaluator(i)

            # Cascade R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = loss_evaluator.subsample(proposals, targets)

            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x = head(features, proposals)
            # final classifier that converts the features into predictions
            class_logits, box_regression = output(x)

            loss_classifier, loss_box_reg = loss_evaluator([class_logits], [box_regression])
            loss_scalar = self.stage_loss_weights[i]
            all_loss['s{}_cls_loss'.format(i + 1)] = loss_classifier * loss_scalar
            all_loss['s{}_bbox_loss'.format(i + 1)] = loss_box_reg * loss_scalar

            with torch.no_grad():
                if i < self.num_stage - 1:
                    post_processor_train = box_post_processor(i, is_train=True)
                    proposals = post_processor_train((class_logits, box_regression), proposals, targets)

        return x, proposals, all_loss

    def _forward_test(self, features, proposals):
        ms_scores = []
        for i in range(self.num_stage):
            head = getattr(self, 'Box_Head_{}'.format(i + 1))
            output = getattr(self, 'Output_{}'.format(i + 1))
            post_processor_test = box_post_processor(i, is_train=False)
            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x = head(features, proposals)
            # final classifier that converts the features into predictions
            class_logits, box_regression = output(x)
            ms_scores.append(class_logits)

            if i < self.test_stage - 1:
                proposals = post_processor_test((class_logits, box_regression), proposals)
            else:
                if self.test_ensemble:
                    assert len(ms_scores) == self.test_stage
                    class_logits = sum(ms_scores) / self.test_stage
                result = post_processor_test((class_logits, box_regression), proposals)
                return x, result, {}
