import math

import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes


class ATSSBoxCoder(object):

    def __init__(self, reg_type, anchor_size, anchor_strides):
        self.reg_type = reg_type
        self.anchor_size = anchor_size
        self.anchor_strides = anchor_strides

    def encode(self, gt_boxes, anchors):
        TO_REMOVE = 1  # TODO remove
        if self.reg_type == 'POINT':
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.anchor_size[0] / self.anchor_strides[0]
            l = w * (anchors_cx - gt_boxes[:, 0]) / anchors_w
            t = w * (anchors_cy - gt_boxes[:, 1]) / anchors_h
            r = w * (gt_boxes[:, 2] - anchors_cx) / anchors_w
            b = w * (gt_boxes[:, 3] - anchors_cy) / anchors_h
            targets = torch.stack([l, t, r, b], dim=1)
        elif self.reg_type == 'BOX':
            ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
            gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
            gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
            gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = ww * torch.log(gt_widths / ex_widths)
            targets_dh = wh * torch.log(gt_heights / ex_heights)
            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        else:
            raise NotImplementedError

        return targets

    def decode(self, preds, anchors):
        TO_REMOVE = 1  # TODO remove
        if self.reg_type == 'POINT':
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.anchor_size[0] / self.anchor_strides[0]
            x1 = anchors_cx - preds[:, 0] / w * anchors_w
            y1 = anchors_cy - preds[:, 1] / w * anchors_h
            x2 = anchors_cx + preds[:, 2] / w * anchors_w
            y2 = anchors_cy + preds[:, 3] / w * anchors_h
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
        elif self.reg_type == 'BOX':
            anchors = anchors.to(preds.dtype)

            widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            dx = preds[:, 0::4] / wx
            dy = preds[:, 1::4] / wy
            dw = preds[:, 2::4] / ww
            dh = preds[:, 3::4] / wh

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=math.log(1000. / 16))
            dh = torch.clamp(dh, max=math.log(1000. / 16))

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]

            pred_boxes = torch.zeros_like(preds)
            pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
            pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
            pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
            pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
        else:
            raise NotImplementedError
        
        return pred_boxes
