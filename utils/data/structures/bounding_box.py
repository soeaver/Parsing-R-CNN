import numpy as np

import torch
# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def move(self, gap):
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = self.size[0]-gap[0], self.size[1]-gap[1]
        moved_xmin = xmin - gap[0]
        moved_ymin = ymin - gap[1]
        moved_xmax = xmax - gap[0]
        moved_ymax = ymax - gap[1]
        moved_bbox = torch.cat(
            (moved_xmin, moved_ymin, moved_xmax, moved_ymax), dim=-1
        )
        bbox = BoxList(moved_bbox, (w, h), mode="xyxy")

        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.move(gap)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, (torch.Tensor, np.ndarray, list)):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, (torch.Tensor, np.ndarray, list)):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def set_size(self, size):
        self.size = size
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v.size = size
                v.set_size(size)
            self.add_field(k, v)

        return self
    
    def transpose(self, method, left_right=[]):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        if len(left_right) > 0:
            labels = self.get_field('labels').numpy()
            for i in left_right:
                left = np.where(labels == i[0])[0]
                right = np.where(labels == i[1])[0]
                labels[left] = i[1]
                labels[right] = i[0]
            bbox.add_field("labels", torch.tensor(labels))
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, (torch.Tensor, np.ndarray, list)):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                if k == "uv":
                    v = v.crop(box, self.bbox)
                else:
                    v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def ssd_crop(self, boxes, roi, roi_width, roi_height,labels):
        boxes_t = boxes.copy()
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]
        ssd_crop_boxes = torch.from_numpy(boxes_t)
        bbox = BoxList(ssd_crop_boxes, (roi_width, roi_height), mode="xyxy")
        self.add_field('labels', torch.tensor(labels))
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def ssd_expand(self, left, top, roi_width, roi_height):
        boxes = self.bbox.numpy()
        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)
        ssd_expand_boxes = torch.from_numpy(boxes_t)
        bbox = BoxList(ssd_expand_boxes, (roi_width, roi_height), mode="xyxy")

        for k, v in self.extra_fields.items():
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def ssd_mirror(self, img_width, left_right):
        w, h = self.size
        boxes_t = self.bbox.numpy()
        if len(left_right) == 0:
            boxes_t[:, 0::2] = img_width - boxes_t[:, 2::-2]
            ssd_mirror_boxes = torch.from_numpy(boxes_t)
            bbox = BoxList(ssd_mirror_boxes, (w, h), mode="xyxy")
            for k, v in self.extra_fields.items():
                bbox.add_field(k, v)
            return bbox.convert(self.mode)
        else:
            labels = self.get_field('labels').numpy()
            boxes_t[:, 0::2] = img_width - boxes_t[:, 2::-2]
            for i in left_right:
                left = np.where(labels == i[0])[0]
                right = np.where(labels == i[1])[0]
                labels[left] = i[1]
                labels[right] = i[0]
            ssd_mirror_boxes = torch.from_numpy(boxes_t)
            bbox = BoxList(ssd_mirror_boxes, (w, h), mode="xyxy")
            self.add_field("labels", torch.tensor(labels))
            for k, v in self.extra_fields.items():
                bbox.add_field(k, v)
            return bbox.convert(self.mode)

    def ssd_collect(self, bboxes, labels):
        w, h = self.size
        boxes = bboxes.copy()
        ssd_resize_boxes = torch.from_numpy(boxes)
        bbox = BoxList(ssd_resize_boxes, (w, h), mode="xyxy")
        self.add_field("labels", torch.tensor(labels))
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def ssd_resize(self, size):
        w, h = self.size
        boxes = self.bbox.numpy()
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
        ssd_resize_boxes = torch.from_numpy(boxes)
        bbox = BoxList(ssd_resize_boxes, (size[1], size[0]), mode="xyxy")
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s

