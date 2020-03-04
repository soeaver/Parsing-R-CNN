import cv2
import os
import copy
import torch
import numpy as np

from torch.nn import functional as F

from cv2 import IMREAD_UNCHANGED

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class SemanticSegmentation(object):

    def __init__(self, semseg, class_ids, size, mode='poly'):
        self.size = size
        self.class_ids = class_ids
        self.mode = mode

        semseg = self.convert_semseg(semseg, class_ids) if isinstance(semseg, list) else semseg
        if isinstance(semseg, torch.Tensor):
            # The raw data representation is passed as argument
            semseg = semseg.clone()
        elif isinstance(semseg, (list, tuple, np.ndarray)):
            semseg = torch.as_tensor(semseg)

        # single channel
        semseg = semseg.unsqueeze(0) if len(semseg.shape) == 2 else semseg
        assert len(semseg.shape) == 3 and semseg.shape[0] == 1

        self.semseg = semseg
        self.extra_fields = {}

    def convert_semseg(self, semsegs_anno, class_ids):
        if self.mode == 'poly':
            img = np.zeros(self.size)
            for semseg_anno_per_instance, class_id in zip(semsegs_anno, class_ids):
                for semseg_anno_per_part in semseg_anno_per_instance:
                    seg = np.array(semseg_anno_per_part).astype(int)
                    seg = seg.reshape((seg.shape[0] // 2, 2))
                    cv2.fillConvexPoly(img, np.array([seg]), int(class_id))
            return img
        else:
            raise Exception('Unsupportable type annotations !')

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT implemented")

        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_semseg = self.semseg.flip(dim)

        if self.mode == 'pic':
            from pet.utils.data.structures.parsing import FLIP_MAP
            flipped_semseg = flipped_semseg.numpy()
            for l_r in FLIP_MAP:
                left = np.where(flipped_semseg == l_r[0])
                right = np.where(flipped_semseg == l_r[1])
                flipped_semseg[left] = l_r[1]
                flipped_semseg[right] = l_r[0]
            flipped_semseg = torch.from_numpy(flipped_semseg)

        return SemanticSegmentation(flipped_semseg, self.class_ids, self.size, mode=self.mode)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        width, height = xmax - xmin, ymax - ymin
        cropped_semseg = self.semseg[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height

        return SemanticSegmentation(cropped_semseg, self.class_ids, cropped_size, mode=self.mode)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)
        # width, height = int(width * float(scale) + 0.5), int(height * float(scale) + 0.5)

        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_semseg = F.interpolate(
            self.semseg[None].float(),
            size=(height, width),
            mode="nearest",
        )[0].type_as(self.semseg)

        resized_size = width, height
        return SemanticSegmentation(resized_semseg, self.class_ids, resized_size, mode=self.mode)

    def to(self, device):
        semseg = self.semseg.to(device)
        return SemanticSegmentation(semseg, self.class_ids, self.size, mode=self.mode)

    def __getitem__(self, item):
        return SemanticSegmentation(self.semseg, self.class_ids, self.size, mode=self.mode)

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def __len__(self):
        return len(self.dp_uvs)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "semseg_shape={}, ".format(self.semseg.shape)
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        return s


def get_semseg(root_dir, parsing_name):
    """
    get picture form annotations when parsing task runs
    """
    parsing_dir = root_dir.replace('img', 'seg')
    parsing_path = os.path.join(parsing_dir, parsing_name.replace('jpg', 'png'))
    return cv2.imread(parsing_path, 0)


def semseg_batch_resize(tensors, size_divisible=0, scale=1 / 8):
    assert isinstance(tensors, list)
    if size_divisible > 0:
        max_size = tuple(max(s) for s in zip(*[semseg.shape for semseg in tensors]))
        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_semsegs = tensors[0].new(*batch_shape).zero_()
        for semseg, pad_semseg in zip(tensors, batched_semsegs):
            pad_semseg[: semseg.shape[0], : semseg.shape[1], : semseg.shape[2]].copy_(semseg)

        _, _, height, width = batched_semsegs.shape

        width, height = int(width * float(scale) + 0.5), int(height * float(scale) + 0.5)
        # Height comes first here!
        batched_resized_semsegs = F.interpolate(
            batched_semsegs.float(),
            size=(height, width),
            mode="nearest",
        ).type_as(batched_semsegs)

        return batched_resized_semsegs
