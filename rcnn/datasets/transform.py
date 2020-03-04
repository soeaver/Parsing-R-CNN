from utils.data import transforms as T

from rcnn.core.config import cfg


def build_transforms(is_train=True):
    if is_train:
        min_size = cfg.TRAIN.SCALES
        max_size = cfg.TRAIN.MAX_SIZE
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.TRAIN.BRIGHTNESS
        contrast = cfg.TRAIN.CONTRAST
        saturation = cfg.TRAIN.SATURATION
        hue = cfg.TRAIN.HUE
        left_right = cfg.TRAIN.LEFT_RIGHT

        # for force resize
        force_test_scale = [-1, -1]
        scale_ratios = cfg.TRAIN.RANDOM_CROP.SCALE_RATIOS

        # for random crop
        preprocess_type = cfg.TRAIN.PREPROCESS_TYPE

        crop_sizes = cfg.TRAIN.RANDOM_CROP.CROP_SCALES
        crop_iou_ths = cfg.TRAIN.RANDOM_CROP.IOU_THS
        pad_pixel = cfg.TRAIN.RANDOM_CROP.PAD_PIXEL
        pad_pixel = (cfg.PIXEL_MEANS if len(pad_pixel) < 3 else pad_pixel)
    else:
        min_size = cfg.TEST.SCALE
        max_size = cfg.TEST.MAX_SIZE
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0
        left_right = ()

        # for force resize
        force_test_scale = cfg.TEST.FORCE_TEST_SCALE
        scale_ratios = ()

        # for random crop
        preprocess_type = "none"

        crop_sizes = ()
        pad_pixel = ()
        crop_iou_ths = ()

    to_bgr255 = cfg.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.PIXEL_MEANS, std=cfg.PIXEL_STDS, to_bgr255=to_bgr255
    )

    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size, preprocess_type, scale_ratios, force_test_scale),
            T.RandomCrop(preprocess_type, crop_sizes, pad_pixel, crop_iou_ths),
            T.RandomHorizontalFlip(flip_prob, left_right),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
