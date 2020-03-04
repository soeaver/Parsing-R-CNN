import os
import bisect
import copy

import torch.utils.data

from utils.data import datasets as D
from utils.data import samplers
from utils.misc import logging_rank
from utils.data.collate_batch import BatchCollator
from utils.comm import get_world_size
from rcnn.datasets import build_transforms
from rcnn.datasets.dataset_catalog import contains, get_im_dir, get_ann_fn
from rcnn.core.config import cfg


def build_dataset(dataset_list, is_train=True, local_rank=0):
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    for dataset_name in dataset_list:
        assert contains(dataset_name), 'Unknown dataset name: {}'.format(dataset_name)
        assert os.path.exists(get_im_dir(dataset_name)), 'Im dir \'{}\' not found'.format(get_im_dir(dataset_name))
        logging_rank('Creating: {}'.format(dataset_name), local_rank=local_rank)

    transforms = build_transforms(is_train)
    datasets = []
    for dataset_name in dataset_list:
        args = {}
        args['root'] = get_im_dir(dataset_name)
        args['ann_file'] = get_ann_fn(dataset_name)
        args['remove_images_without_annotations'] = is_train
        ann_types = ('bbox',)
        if cfg.MODEL.MASK_ON:
            ann_types = ann_types + ('segm',)
        if cfg.MODEL.SEMSEG_ON:
            ann_types = ann_types + ('semseg',)
        if cfg.MODEL.KEYPOINT_ON:
            ann_types = ann_types + ('keypoints',)
        if cfg.MODEL.PARSING_ON:
            ann_types = ann_types + ('parsing',)
        if cfg.MODEL.UV_ON:
            ann_types = ann_types + ('uv',)
        args['ann_types'] = ann_types
        args['transforms'] = transforms
        # make dataset from factory
        dataset = D.COCODataset(**args)
        datasets.append(dataset)

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return dataset


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        if cfg.DATALOADER.SAMPLER_TRAIN == "RepeatFactorTrainingSampler":
            return samplers.RepeatFactorTrainingSampler(dataset, cfg.DATALOADER.RFTSAMPLER, shuffle=shuffle)
        else:
            return samplers.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_train_data_loader(datasets, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    ims_per_gpu = int(cfg.TRAIN.BATCH_SIZE / num_gpus)
    shuffle = True
    num_iters = cfg.SOLVER.MAX_ITER

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    sampler = make_data_sampler(datasets, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        datasets, sampler, aspect_grouping, ims_per_gpu, num_iters, start_iter
    )
    collator = BatchCollator(cfg.TRAIN.SIZE_DIVISIBILITY)
    num_workers = cfg.TRAIN.LOADER_THREADS
    data_loader = torch.utils.data.DataLoader(
        datasets,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )

    return data_loader


def make_test_data_loader(datasets, start_ind, end_ind, is_distributed=True):
    ims_per_gpu = cfg.TEST.IMS_PER_GPU
    if start_ind == -1 or end_ind == -1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(datasets) if is_distributed else None
    else:
        test_sampler = samplers.RangeSampler(start_ind, end_ind)
    num_workers = cfg.TEST.LOADER_THREADS
    collator = BatchCollator(cfg.TEST.SIZE_DIVISIBILITY)
    data_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=ims_per_gpu,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return data_loader
