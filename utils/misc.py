import errno
import os
import logging
import numpy as np
from six.moves import cPickle as pickle

import torch
import torch.distributed as dist

# Set up logging and load config options
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# logging only in rank 0
def logging_rank(sstr, distributed=True, local_rank=0):
    if distributed and local_rank == 0:
        logger.info(sstr)
    elif not distributed:
        logger.info(sstr)
    return 0


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def mkdir_p(path):
    """make dir if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

            
def save_object(obj, file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
