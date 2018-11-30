from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
import pprint
import shutil
import yaml
import random
import cv2
import re
import numpy as np

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths  # pylint: disable=unused-import
import parsingrcnn.nn as mynn
import parsingrcnn.utils.net as net_utils
import parsingrcnn.utils.misc as misc_utils
from parsingrcnn.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from parsingrcnn.datasets.roidb import combined_roidb_for_training
from parsingrcnn.roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from parsingrcnn.modeling.model_builder import Generalized_RCNN, RetinaNet
from parsingrcnn.utils.detectron_weight_helper import load_detectron_weight
from parsingrcnn.utils.pytorch_weight_helper import load_pytorch_weight
from parsingrcnn.utils.logging import setup_logging
from parsingrcnn.utils.timer import Timer
from parsingrcnn.utils.misc import mkdir_p
from parsingrcnn.utils.training_stats import TrainingStats

# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/maskrcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml', type=str)
parser.add_argument('--multi-gpu-testing', dest='multi_gpu_testing',
                    help='Use cfg.NUM_GPUS GPUs for inference',
                    action='store_true')  # TODO
parser.add_argument('--skip-test', dest='skip_test',
                    help='Do not test the final model',
                    action='store_true')  # TODO
parser.add_argument('--tensorboard', action='store_true',
                    help='use tensorboardX for visualization?')  # TODO
parser.add_argument('opts', help='See parsingrcnn/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()
args.iter_size = 1
args.start_step = 0
args.run_name = 'Training'
args.cfg_filename = args.cfg_file.split('/')[-1]
print('==> Called with args:')
print(args)
if args.cfg_file is not None:
    merge_cfg_from_file(args.cfg_file)
if args.opts is not None:
    merge_cfg_from_list(args.opts)
print('==> Using config:')
pprint.pprint(cfg)

if cfg.NUM_GPUS > 0:
    cfg.CUDA = True
else:
    raise ValueError("Need Cuda device to run !")

# Random seed
if cfg.RNG_SEED is None:
    cfg.RNG_SEED = random.randint(1, 10000)
random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)
if cfg.CUDA:
    torch.cuda.manual_seed_all(cfg.RNG_SEED)


def save_ckpt(ckpt_dir, batch_size, step, train_size, model, optimizer):
    """Save checkpoint"""
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


def main():
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    # Calculating total_batch_size
    total_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    assert (total_batch_size % cfg.NUM_GPUS) == 0, 'batch_size: %d, NUM_GPUS: %d' % (total_batch_size, cfg.NUM_GPUS)
    assert_and_infer_cfg()

    if not os.path.isdir(cfg.CKPT):
        mkdir_p(cfg.CKPT)
    if args.cfg_file is not None:
        shutil.copyfile(args.cfg_file, os.path.join(cfg.CKPT, args.cfg_file.split('/')[-1]))

    # Dataset and Loader
    timers = defaultdict(Timer)
    timers['roidb'].tic()
    roidb, ratio_list, ratio_index = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb'].toc()
    roidb_size = len(roidb)
    logger.info('{:d} roidb entries'.format(roidb_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

    train_size = roidb_size // total_batch_size * total_batch_size  # Effective training sample size for one epoch
    batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index),
        batch_size=total_batch_size,
        drop_last=True
    )
    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batchSampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)
    dataiterator = iter(dataloader)

    # Create model
    if cfg.RETINANET.RETINANET_ON:
        maskRCNN = RetinaNet()
    else:
        maskRCNN = Generalized_RCNN()
    if cfg.CUDA:
        maskRCNN.cuda()

    # Define Optimizer
    gn_param_nameset = set()
    for name, module in maskRCNN.named_modules():
        if isinstance(module, nn.GroupNorm):
            gn_param_nameset.add(name + '.weight')
            gn_param_nameset.add(name + '.bias')
    gn_params = []
    gn_param_names = []
    bias_params = []
    bias_param_names = []
    nonbias_params = []
    nonbias_param_names = []
    nograd_param_names = []
    for key, value in maskRCNN.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
                bias_param_names.append(key)
            elif key in gn_param_nameset:
                gn_params.append(value)
                gn_param_names.append(key)
            else:
                nonbias_params.append(value)
                nonbias_param_names.append(key)
        else:
            nograd_param_names.append(key)
    assert (gn_param_nameset - set(nograd_param_names) - set(bias_param_names)) == set(gn_param_names)

    params = [
        {'params': nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': gn_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
    ]  # Learning rate of 0 is a dummy value to be set properly at the start of training

    param_names = [nonbias_param_names, bias_param_names, gn_param_names]  # names of paramerters for each paramter

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)

    files = os.listdir(cfg.CKPT)
    target_file = [name for name in files if name.endswith(('.pth'))]
    if cfg.TRAIN.AUTO_RESUME and target_file:
        _iter = 0
        for f in files:
            iter_string = re.findall(r'\d+(?=\.pth)', f)
            if len(iter_string) > 0:
                checkpoint_iter = int(iter_string[0])
                if checkpoint_iter > _iter:
                    _iter = checkpoint_iter
                    resume_weights_file = os.path.join(cfg.CKPT, f)

        logging.info("resume from %s", resume_weights_file)
        checkpoint = torch.load(resume_weights_file, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'], mapping=False)
        args.start_step = checkpoint['step'] + 1
        if 'train_size' in checkpoint:  # For backward compatibility
            if checkpoint['train_size'] != train_size:
                print('train_size value: %d different from the one in checkpoint: %d' % (
                    train_size, checkpoint['train_size']))

        # reorder the params in optimizer checkpoint's params_groups if needed
        # misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)

        # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
        # However it's fixed on master.
        optimizer.load_state_dict(checkpoint['optimizer'])
        # misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
    else:
        # Load pre-train model
        _, ext = os.path.splitext(cfg.TRAIN.WEIGHTS)
        if ext == '.pth':
            logging.info("loading checkpoint %s", cfg.TRAIN.WEIGHTS)
            if cfg.MODEL.PYTORCH_WEIGHTS_MAPPING:
                load_pytorch_weight(maskRCNN, cfg.TRAIN.WEIGHTS)
            else:
                checkpoint = torch.load(cfg.TRAIN.WEIGHTS, map_location=lambda storage, loc: storage)
                net_utils.load_ckpt(maskRCNN, checkpoint['model'], mapping=False)
                del checkpoint
                torch.cuda.empty_cache()
        elif ext == '.pkl':
            logging.info("loading Detectron weights %s", cfg.TRAIN.WEIGHTS)
            load_detectron_weight(maskRCNN, cfg.TRAIN.WEIGHTS)
        else:
            logging.info("unknown pre-train weights %s", cfg.TRAIN.WEIGHTS)

    lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

    if cfg.RPN.RPN_ON:
        maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True)
    else:
        maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info'], minibatch=True)  # no rpn means no roidb

    # Training Loop
    maskRCNN.train()

    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)
    # Set index for decay steps
    decay_steps_ind = None
    for i in range(1, len(cfg.SOLVER.STEPS)):
        if cfg.SOLVER.STEPS[i] >= args.start_step:
            decay_steps_ind = i
            break
    if decay_steps_ind is None:
        decay_steps_ind = len(cfg.SOLVER.STEPS)

    training_stats = TrainingStats(args)
    try:
        logger.info('Training starts !')
        step = args.start_step
        for step in range(args.start_step, cfg.SOLVER.MAX_ITER):

            # Warm up
            if step < cfg.SOLVER.WARM_UP_ITERS:
                method = cfg.SOLVER.WARM_UP_METHOD
                if method == 'constant':
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
                elif method == 'linear':
                    alpha = step / cfg.SOLVER.WARM_UP_ITERS
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                else:
                    raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                lr_new = cfg.SOLVER.BASE_LR * warmup_factor
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
            elif step == cfg.SOLVER.WARM_UP_ITERS:
                net_utils.update_learning_rate(optimizer, lr, cfg.SOLVER.BASE_LR)
                lr = optimizer.param_groups[0]['lr']
                assert lr == cfg.SOLVER.BASE_LR
            elif step > cfg.SOLVER.WARM_UP_ITERS and cfg.SOLVER.LR_POLICY == 'cosine':    # Cosine learning rate decay
                total_iter = cfg.SOLVER.MAX_ITER - cfg.SOLVER.WARM_UP_ITERS
                lr_new = 0.5 * cfg.SOLVER.BASE_LR * \
                         (np.cos((step - cfg.SOLVER.WARM_UP_ITERS) * np.pi / total_iter) + 1.0)
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
                
            # Learning rate decay
            if cfg.SOLVER.LR_POLICY == 'steps_with_decay' and decay_steps_ind < len(cfg.SOLVER.STEPS) and step == cfg.SOLVER.STEPS[decay_steps_ind]:
                logger.info('Decay the learning on step %d', step)
                lr_new = lr * cfg.SOLVER.GAMMA
                net_utils.update_learning_rate(optimizer, lr, lr_new)
                lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
                decay_steps_ind += 1

            training_stats.IterTic()
            optimizer.zero_grad()
            for inner_iter in range(args.iter_size):
                try:
                    input_data = next(dataiterator)
                except StopIteration:
                    dataiterator = iter(dataloader)
                    input_data = next(dataiterator)

                for key in input_data:
                    if key != 'roidb':  # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))

                net_outputs = maskRCNN(**input_data)
                training_stats.UpdateIterStats(net_outputs, inner_iter)
                loss = net_outputs['total_loss']
                loss.backward()
            optimizer.step()
            training_stats.IterToc()

            training_stats.LogIterStats(step, lr)

            if (step + 1) % CHECKPOINT_PERIOD == 0:
                save_ckpt(cfg.CKPT, total_batch_size, step, train_size, maskRCNN, optimizer)

        # ---- Training ends ----
        # Save last checkpoint
        save_ckpt(cfg.CKPT, total_batch_size, step, train_size, maskRCNN, optimizer)
    except (RuntimeError, KeyboardInterrupt):
        del dataiterator
        logger.info('Quit training on exception ...')
        '''
        logger.info('Save ckpt on exception ...')
        save_ckpt(cfg.CKPT, total_batch_size, step, train_size, maskRCNN, optimizer)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)
        '''


if __name__ == '__main__':
    main()
