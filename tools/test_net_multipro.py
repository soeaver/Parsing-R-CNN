from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import os
import pprint
import sys

import torch

import _init_paths  # pylint: disable=unused-import
from parsingrcnn.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from parsingrcnn.core.test_engine import run_inference
import parsingrcnn.utils.logging as logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluat the imagenet validation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/maskrcnn/mscoco/e2e_mask_rcnn_R-50-FPN_1x.yaml', type=str)
parser.add_argument('--gpu_id', type=str, default='0', help='gpu id for evaluation')
parser.add_argument('--range', help='start (inclusive) and end (exclusive) indices', type=int, nargs=2)
parser.add_argument('opts', help='See parsingrcnn/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def get_weights(mode='final'):
    final_step = int(cfg.SOLVER.MAX_ITER - 1)
    if os.path.exists(cfg.TEST.WEIGHTS):
        weights = cfg.TEST.WEIGHTS
    else:        
        if os.path.exists(os.path.join(cfg.CKPT, 'model_{}.pkl'.format(mode))): # pkl first
            weights = os.path.join(cfg.CKPT, 'model_{}.pkl'.format(mode))
        elif os.path.exists(os.path.join(cfg.CKPT, 'model_step{}.pth'.format(final_step))):  # last pth second
            weights = os.path.join(cfg.CKPT, 'model_step{}.pth'.format(final_step))
        else:
            all_files = os.listdir(cfg.CKPT)
            max_step = 0
            for _ in all_files:
                if _.endswith('.pth'):
                    cur_step = int(_.replace('model_step', '').replace('.pth', ''))
                    if cur_step > max_step:
                        max_step = cur_step
            if os.path.exists(os.path.join(cfg.CKPT, 'model_step{}.pth'.format(max_step))):
                weights = os.path.join(cfg.CKPT, 'model_step{}.pth'.format(max_step))
            else:
                raise 'No weights file found'
          
    return weights

  
if __name__ == '__main__':
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = logging.setup_logging(__name__)
    logger.info('Called with args:')
    logger.info(args)

    cfg.NUM_GPUS = len(args.gpu_id.split(','))
    multi_gpu_testing = True if cfg.NUM_GPUS > 1 else False
    # assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    output_dir = os.path.join(cfg.CKPT, 'test')
    logger.info('Automatically set output directory to %s', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.output_dir = output_dir
    args.test_net_file, _ = os.path.splitext(__file__)  # For test_engine.multi_gpu_test_net_on_dataset
    args.cuda = True    # manually set args.cuda

    cfg.TEST.WEIGHTS = get_weights()
    logger.info('Loading weights %s', cfg.TEST.WEIGHTS)
    assert_and_infer_cfg()
    _, ext = os.path.splitext(cfg.TEST.WEIGHTS)
    if ext == '.pth':
        args.load_detectron = ''
        args.load_ckpt = cfg.TEST.WEIGHTS
    elif ext == '.pkl':
        args.load_detectron = cfg.TEST.WEIGHTS
        args.load_ckpt = ''
    else:
        raise KeyError('Unknown Model Type: {}'.format(ext))

    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=multi_gpu_testing,
        check_expected_results=True)
