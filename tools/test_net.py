import os
import argparse

import _init_paths  # pylint: disable=unused-import
from utils.misc import mkdir_p, logging_rank

from rcnn.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from rcnn.core.test_engine import run_inference

# Parse arguments
parser = argparse.ArgumentParser(description='Hier R-CNN Model Testing')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/mscoco_humanparts/e2e_hier_rcnn_R-50-FPN_1x.yaml', type=str)
parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7', help='gpu id for evaluation')
parser.add_argument('--range', help='start (inclusive) and end (exclusive) indices', type=int, nargs=2)
parser.add_argument('opts', help='See rcnn/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def main():
    if len(args.gpu_id.split(',')) == 1:
        local_rank = int(args.gpu_id.split(',')[0])
    else:
        local_rank = -1
    args.local_rank = local_rank

    num_gpus = len(args.gpu_id.split(','))
    multi_gpu_testing = True if num_gpus > 1 else False

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)

    if not os.path.isdir(os.path.join(cfg.CKPT, 'test')):
        mkdir_p(os.path.join(cfg.CKPT, 'test'))
    if cfg.VIS.ENABLED:
        if not os.path.exists(os.path.join(cfg.CKPT, 'vis')):
            mkdir_p(os.path.join(cfg.CKPT, 'vis'))

    assert_and_infer_cfg(make_immutable=False)
    args.test_net_file, _ = os.path.splitext(__file__)
    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=multi_gpu_testing
    )


if __name__ == '__main__':
    main()
