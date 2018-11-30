# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""PASCAL VOC dataset evaluation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import os
import shutil
import uuid

from parsingrcnn.core.config import cfg
from parsingrcnn.datasets.voc_eval import voc_eval, mkdir_p
from parsingrcnn.utils.io import save_object

logger = logging.getLogger(__name__)


def evaluate_boxes(
    json_dataset,
    all_boxes,
    output_dir,
    use_salt=False,
    cleanup=False
):
    salt = '_{}'.format(str(uuid.uuid4())) if use_salt else ''
    filenames = _write_voc_results_files(json_dataset, all_boxes, salt)
    map50 = _do_python_eval(json_dataset, salt, output_dir)
    if cleanup:
        for filename in filenames:
            shutil.copy(filename, output_dir)
            os.remove(filename)
    return map50


def _write_voc_results_files(json_dataset, all_boxes, salt):
    filenames = []
    # image set
    image_index = []
    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        index = os.path.splitext(os.path.split(entry['image'])[-1])[0]
        image_index.append(index)
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        logger.info('Writing VOC results for: {}'.format(cls))
        filename = _get_voc_results_file_template(json_dataset,
                                                  salt).format(cls)
        filenames.append(filename)
        assert len(all_boxes[cls_ind]) == len(image_index)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind][im_ind]
                if type(dets) == list:
                    assert len(dets) == 0, \
                        'dets should be numpy.ndarray or empty list'
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
    return filenames


def _get_voc_results_file_template(json_dataset, salt):
    info = voc_info(json_dataset)
    year = info['year']
    image_set = info['image_set']
    # cfg.CKPT/test/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = 'comp4' + salt + '_det_' + image_set + '_{:s}.txt'
    if not os.path.isdir(os.path.join(cfg.CKPT, 'test', 'results', 'VOC' + year, 'Main')):
        mkdir_p(os.path.join(cfg.CKPT, 'test', 'results', 'VOC' + year, 'Main'))
    return os.path.join(cfg.CKPT, 'test', 'results', 'VOC' + year, 'Main', filename)


def parse_entry(entry, cls_mapping=None):
    """Parse a entry of json_dataset."""
    boxes = entry['boxes']
    gt_classes = entry['gt_classes']
    difficults = entry['difficult']
    objects = []
    for ids in range(boxes.shape[0]):
        obj_struct = {}
        gt_classs = gt_classes[ids]
        obj_struct['name'] = cls_mapping[gt_classs]
        obj_struct['pose'] = 'Unspecified'
        obj_struct['truncated'] = 0
        obj_struct['difficult'] = difficults[ids]
        obj_struct['bbox'] = [boxes[ids][0], boxes[ids][1], boxes[ids][2], boxes[ids][3]]
        objects.append(obj_struct)

    return objects


def _do_python_eval(json_dataset, salt, output_dir='output'):
    info = voc_info(json_dataset)
    year = info['year']
    aps = []
    # The PASCAL VOC metric changed in 2010
    if int(year) not in [2007, 2012]:
        year = 2007
    use_07_metric = True if int(year) < 2010 else False
    logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        mkdir_p(output_dir)

    roidb = json_dataset.get_roidb(gt=True)
    cls_mapping = json_dataset.classes
    imagenames = []
    recs = {}
    for i, entry in enumerate(roidb):
        imagename = os.path.splitext(os.path.split(entry['image'])[-1])[0]
        imagenames.append(imagename)
        recs[imagename] = parse_entry(entry, cls_mapping)
        if i % 100 == 0:
            logger.info('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))

    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(json_dataset, salt).format(cls)
        rec, prec, ap = voc_eval(recs, imagenames, filename, cls, ovthresh=0.5,
                                 use_07_metric=use_07_metric)
        aps += [ap]
        logger.info('AP for {} = {:.4f}'.format(cls, ap))
        res_file = os.path.join(output_dir, cls + '_pr.pkl')
        save_object({'rec': rec, 'prec': prec, 'ap': ap}, res_file)
    logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('Results:')
    for ap in aps:
        logger.info('{:.3f}'.format(ap))
    logger.info('~~~~~~~~')
    logger.info('{:.3f}'.format(np.mean(aps)))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info('----------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info('Results should be very close to the official MATLAB code.')
    logger.info('-- Thanks, The Management')
    logger.info('----------------------------------------------------------')

    return np.mean(aps)


def voc_info(json_dataset):
    year = json_dataset.name[4:8]
    image_set = json_dataset.name[9:]
    return dict(
        year=year,
        image_set=image_set)
