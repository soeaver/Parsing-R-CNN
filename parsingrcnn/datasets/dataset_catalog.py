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

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from parsingrcnn.core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

# Optional dataset entry keys
_IM_PREFIX = 'image_prefix'
_DEVKIT_DIR = 'devkit_directory'
_RAW_DIR = 'raw_dir'
_PARS_DIR = 'parsing_directory'
_FSEG_DIR = 'fseg_directory'

# Available datasets
_DATASETS = {
    # 'cityscapes_fine_instanceonly_seg_train': {
    #     _IM_DIR:
    #         _DATA_DIR + '/cityscapes/images',
    #     _ANN_FN:
    #         _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
    #     _RAW_DIR:
    #         _DATA_DIR + '/cityscapes/raw'
    # },
    # 'cityscapes_fine_instanceonly_seg_val': {
    #     _IM_DIR:
    #         _DATA_DIR + '/cityscapes/images',
    #     # use filtered validation as there is an issue converting contours
    #     _ANN_FN:
    #         _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
    #     _RAW_DIR:
    #         _DATA_DIR + '/cityscapes/raw'
    # },
    # 'cityscapes_fine_instanceonly_seg_test': {
    #     _IM_DIR:
    #         _DATA_DIR + '/cityscapes/images',
    #     _ANN_FN:
    #         _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
    #     _RAW_DIR:
    #         _DATA_DIR + '/cityscapes/raw'
    # },
    'cityscapes_fine_instanceonly_seg_train': {
        _IM_DIR:
            _DATA_DIR + '/CityScape/images',
        _ANN_FN:
            _DATA_DIR + '/CityScape/annotations/instancesonly_filtered_gtFine_train.json',
    },
    'cityscapes_fine_instanceonly_seg_val': {
        _IM_DIR:
            _DATA_DIR + '/CityScape/images',
        # use filtered validation as there is an issue converting contours
        _ANN_FN:
            _DATA_DIR + '/CityScape/annotations/instancesonly_filtered_gtFine_val.json',
    },
    'cityscapes_fine_instanceonly_seg_test': {
        _IM_DIR:
            _DATA_DIR + '/CityScape/images',
        _ANN_FN:
            _DATA_DIR + '/CityScape/annotations/instancesonly_filtered_gtFine_test.json',
    },
    'coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
    },
    'coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
    },
    'coco_2017_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
    },
    'coco_stuff_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_train.json'
    },
    'coco_stuff_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'keypoints_coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json'
    },
    'keypoints_coco_2017_train-6k': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017-6k.json'
    },
    'keypoints_coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json'
    },
    'keypoints_coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json'
    },
    'keypoints_coco_2017_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
    },
    'dense_coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/densepose_coco_2014_train.json',
        _IM_PREFIX:
            'COCO_train2014_'
    },
    'dense_coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/densepose_coco_2014_minival.json',
        _IM_PREFIX:
            'COCO_val2014_'
    },
    'dense_coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/densepose_coco_2014_valminusminival.json',
        _IM_PREFIX:
            'COCO_val2014_'
    },
    'dense_coco_2014_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/densepose_coco_2014_test.json',
        _IM_PREFIX:
            'COCO_test2015_'
    },
    'keypoints_densepose_train2017': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/keypoints_densepose_train2017.json',
        _IM_PREFIX:
            ''
    },
    'keypoints_densepose_val2017': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/keypoints_densepose_val2017.json',
        _IM_PREFIX:
            ''
    },
    # 'voc_2007_trainval': {
    #     _IM_DIR:
    #         _DATA_DIR + '/VOC2007/JPEGImages',
    #     _ANN_FN:
    #         _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
    #     _DEVKIT_DIR:
    #         _DATA_DIR + '/VOC2007/VOCdevkit2007'
    # },
    # 'voc_2007_test': {
    #     _IM_DIR:
    #         _DATA_DIR + '/VOC2007/JPEGImages',
    #     _ANN_FN:
    #         _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
    #     _DEVKIT_DIR:
    #         _DATA_DIR + '/VOC2007/VOCdevkit2007'
    # },
    # 'voc_2012_trainval': {
    #     _IM_DIR:
    #         _DATA_DIR + '/VOC2012/JPEGImages',
    #     _ANN_FN:
    #         _DATA_DIR + '/VOC2012/annotations/voc_2012_trainval.json',
    #     _DEVKIT_DIR:
    #         _DATA_DIR + '/VOC2012/VOCdevkit2012'
    # },
    'voc_2007_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2007_trainval/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2007_trainval/Json_Annos/voc_2007_train.json',
    },
    'voc_2007_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2007_trainval/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2007_trainval/Json_Annos/voc_2007_val.json',
    },
    'voc_2007_te-st': {  # new addition by wzh, 'test' will not be evaluated
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2007_test/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2007_test/Json_Annos/voc_2007_test.json',
    },
    'voc_2012_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2012_trainval/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2012_trainval/Json_Annos/voc_2012_train.json',
    },
    'voc_2012_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2012_trainval/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2012_trainval/Json_Annos/voc_2012_val.json',
    },
    'voc_2012_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2012_test/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2012_test/Json_Annos/voc_2012_test.json',
    },
    'SBD_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/SBD/img',
        _ANN_FN:
            _DATA_DIR + '/SBD/annotations/SBD_train.json',
    },
    'SBD_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/SBD/img',
        _ANN_FN:
            _DATA_DIR + '/SBD/annotations/SBD_val.json',
    },
    'SBD_train-2': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/SBD/img',
        _ANN_FN:
            _DATA_DIR + '/SBD/annotations/SBD_train-2.json',
    },
    'SBD_val-2': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/SBD/img',
        _ANN_FN:
            _DATA_DIR + '/SBD/annotations/SBD_val-2.json',
    },
    'CIHP_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP/train_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_train.json',
        _PARS_DIR:
            _DATA_DIR + '/CIHP/train_parsing',
        _FSEG_DIR:
            _DATA_DIR + '/CIHP/train_seg',
    },
    'CIHP_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP/val_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
        _PARS_DIR:
            _DATA_DIR + '/CIHP/val_parsing',
        _FSEG_DIR:
            _DATA_DIR + '/CIHP/val_seg',
    },
    'CIHP_val_200': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP/val_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_val_200.json',
        _PARS_DIR:
            _DATA_DIR + '/CIHP/val_parsing',
        _FSEG_DIR:
            _DATA_DIR + '/CIHP/val_seg',
    },
    'CIHP_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP/test_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_test.json',
        _PARS_DIR:
            _DATA_DIR + '/CIHP',
        _FSEG_DIR:
            _DATA_DIR + '/CIHP',
    },
    'MHP-v2_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/train_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_train.json',
        _PARS_DIR:
            _DATA_DIR + '/MHP-v2/train_parsing',
        _FSEG_DIR:
            _DATA_DIR + '/MHP-v2/train_seg',
    },
    'MHP-v2_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/val_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_val.json',
        _PARS_DIR:
            _DATA_DIR + '/MHP-v2/val_parsing',
        _FSEG_DIR:
            _DATA_DIR + '/MHP-v2/val_seg',
    },
    'MHP-v2_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test_all.json',
        _PARS_DIR:
            _DATA_DIR + '/MHP-v2',
        _FSEG_DIR:
            _DATA_DIR + '/MHP-v2',
    },
    'MHP-v2_test_inter_top10': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test_inter_top10.json',
        _PARS_DIR:
            _DATA_DIR + '/MHP-v2',
        _FSEG_DIR:
            _DATA_DIR + '/MHP-v2',
    },
    'MHP-v2_test_inter_top20': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test_inter_top20.json',
        _PARS_DIR:
            _DATA_DIR + '/MHP-v2',
        _FSEG_DIR:
            _DATA_DIR + '/MHP-v2',
    }
}


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in _DATASETS.keys()


def get_im_dir(name):
    """Retrieve the image directory for the dataset."""
    return _DATASETS[name][_IM_DIR]


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return _DATASETS[name][_ANN_FN]


def get_im_prefix(name):
    """Retrieve the image prefix for the dataset."""
    return _DATASETS[name][_IM_PREFIX] if _IM_PREFIX in _DATASETS[name] else ''


def get_devkit_dir(name):
    """Retrieve the devkit dir for the dataset."""
    return _DATASETS[name][_DEVKIT_DIR]


def get_raw_dir(name):
    """Retrieve the raw dir for the dataset."""
    return _DATASETS[name][_RAW_DIR]


def get_parsing_dir(name):
    """Retrieve the parsing dir for the dataset."""
    return _DATASETS[name][_PARS_DIR]


def get_fseg_dir(name):
    """Retrieve the fseg dir for the dataset."""
    return _DATASETS[name][_FSEG_DIR]
