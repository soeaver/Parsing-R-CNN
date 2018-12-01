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

"""Construct minibatches for Mask R-CNN training. Handles the minibatch blobs
that are specific to Mask R-CNN. Other blobs that are generic to RPN or
Fast/er R-CNN are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import numpy.random as npr

from parsingrcnn.core.config import cfg
import parsingrcnn.utils.blob as blob_utils
import parsingrcnn.utils.boxes as box_utils
import parsingrcnn.utils.parsing as parsing_utils

logger = logging.getLogger(__name__)


def add_parsing_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add parsing R-CNN specific blobs to the input blob dictionary."""
    # Prepare the parsing targets by associating one gt parsing to each training roi
    # that has a fg (non-bg) class label.
    M = cfg.PRCNN.RESOLUTION
    polys_gt_inds = np.where(
        (roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0)
    )[0]

    parsing_gt = [roidb['parsing'][i] for i in polys_gt_inds]
    boxes_from_png = parsing_utils.parsing_to_boxes(parsing_gt, roidb['flipped'])

    fg_inds = np.where(blobs['labels_int32'] > 0)[0]

    if fg_inds.shape[0] > 0:
        if cfg.PRCNN.ROI_BATCH_SIZE > 0:
            fg_rois_per_this_image = np.minimum(cfg.PRCNN.ROI_BATCH_SIZE, fg_inds.shape[0])
            fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False
            )
        parsings = blob_utils.zeros((fg_inds.shape[0], M**2), int32=True)

        # Find overlap between all foreground rois and the bounding boxes
        # enclosing each segmentation
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_png.astype(np.float32, copy=False)
        )
        # Map from each fg rois to the index of the parsing with highest overlap
        # (measured by bbox overlap)
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # add fg targets
        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            parsing_gt_fg = parsing_gt[fg_polys_ind]
            roi_fg = rois_fg[i]
            # Rasterize the portion of the polygon mask within the given fg roi
            # to an M x M binary image

            parsing = parsing_utils.parsing_wrt_box(parsing_gt_fg, roi_fg, M, roidb['flipped'])
            parsings[i, :] = parsing
        weights = blob_utils.ones((rois_fg.shape[0], M**2))
    else:  # If there are no fg masks (it does happen)
        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        # rois_fg is actually one background roi, but that's ok because ...
        if(len(bg_inds)==0):
            rois_fg = sampled_boxes[0].reshape((1, -1))
        else:
            rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # We give it an -1's blob (ignore label)
        parsings = blob_utils.zeros((1, M**2), int32=True)
        # Mark that the first roi has a mask
        weights = blob_utils.zeros((1, M**2))

    parsings = np.reshape(parsings, (-1, 1))
    weights = np.reshape(weights, (-1, 1))

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))

    # Update blobs dict with Mask R-CNN blobs
    blobs['parsing_rois'] = rois_fg
    blobs['parsing_weights'] = weights
    blobs['parsing_int32'] = parsings
