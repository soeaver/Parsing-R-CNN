import cv2
import numpy as np
import pycocotools.mask as mask_util

import torch

from utils.data.structures.bounding_box import BoxList
from utils.data.structures.boxlist_ops import cat_boxlist, boxlist_nms, \
    boxlist_ml_nms, boxlist_soft_nms, boxlist_box_voting
from utils.data.structures.parsing import flip_parsing_featuremap
from utils.data.structures.densepose_uv import flip_uv_featuremap
from rcnn.core.config import cfg


def im_detect_bbox(model, ims):
    box_results = [[] for _ in range(len(ims))]
    features = []
    semseg_pred_results = []
    results, net_imgs_size, blob_conv, semseg_pred = im_detect_bbox_net(model, ims, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    if cfg.RPN.RPN_ONLY:
        return results, None
    add_results(box_results, results)
    features.append((net_imgs_size, blob_conv))
    semseg_pred_results.append(semseg_pred)

    if cfg.TEST.BBOX_AUG.ENABLED:
        if cfg.TEST.BBOX_AUG.H_FLIP:
            results_hf, net_imgs_size_hf, blob_conv_hf, semseg_pred_hf = im_detect_bbox_net(
                model, ims, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, True, net_imgs_size
            )
            add_results(box_results, results_hf)
            features.append((net_imgs_size_hf, blob_conv_hf))
            semseg_pred_results.append(semseg_pred_hf)

        for scale in cfg.TEST.BBOX_AUG.SCALES:
            max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
            results_scl, net_imgs_size_scl, blob_conv_scl, semseg_pred_scl = im_detect_bbox_net(
                model, ims, scale, max_size, False, net_imgs_size
            )
            add_results(box_results, results_scl)
            features.append((net_imgs_size_scl, blob_conv_scl))
            semseg_pred_results.append(semseg_pred_scl)

            if cfg.TEST.BBOX_AUG.H_FLIP:
                results_scl_hf, net_imgs_size_scl_hf, blob_conv_scl_hf, semseg_pred_scl_hf = im_detect_bbox_net(
                    model, ims, scale, max_size, True, net_imgs_size
                )
                add_results(box_results, results_scl_hf)
                features.append((net_imgs_size_scl_hf, blob_conv_scl_hf))
                semseg_pred_results.append(semseg_pred_scl_hf)

    box_results = [cat_boxlist(result) for result in box_results]

    if cfg.MODEL.FASTER_ON:
        box_results = [filter_results(result) for result in box_results]

    if cfg.MODEL.SEMSEG_ON:
        semseg_pred_results = np.asarray(semseg_pred_results).transpose((1, 0, 2, 3, 4))
        for i in range(len(box_results)):
            semseg_pred = np.mean(semseg_pred_results[i], axis=0)
            box_results[i].add_field("semseg", semseg_pred)

    return box_results, features


def im_detect_mask(model, rois, features):
    _idx = 0
    mask_results = [[] for _ in range(len(rois))]
    mask_scores = [[] for _ in range(len(rois))]
    conv_features = features[_idx][1]
    _idx += 1
    results = model.mask_net(conv_features, rois, targets=None)

    if cfg.TEST.BBOX_AUG.ENABLED and cfg.TEST.MASK_AUG.ENABLED:
        if len(rois[0]) == 0:
            return results
        masks = [result.get_field("mask") for result in results]
        add_results(mask_results, masks)
        scores = [result.get_field("mask_scores") for result in results]
        add_results(mask_scores, scores)
        if cfg.TEST.BBOX_AUG.H_FLIP:
            rois_hf = [roi.transpose(0) for roi in rois]
            features_hf = features[_idx][1]
            _idx += 1
            results_hf = model.mask_net(features_hf, rois_hf, targets=None)
            masks_hf = [result_hf.get_field("mask") for result_hf in results_hf]
            masks_hf = [mask_hf[:, :, :, ::-1] for mask_hf in masks_hf]
            add_results(mask_results, masks_hf)
            scores_hf = [result_hf.get_field("mask_scores") for result_hf in results_hf]
            add_results(mask_scores, scores_hf)

        for scale in cfg.TEST.BBOX_AUG.SCALES:
            rois_scl = [roi.resize(size) for roi, size in zip(rois, features[_idx][0])]
            features_scl = features[_idx][1]
            _idx += 1
            results_scl = model.mask_net(features_scl, rois_scl, targets=None)
            masks_scl = [result_scl.get_field("mask") for result_scl in results_scl]
            add_results(mask_results, masks_scl)
            scores_scl = [result_scl.get_field("mask_scores") for result_scl in results_scl]
            add_results(mask_scores, scores_scl)

            if cfg.TEST.BBOX_AUG.H_FLIP:
                rois_scl_hf = [roi.resize(size) for roi, size in zip(rois, features[_idx][0])]
                rois_scl_hf = [roi.transpose(0) for roi in rois_scl_hf]
                features_scl_hf = features[_idx][1]
                _idx += 1
                results_scl_hf = model.mask_net(features_scl_hf, rois_scl_hf, targets=None)
                masks_scl_hf = [result_scl_hf.get_field("mask") for result_scl_hf in results_scl_hf]
                masks_scl_hf = [mask_scl_hf[:, :, :, ::-1] for mask_scl_hf in masks_scl_hf]
                add_results(mask_results, masks_scl_hf)
                scores_scl_hf = [result_scl_hf.get_field("mask_scores") for result_scl_hf in results_scl_hf]
                add_results(mask_scores, scores_scl_hf)

        for masks_ts, scores_ts, result in zip(mask_results, mask_scores, results):
            scores_c = np.mean(scores_ts, axis=0)
            # Combine the predicted soft masks
            if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
                masks_c = np.mean(masks_ts, axis=0)
            elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
                masks_c = np.amax(masks_ts, axis=0)
            elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

                def logit(y):
                    return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

                logit_masks = [logit(y) for y in masks_ts]
                logit_masks = np.mean(logit_masks, axis=0)
                masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
            else:
                raise NotImplementedError('Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR))
            result.add_field("mask", masks_c)
            result.add_field("mask_scores", scores_c)

    return results


def im_detect_keypoint(model, rois, features):
    _idx = 0
    keypoint_results = [[] for _ in range(len(rois))]
    conv_features = features[_idx][1]
    _idx += 1
    results = model.keypoint_net(conv_features, rois, targets=None)

    if cfg.TEST.BBOX_AUG.ENABLED and cfg.TEST.KPS_AUG.ENABLED:
        if len(rois[0]) == 0:
            return results
        keypoints = [result.get_field("keypoints") for result in results]
        add_results(keypoint_results, keypoints)
        if cfg.TEST.BBOX_AUG.H_FLIP:
            rois_hf = [roi.transpose(0) for roi in rois]
            features_hf = features[_idx][1]
            _idx += 1
            results_hf = model.keypoint_net(features_hf, rois_hf, targets=None)
            keypoints_hf = [result_hf.get_field("keypoints") for result_hf in results_hf]
            keypoints_hf = [flip_keypoint(keypoint_hf) for keypoint_hf in keypoints_hf]
            add_results(keypoint_results, keypoints_hf)

        for scale in cfg.TEST.BBOX_AUG.SCALES:
            rois_scl = [roi.resize(size) for roi, size in zip(rois, features[_idx][0])]
            features_scl = features[_idx][1]
            _idx += 1
            results_scl = model.keypoint_net(features_scl, rois_scl, targets=None)
            keypoints_scl = [result_scl.get_field("keypoints") for result_scl in results_scl]
            add_results(keypoint_results, keypoints_scl)

            if cfg.TEST.BBOX_AUG.H_FLIP:
                rois_scl_hf = [roi.resize(size) for roi, size in zip(rois, features[_idx][0])]
                rois_scl_hf = [roi.transpose(0) for roi in rois_scl_hf]
                features_scl_hf = features[_idx][1]
                _idx += 1
                results_scl_hf = model.keypoint_net(features_scl_hf, rois_scl_hf, targets=None)
                keypoints_scl_hf = [result_scl_hf.get_field("keypoints") for result_scl_hf in results_scl_hf]
                keypoints_scl_hf = [flip_keypoint(keypoint_scl_hf) for keypoint_scl_hf in keypoints_scl_hf]
                add_results(keypoint_results, keypoints_scl_hf)

        for keypoints_ts, result in zip(keypoint_results, results):
            # Combine the predicted soft keypoints
            if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
                keypoints_c = np.mean(keypoints_ts, axis=0)
            elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
                keypoints_c = np.amax(keypoints_ts, axis=0)
            else:
                raise NotImplementedError('Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR))
            result.add_field("keypoints", keypoints_c)

    return results


def im_detect_parsing(model, rois, features):
    _idx = 0
    parsing_results = [[] for _ in range(len(rois))]
    parsing_scores = [[] for _ in range(len(rois))]
    conv_features = features[_idx][1]
    _idx += 1
    results = model.parsing_net(conv_features, rois, targets=None)

    if cfg.TEST.BBOX_AUG.ENABLED and cfg.TEST.PARSING_AUG.ENABLED:
        if len(rois[0]) == 0:
            return results
        parsings = [result.get_field("parsing") for result in results]
        add_results(parsing_results, parsings)
        scores = [result.get_field("parsing_scores") for result in results]
        add_results(parsing_scores, scores)
        if cfg.TEST.BBOX_AUG.H_FLIP:
            rois_hf = [roi.transpose(0) for roi in rois]
            features_hf = features[_idx][1]
            _idx += 1
            results_hf = model.parsing_net(features_hf, rois_hf, targets=None)
            parsings_hf = [result_hf.get_field("parsing") for result_hf in results_hf]
            parsings_hf = [flip_parsing_featuremap(parsing_hf) for parsing_hf in parsings_hf]
            add_results(parsing_results, parsings_hf)
            scores_hf = [result_hf.get_field("parsing_scores") for result_hf in results_hf]
            add_results(parsing_scores, scores_hf)

        for scale in cfg.TEST.BBOX_AUG.SCALES:
            rois_scl = [roi.resize(size) for roi, size in zip(rois, features[_idx][0])]
            features_scl = features[_idx][1]
            _idx += 1
            results_scl = model.parsing_net(features_scl, rois_scl, targets=None)
            parsings_scl = [result_scl.get_field("parsing") for result_scl in results_scl]
            add_results(parsing_results, parsings_scl)
            scores_scl = [result_scl.get_field("parsing_scores") for result_scl in results_scl]
            add_results(parsing_scores, scores_scl)

            if cfg.TEST.BBOX_AUG.H_FLIP:
                rois_scl_hf = [roi.resize(size) for roi, size in zip(rois, features[_idx][0])]
                rois_scl_hf = [roi.transpose(0) for roi in rois_scl_hf]
                features_scl_hf = features[_idx][1]
                _idx += 1
                results_scl_hf = model.parsing_net(features_scl_hf, rois_scl_hf, targets=None)
                parsings_scl_hf = [result_scl_hf.get_field("parsing") for result_scl_hf in results_scl_hf]
                parsings_scl_hf = [flip_parsing_featuremap(parsing_scl_hf) for parsing_scl_hf in parsings_scl_hf]
                add_results(parsing_results, parsings_scl_hf)
                scores_scl_hf = [result_scl_hf.get_field("parsing_scores") for result_scl_hf in results_scl_hf]
                add_results(parsing_scores, scores_scl_hf)

        for parsings_ts, scores_ts, result in zip(parsing_results, parsing_scores, results):
            scores_c = np.mean(scores_ts, axis=0)
            # Combine the predicted soft parsings
            if cfg.TEST.PARSING_AUG.HEUR == 'SOFT_AVG':
                parsings_c = np.mean(parsings_ts, axis=0)
            elif cfg.TEST.PARSING_AUG.HEUR == 'SOFT_MAX':
                parsings_c = np.amax(parsings_ts, axis=0)
            elif cfg.TEST.PARSING_AUG.HEUR == 'LOGIT_AVG':

                def logit(y):
                    return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

                logit_parsings = [logit(y) for y in parsings_ts]
                logit_parsings = np.mean(logit_parsings, axis=0)
                parsings_c = 1.0 / (1.0 + np.exp(-logit_parsings))
            else:
                raise NotImplementedError('Heuristic {} not supported'.format(cfg.TEST.PARSING_AUG.HEUR))
            result.add_field("parsing", parsings_c)
            result.add_field("parsing_scores", scores_c)

    return results


def im_detect_uv(model, rois, features):
    _idx = 0
    uv_results = [[[], [], [], []] for _ in range(len(rois))]
    conv_features = features[_idx][1]
    _idx += 1
    results = model.uv_net(conv_features, rois, targets=None)

    if cfg.TEST.BBOX_AUG.ENABLED and cfg.TEST.UV_AUG.ENABLED:
        if len(rois[0]) == 0:
            return results
        uvs = [result.get_field("uv") for result in results]
        add_uv_results(uv_results, uvs)
        if cfg.TEST.BBOX_AUG.H_FLIP:
            rois_hf = [roi.transpose(0) for roi in rois]
            features_hf = features[_idx][1]
            _idx += 1
            results_hf = model.uv_net(features_hf, rois_hf, targets=None)
            uvs_hf = [result_hf.get_field("uv") for result_hf in results_hf]
            uvs_hf = [flip_uv_featuremap(uv_hf) for uv_hf in uvs_hf]
            add_uv_results(uv_results, uvs_hf)

        for scale in cfg.TEST.BBOX_AUG.SCALES:
            rois_scl = [roi.resize(size) for roi, size in zip(rois, features[_idx][0])]
            features_scl = features[_idx][1]
            _idx += 1
            results_scl = model.uv_net(features_scl, rois_scl, targets=None)
            uvs_scl = [result_scl.get_field("uv") for result_scl in results_scl]
            add_uv_results(uv_results, uvs_scl)

            if cfg.TEST.BBOX_AUG.H_FLIP:
                rois_scl_hf = [roi.resize(size) for roi, size in zip(rois, features[_idx][0])]
                rois_scl_hf = [roi.transpose(0) for roi in rois_scl_hf]
                features_scl_hf = features[_idx][1]
                _idx += 1
                results_scl_hf = model.uv_net(features_scl_hf, rois_scl_hf, targets=None)
                uvs_scl_hf = [result_scl_hf.get_field("uv") for result_scl_hf in results_scl_hf]
                uvs_scl_hf = [flip_uv_featuremap(uv_scl_hf) for uv_scl_hf in uvs_scl_hf]
                add_uv_results(uv_results, uvs_scl_hf)

        uvs_c = []
        for uvs_ts, result in zip(uv_results, results):
            # Combine the predicted soft uvs
            if cfg.TEST.UV_AUG.HEUR == 'SOFT_AVG':
                for i in range(4):
                    uvs_c.append(np.mean(uvs_ts[i], axis=0))
            elif cfg.TEST.UV_AUG.HEUR == 'SOFT_MAX':
                for i in range(4):
                    uvs_c.append(np.amax(uvs_ts[i], axis=0))
            else:
                raise NotImplementedError('Heuristic {} not supported'.format(cfg.TEST.UV_AUG.HEUR))
            result.add_field("uv", uvs_c)

    return results


def im_detect_bbox_net(model, ims, target_scale, target_max_size, flip=False, size=None):
    net_imgs_size = []
    results = []
    ims_blob = get_blob(ims, target_scale, target_max_size, flip)
    blob_conv, _results, semseg_pred = model.box_net(ims_blob)

    if cfg.MODEL.SEMSEG_ON:
        semseg_pred = semseg_pred.cpu().numpy()
        if flip:
            semseg_pred = flip_parsing_featuremap(semseg_pred)
        semseg_pred = semseg_pred.transpose((0, 2, 3, 1))
        im_h, im_w = ims[0].shape[0:2]
        semseg_pred_resized = [cv2.resize(pred, (im_w, im_h), interpolation=cv2.INTER_LINEAR) for pred in semseg_pred]
    else:
        semseg_pred_resized = None

    for i, im_result in enumerate(_results):
        net_img_size = im_result.size
        net_imgs_size.append(net_img_size)
        if flip:
            im_result = im_result.transpose(0)
            if len(cfg.TRAIN.LEFT_RIGHT) > 0:
                scores = im_result.get_field("scores").reshape(-1, cfg.MODEL.NUM_CLASSES)
                boxes = im_result.bbox.reshape(-1, cfg.MODEL.NUM_CLASSES, 4)
                idx = torch.arange(cfg.MODEL.NUM_CLASSES)
                for j in cfg.TRAIN.LEFT_RIGHT:
                    idx[j[0]] = j[1]
                    idx[j[1]] = j[0]
                boxes = boxes[:, idx].reshape(-1, 4)
                scores = scores[:, idx].reshape(-1)
                im_result.bbox = boxes
                im_result.add_field("scores", scores)
        if size:
            im_result = im_result.resize(size[i])
        results.append(im_result)

    return results, net_imgs_size, blob_conv, semseg_pred_resized


def add_results(all_results, results):
    for i in range(len(all_results)):
        all_results[i].append(results[i])


def add_uv_results(all_results, results):
    for i in range(len(all_results)):
        for j in range(4):
            all_results[i][j].append(results[i][j])


def get_blob(ims, target_scale, target_max_size, flip):
    ims_processed = []
    for im in ims:
        if flip:
            im = im[:, ::-1, :]
        im = im.astype(np.float32, copy=False)
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        im_scale = float(target_scale) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > target_max_size:
            im_scale = float(target_max_size) / float(im_size_max)
        im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_processed = im_resized.transpose(2, 0, 1)
        im_processed = torch.from_numpy(im_processed).to(torch.device(cfg.DEVICE))

        ims_processed.append(im_processed)

    return ims_processed


def filter_results(boxlist):
    num_classes = cfg.MODEL.NUM_CLASSES
    if not cfg.TEST.SOFT_NMS.ENABLED and not cfg.TEST.BBOX_VOTE.ENABLED:
        # multiclass nms
        scores = boxlist.get_field("scores")
        device = scores.device
        num_repeat = int(boxlist.bbox.shape[0] / num_classes)
        labels = np.tile(np.arange(num_classes), num_repeat)
        boxlist.add_field("labels", torch.from_numpy(labels).to(dtype=torch.int64, device=device))
        fg_labels = torch.from_numpy(
            (np.arange(boxlist.bbox.shape[0]) % num_classes != 0).astype(int)
        ).to(dtype=torch.uint8, device=device)
        _scores = scores > cfg.FAST_RCNN.SCORE_THRESH
        inds_all = _scores & fg_labels
        result = boxlist_ml_nms(boxlist[inds_all], cfg.FAST_RCNN.NMS)
    else:
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > cfg.FAST_RCNN.SCORE_THRESH
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4: (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class_old = boxlist_for_class
            if cfg.TEST.SOFT_NMS.ENABLED:
                boxlist_for_class = boxlist_soft_nms(
                    boxlist_for_class,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=cfg.FAST_RCNN.NMS,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
            else:
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, cfg.FAST_RCNN.NMS
                )
            # Refine the post-NMS boxes using bounding-box voting
            if cfg.TEST.BBOX_VOTE.ENABLED and boxes_j.shape[0] > 0:
                boxlist_for_class = boxlist_box_voting(
                    boxlist_for_class,
                    boxlist_for_class_old,
                    cfg.TEST.BBOX_VOTE.VOTE_TH,
                    scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)

    number_of_detections = len(result)

    # Limit to max_per_image detections **over all classes**
    if number_of_detections > cfg.FAST_RCNN.DETECTIONS_PER_IMG > 0:
        cls_scores = result.get_field("scores")
        image_thresh, _ = torch.kthvalue(
            cls_scores.cpu(), number_of_detections - cfg.FAST_RCNN.DETECTIONS_PER_IMG + 1
        )
        keep = cls_scores >= image_thresh.item()
        keep = torch.nonzero(keep).squeeze(1)
        result = result[keep]
    return result


def flip_keypoint(heatmaps):
    """Flip heatmaps horizontally."""
    keypoints, flip_map = get_keypoints()
    heatmaps_flipped = heatmaps.copy()
    for lkp, rkp in flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        heatmaps_flipped[:, rid, :, :] = heatmaps[:, lid, :, :]
        heatmaps_flipped[:, lid, :, :] = heatmaps[:, rid, :, :]
    heatmaps_flipped = heatmaps_flipped[:, :, :, ::-1]
    return heatmaps_flipped


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map
