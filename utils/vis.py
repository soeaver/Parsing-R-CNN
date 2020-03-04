import cv2
import numpy as np
import os
import pycocotools.mask as mask_util
from collections import defaultdict
import matplotlib.pyplot as plt

from utils.timer import Timer
import utils.colormap as colormap_utils
from rcnn.core.config import cfg

_GRAY = [218, 227, 218]
_GREEN = [18, 127, 15]
_WHITE = [255, 255, 255]


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


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


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_bbox(img, bbox, bbox_color):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), bbox_color, thickness=cfg.VIS.SHOW_BOX.BORDER_THICK)

    return img


def vis_class(img, pos, class_str, bg_color):
    """Visualizes the class."""
    font_color = cfg.VIS.SHOW_CLASS.COLOR
    font_scale = cfg.VIS.SHOW_CLASS.FONT_SCALE

    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, bg_color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, font_color, lineType=cv2.LINE_AA)

    return img


def vis_mask(img, mask, bbox_color, show_parss=False):
    """Visualizes a single binary mask."""
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    border_color = cfg.VIS.SHOW_SEGMS.BORDER_COLOR
    border_thick = cfg.VIS.SHOW_SEGMS.BORDER_THICK

    mask_color = bbox_color if cfg.VIS.SHOW_SEGMS.MASK_COLOR_FOLLOW_BOX else _WHITE
    mask_color = np.asarray(mask_color)
    mask_alpha = cfg.VIS.SHOW_SEGMS.MASK_ALPHA

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if cfg.VIS.SHOW_SEGMS.SHOW_BORDER:
        cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)

    if cfg.VIS.SHOW_SEGMS.SHOW_MASK and not show_parss:
        img[idx[0], idx[1], :] *= 1.0 - mask_alpha
        img[idx[0], idx[1], :] += mask_alpha * mask_color

    return img.astype(np.uint8)


def vis_keypoints(img, kps, show_parss=False):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints, _ = get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    if show_parss:
        colors = [cfg.VIS.SHOW_KPS.KPS_COLOR_WITH_PARSING for c in colors]
    else:
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (kps[:2, dataset_keypoints.index('right_shoulder')] +
                    kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (kps[:2, dataset_keypoints.index('right_hip')] +
               kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > cfg.VIS.SHOW_KPS.KPS_TH and kps[2, nose_idx] > cfg.VIS.SHOW_KPS.KPS_TH:
        cv2.line(kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]), color=colors[len(kp_lines)],
                 thickness=cfg.VIS.SHOW_KPS.LINK_THICK, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > cfg.VIS.SHOW_KPS.KPS_TH and sc_mid_hip > cfg.VIS.SHOW_KPS.KPS_TH:
        cv2.line(kp_mask, tuple(mid_shoulder), tuple(mid_hip), color=colors[len(kp_lines) + 1],
                 thickness=cfg.VIS.SHOW_KPS.LINK_THICK, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > cfg.VIS.SHOW_KPS.KPS_TH and kps[2, i2] > cfg.VIS.SHOW_KPS.KPS_TH:
            cv2.line(kp_mask, p1, p2, color=colors[l],
                     thickness=cfg.VIS.SHOW_KPS.LINK_THICK, lineType=cv2.LINE_AA)
        if kps[2, i1] > cfg.VIS.SHOW_KPS.KPS_TH:
            cv2.circle(kp_mask, p1, radius=cfg.VIS.SHOW_KPS.CIRCLE_RADIUS, color=colors[l],
                       thickness=cfg.VIS.SHOW_KPS.CIRCLE_THICK, lineType=cv2.LINE_AA)
        if kps[2, i2] > cfg.VIS.SHOW_KPS.KPS_TH:
            cv2.circle(kp_mask, p2, radius=cfg.VIS.SHOW_KPS.CIRCLE_RADIUS, color=colors[l],
                       thickness=cfg.VIS.SHOW_KPS.CIRCLE_THICK, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - cfg.VIS.SHOW_KPS.KPS_ALPHA, kp_mask, cfg.VIS.SHOW_KPS.KPS_ALPHA, 0)


def vis_parsing(img, parsing, colormap, show_segms=True):
    """Visualizes a single binary parsing."""
    img = img.astype(np.float32)
    idx = np.nonzero(parsing)

    parsing_alpha = cfg.VIS.SHOW_PARSS.PARSING_ALPHA
    colormap = colormap_utils.dict2array(colormap)
    parsing_color = colormap[parsing.astype(np.int)]

    border_color = cfg.VIS.SHOW_PARSS.BORDER_COLOR
    border_thick = cfg.VIS.SHOW_PARSS.BORDER_THICK

    img[idx[0], idx[1], :] *= 1.0 - parsing_alpha
    # img[idx[0], idx[1], :] += alpha * parsing_color
    img += parsing_alpha * parsing_color

    if cfg.VIS.SHOW_PARSS.SHOW_BORDER and not show_segms:
        _, contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def vis_uv_temp(img, uv, bbox, show_segms=True):
    """Visualizes a single binary parsing."""
    padded_uv = np.zeros((img.shape), dtype=np.float32)
    uv_temp = np.array([uv[0], uv[1] * 256, uv[2] * 256]).transpose(1, 2, 0)
    y2 = int(bbox[1]) + int(bbox[3] - bbox[1])
    x2 = int(bbox[0]) + int(bbox[2] - bbox[0])
    padded_uv[int(bbox[1]):y2, int(bbox[0]):x2] = uv_temp

    img = img.astype(np.float32)
    idx = np.nonzero(padded_uv[:, :, 0])

    uv_alpha = cfg.VIS.SHOW_UV.UV_ALPHA

    border_color = cfg.VIS.SHOW_UV.BORDER_COLOR
    border_thick = cfg.VIS.SHOW_UV.BORDER_THICK

    img[idx[0], idx[1], :] *= 1.0 - uv_alpha
    img += uv_alpha * padded_uv

    if cfg.VIS.SHOW_UV.SHOW_BORDER and not show_segms:
        _, contours, _ = cv2.findContours(
            padded_uv[:, :, 0].astype(np.uint8).copy(),
            cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def vis_uv(img, uv, bbox):
    border_thick = cfg.VIS.SHOW_UV.BORDER_THICK
    grid_thick = cfg.VIS.SHOW_UV.GRID_THICK
    lines_num = cfg.VIS.SHOW_UV.LINES_NUM

    uv = np.transpose(uv, (1, 2, 0))
    uv = cv2.resize(uv, (int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1)), interpolation=cv2.INTER_LINEAR)
    roi_img = img[int(bbox[1]):int(bbox[3] + 1), int(bbox[0]):int(bbox[2] + 1), :]

    roi_img_resize = cv2.resize(roi_img, (2 * roi_img.shape[1], 2 * roi_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    I = uv[:, :, 0]
    for i in range(1, 25):
        if (len(I[I == i]) == 0):
            continue

        u = np.zeros_like(I)
        v = np.zeros_like(I)
        u[I == i] = uv[:, :, 1][I == i]
        v[I == i] = uv[:, :, 2][I == i]

        for ind in range(1, lines_num):
            thred = 1.0 * ind / lines_num
            _, thresh = cv2.threshold(u, u.min() + thred * (u.max() - u.min()), 255, 0)
            dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)
            dist_transform = np.uint8(dist_transform)

            _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours = [(col * 2) for col in contours]
            cv2.drawContours(roi_img_resize, contours, -1, ((1 - thred) * 255, thred * 255, thred * 200), grid_thick)

            _, thresh = cv2.threshold(v, v.min() + thred * (v.max() - v.min()), 255, 0)
            dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)
            dist_transform = np.uint8(dist_transform)

            _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours = [(col * 2) for col in contours]
            cv2.drawContours(roi_img_resize, contours, -1, (thred * 255, (1 - thred) * 255, thred * 200), grid_thick)

    _, thresh = cv2.threshold(I, 0.5, 255, 0)
    dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)
    dist_transform = np.uint8(dist_transform)
    _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = [(col * 2) for col in contours]
    cv2.drawContours(roi_img_resize, contours, -1, (70, 150, 0), border_thick)

    roi_img[:] = cv2.resize(roi_img_resize, (roi_img.shape[1], roi_img.shape[0]), interpolation=cv2.INTER_LINEAR)[:]

    return img


def get_instance_parsing_colormap(rgb=False):
    instance_colormap = eval('colormap_utils.{}'.format(cfg.VIS.SHOW_BOX.COLORMAP))
    parsing_colormap = eval('colormap_utils.{}'.format(cfg.VIS.SHOW_PARSS.COLORMAP))
    if rgb:
        instance_colormap = colormap_utils.dict_bgr2rgb(instance_colormap)
        parsing_colormap = colormap_utils.dict_bgr2rgb(parsing_colormap)

    return instance_colormap, parsing_colormap


def vis_one_image_opencv(im, config, boxes, classes, segms=None, keypoints=None, parsing=None, uv=None, dataset=None):
    """Constructs a numpy array with the detections visualized."""
    timers = defaultdict(Timer)
    timers['bbox_prproc'].tic()

    global cfg
    cfg = config

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < cfg.VIS.VIS_TH:
        return im

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    # get color map
    ins_colormap, parss_colormap = get_instance_parsing_colormap()

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    timers['bbox_prproc'].toc()

    instance_id = 1
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < cfg.VIS.VIS_TH:
            continue

        # get instance color (box, class_bg)
        if cfg.VIS.SHOW_BOX.COLOR_SCHEME == 'category':
            ins_color = ins_colormap[classes[i]]
        elif cfg.VIS.SHOW_BOX.COLOR_SCHEME == 'instance':
            instance_id = instance_id % len(ins_colormap.keys())
            ins_color = ins_colormap[instance_id]
        else:
            ins_color = _GREEN
        instance_id += 1

        # show box (off by default)
        if cfg.VIS.SHOW_BOX.ENABLED:
            timers['show_box'].tic()
            im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), ins_color)
            timers['show_box'].toc()

        # show class (off by default)
        if cfg.VIS.SHOW_CLASS.ENABLED:
            timers['show_class'].tic()
            class_str = get_class_string(classes[i], score, dataset)
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str, ins_color)
            timers['show_class'].toc()

        show_segms = True if cfg.VIS.SHOW_SEGMS.ENABLED and segms is not None and len(segms) > i else False
        show_kpts = True if cfg.VIS.SHOW_KPS.ENABLED and keypoints is not None and len(keypoints) > i else False
        show_parss = True if cfg.VIS.SHOW_PARSS.ENABLED and parsing is not None and len(parsing) > i else False
        show_uv = True if cfg.VIS.SHOW_UV.ENABLED and uv is not None and len(uv) > i else False
        # show mask
        if show_segms:
            timers['show_segms'].tic()
            color_list = colormap_utils.colormap()
            im = vis_mask(im, masks[..., i], ins_color, show_parss=show_parss)
            timers['show_segms'].toc()

        # show keypoints
        if show_kpts:
            timers['show_kpts'].tic()
            im = vis_keypoints(im, keypoints[i], show_parss=show_parss)
            timers['show_kpts'].toc()

        # show parsing
        if show_parss:
            timers['show_parss'].tic()
            im = vis_parsing(im, parsing[i], parss_colormap, show_segms=show_segms)
            timers['show_parss'].toc()

        # show uv
        if show_uv:
            timers['show_uv'].tic()
            im = vis_uv(im, uv[i], bbox)
            timers['show_uv'].toc()

    # for k, v in timers.items():
    #     print(' | {}: {:.3f}s'.format(k, v.total_time))

    return im
