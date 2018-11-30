from pycocotools.coco import COCO

import os
import cv2
import glob
import json
import copy
import numpy as np
from tqdm import trange, tqdm

import warnings
warnings.filterwarnings("ignore")

from parsingrcnn.core.config import cfg
import parsingrcnn.datasets.dataset_catalog as dataset_catalog


def get_parsing():
    name = cfg.TEST.DATASETS[0]
    _json_path = dataset_catalog.get_ann_fn(name)
    with open(_json_path, 'r') as f:
        _json = json.load(f)
    parsing_name = _json['categories'][0]['parsing']
  
    return parsing_name


def get_colormap():
    LIP_colormap = np.array([[  0,   0,   0], [  0,   0, 128],
                             [  0,   0, 255], [  0,  85,   0],
                             [ 51,   0, 170], [  0,  85, 255],
                             [ 85,   0,   0], [221, 119,   0],
                             [  0,  85,  85], [ 85,  85,   0],
                             [  0,  51,  85], [128,  86,  52],
                             [  0, 128,   0], [255,   0,   0],
                             [221, 170,  51], [255, 255,   0],
                             [170, 255,  85], [ 85, 255, 170],
                             [  0, 255, 255], [  0, 170, 255]
                             ])

    MHP_colormap = np.array([[  0,   0,   0], [196, 196, 225],
                             [ 32,  32,  63], [  0,   0, 253],
                             [  0,  27, 253], [  0,  55, 253],
                             [  0,  83, 253], [  0, 110, 253],
                             [  0, 138, 253], [  0, 165, 253],
                             [  0, 192, 253], [  0, 220, 254],
                             [  0, 248, 254], [  0, 255, 237],
                             [  0, 255, 209], [  0, 255, 183],
                             [  0, 255, 157], [  0, 255, 131],
                             [  0, 255, 104], [  0, 255,  80],
                             [  0, 255,  57], [  0, 255,  39],
                             [  8, 255,  32], [ 36, 255,  33],
                             [ 64, 255,  33], [ 91, 255,  33],
                             [118, 255,  34], [146, 255,  35],
                             [173, 255,  36], [200, 255,  37],
                             [228, 255,  39], [255, 255,  40],
                             [255, 228,  37], [255, 199,  33],
                             [255, 171,  30], [255, 144,  27],
                             [255, 114,  25], [255,  85,  23],
                             [255,  54,  21], [255,  10,  20],
                             [255,   0,  20], [255,   0,  30],
                             [255,   0,  51], [255,   0,  76],
                             [255,   0, 102], [255,   0, 129],
                             [255,   0, 155], [255,   0, 181],
                             [255,   0, 208], [255,   0, 236],
                             [246,   0, 253], [219,   0, 253],
                             [191,   0, 253], [164,   0, 253],
                             [137,   0, 253], [109,   0, 253],
                             [ 82,   0, 253], [ 55,   0, 253],
                             [ 27,   0, 253]
                             ])

    name = cfg.TRAIN.DATASETS[0]
    if 'LIP' in name:
        _colormap = LIP_colormap
    elif 'MHP' in name:
        _colormap = MHP_colormap
    else:
        assert False, \
        'This dataset {} has not be supported now'.format(name)

    return _colormap


def label_to_bbox(mask):
    """Compute the tight bounding box of a binary mask."""
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]
    return np.array((x0, y0, x1, y1), dtype=np.float32)


def parsing_to_boxes(parsing_gt, flipped):
    label_boxes = []
    for i in range(len(parsing_gt)):
        _label = cv2.imread(parsing_gt[i], 0)
        if flipped:
            _label = _label[:, ::-1]
        label_boxes.append(label_to_bbox(_label).copy())

    return np.array(label_boxes, dtype=np.float32)


def flip_left2right(parsing):
    l_r = cfg.PRCNN.LEFT_RIGHT
    for i in l_r:
        left = np.where(parsing == i[0])[0]
        right = np.where(parsing == i[1])[0]
        parsing[left] = i[1]
        parsing[right] = i[0]
    
    return parsing


def flip_left2right_featuremap(parsing):
    l_r = cfg.PRCNN.LEFT_RIGHT
    index = np.arange(cfg.PRCNN.NUM_PARSING)
    for i in l_r:
        index[i[0]] = i[1]
        index[i[1]] = i[0]

    parsing = parsing[:, :, :, index]
    
    return parsing


def parsing_wrt_box(parsing_gt, box, M, flipped):
    _label = cv2.imread(parsing_gt, 0)
    if flipped:
        _label = _label[:, ::-1]
    parsing = _label[int(box[1]):int(box[3]) + 1, int(box[0]):int(box[2]) + 1]
    parsing = cv2.resize(parsing, (M, M), interpolation=cv2.INTER_NEAREST)

    parsing = parsing.flatten()

    if flipped:
        parsing = flip_left2right(parsing)

    return np.array(parsing, dtype=np.int32)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def compute_hist(predict_list):
    if cfg.MODEL.PARSING_ON:
        n_class = cfg.PRCNN.NUM_PARSING
    hist = np.zeros((n_class, n_class))
    hist_s = np.zeros((n_class, n_class))
    hist_m = np.zeros((n_class, n_class))
    hist_l = np.zeros((n_class, n_class))

    name = cfg.TEST.DATASETS[0]
    gt_root = dataset_catalog.get_fseg_dir(name)

    for predict_png in tqdm(predict_list, desc='Calculating IoU ..'):
        gt_png = os.path.join(gt_root, predict_png.split('/')[-1])

        label = cv2.imread(gt_png, 0)        
        tmp = cv2.imread(predict_png, 0)
        label_s = label
        label_m = label
        label_l = label 

        assert label.shape == tmp.shape, '{} VS {}'.format(str(label.shape), str(tmp.shape))

        im_shape = label.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        im_scale = float(cfg.TEST.SCALE) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        for i in range(1, n_class):
            mask = np.where(label == i, 1, 0).astype(np.uint8).copy()
            _eare = np.sum(mask)
            _eare = _eare * im_scale ** 2
            
            if _eare >= 0 ** 2 and _eare < 32 ** 2:
                label_m = np.where(mask > 0, 255, label_m)
                label_l = np.where(mask > 0, 255, label_l)
            elif _eare >= 32 ** 2 and _eare < 96 ** 2:
                label_s = np.where(mask > 0, 255, label_s)
                label_l = np.where(mask > 0, 255, label_l)
            elif _eare >= 96 ** 2 and _eare < 1e5 ** 2:
                label_s = np.where(mask > 0, 255, label_s)
                label_m = np.where(mask > 0, 255, label_m)


        gt = label.flatten()
        gt_s = label_s.flatten()
        gt_m = label_m.flatten()
        gt_l = label_l.flatten()
        pre = tmp.flatten()
        
        hist += fast_hist(gt, pre, n_class)
        hist_s += fast_hist(gt_s, pre, n_class)
        hist_m += fast_hist(gt_m, pre, n_class)
        hist_l += fast_hist(gt_l, pre, n_class)

    # return hist[1:, 1:]
    return hist, hist_s, hist_m, hist_l


def mean_IoU(overall_h):
    iu = np.diag(overall_h) / (overall_h.sum(1) + overall_h.sum(0) - np.diag(overall_h))
    return iu, np.nanmean(iu)


def per_class_acc(overall_h):
    acc = np.diag(overall_h) / overall_h.sum(1)
    return np.nanmean(acc)


def pixel_wise_acc(overall_h):
    return np.diag(overall_h).sum() / overall_h.sum()


def parsing_iou(predict_root):
    predict_list = glob.glob(predict_root + '/*.png')
    print('The predict size: {}'.format(len(predict_list)))

    hist, hist_s, hist_m, hist_l = compute_hist(predict_list)
    _iou, _miou = mean_IoU(hist)
    _iou_s, _miou_s = mean_IoU(hist_s)
    _iou_m, _miou_m = mean_IoU(hist_m)
    _iou_l, _miou_l = mean_IoU(hist_l)

    return _iou, _miou, _miou_s, _miou_m, _miou_l


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def cal_one_mean_iou(image_array, label_array, NUM_CLASSES):
    hist = fast_hist(label_array, image_array, NUM_CLASSES).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu


def get_gt():
    assert len(cfg.TEST.DATASETS) == 1, \
        'Parsing only support one dataset now'
    name = cfg.TEST.DATASETS[0]
    _json_path = dataset_catalog.get_ann_fn(name)
    parsing_directory = dataset_catalog.get_parsing_dir(name)

    class_recs = []
    npos = 0
    parsing_COCO = COCO(_json_path)
    image_ids = parsing_COCO.getImgIds()
    image_ids.sort()

    for image_id in image_ids:
        # imagename = parsing_COCO.loadImgs(image_id)[0]['file_name']
        ann_ids = parsing_COCO.getAnnIds(imgIds=image_id, iscrowd=None)
        objs = parsing_COCO.loadAnns(ann_ids)
        # gt_box = []
        anno_adds = []
        for obj in objs:
            # gt_box.append(obj['bbox'])
            parsing_path = os.path.join(parsing_directory, obj['parsing'])
            anno_adds.append(parsing_path)
            npos = npos + 1

        det = [False] * len(anno_adds)
        # class_recs.append({'gt_box': np.array(gt_box),
        #                    'anno_adds': anno_adds, 
        #                    'det': det})
        class_recs.append({'anno_adds': anno_adds, 
                           'det': det})
  
    return class_recs, npos


def eval_seg_ap(all_boxes, all_parsings):
    '''
    From_pkl: load results from pickle files 
    Sparse: Indicate that the masks in the results are sparse matrices
    '''
    nb_class = cfg.PRCNN.NUM_PARSING
    ovthresh_seg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    confidence = []
    image_ids  = []
    Local_segs_ptr = []

    for img_index, parsings in enumerate(all_parsings):
        for idx, rect in enumerate(parsings):
            score = all_boxes[img_index][idx][4]
            image_ids.append(img_index)
            confidence.append(score)
            Local_segs_ptr.append(idx)

    confidence = np.array(confidence)
    Local_segs_ptr = np.array(Local_segs_ptr)

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    Local_segs_ptr = Local_segs_ptr[sorted_ind]
    image_ids =  [image_ids[x]  for x in sorted_ind]

    class_recs_temp, npos = get_gt()
    class_recs = [copy.deepcopy(class_recs_temp) for _ in range(len(ovthresh_seg))]
    nd = len(image_ids)
    tp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    fp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    pcp_list= [[] for _ in range(len(ovthresh_seg))]

    for d in trange(nd, desc='Calculating AP and PCP ..'):
        R = []
        for j in range(len(ovthresh_seg)):
            R.append(class_recs[j][image_ids[d]])
        ovmax = -np.inf
        jmax = -1

        parsings = all_parsings[image_ids[d]]
        mask0 = parsings[Local_segs_ptr[d]]

        mask_pred = mask0.astype(np.int)

        for i in range(len(R[0]['anno_adds'])):
            mask_gt = cv2.imread(R[0]['anno_adds'][i], 0)

            seg_iou= cal_one_mean_iou(mask_pred.astype(np.uint8), mask_gt, nb_class)

            mean_seg_iou = np.nanmean(seg_iou)
            if mean_seg_iou > ovmax:
                ovmax =  mean_seg_iou
                seg_iou_max = seg_iou 
                jmax = i
                mask_gt_u = np.unique(mask_gt)

        for j in range(len(ovthresh_seg)):   
            if ovmax > ovthresh_seg[j]:
                if not R[j]['det'][jmax]:
                    tp_seg[j][d] = 1.
                    R[j]['det'][jmax] = 1
                    pcp_d = len(mask_gt_u[np.logical_and(mask_gt_u>0, mask_gt_u<nb_class)])
                    pcp_n = float(np.sum(seg_iou_max[1:]>ovthresh_seg[j]))
                    if pcp_d > 0:
                        pcp_list[j].append(pcp_n/pcp_d)
                    else:
                        pcp_list[j].append(0.0)
                else:
                    fp_seg[j][d] =  1.
            else:
                fp_seg[j][d] =  1.

    # compute precision recall
    all_ap_seg = []
    all_pcp = []
    for j in range(len(ovthresh_seg)):
        fp_seg[j] = np.cumsum(fp_seg[j])
        tp_seg[j] = np.cumsum(tp_seg[j])
        rec_seg = tp_seg[j] / float(npos)
        prec_seg = tp_seg[j] / np.maximum(tp_seg[j] + fp_seg[j], np.finfo(np.float64).eps)

        ap_seg = voc_ap(rec_seg, prec_seg)
        all_ap_seg.append(ap_seg)

        assert(np.max(tp_seg[j]) == len(pcp_list[j])), "%d vs %d"%(np.max(tp_seg[j]),len(pcp_list[j]))
        pcp_list[j].extend([0.0]*(npos - len(pcp_list[j])))
        pcp = np.mean(pcp_list[j])
        all_pcp.append(pcp)

    return all_ap_seg, all_pcp


def parsing2png(cls_boxes_i, cls_parss_i, output_dir, img_name, img_shape):           
    parsing_output_dir = os.path.join(output_dir, 'parsing_predict')
    if not os.path.exists(parsing_output_dir):
        os.makedirs(parsing_output_dir)
    parsing_ins_dir = os.path.join(output_dir, 'parsing_instance')
    if not os.path.exists(parsing_ins_dir):
        os.makedirs(parsing_ins_dir)

    if cls_parss_i is not None:
        im_name = os.path.splitext(os.path.basename(img_name))[0]
        txt_result = im_name
        save_name = os.path.join(parsing_output_dir, '{}.png'.format(im_name))
        save_ins = os.path.join(parsing_ins_dir, '{}.png'.format(im_name))
        parsing_png = np.zeros(img_shape)
        parsing_ins = np.zeros(img_shape)
        parsings = []
        scores = []
        for i in range(len(cls_boxes_i[1])):
            scores.append(cls_boxes_i[1][i][4])
            parsings.append(cls_parss_i[1][i])

        _inx = np.argsort(np.array(scores))
        ins_id = 1
        for k in range(len(_inx)):
            if scores[_inx[k]] < cfg.PRCNN.SCORE:
                continue
            parsing = parsings[_inx[k]]
            txt_result += ' {} {}'.format(str(ins_id), str(scores[_inx[k]]))
            parsing_png = np.where(parsing > 0, parsing, parsing_png)
            parsing_ins = np.where(parsing > 0, ins_id, parsing_ins)
            ins_id += 1
        txt_result += '\n'
        cv2.imwrite(save_name, parsing_png)
        cv2.imwrite(save_ins, parsing_ins)
    else:
        return [], ''
    
    return parsings, txt_result
