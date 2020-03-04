from pycocotools.coco import COCO

import cv2
import os
import glob
import json
import copy
import numpy as np
from tqdm import trange, tqdm

import warnings

warnings.filterwarnings("ignore")

from utils.misc import logging_rank


def parsing_png(parsing, scores, semseg_thresh, img_info, output_folder, semseg=None):
    parsing_output_dir = os.path.join(output_folder, 'parsing_predict')
    if not os.path.exists(parsing_output_dir):
        os.makedirs(parsing_output_dir)
    parsing_ins_dir = os.path.join(output_folder, 'parsing_instance')
    if not os.path.exists(parsing_ins_dir):
        os.makedirs(parsing_ins_dir)

    im_name = img_info['file_name']
    save_name = os.path.join(parsing_output_dir, im_name.replace('jpg', 'png'))
    save_ins = os.path.join(parsing_ins_dir, im_name.replace('jpg', 'png'))
    if semseg is not None:
        semseg = cv2.resize(semseg, (img_info["width"], img_info["height"]), interpolation=cv2.INTER_LINEAR)
        parsing_max = np.max(semseg, axis=2)
        max_map = np.where(parsing_max > 0.7, 1, 0)
        parsing_seg = np.argmax(semseg, axis=2).astype(np.uint8) * max_map
    else:
        parsing_seg = np.zeros((img_info["height"], img_info["width"]))
    parsing_ins = np.zeros((img_info["height"], img_info["width"]))

    _inx = scores.argsort()
    ins_id = 1
    for k in range(len(_inx)):
        if scores[_inx[k]] < semseg_thresh:
            continue
        _parsing = parsing[_inx[k]]
        parsing_seg = np.where(_parsing > 0, _parsing, parsing_seg)
        parsing_ins = np.where(_parsing > 0, ins_id, parsing_ins)
        ins_id += 1

    cv2.imwrite(save_name, parsing_seg.astype(np.uint8))
    cv2.imwrite(save_ins, parsing_ins.astype(np.uint8))


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def compute_hist(predict_list, im_dir, num_parsing):
    n_class = num_parsing
    hist = np.zeros((n_class, n_class))

    gt_root = im_dir.replace('img', 'seg')

    for predict_png in tqdm(predict_list, desc='Calculating IoU ..'):
        gt_png = os.path.join(gt_root, predict_png.split('/')[-1])

        label = cv2.imread(gt_png, 0)
        tmp = cv2.imread(predict_png, 0)

        assert label.shape == tmp.shape, '{} VS {}'.format(str(label.shape), str(tmp.shape))

        gt = label.flatten()
        pre = tmp.flatten()
        hist += fast_hist(gt, pre, n_class)

    return hist


def mean_IoU(overall_h):
    iu = np.diag(overall_h) / (overall_h.sum(1) + overall_h.sum(0) - np.diag(overall_h))
    return iu, np.nanmean(iu)


def per_class_acc(overall_h):
    acc = np.diag(overall_h) / overall_h.sum(1)
    return np.nanmean(acc)


def pixel_wise_acc(overall_h):
    return np.diag(overall_h).sum() / overall_h.sum()


def parsing_iou(predict_root, im_dir, num_parsing):
    predict_list = glob.glob(predict_root + '/*.png')
    logging_rank('The predict size: {}'.format(len(predict_list)))

    hist = compute_hist(predict_list, im_dir, num_parsing)
    _iou, _miou = mean_IoU(hist)
    mean_acc = per_class_acc(hist)
    pixel_acc = pixel_wise_acc(hist)

    return _iou, _miou, mean_acc, pixel_acc


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


def cal_one_mean_iou(image_array, label_array, num_parsing):
    hist = fast_hist(label_array, image_array, num_parsing).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu


def get_gt(im_dir, ann_fn):
    parsing_directory = im_dir.replace('img', 'parsing')

    class_recs = []
    npos = 0
    parsing_COCO = COCO(ann_fn)
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
        class_recs.append({'anno_adds': anno_adds, 'det': det})

    return class_recs, npos


def eval_parsing_ap(all_parsings, all_scores, score_thresh, im_dir, ann_fn, num_parsing):
    nb_class = num_parsing
    ovthresh_seg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    confidence = []
    image_ids = []
    Local_segs_ptr = []

    for img_index, parsings in enumerate(all_parsings):
        for idx, rect in enumerate(parsings):
            score = all_scores[img_index][idx]
            image_ids.append(img_index)
            confidence.append(score)
            Local_segs_ptr.append(idx)

    confidence = np.array(confidence)
    Local_segs_ptr = np.array(Local_segs_ptr)

    sorted_ind = np.argsort(-confidence)
    sorted_scores = confidence[sorted_ind]
    Local_segs_ptr = Local_segs_ptr[sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    class_recs_temp, npos = get_gt(im_dir, ann_fn)
    class_recs = [copy.deepcopy(class_recs_temp) for _ in range(len(ovthresh_seg))]
    nd = len(image_ids)
    tp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    fp_seg = [np.zeros(nd) for _ in range(len(ovthresh_seg))]
    pcp_list = [[] for _ in range(len(ovthresh_seg))]

    for d in trange(nd, desc='Calculating AP and PCP ..'):
        if sorted_scores[d] < score_thresh:
            continue
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

            seg_iou = cal_one_mean_iou(mask_pred.astype(np.uint8), mask_gt, nb_class)

            mean_seg_iou = np.nanmean(seg_iou)
            if mean_seg_iou > ovmax:
                ovmax = mean_seg_iou
                seg_iou_max = seg_iou
                jmax = i
                mask_gt_u = np.unique(mask_gt)

        for j in range(len(ovthresh_seg)):
            if ovmax > ovthresh_seg[j]:
                if not R[j]['det'][jmax]:
                    tp_seg[j][d] = 1.
                    R[j]['det'][jmax] = 1
                    pcp_d = len(mask_gt_u[np.logical_and(mask_gt_u > 0, mask_gt_u < nb_class)])
                    pcp_n = float(np.sum(seg_iou_max[1:] > ovthresh_seg[j]))
                    if pcp_d > 0:
                        pcp_list[j].append(pcp_n / pcp_d)
                    else:
                        pcp_list[j].append(0.0)
                else:
                    fp_seg[j][d] = 1.
            else:
                fp_seg[j][d] = 1.

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

        assert (np.max(tp_seg[j]) == len(pcp_list[j])), "%d vs %d" % (np.max(tp_seg[j]), len(pcp_list[j]))
        pcp_list[j].extend([0.0] * (npos - len(pcp_list[j])))
        pcp = np.mean(pcp_list[j])
        all_pcp.append(pcp)

    return all_ap_seg, all_pcp


def get_parsing(ann_fn):
    with open(ann_fn, 'r') as f:
        _json = json.load(f)
    parsing_name = _json['categories'][0]['parsing']

    return parsing_name


def evaluate_parsing(all_results, eval_ap, score_thresh, num_parsing, im_dir, ann_fn, output_folder):
    logging_rank('Evaluating parsing')
    predict_dir = os.path.join(output_folder, 'parsing_predict')
    assert os.path.exists(predict_dir), 'predict dir \'{}\' not found'.format(predict_dir)
    _iou, _miou, mean_acc, pixel_acc = parsing_iou(predict_dir, im_dir, num_parsing)

    parsing_result = {'mIoU': _miou, 'pixel_acc': pixel_acc, 'mean_acc': mean_acc}

    parsing_name = get_parsing(ann_fn)
    logging_rank('IoU for each category:')
    assert len(parsing_name) == len(_iou), '{} VS {}'.format(str(len(parsing_name)), str(len(_iou)))

    for i, iou in enumerate(_iou):
        print(' {:<30}:  {:.2f}'.format(parsing_name[i], 100 * iou))

    print('----------------------------------------')
    print(' {:<30}:  {:.2f}'.format('mean IoU', 100 * _miou))
    print(' {:<30}:  {:.2f}'.format('pixel acc', 100 * pixel_acc))
    print(' {:<30}:  {:.2f}'.format('mean acc', 100 * mean_acc))
    
    if eval_ap:
        all_ap_p, all_pcp = eval_parsing_ap(all_results[0], all_results[1], score_thresh, im_dir, ann_fn, num_parsing)
        ap_p_vol = np.mean(all_ap_p)

        print('~~~~ Summary metrics ~~~~')
        print(' Average Precision based on part (APp)               @[mIoU=0.10:0.90 ] = {:.3f}'.format(ap_p_vol))
        print(' Average Precision based on part (APp)               @[mIoU=0.10      ] = {:.3f}'.format(all_ap_p[0]))
        print(' Average Precision based on part (APp)               @[mIoU=0.30      ] = {:.3f}'.format(all_ap_p[2]))
        print(' Average Precision based on part (APp)               @[mIoU=0.50      ] = {:.3f}'.format(all_ap_p[4]))
        print(' Average Precision based on part (APp)               @[mIoU=0.70      ] = {:.3f}'.format(all_ap_p[6]))
        print(' Average Precision based on part (APp)               @[mIoU=0.90      ] = {:.3f}'.format(all_ap_p[8]))
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.50      ] = {:.3f}'.format(all_pcp[4]))
        parsing_result['APp50'] = all_ap_p[4]
        parsing_result['APpvol'] = ap_p_vol
        parsing_result['PCP'] = all_pcp[4]

    return parsing_result
