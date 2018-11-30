import logging
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from parsingrcnn.core.config import cfg
import parsingrcnn.nn as mynn

logger = logging.getLogger(__name__)


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, beta=1.0):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    1 / N * sum_i alpha_out[i] * SmoothL1(alpha_in[i] * (y_hat[i] - y[i])).
    N is the number of batch elements in the input predictions
    """
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < beta).detach().float()
    in_loss_box = smoothL1_sign * 0.5 * torch.pow(in_box_diff, 2) / beta + \
                  (1 - smoothL1_sign) * (abs_in_box_diff - (0.5 * beta))
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    N = loss_box.size(0)  # batch size
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


def help_target(bbox_targets, locs_not_target):
    N, L, Four = bbox_targets.shape
    if locs_not_target:
        for i in range(1, N):
            bbox_targets[i][:, 0] = i
    bbox_targets = bbox_targets.reshape(N * L, Four)
    return bbox_targets


# ONE LOOP VERSION
def retinanet_smooth_l1_loss(bbox_pred, bbox_targets, fg_bbox_locs, fg_num, beta=1.0, scale=1.0):
    N, D, H, W = bbox_pred.shape
    fg_num = sum(fg_num)
    Y_hat = bbox_pred  # .to('cuda:0')
    L = help_target(fg_bbox_locs, True)
    Y = help_target(bbox_targets, False)
    S = fg_num

    M = Y.shape[0]
    out = torch.zeros(M).cuda('cuda:' + str(bbox_pred.get_device()))
    mask = (Y.sum(dim=1) != 0)

    for j in range(4):
        tmp_idx = L.long()  # n, c+j, y, x
        tmp_idx[:, 1] += j
        y_hat = Y_hat[tmp_idx[:, 0], tmp_idx[:, 1], tmp_idx[:, 2], tmp_idx[:, 3]]
        yy = Y[:, j]
        val = y_hat - yy
        abs_val = abs(val)
        out[abs_val < beta] += ((0.5 * val * val / beta) / max(S, 1.0))[abs_val < beta]
        out[abs_val >= beta] += ((abs_val - 0.5 * beta) / max(S, 1.0))[abs_val >= beta]
    out = out[mask]
    return out.sum()


# CURRENT GPU VERSION
def sigmoid_focal_loss(cls_pred, cls_targets, fg_num, alpha=0.25, num_classes=80, gamma=2.0, lvl=3, scale=1):
    # t0 = time.time()
    N, D, H, W = cls_pred.shape
    _, A, _, _ = cls_targets.shape
    C = num_classes
    cls = torch.transpose(cls_pred, 1, 3).reshape(N * H * W * A, -1)
    # print('cls', cls.shape)
    lbl = torch.transpose(cls_targets, 1, 3).reshape(-1)
    fg_num = sum(fg_num)

    t = one_hot_embedding(lbl.data.cpu().long(), 1 + C)
    t = t[:, 1:]  # exclude background #[NN, 80]
    t = Variable(t).cuda('cuda:' + str(lbl.get_device()))
    p = cls.sigmoid()
    # matched_t = (t > 0)
    pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
    # pt[~matched_t] = (1- p)[~matched_t]
    w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
    # w[~matched_t] = 1 - alpha
    w = w * (1 - pt).pow(gamma)
    # non_negative_logit = (cls >= 0)

    # loss = torch.zeros(cls.shape).cuda('cuda:'+str(cls.get_device()))
    # loss[(~matched_t) & non_negative_logit] = (-w * (-cls + torch.log(pt)))[(~matched_t) & non_negative_logit]
    # loss[(~matched_t) & (~non_negative_logit)] =  (-w * torch.log(pt))[(~matched_t) & (~non_negative_logit)]
    # loss[matched_t] = -w * torch.log(pt)[matched_t]
    # loss = (loss[lbl != -1]).sum()
    loss = ((-w * torch.log(pt))[lbl != -1]).sum()  # F.binary_cross_entropy_with_logits(cls, t, w, size_average=False)
    loss /= fg_num
    # print("Elasped", time.time() - t0)
    # loss = torch.FloatTensor(2).random_(1).cuda('cuda:'+str(cls_pred.get_device()))
    return loss


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]  # [N,D]


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def decay_learning_rate(optimizer, cur_lr, decay_rate):
    """Decay learning rate"""
    new_lr = cur_lr * decay_rate
    # ratio = _get_lr_change_ratio(cur_lr, new_lr)
    ratio = 1 / decay_rate
    if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
        logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
    # Update learning rate, note that different parameter may have different learning rate
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
        new_lr = decay_rate * param_group['lr']
        param_group['lr'] = new_lr
        if cfg.SOLVER.TYPE in ['SGD']:
            if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                            ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
                _CorrectMomentum(optimizer, param_group['params'], new_lr / cur_lr)


def update_learning_rate(optimizer, cur_lr, new_lr):
    """Update learning rate"""
    if cur_lr != new_lr:
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
            logger.info('Changing learning rate %.6f -> %.6f', cur_lr, new_lr)
        # Update learning rate, note that different parameter may have different learning rate
        param_keys = []
        for ind, param_group in enumerate(optimizer.param_groups):
            if ind == 1 and cfg.SOLVER.BIAS_DOUBLE_LR:  # bias params
                param_group['lr'] = new_lr * 2
            else:
                param_group['lr'] = new_lr
            param_keys += param_group['params']
        if cfg.SOLVER.TYPE in ['SGD'] and cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                        ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            _CorrectMomentum(optimizer, param_keys, new_lr / cur_lr)


def _CorrectMomentum(optimizer, param_keys, correction):
    """The MomentumSGDUpdate op implements the update V as

        V := mu * V + lr * grad,

    where mu is the momentum factor, lr is the learning rate, and grad is
    the stochastic gradient. Since V is not defined independently of the
    learning rate (as it should ideally be), when the learning rate is
    changed we should scale the update history V in order to make it
    compatible in scale with lr * grad.
    """
    logger.info('Scaling update history by %.6f (new lr / old lr)', correction)
    for p_key in param_keys:
        optimizer.state[p_key]['momentum_buffer'] *= correction


def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio


def affine_grid_gen(rois, input_size, grid_size):
    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([ \
        (x2 - x1) / (width - 1),
        zero,
        (x1 + x2 - width + 1) / (width - 1),
        zero,
        (y2 - y1) / (height - 1),
        (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid


def save_ckpt(output_dir, args, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_{}_{}.pth'.format(args.epoch, args.step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    # TODO: (maybe) Do not save redundant shared params
    # model_state_dict = model.state_dict()
    torch.save({
        'epoch': args.epoch,
        'step': args.step,
        'iters_per_epoch': args.iters_per_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


def load_ckpt(model, ckpt, mapping=False):
    """Load checkpoint"""
    if mapping:
        mapping, _ = model.detectron_weight_mapping
        state_dict = {}
        for name in ckpt:
            if mapping[name]:
                state_dict[name] = ckpt[name]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)


def get_group_gn(dim):
    """
    get number of groups used by GroupNorm, based on number of channels
    """
    dim_per_gp = cfg.GROUP_NORM.DIM_PER_GP
    num_groups = cfg.GROUP_NORM.NUM_GROUPS

    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0
        group_gn = num_groups
    return group_gn
