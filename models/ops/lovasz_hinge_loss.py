import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszHinge(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszHinge, self).__init__()
        self.reduction = reduction

    def flatten(self, input, target, mask=None):
        if mask is None:
            input_flatten = input.view(-1)
            target_flatten = target.view(-1)
        else:
            input_flatten = input[mask].view(-1)
            target_flatten = target[mask].view(-1)
        return input_flatten, target_flatten

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    def forward(self, inputs, targets, mask=None, act=False):
        losses = []
        for id in range(len(inputs)):
            if mask is not None:
                input_flatten, target_flatten = self.flatten(inputs[id], targets[id], mask[id])
            else:
                input_flatten, target_flatten = self.flatten(inputs[id], targets[id])
            if act:
                # map [0, 1] to [-inf, inf]
                input_flatten = torch.log(input_flatten) - torch.log(1 - input_flatten)
            losses.append(self.lovasz_hinge_flat(input_flatten, target_flatten))
        losses = torch.stack(losses).to(device=inputs.device)
        if self.reduction == "mean":
            losses = losses.mean()
        elif self.reduction == "sum":
            losses = losses.sum()

        return losses
