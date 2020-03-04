import numpy as np
from bisect import bisect_right

from torch.optim.optimizer import Optimizer

from utils.misc import logging_rank


def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio


class LearningRateScheduler(object):
    """We re-implement the _LRScheduler class, and support warm up with three kinds of lr decay strategies.
    Pytorch official implementation can be found in:
    https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
    """
    def __init__(self, optimizer, solver, start_iter=1, iter_per_epoch=-1, local_rank=0):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.solver = solver
        assert self.solver.LR_POLICY in ['STEP', 'COSINE', 'STEP_COSINE', 'POLY']
        assert self.solver.WARM_UP_METHOD in ['CONSTANT', 'LINEAR']
        self.base_lr = self.solver.BASE_LR
        self.new_lr = self.base_lr

        self.iteration = start_iter
        self.iter_per_epoch = iter_per_epoch
        self.local_rank = local_rank

        self.info = dict(best_acc=0.0, best_epoch=1, cur_acc=0.0, cur_epoch=1)

        if 'MAX_ITER' in self.solver:
            self.max_iter = self.solver.MAX_ITER
            self.warm_up_iters = self.solver.WARM_UP_ITERS
            self.steps = self.solver.STEPS  # only useful for step policy
        else:
            assert self.iter_per_epoch > 0  # need to specify the iter_per_epoch
            self.conver_epoch2iter()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def conver_epoch2iter(self):
        """Convert the epoch style parameters to corresponding iteration.
        """
        self.max_iter = self.solver.MAX_EPOCHS * self.iter_per_epoch
        self.warm_up_iters = self.solver.WARM_UP_EPOCH * self.iter_per_epoch
        self.steps = [epoch * self.iter_per_epoch for epoch in self.solver.STEPS]  # only useful for step policy

    def get_lr(self):
        new_lr = self.base_lr
        if self.iteration <= self.warm_up_iters:  # warm up
            if self.solver.WARM_UP_METHOD == 'CONSTANT':
                warmup_factor = self.solver.WARM_UP_FACTOR
            elif self.solver.WARM_UP_METHOD == 'LINEAR':
                alpha = self.iteration / self.warm_up_iters
                warmup_factor = self.solver.WARM_UP_FACTOR * (1 - alpha) + alpha
            else:
                raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(self.solver.WARM_UP_METHOD))
            new_lr = self.base_lr * warmup_factor
        elif self.iteration > self.warm_up_iters:
            if self.solver.LR_POLICY == 'STEP':
                new_lr = self.base_lr * self.solver.GAMMA ** bisect_right(self.steps, self.iteration)
            elif self.solver.LR_POLICY == 'COSINE':
                actual_iter = self.max_iter - self.warm_up_iters  # except warm up
                new_lr = 0.5 * self.base_lr * (
                    np.cos((self.iteration - self.warm_up_iters - 1) * np.pi / actual_iter) + 1.0)
            elif self.solver.LR_POLICY == 'STEP_COSINE':
                if self.iteration < self.steps[-1]:
                    new_lr = self.base_lr * self.solver.GAMMA ** bisect_right(self.steps, self.iteration)
                else:
                    new_base_lr = self.base_lr * self.solver.GAMMA ** bisect_right(self.steps, self.steps[-1] - 1)
                    actual_iter = self.max_iter - self.steps[-1]  # except step schedule iterations
                    new_lr = 0.5 * new_base_lr * (
                            np.cos((self.iteration - self.steps[-1] - 1) * np.pi / actual_iter) + 1.0)
            elif self.solver.LR_POLICY == 'POLY':
                actual_iter = self.max_iter - self.warm_up_iters  # except warm up
                new_lr = self.base_lr * (
                    (1. - float(self.iteration - self.warm_up_iters - 1) / actual_iter) ** self.solver.LR_POW)
            else:
                raise KeyError('Unknown SOLVER.LR_POLICY: {}'.format(self.solver.LR_POLICY))
        return new_lr

    def update_learning_rate(self):
        """Update learning rate
        """
        cur_lr = self.optimizer.param_groups[0]['lr']
        if cur_lr != self.new_lr:
            ratio = _get_lr_change_ratio(cur_lr, self.new_lr)
            if ratio > self.solver.LOG_LR_CHANGE_THRESHOLD and self.new_lr >= 1e-7:
                logging_rank('Changing learning rate {:.6f} -> {:.6f}'.format(cur_lr, self.new_lr),
                             local_rank=self.local_rank)
            # Update learning rate, note that different parameter may have different learning rate
            for ind, param_group in enumerate(self.optimizer.param_groups):
                if 'lr_scale' in param_group:
                    lr_scale = param_group['lr_scale']
                else:
                    lr_scale = 1
                param_group['lr'] = self.new_lr * lr_scale

    def step(self, cur_iter=None):
        if cur_iter is None:
            cur_iter = self.iteration + 1
        self.iteration = cur_iter

        # update learning rate
        self.new_lr = self.get_lr()
        self.update_learning_rate()
