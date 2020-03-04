import torch
import torch.nn as nn

from utils.misc import logging_rank


class Optimizer(object):
    def __init__(self, model, solver, local_rank=0):
        self.model = model
        self.solver = solver
        self.local_rank = local_rank

        self.bias_params_list = []
        self.gn_params_list = []
        self.nonbias_params_list = []

        self.params = []
        self.gn_param_nameset = self.get_gn_param_nameset()

    def get_gn_param_nameset(self):
        gn_param_nameset = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.GroupNorm):
                gn_param_nameset.add(name + '.weight')
                gn_param_nameset.add(name + '.bias')
        return gn_param_nameset

    def get_params_list(self):
        for key, value in self.model.named_parameters():
            if value.requires_grad:
                if 'bias' in key:
                    self.bias_params_list.append(value)
                elif key in self.gn_param_nameset:
                    self.gn_params_list.append(value)
                else:
                    self.nonbias_params_list.append(value)
            else:
                logging_rank('{} does not need grad.'.format(key), local_rank=self.local_rank)

    def get_params(self):
        self.params += [
            {'params': self.nonbias_params_list,
             'lr': 0,
             'weight_decay': self.solver.WEIGHT_DECAY,
             'lr_scale': 1},
            {'params': self.bias_params_list,
             'lr': 0 * (self.solver.BIAS_DOUBLE_LR + 1),
             'weight_decay': self.solver.WEIGHT_DECAY if self.solver.BIAS_WEIGHT_DECAY else 0,
             'lr_scale': self.solver.BIAS_DOUBLE_LR + 1},
            {'params': self.gn_params_list,
             'lr': 0,
             'weight_decay': self.solver.WEIGHT_DECAY_GN * self.solver.WEIGHT_DECAY,
             'lr_scale': 1}
        ]

    def build(self):
        assert self.solver.OPTIMIZER in ['SGD', 'RMSPROP', 'ADAM']
        self.get_params_list()
        self.get_params()

        if self.solver.OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(
                self.params,
                momentum=self.solver.MOMENTUM
            )
        elif self.solver.OPTIMIZER == 'RMSPROP':
            optimizer = torch.optim.RMSprop(
                self.params,
                momentum=self.solver.MOMENTUM
            )
        elif self.solver.OPTIMIZER == 'ADAM':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.solver.BASE_LR
            )
        else:
            optimizer = None
        return optimizer
