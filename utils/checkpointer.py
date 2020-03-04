import os
import shutil
from collections import OrderedDict

import torch
import torch.nn.init as init

import models.ops as ops
from utils.misc import logging_rank
from utils.net import mismatch_params_filter


def get_weights(ckpt_path, cfg_test_weights, mode='latest'):
    if os.path.exists(cfg_test_weights):
        weights = cfg_test_weights
    else:
        weights = os.path.join(ckpt_path, 'model_{}.pth'.format(mode))
    return weights


def load_weights(model, weights_path, local_rank=0):
    try:
        weights_dict = torch.load(weights_path, map_location=torch.device("cpu"))['model']
    except:
        weights_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    weights_dict = strip_prefix_if_present(weights_dict, prefix='module.')
    model_state_dict = model.state_dict()
    model_state_dict, mismatch_keys = align_and_update_state_dicts(model_state_dict, weights_dict, -1)
    model.load_state_dict(model_state_dict)
    logging_rank('The mismatch keys: {}.'.format(list(mismatch_params_filter(sorted(mismatch_keys)))),
                 local_rank=local_rank)
    logging_rank('Loading from weights: {}.'.format(weights_path), local_rank=local_rank)


class CheckPointer(object):
    def __init__(self, ckpt, weights_path=None, auto_resume=True, local_rank=0):
        self.ckpt = ckpt
        self.weights_path = weights_path
        self.auto_resume = auto_resume
        self.local_rank = local_rank

        self.mismatch_keys = set()
        self.resume = self.get_model_latest()
        if self.weights_path:
            self.checkpoint = self._load_file()

    def get_model_latest(self):
        model_latest = os.path.join(self.ckpt, 'model_latest.pth') \
            if os.path.exists(os.path.join(self.ckpt, 'model_latest.pth')) else ''
        if self.auto_resume and model_latest:
            self.weights_path = model_latest
            return True
        else:
            return False

    def _load_file(self):
        return torch.load(self.weights_path, map_location=torch.device("cpu"))

    def weight_mapping(self, weights_dict):
        """Support caffe trained vgg16 model"""
        if 'vgg16_reducedfc' in self.weights_path:
            mapping = VGG16_NAME_MAPPING
        else:
            mapping = None

        if mapping:
            weights_dict_new = {}
            for old_name in weights_dict:
                weights_dict_new[mapping[old_name]] = weights_dict[old_name]
            return weights_dict_new
        else:
            return weights_dict

    def convert_conv1_rgb2bgr(self, weights_dict):
        """Support caffe trained models: include resnet50/101/152 and vgg16"""
        conv1_weight = weights_dict['conv1.weight'].cpu().numpy().copy()
        conv1_weight[:, [0, 2], :, :] = conv1_weight[:, [2, 0], :, :]
        weights_dict['conv1.weight'] = torch.from_numpy(conv1_weight)
        logging_rank('Convert conv1.weight channel: {}'.format(conv1_weight.shape), local_rank=self.local_rank)
        return weights_dict

    def load_model(self, model, convert_conv1=False):
        if self.resume:
            weights_dict = self.checkpoint.pop('model')
            weights_dict = strip_prefix_if_present(weights_dict, prefix='module.')
            model_state_dict = model.state_dict()
            model_state_dict, self.mismatch_keys = align_and_update_state_dicts(model_state_dict, weights_dict,
                                                                                self.local_rank)
            model.load_state_dict(model_state_dict)
            logging_rank('Resuming from weights: {}.'.format(self.weights_path), local_rank=self.local_rank)
        else:
            if self.weights_path:
                weights_dict = self.checkpoint
                weights_dict = strip_prefix_if_present(weights_dict, prefix='module.')
                weights_dict = self.weight_mapping(weights_dict)    # only for pre-training
                if convert_conv1:   # only for pre-training
                    weights_dict = self.convert_conv1_rgb2bgr(weights_dict)
                model_state_dict = model.state_dict()
                model_state_dict, self.mismatch_keys = align_and_update_state_dicts(model_state_dict, weights_dict,
                                                                                    self.local_rank)
                model.load_state_dict(model_state_dict)
                logging_rank('Pre-training on weights: {}.'.format(self.weights_path), local_rank=self.local_rank)
            else:
                logging_rank('Training from scratch.', local_rank=self.local_rank)
        return model

    def load_optimizer(self, optimizer):
        if self.resume:
            optimizer.load_state_dict(self.checkpoint.pop('optimizer'))
            logging_rank('Loading optimizer done.', local_rank=self.local_rank)
        else:
            logging_rank('Initializing optimizer done.', local_rank=self.local_rank)
        return optimizer

    def load_scheduler(self, scheduler):
        if self.resume:
            scheduler.iteration = self.checkpoint['scheduler']['iteration']
            scheduler.info = self.checkpoint['scheduler']['info']
            logging_rank('Loading scheduler done.', local_rank=self.local_rank)
        else:
            logging_rank('Initializing scheduler done.', local_rank=self.local_rank)
        return scheduler

    def save(self, model, optimizer=None, scheduler=None, copy_latest=True, infix='epoch'):
        save_dict = {'model': model.state_dict()}
        if optimizer is not None:
            save_dict['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            save_dict['scheduler'] = scheduler.state_dict()

        torch.save(save_dict, os.path.join(self.ckpt, 'model_latest.pth'))
        logg_sstr = 'Saving checkpoint done.'
        if copy_latest and scheduler:
            shutil.copyfile(os.path.join(self.ckpt, 'model_latest.pth'),
                            os.path.join(self.ckpt, 'model_{}{}.pth'.format(infix, str(scheduler.iteration))))
            logg_sstr += ' And copy "model_latest.pth" to "model_{}{}.pth".'.format(infix, str(scheduler.iteration))
        logging_rank(logg_sstr, local_rank=self.local_rank)

    def save_best(self, model, optimizer=None, scheduler=None, remove_old=True, infix='epoch'):
        if scheduler.info['cur_acc'] < scheduler.info['best_acc']:
            return False

        old_name = 'model_{}{}-{:4.2f}.pth'.format(infix, scheduler.info['best_epoch'], scheduler.info['best_acc'])
        new_name = 'model_{}{}-{:4.2f}.pth'.format(infix, scheduler.info['cur_epoch'], scheduler.info['cur_acc'])
        if os.path.exists(os.path.join(self.ckpt, old_name)) and remove_old:
            os.remove(os.path.join(self.ckpt, old_name))
        scheduler.info['best_acc'] = scheduler.info['cur_acc']
        scheduler.info['best_epoch'] = scheduler.info['cur_epoch']

        save_dict = {'model': model.state_dict()}
        if optimizer is not None:
            save_dict['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            save_dict['scheduler'] = scheduler.state_dict()
        torch.save(save_dict, os.path.join(self.ckpt, new_name))
        shutil.copyfile(os.path.join(self.ckpt, new_name), os.path.join(self.ckpt, 'model_latest.pth'))
        logging_rank('Saving best checkpoint done: {}.'.format(new_name), local_rank=self.local_rank)
        return True


def strip_prefix_if_present(state_dict, prefix='module.'):
    """
    This function is taken from the maskrcnn_benchmark repo.
    It can be seen here:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/model_serialization.py
    """
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped_state_dict[key[len(prefix):]] = value
            # stripped_state_dict[key.replace(prefix, "")] = value
        else:
            pass
    return stripped_state_dict


def align_and_update_state_dicts(model_state_dict, weights_dict, local_rank):
    """
    This function is taken from the maskrcnn_benchmark repo.
    It can be seen here:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/model_serialization.py

    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    model_keys = sorted(list(model_state_dict.keys()))
    weights_keys = sorted(list(weights_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in model_keys for j in weights_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(weights_keys))
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size_model = max([len(key) for key in model_keys]) if model_keys else 1
    max_size_weights = max([len(key) for key in weights_keys]) if weights_keys else 1
    match_keys = set()
    for idx_model, idx_weights in enumerate(idxs.tolist()):
        if idx_weights == -1:
            continue
        key_model = model_keys[idx_model]
        key_weights = weights_keys[idx_weights]
        model_state_dict[key_model] = weights_dict[key_weights]
        match_keys.add(key_model)
        logging_rank(
            '{: <{}} loaded from {: <{}} of shape {}'.format(key_model, max_size_model, key_weights, max_size_weights,
                                                             tuple(weights_dict[key_weights].shape)),
            local_rank=local_rank)
    mismatch_keys = set(model_keys) - match_keys
    return model_state_dict, mismatch_keys


def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0


VGG16_NAME_MAPPING = {
    '0.weight': 'features1.0.weight',
    '0.bias': 'features1.0.bias',
    '2.weight': 'features1.2.weight',
    '2.bias': 'features1.2.bias',
    '5.weight': 'features1.5.weight',
    '5.bias': 'features1.5.bias',
    '7.weight': 'features1.7.weight',
    '7.bias': 'features1.7.bias',
    '10.weight': 'features1.10.weight',
    '10.bias': 'features1.10.bias',
    '12.weight': 'features1.12.weight',
    '12.bias': 'features1.12.bias',
    '14.weight': 'features1.14.weight',
    '14.bias': 'features1.14.bias',
    '17.weight': 'features1.17.weight',
    '17.bias': 'features1.17.bias',
    '19.weight': 'features1.19.weight',
    '19.bias': 'features1.19.bias',
    '21.weight': 'features1.21.weight',
    '21.bias': 'features1.21.bias',
    '24.weight': 'features2.1.weight',
    '24.bias': 'features2.1.bias',
    '26.weight': 'features2.3.weight',
    '26.bias': 'features2.3.bias',
    '28.weight': 'features2.5.weight',
    '28.bias': 'features2.5.bias',
    '31.weight': 'conv6.weight',
    '31.bias': 'conv6.bias',
    '33.weight': 'conv7.weight',
    '33.bias': 'conv7.bias',
}
