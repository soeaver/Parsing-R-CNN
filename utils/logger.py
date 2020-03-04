import os
import datetime
from collections import defaultdict, OrderedDict, deque

import torch

from utils.timer import Timer
from utils.misc import logging_rank
from utils.net import reduce_tensor


class TrainingLogger(object):
    """Track vital training statistics."""

    def __init__(self, cfg_filename, scheduler=None, log_period=20, iter_per_epoch=-1):
        self.cfg_filename = cfg_filename
        self.scheduler = scheduler
        self.log_period = log_period
        self.window_size = iter_per_epoch if iter_per_epoch != -1 else log_period

        self.data_timer = Timer()
        self.iter_timer = Timer()

        def create_smoothed_value():
            return SmoothedValue(self.window_size)

        self.smoothed_losses = defaultdict(create_smoothed_value)
        self.smoothed_metrics = defaultdict(create_smoothed_value)
        self.smoothed_total_loss = SmoothedValue(self.window_size)

    def data_tic(self):
        self.data_timer.tic()

    def data_toc(self):
        return self.data_timer.toc(average=False)

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        return self.iter_timer.toc(average=False)

    def reset_timer(self):
        self.data_timer.reset()
        self.iter_timer.reset()

    def update_stats(self, output, distributed=True, world_size=1):
        total_loss = 0
        for k, loss in output['losses'].items():
            total_loss += loss
            loss_data = loss.data
            if distributed:
                loss_data = reduce_tensor(loss_data, world_size=world_size)
            self.smoothed_losses[k].update(loss_data)
        output['total_loss'] = total_loss  # add the total loss for back propagation
        self.smoothed_total_loss.update(total_loss.data)

        for k, metric in output['metrics'].items():
            metric = metric.mean(dim=0, keepdim=True)
            self.smoothed_metrics[k].update(metric.data[0])

    def log_stats(self, cur_iter, lr, skip_metrics=False, skip_losses=False, suffix=None):
        """Log the tracked statistics."""
        if self.scheduler.iter_per_epoch == -1:
            log_flag = not cur_iter % self.log_period
        else:
            log_flag = not (cur_iter % self.scheduler.iter_per_epoch) % self.log_period
        if log_flag:
            stats = self.get_stats(cur_iter, lr)
            lines = '[Training][{}]'.format(stats['cfg_filename'])
            if 'epoch' in stats.keys():
                lines += '[epoch: {}/{}]'.format(stats['epoch'], stats['max_epoch'])
            lines += '[iter: {}/{}]'.format(stats['iter'], stats['max_iter'])
            lines += '[lr: {:.6f}][eta: {}]'.format(stats['lr'], stats['eta'])
            if suffix is not None:
                lines += suffix
            lines += '\n'

            lines += '\t  total_loss: {:.6f} ({:.6f}), '.format(stats['total_loss_cur'], stats['total_loss'])
            lines += 'iter_time: {:.4f} ({:.4f}), data_time: {:.4f} ({:.4f})\n'. \
                format(stats['iter_time_cur'], stats['iter_time'], stats['data_time_cur'], stats['data_time'])

            if stats['metrics'] and not skip_metrics:
                lines += '\t  ' + ', '.join('{}: {:.4f} ({:.4f})'.format(k, float(v.split(' ')[0]), 
                                                                         float(v.split(' ')[1])) 
                                            for k, v in stats['metrics'].items()) + '\n'
            if stats['losses'] and not skip_losses:
                lines += '\t  ' + ', '.join('{}: {:.6f} ({:.6f})'.format(k, float(v.split(' ')[0]), 
                                                                         float(v.split(' ')[1])) 
                                            for k, v in stats['losses'].items()) + '\n'
            print(lines[:-1])  # remove last new line
        return None

    def get_stats(self, cur_iter, lr):
        eta_seconds = self.iter_timer.average_time * (self.scheduler.max_iter - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        stats = OrderedDict(cfg_filename=self.cfg_filename)
        if self.scheduler.iter_per_epoch == -1:
            stats['iter'] = cur_iter
            stats['max_iter'] = self.scheduler.max_iter
        else:
            stats['epoch'] = cur_iter // self.scheduler.iter_per_epoch + 1
            stats['max_epoch'] = self.scheduler.max_iter // self.scheduler.iter_per_epoch
            stats['iter'] = cur_iter % self.scheduler.iter_per_epoch
            stats['max_iter'] = self.scheduler.iter_per_epoch
        stats['lr'] = lr
        stats['eta'] = eta
        stats['data_time'] = self.data_timer.average_time
        stats['data_time_cur'] = self.data_timer.diff
        stats['iter_time'] = self.iter_timer.average_time
        stats['iter_time_cur'] = self.iter_timer.diff
        stats['total_loss'] = self.smoothed_total_loss.avg
        stats['total_loss_cur'] = self.smoothed_total_loss.latest

        metrics = []
        for k, v in self.smoothed_metrics.items():
            metrics.append((k, str(v.latest) + ' ' + str(v.avg)))
        stats['metrics'] = OrderedDict(metrics)

        losses = []
        for k, v in self.smoothed_losses.items():
            losses.append((k, str(v.latest) + ' ' + str(v.avg)))
        stats['losses'] = OrderedDict(losses)

        return stats


class TestingLogger(object):
    """Track vital testing statistics."""

    def __init__(self, cfg_filename, log_period=10):
        self.cfg_filename = cfg_filename
        self.log_period = log_period

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.infer_timer = Timer()
        self.post_timer = Timer()

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        return self.iter_timer.toc(average=False)

    def data_tic(self):
        self.data_timer.tic()

    def data_toc(self):
        return self.data_timer.toc(average=False)

    def infer_tic(self):
        self.infer_timer.tic()

    def infer_toc(self):
        return self.infer_timer.toc(average=False)

    def post_tic(self):
        self.post_timer.tic()

    def post_toc(self):
        return self.post_timer.toc(average=False)

    def reset_timer(self):
        self.iter_timer.reset()
        self.data_timer.reset()
        self.infer_timer.reset()
        self.post_timer.reset()

    def log_stats(self, cur_idx, start_ind, end_ind, total_num_images, suffix=None):
        """Log the tracked statistics."""
        if cur_idx % self.log_period == 0:
            eta_seconds = self.iter_timer.average_time * (end_ind - cur_idx - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            lines = '[Testing][range:{}-{} of {}][{}/{}]'. \
                format(start_ind + 1, end_ind, total_num_images, cur_idx + 1, end_ind)

            lines += '[{:.3f}s = {:.3f}s + {:.3f}s + {:.3f}s][eta: {}]'. \
                format(self.iter_timer.average_time, self.data_timer.average_time, self.infer_timer.average_time,
                       self.post_timer.average_time, eta)
            if suffix is not None:
                lines += suffix
            logging_rank(lines)
        return None


class MetricLogger(object):
    """Save training metric to log file with simple plot function."""

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume if os.path.exists(fpath) else False
        self.title = '' if title == None else title
        if fpath is not None:
            if self.resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def latest(self):
        return torch.tensor(self.series[-1]).item()

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
