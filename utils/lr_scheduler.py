import logging
from math import cos, pi
from mxnet.lr_scheduler import LRScheduler

class WarmupMultiFactorScheduler(LRScheduler):
    def __init__(self, step, factor=1, warmup=False, warmup_type='constant', warmup_lr=0, warmup_step=0):
        super().__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        if warmup:
            if warmup_step >= step[0]:
                raise ValueError("Warmup step must be smaller than schedule step")
            if warmup_type not in ['constant', 'gradual']:
                raise ValueError("Warmup scheduler only support constant or gradual")

        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.warmup = warmup
        self.warmup_type = warmup_type
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step

    def __call__(self, num_update):
        if self.warmup and num_update <= self.warmup_step:
            if self.warmup_type == 'constant':
                return self.warmup_lr
            elif self.warmup_type == 'gradual':
                return (self.base_lr - self.warmup_lr) / self.warmup_step * num_update + self.warmup_lr

        while self.cur_step_ind <= len(self.step) - 1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr


# from gluoncv
class LRSequential(LRScheduler):
    r"""Compose Learning Rate Schedulers
    Parameters
    ----------
    schedulers: list
        list of LRScheduler objects
    """
    def __init__(self, schedulers):
        super().__init__()
        assert(len(schedulers) > 0)

        self.update_sep = []
        self.count = 0
        self.learning_rate = 0
        self.schedulers = []
        for lr in schedulers:
            self.add(lr)

    def add(self, scheduler):
        assert(isinstance(scheduler, LRScheduler))

        scheduler.offset = self.count
        self.count += scheduler.niters
        self.update_sep.append(self.count)
        self.schedulers.append(scheduler)

    def __call__(self, num_update):
        self.update(num_update)
        return self.learning_rate

    def update(self, num_update):
        num_update = min(num_update, self.count - 1)
        ind = len(self.schedulers) - 1
        for i, sep in enumerate(self.update_sep):
            if sep > num_update:
                ind = i
                break
        lr = self.schedulers[ind]
        lr.update(num_update)
        self.learning_rate = lr.learning_rate


# from gluoncv
class AdvancedLRScheduler(LRScheduler):
    r"""Learning Rate Scheduler
    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'constant', 'step', 'linear', 'poly' and 'cosine'.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    target_lr : float
        Target learning rate, i.e. the ending learning rate.
        With constant mode target_lr is ignored.
    niters : int
        Number of iterations to be scheduled.
    nepochs : int
        Number of epochs to be scheduled.
    iters_per_epoch : int
        Number of iterations in each epoch.
    offset : int
        Number of iterations before this scheduler.
    power : float
        Power parameter of poly scheduler.
    step_iter : list
        A list of iterations to decay the learning rate.
    step_epoch : list
        A list of epochs to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    """
    def __init__(self, mode, base_lr=0.1, target_lr=0,
                 niters=0, nepochs=0, iters_per_epoch=0, offset=0,
                 power=2, step_iter=None, step_epoch=None, step_factor=0.1,
                 baselr=None, targetlr=None):
        super().__init__()
        assert(mode in ['constant', 'step', 'linear', 'poly', 'cosine'])

        self.mode = mode
        if mode == 'step':
            assert(step_iter is not None or step_epoch is not None)
        if baselr is not None:
            warnings.warn("baselr is deprecated. Please use base_lr.")
            if base_lr == 0.1:
                base_lr = baselr
        self.base_lr = base_lr
        if targetlr is not None:
            warnings.warn("targetlr is deprecated. Please use target_lr.")
            if target_lr == 0:
                target_lr = targetlr
        self.target_lr = target_lr
        if self.mode == 'constant':
            self.target_lr = self.base_lr

        self.niters = niters
        self.step = step_iter
        epoch_iters = nepochs * iters_per_epoch
        if epoch_iters > 0:
            self.niters = epoch_iters
            if step_epoch is not None:
                self.step = [s*iters_per_epoch for s in step_epoch]

        self.offset = offset
        self.power = power
        self.step_factor = step_factor

    def __call__(self, num_update):
        self.update(num_update)
        return self.learning_rate

    def update(self, num_update):
        N = self.niters - 1
        T = num_update - self.offset
        T = min(max(0, T), N)

        if self.mode == 'constant':
            factor = 0
        elif self.mode == 'linear':
            factor = 1 - T / N
        elif self.mode == 'poly':
            factor = pow(1 - T / N, self.power)
        elif self.mode == 'cosine':
            factor = (1 + cos(pi * T / N)) / 2
        elif self.mode == 'step':
            if self.step is not None:
                count = sum([1 for s in self.step if s <= T])
                factor = pow(self.step_factor, count)
            else:
                factor = 1
        else:
            raise NotImplementedError

        if self.mode == 'step':
            self.learning_rate = self.base_lr * factor
        else:
            self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * factor
