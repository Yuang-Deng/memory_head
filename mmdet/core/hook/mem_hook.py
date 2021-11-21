# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from math import inf
import time

from mmcv.runner.hooks import HOOKS, Hook
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from mmcv.utils import is_seq_of
from mmcv.runner.hooks import Hook
# from mmcv.runner.logger import LoggerHook


class MEMHook(Hook):
    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 ):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f'dataloader must be a pytorch DataLoader, '
                            f'but got {type(dataloader)}')

        if interval <= 0:
            raise ValueError(f'interval must be a positive number, '
                             f'but got {interval}')

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')

        self.dataloader = dataloader
        self.interval = interval
        self.start = start

    def _should_mem_forward(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        current = runner.epoch
        check_time = self.every_n_epochs

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True

    def _mem_forward(self, runner):
        for i, data in enumerate(self.dataloader):
            runner.model.train_step(data)

    def before_train_epoch(self, runner):
        if self._should_mem_forward(runner):
            self._mem_forward(runner)
        if hasattr(runner.model.module.roi_head, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.model.module.roi_head.set_epoch(runner.epoch)
        



class DistMEMHook(MEMHook):
    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,):

        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            )

        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def _mem_forward(self, runner):
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)
        time_start=time.time()

        # print(len(self.dataloader))
        for i, data in enumerate(self.dataloader):
            runner.model.train_step(data, None, mem_forward=True)

        time_end=time.time()
        print('time cost',time_end-time_start,'s')
    
@HOOKS.register_module()
class MEMEMAHook(Hook):
    def before_run(self, runner):
        if hasattr(runner.model.module.roi_head, 'mem_weight_init'):
                # in case the data loader uses `SequentialSampler` in Pytorch
                runner.model.module.roi_head.mem_weight_init()

    def before_train_epoch(self, runner):
        runner.model.module.roi_head.dataset = runner.data_loader.dataset

    def after_train_iter(self, runner):
        if hasattr(runner.model.module.roi_head, 'mem_weight_update'):
                # in case the data loader uses `SequentialSampler` in Pytorch
                runner.model.module.roi_head.mem_weight_update()
        
        if hasattr(runner.model.module, 'ema_update'):
                # in case the data loader uses `SequentialSampler` in Pytorch
                runner.model.module.ema_update()
        
        
