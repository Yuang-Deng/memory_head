# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner.hooks import HOOKS, Hook
from torch.utils.data import DataLoader
import json
from mmcv.runner import get_dist_info

@HOOKS.register_module()
class PseudoHook(Hook):

    def evaluate(self, runner, results):
        """Evaluate the results.
        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        return None

    def _should_evaluate(self, runner):
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

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner) and runner.rank != 0:
            return
        
        from mmdet.apis import single_gpu_test
        filename = self.dataloader.dataset.ann_file
        results, img_metas, boxex, labels = single_gpu_test(runner.model, self.dataloader, show=False, return_meta=True, annfile=filename, rank=runner.rank)
        filename = self.dataloader.dataset.ann_file
        with open("C:/Users/Alex/WorkSpace/dataset/coco/annotations/instances_val2017.json", 'r') as load_f:
            load_dict = json.load(load_f)
            load_dict['annotations'] = []
            id = 0
            for i, box, label in zip(img_metas, boxex, labels):
                for b, l in zip(box, label):
                    ann = {}
                    ann['image_id'] = int(i['ori_filename'].split('.')[0])
                    ann['bbox'] = list(map(int, b[0:4].cpu().numpy().tolist()))
                    ann['category_id'] = l.item()
                    ann['id'] = id
                    ann['area'] = (ann['bbox'][2] - ann['bbox'][0]) * (ann['bbox'][3] - ann['bbox'][1])
                    id += 1
                    load_dict['annotations'].append(ann)
        json.dump(load_dict, open('C:/Users/Alex/WorkSpace/dataset/coco/annotations/test.json', 'w'))
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
    
    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 **kwargs
                 ):
        self.eval_kwargs = kwargs
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

    def after_train_epoch(self, runner):
        self._do_evaluate(runner)


@HOOKS.register_module()
class DistPseudoHook(BaseDistEvalHook):

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return
        
        rank, world_size = get_dist_info()

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test
        filename = self.dataloader.dataset.ann_file
        results = multi_gpu_test(runner.model, self.dataloader, gpu_collect=False, pseudo_gen=True, annfile=filename)
        if runner.rank == 0:
            load_dicts = []
            for i in range(world_size):
                wfile = filename.split('.')[0] + str(i) + '.' + filename.split('.')[1]
                load_dicts[i] = json.load(open(wfile, 'r'))
            load_dict = load_dicts[0]
            for i in range(1, world_size):
                load_dict['annotations'].extend(load_dicts[i]['annotations'])
            json.dump(load_dict, open(filename, 'w'))
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)