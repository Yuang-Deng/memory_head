from __future__ import division
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler


class SSODGroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1, labeled_rate=0.5, warm_epoch=-1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.labeled_num = samples_per_gpu // labeled_rate
        self.unlabeled_num = samples_per_gpu - self.labeled_num
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        self.epoch = 0
        self.warm_epoch = warm_epoch
        if len(self.group_sizes) == 4:
            samples = self.samples_per_gpu // 2
            labeled_size = self.group_sizes[0]
            unlabeled_size = self.group_sizes[1]
            larger_size = max(labeled_size, unlabeled_size)
            self.num_samples += int(np.ceil(larger_size / samples)) * samples
            self.num_samples *= 2
        else:
            for i, size in enumerate(self.group_sizes):
                self.num_samples += int(np.ceil(
                    size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        if len(self.group_sizes) == 4:
            self.num_samples = 0
            samples = self.samples_per_gpu // 2
            hori_labeled_indice = np.where(self.flag == 0)[0]
            hori_unlabeled_indice = np.where(self.flag == 2)[0]
            np.random.shuffle(hori_labeled_indice)
            np.random.shuffle(hori_unlabeled_indice)

            vert_labeled_indice = np.where(self.flag == 1)[0]
            vert_unlabeled_indice = np.where(self.flag == 3)[0]
            np.random.shuffle(vert_labeled_indice)
            np.random.shuffle(vert_unlabeled_indice)

            tmp = hori_labeled_indice.copy()
            extra = samples - (len(hori_labeled_indice) % samples)
            for _ in range(extra // len(hori_labeled_indice)):
                hori_labeled_indice = np.concatenate([hori_labeled_indice, tmp])
            hori_labeled_indice = np.concatenate([hori_labeled_indice, tmp[:extra % len(hori_labeled_indice)]])

            tmp = hori_unlabeled_indice.copy()
            extra = samples - (len(hori_unlabeled_indice) % samples)
            for _ in range(extra // len(hori_unlabeled_indice)):
                hori_unlabeled_indice = np.concatenate([hori_unlabeled_indice, tmp])
            hori_unlabeled_indice = np.concatenate([hori_unlabeled_indice, tmp[:extra % len(hori_unlabeled_indice)]])

            tmp = vert_labeled_indice.copy()
            extra = samples - (len(vert_labeled_indice) % samples)
            for _ in range(extra // len(vert_labeled_indice)):
                vert_labeled_indice = np.concatenate([vert_labeled_indice, tmp])
            vert_labeled_indice = np.concatenate([vert_labeled_indice, tmp[:extra % len(vert_labeled_indice)]])

            tmp = vert_unlabeled_indice.copy()
            extra = samples - (len(vert_unlabeled_indice) % samples)
            for _ in range(extra // len(vert_unlabeled_indice)):
                vert_unlabeled_indice = np.concatenate([vert_unlabeled_indice, tmp])
            vert_unlabeled_indice = np.concatenate([vert_unlabeled_indice, tmp[:extra % len(vert_unlabeled_indice)]])

            labeled_indice = np.concatenate([hori_labeled_indice, vert_labeled_indice])
            unlabeled_indice = np.concatenate([hori_unlabeled_indice, vert_unlabeled_indice])

            if self.epoch >= self.warm_epoch:
                larger_size = max(len(labeled_indice), len(unlabeled_indice))
            else:
                larger_size = len(labeled_indice)
            
            labeled_num_extra = int(np.ceil(larger_size / samples)
                                ) * samples - len(labeled_indice)
            unlabeled_num_extra = int(np.ceil(larger_size / samples)
                            ) * samples - len(unlabeled_indice)
            labeled_num_extra = 0 if labeled_num_extra < 0 else labeled_num_extra
            unlabeled_num_extra = 0 if unlabeled_num_extra < 0 else unlabeled_num_extra
            
            self.num_samples += int(np.ceil(larger_size / samples)) * samples
            self.num_samples *= 2
            labeled_indice = np.concatenate(
                [labeled_indice, np.random.choice(labeled_indice, labeled_num_extra)])
            unlabeled_indice = np.concatenate(
                [unlabeled_indice, np.random.choice(unlabeled_indice, unlabeled_num_extra)])
            indices = []
            for i in np.random.permutation(range(len(labeled_indice) // samples)):
                indices.append(labeled_indice[i * samples:(i + 1) * samples])
                indices.append(unlabeled_indice[i * samples:(i + 1) * samples])
            indices = np.concatenate(indices)
            indices = indices.astype(np.int64).tolist()
            return iter(indices)
        else:
            indices = []
            # self.group_sizes = self.group_sizes[:2]
            for i, size in enumerate(self.group_sizes):
                if size == 0:
                    continue
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                np.random.shuffle(indice)
                num_extra = int(np.ceil(size / self.samples_per_gpu)
                                ) * self.samples_per_gpu - len(indice)
                indice = np.concatenate(
                    [indice, np.random.choice(indice, num_extra)])
                indices.append(indice)
            indices = np.concatenate(indices)
            indices = [
                indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
                for i in np.random.permutation(
                    range(len(indices) // self.samples_per_gpu))
            ]
            indices = np.concatenate(indices)
            indices = indices.astype(np.int64).tolist()
            return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class SSODDistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 warm_epoch=-1,
                 postive_per_gpu=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0
        self.warm_epoch = warm_epoch
        self.postive_per_gpu = postive_per_gpu


        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        if len(self.group_sizes) == 4:
            samples = self.samples_per_gpu // 2
            labeled_size = self.group_sizes[0]
            unlabeled_size = self.group_sizes[1]
            larger_size = max(labeled_size, unlabeled_size)
            self.num_samples += int(
                math.ceil(larger_size * 1.0 / samples /
                          self.num_replicas)) * samples
            self.num_samples *= 2
            self.total_size = self.num_samples * self.num_replicas
        else:
            for i, j in enumerate(self.group_sizes):
                self.num_samples += int(
                    math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                              self.num_replicas)) * self.samples_per_gpu
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        if len(self.group_sizes) == 4:
            self.num_samples = 0
            # labeled_samples = self.samples_per_gpu // 2
            # unlabeled_samples = self.samples_per_gpu // 2
            labeled_samples = self.postive_per_gpu
            unlabeled_samples = self.samples_per_gpu - self.postive_per_gpu
            
            # labeled_size_hori = self.group_sizes[0]
            # unlabeled_size_hori = self.group_sizes[1]
            labeled_indice_hori = np.where(self.flag == 0)[0]
            unlabeled_indice_hori = np.where(self.flag == 2)[0]
            labeled_indice_hori = labeled_indice_hori[list(
                torch.randperm(int(len(labeled_indice_hori)), generator=g).numpy())].tolist()
            unlabeled_indice_hori = unlabeled_indice_hori[list(
                torch.randperm(int(len(unlabeled_indice_hori)), generator=g).numpy())].tolist()
            
            
            # labeled_size_vert = self.group_sizes[2]
            # unlabeled_size_vert = self.group_sizes[3]
            labeled_indice_vert = np.where(self.flag == 1)[0]
            unlabeled_indice_vert = np.where(self.flag == 3)[0]
            labeled_indice_vert = labeled_indice_vert[list(
                torch.randperm(int(len(labeled_indice_vert)), generator=g).numpy())].tolist()
            unlabeled_indice_vert = unlabeled_indice_vert[list(
                torch.randperm(int(len(unlabeled_indice_vert)), generator=g).numpy())].tolist()

            tmp = labeled_indice_hori.copy()
            labeled_extra_hori = labeled_samples - (len(labeled_indice_hori) % labeled_samples)
            for _ in range(labeled_extra_hori // len(labeled_indice_hori)):
                labeled_indice_hori.extend(tmp)
            labeled_indice_hori.extend(tmp[:labeled_extra_hori % len(labeled_indice_hori)])

            tmp = unlabeled_indice_hori.copy()
            unlabeled_extra_hori = unlabeled_samples - (len(unlabeled_indice_hori) % unlabeled_samples)
            for _ in range(unlabeled_extra_hori // len(unlabeled_indice_hori)):
                unlabeled_indice_hori.extend(tmp)
            unlabeled_indice_hori.extend(tmp[:unlabeled_extra_hori % len(unlabeled_indice_hori)])

            tmp = labeled_indice_vert.copy()
            labeled_extra_vert = labeled_samples - (len(labeled_indice_vert) % labeled_samples)
            for _ in range(labeled_extra_vert // len(labeled_indice_vert)):
                labeled_indice_vert.extend(tmp)
            labeled_indice_vert.extend(tmp[:labeled_extra_vert % len(labeled_indice_vert)])

            tmp = unlabeled_indice_vert.copy()
            unlabeled_extra_vert = unlabeled_samples - (len(unlabeled_indice_vert) % unlabeled_samples)
            for _ in range(unlabeled_extra_vert // len(unlabeled_indice_vert)):
                unlabeled_indice_vert.extend(tmp)
            unlabeled_indice_vert.extend(tmp[:unlabeled_extra_vert % len(unlabeled_indice_vert)])
            labeled_indice_hori.extend(labeled_indice_vert)
            unlabeled_indice_hori.extend(unlabeled_indice_vert)
            labeled_indice = labeled_indice_hori
            unlabeled_indice = unlabeled_indice_hori
            labeled_size = len(labeled_indice)
            unlabeled_size = len(unlabeled_indice)
            
            # if self.epoch >= self.warm_epoch:
            #     larger = max(labeled_size, unlabeled_size)
            # else:
            #     larger = labeled_size

            larger = unlabeled_size

            all_round = max(int(math.ceil(labeled_size * 1.0 / labeled_samples / self.num_replicas)), int(
                    math.ceil(unlabeled_size * 1.0 / unlabeled_samples / self.num_replicas)))
                
            labeled_extra = all_round * labeled_samples * self.num_replicas - len(labeled_indice)
            unlabeled_extra = all_round * unlabeled_samples * self.num_replicas - len(unlabeled_indice)
                
            unlabeled_extra = 0 if unlabeled_extra < 0 else unlabeled_extra
            labeled_extra = 0 if labeled_extra < 0 else labeled_extra
            # self.num_samples += int(math.ceil(labeled_size * 1.0 / labeled_samples /self.num_replicas)) * labeled_samples + int(math.ceil(unlabeled_size * 1.0 / unlabeled_samples /self.num_replicas)) * unlabeled_samples
            # self.num_samples *= 2

            tmp = labeled_indice.copy()
            for _ in range(labeled_extra // labeled_size):
                labeled_indice.extend(tmp)
            labeled_indice.extend(tmp[:labeled_extra % labeled_size])

            tmp = unlabeled_indice.copy()
            for _ in range(unlabeled_extra // unlabeled_size):
                unlabeled_indice.extend(tmp)
            unlabeled_indice.extend(tmp[:unlabeled_extra % unlabeled_size])
            # assert len(indices) == self.total_size
            self.num_samples += int(math.ceil(len(labeled_indice) * 1.0 / labeled_samples /self.num_replicas)) * labeled_samples + int(math.ceil(len(unlabeled_indice) * 1.0 / unlabeled_samples /self.num_replicas)) * unlabeled_samples


            for i in list(torch.randperm(len(labeled_indice) // labeled_samples, generator=g)):
                indices.extend(labeled_indice[i * labeled_samples:(i + 1) * labeled_samples])
                indices.extend(unlabeled_indice[i * unlabeled_samples:(i + 1) * unlabeled_samples])
            # subsample
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]
            return iter(indices)
            # assert len(indices) == self.num_samples
        else:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
    
            indices = []
            for i, size in enumerate(self.group_sizes):
                if size > 0:
                    indice = np.where(self.flag == i)[0]
                    assert len(indice) == size
                    # add .numpy() to avoid bug when selecting indice in parrots.
                    # TODO: check whether torch.randperm() can be replaced by
                    # numpy.random.permutation().
                    indice = indice[list(
                        torch.randperm(int(size), generator=g).numpy())].tolist()
                    extra = int(
                        math.ceil(
                            size * 1.0 / self.samples_per_gpu / self.num_replicas)
                    ) * self.samples_per_gpu * self.num_replicas - len(indice)
                    # pad indice
                    tmp = indice.copy()
                    for _ in range(extra // size):
                        indice.extend(tmp)
                    indice.extend(tmp[:extra % size])
                    indices.extend(indice)
    
            assert len(indices) == self.total_size
    
            indices = [
                indices[j] for i in list(
                    torch.randperm(
                        len(indices) // self.samples_per_gpu, generator=g))
                for j in range(i * self.samples_per_gpu, (i + 1) *
                               self.samples_per_gpu)
            ]
    
            # subsample
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]
            assert len(indices) == self.num_samples
    
            return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch