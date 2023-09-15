##############################################################################################################
# Code reproducity is essential for DL. The MyDistributedSampler tries to make datasets reproducible by 
# generating a seed for each dataset item at specific epoch.
#
# Copyright: Changsheng Quan @ Audio Lab of Westlake University 2023
##############################################################################################################



import math
from typing import Iterator, Optional

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler, T_co


class MyDistributedSampler(DistributedSampler[T_co]):
    r"""Sampler for single GPU and multi GPU (or Distributed) cases. Change int index to a tuple (index, random seed for this index).
    This sampler is used to enhance the reproducibility of datasets by generating random seed for each item at each epoch.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        try:
            super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        except:
            # if error raises, it is running on single GPU
            # thus, set num_replicas=1, rank=0
            super().__init__(dataset, 1, 0, shuffle, seed, drop_last)
        self.last_epoch = -1

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            if self.last_epoch >= self.epoch:
                if self.epoch != 0:
                    rank_zero_warn(f'shuffle is true but the epoch value doesn\'t get update, thus the order of training data won\'t change at epoch={self.epoch}')
            else:
                self.last_epoch = self.epoch
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            g = torch.Generator()
            g.manual_seed(self.seed)

            indices = list(range(len(self.dataset)))  # type: ignore

        seeds = []
        for i in range(len(indices)):
            seed = torch.randint(high=9999999999, size=(1,), generator=g)[0].item()
            seeds.append(seed)
        indices = list(zip(indices, seeds))

        # drop last
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)  # type: ignore

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
