from torch.utils.data.distributed import DistributedSampler, T_co
import torch.distributed as dist
from typing import TypeVar, Optional, Iterator

from .ss_semi_online_dataset import SS_SemiOnlineDataset

import torch
import math
import copy


class SS_SemiOnlineSampler(DistributedSampler[T_co]):
    r"""Sampler for SS_SemiOnlineDataset for single GPU and multi GPU (or Distributed) cases.
    If shuffle == True, the speech pair sequence and seed changes along epochs, else the speech pair and seed won't change along epochs
    If shuffle_rir == True, the rir will be shuffled, otherwise not

    No matter what is ``shuffle`` or ``shuffle_rir``, the speech sequence, rir sequence, seed generated for dataset are all deterministic.
    They all determined by the parameter ``seed`` and ``epoch`` (for shuffle == True)
    """

    def __init__(self,
                 dataset: SS_SemiOnlineDataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 shuffle_rir: bool = False,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        try:
            super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        except:
            # if error raises, it is running on single GPU
            # thus, set num_replicas=1, rank=0
            super().__init__(dataset, 1, 0, shuffle, seed, drop_last)
        self.shuffle_rir = shuffle_rir
        self.last_epoch = -1

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            speech_indices = torch.randperm(self.dataset.speech_num(), generator=g).tolist()  # type: ignore
            if self.shuffle_rir:
                rir_indices = torch.randperm(self.dataset.rir_num(), generator=g).tolist()  # type: ignore
            else:
                rir_indices = copy.deepcopy(speech_indices)

            if self.last_epoch >= self.epoch:
                import warnings
                if self.epoch != 0:
                    warnings.warn('the epoch value doesn\'t update when shuffle is true, the training data and sequence won\'t change along with different epochs')
            else:
                self.last_epoch = self.epoch
        else:
            g = torch.Generator()
            g.manual_seed(self.seed)

            speech_indices = list(range(len(self.dataset)))  # type: ignore
            if self.shuffle_rir:
                rir_indices = torch.randperm(self.dataset.rir_num(), generator=g).tolist()  # type: ignore
            else:
                rir_indices = list(range(self.dataset.rir_num()))  # type: ignore

        # make rir_indices and speech_indices have the same length as the dataset
        if len(speech_indices) > len(self.dataset):  # type: ignore
            from .spk4_wsj0_mix_sp import Spk4Wsj0mixSp
            assert isinstance(self.dataset, Spk4Wsj0mixSp), type(self.dataset)

            speech_indices = speech_indices[:len(self.dataset)]  # type: ignore

        if len(rir_indices) < len(speech_indices):
            to_add_num = len(speech_indices) - len(rir_indices)
            to_add_rirs = []
            for i in range(to_add_num):
                to_add_rirs.append(rir_indices[i % len(rir_indices)])
            rir_indices = rir_indices + to_add_rirs
        else:
            rir_indices = rir_indices[:len(speech_indices)]

        # construct indices
        indices = []
        for sidx, ridx in zip(speech_indices, rir_indices):
            seed = torch.randint(high=9999999999, size=(1,), generator=g)[0].item()
            indices.append({'speech_index': sidx, 'rir_index': ridx, 'seed': seed})

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
