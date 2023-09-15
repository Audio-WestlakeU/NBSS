from torch import Tensor
import torch
from torch import nn

from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_noise_ratio as snr
from torchmetrics.functional.audio import source_aggregated_signal_distortion_ratio as sa_sdr
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate as permutate

from typing import *


def neg_sa_sdr(preds: Tensor, target: Tensor, scale_invariant: bool = False) -> Tensor:
    batch_size = target.shape[0]
    sa_sdr_val = sa_sdr(preds=preds, target=target, scale_invariant=scale_invariant)
    return -torch.mean(sa_sdr_val.view(batch_size, -1), dim=1)


def neg_si_sdr(preds: Tensor, target: Tensor) -> Tensor:
    """calculate neg_si_sdr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    si_sdr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_sdr_val.view(batch_size, -1), dim=1)


def neg_snr(preds: Tensor, target: Tensor) -> Tensor:
    """calculate neg_snr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    snr_val = snr(preds=preds, target=target)
    return -torch.mean(snr_val.view(batch_size, -1), dim=1)


class Loss(nn.Module):
    is_scale_invariant_loss: bool
    name: str

    def __init__(self, loss_func: Callable, pit: bool, loss_func_kwargs: Dict[str, Any] = dict()):
        super().__init__()

        self.loss_func = loss_func
        self.pit = pit
        self.loss_func_kwargs = loss_func_kwargs
        self.is_scale_invariant_loss = {
            neg_sa_sdr: True if 'scale_invariant' in loss_func_kwargs and loss_func_kwargs['scale_invariant'] == True else False,
            neg_si_sdr: True,
            neg_snr: False,
        }[loss_func]
        self.name = loss_func.__name__

    def forward(self, preds: Tensor, target: Tensor, reorder: bool = None, reduce_batch: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        perms = None
        if self.pit:
            losses, perms = pit(preds=preds, target=target, metric_func=self.loss_func, eval_func='min', mode="permutation-wise", **self.loss_func_kwargs)
            if reorder:
                preds = permutate(preds, perm=perms)
        else:
            losses = self.loss_func(preds=preds, target=target, **self.loss_func_kwargs)

        return losses.mean() if reduce_batch else losses, perms, preds

    def extra_repr(self) -> str:
        kwargs = ""
        for k, v in self.loss_func_kwargs.items():
            kwargs += f'{k}={v},'

        return f"loss_func={self.loss_func.__name__}({kwargs}), pit={self.pit}"
