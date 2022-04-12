from typing import Any, Callable, Tuple

import torch
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_noise_ratio as snr

from models.io.narrow_band.stft_coefficient_nb import STFTCoefficientNB


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


class TimeDomainSignalNB(STFTCoefficientNB):
    """time domain signal IO
    """

    def __init__(self, ft_len: int, ft_overlap: int, ref_chn_idx: int = 0, spk_num: int = 2, loss_func: Callable = neg_si_sdr, loss_name=None) -> None:
        """init

        Args:
            ft_len: for STFT
            ft_overlap: for STFT
            ref_chn_idx: the index of the reference channel
            spk_num: the speaker num
            loss_func: loss function. Defaults to neg_si_sdr_batch
            loss_name: the name of loss function, if is None, the name of `loss_func` will be used
        """
        super().__init__(ft_len, ft_overlap, ref_chn_idx, spk_num, loss_func=loss_func)

    def prepare_input(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, int]:
        """prepare input for network

        Args:
            x: time domain signal, shape [batch, chn, time]

        returns:
            X, XrMM, original_len
        """
        return super().prepare_input(x, *args, **kwargs)

    def prepare_target(self, ys: Tensor, input: Any, *args, **kwargs) -> Tensor:
        """prepare target

        Args:
            ys: time domain signal, shape [batch, spk, time]
            input: the output of prepare_input

        returns:
            ys
        """
        return ys

    def prepare_prediction(self, o: Tensor, input: Any, *args, **kwargs) -> Tensor:
        """prepare prediction from the output of network

        Args:
            o: raw output from network
            input: the output of prepare_input

        returns:
            ys_hat
        """
        Ys_hat = super().prepare_prediction(o, input, *args, **kwargs)
        ys_hat = super().prepare_time_domain(o, input, Ys_hat, *args, **kwargs)
        return ys_hat

    def prepare_time_domain(self, o: Tensor, input: Any, preds: Tensor, *args, **kwargs) -> Tensor:  # type:ignore
        """prepare time domain prediction

        Args:
            o: raw output from network
            input: the output of prepare_input
            pred: the output of prepare_prediction

        returns:
            ys_hat
        """
        return preds

    def loss(self, preds: Tensor, target: Tensor, reduce_batch: bool = True, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """loss for preds and target

        Args:
            preds: prediction, ys_hat
            target: target, ys

        Returns:
            loss value(s), shape [batch] if reduce_batch==False, else a single value
            perms: returned by pit
        """
        losses, perms = pit(preds=preds, target=target, metric_func=self.loss_func, eval_func='min')
        if reduce_batch:
            return losses.mean(), perms
        else:
            return losses, perms
