from typing import Any, Callable, Tuple

import torch
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate

from models.io.narrow_band.nbio import NBIO


def mse_mrm(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """calculate mse loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    res = (target - preds)**2
    return torch.mean(res.view(batch_size, -1), dim=1)


class MagnitudeRatioMaskNB(NBIO):
    """magnitude ratio mask target
    """

    size_per_spk: int = 1

    def __init__(self, ft_len: int, ft_overlap: int, ref_chn_idx: int, spk_num: int, loss_func: Callable = mse_mrm, loss_name=None) -> None:
        """init

        Args:
            ft_len: for STFT
            ft_overlap: for STFT
            ref_chn_idx: the index of the reference channel
            spk_num: the speaker num
            loss_func: loss function. Defaults to mse_batch
            loss_name: the name of loss function, if is None, the name of `loss_func` will be used
        """
        super().__init__(ft_len, ft_overlap, ref_chn_idx, spk_num, loss_func)

    def prepare_input(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, int]:
        """prepare input for network

        Args:
            x: time domain signal, shape [batch, chn, time]

        returns:
            X: STFT cofficients, shape [batch * freq, frame, channel*2]
            Xr: reference channel STFT coefficients of X, shape [batch, freq, frame]
            original_len: the original length (time) of x
        """
        batch_size, chn_num, time = x.shape
        # stft x
        window = self._get_window(x.device)
        x = x.reshape((batch_size * chn_num, time))
        X = torch.stft(x, n_fft=self.ft_len, hop_length=self.ft_overlap, window=window, win_length=self.ft_len, return_complex=True)
        X = X.reshape((batch_size, chn_num, X.shape[-2], X.shape[-1]))  # (batch, channel, freq, time)
        X = X.permute(0, 2, 3, 1)  # (batch, freq, time frame, channel)

        # normalization by using ref_channel
        freq_num, frame_num = X.shape[1], X.shape[2]
        Xr = X[..., self.ref_chn_idx].clone()  # copy
        XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of Xm
        X[:, :, :, :] /= (XrMM.reshape(batch_size, freq_num, 1, 1) + 1e-8)
        X = torch.view_as_real(X).reshape(batch_size * freq_num, frame_num, chn_num * 2)
        return X, Xr, time

    def prepare_target(self, ys: Tensor, input: Any, *args, **kwargs) -> Tensor:
        """prepare target

        Args:
            ys: time domain signal, shape [batch, spk, time]
            input: the output of prepare_input
        returns:
            mrm: magnitude ratio mask, shape [batch, spk, freq, frame], real
        """
        Xr = input[1]  # shape [batch, freq]
        batch_size, spk_num, time = ys.shape

        ys = ys.reshape((batch_size * spk_num, -1))
        window = self._get_window(ys.device)
        Ys = torch.stft(ys, n_fft=self.ft_len, hop_length=self.ft_overlap, window=window, win_length=self.ft_len, return_complex=True)
        Ys = Ys.reshape((batch_size, spk_num, Ys.shape[-2], Ys.shape[-1]))  # (batch, spk, freq, time frame)

        mrm = torch.empty(size=Ys.shape, dtype=torch.float32, device=ys.device)
        Xr_abs = torch.abs(Xr) + 1e-8
        Ys_abs = torch.abs(Ys)
        for spk in range(spk_num):
            mrm[:, spk, :, :] = Ys_abs[:, spk, :, :] / Xr_abs
        mrm = (mrm >= 1) + ((mrm < 1) * mrm)

        return mrm

    def prepare_prediction(self, o: Tensor, input: Any, *args, **kwargs) -> Tensor:
        """prepare prediction from the output of network

        Args:
            o: raw output from network
            input: the output of prepare_input

        returns:
            mrm_hat: magnitude ratio mask, shape [batch, spk, freq, frame], real
        """
        Xr = input[1]  # shape [batch, freq, frame]
        batch_size, freq_num, frame_num = Xr.shape

        mrm_hat = o.reshape(batch_size, freq_num, frame_num, self.spk_num)
        mrm_hat = mrm_hat.permute(0, 3, 1, 2)

        return mrm_hat

    def prepare_time_domain(self, o: Tensor, input: Any, preds: Tensor, *args, **kwargs) -> Tensor:
        """prepare time domain prediction

        Args:
            o: raw output from network
            input: the output of prepare_input
            preds: the output of prepare_prediction

        returns:
            ys_hat: time domain signal, shape [batch, spk, time]
        """
        Xr = input[1]  # shape [batch, freq, frame]
        original_len = input[2]

        mrm_hat = preds
        batch_size, spk_num, freq_num, frame_num = mrm_hat.shape

        # to STFT coefficients
        Ys_hat = torch.empty(size=mrm_hat.shape, dtype=torch.complex64, device=mrm_hat.device)
        for spk in range(self.spk_num):  # Xr_bands * mask
            Ys_hat[:, spk, :, :] = Xr[:, :, :] * mrm_hat[:, spk, :, :]

        # to time domain
        window = self._get_window(mrm_hat.device)
        ys_hat = torch.istft(
            Ys_hat.reshape(batch_size * spk_num, freq_num, frame_num),
            n_fft=self.ft_len,
            hop_length=self.ft_overlap,
            window=window,
            win_length=self.ft_len,
            length=original_len,
        )
        ys_hat = ys_hat.reshape(batch_size, spk_num, ys_hat.shape[1])
        return ys_hat

    def loss(self, preds: Tensor, target: Tensor, reduce_batch: bool = True, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """loss for preds and target

        Args:
            preds: prediction, mrm_hat
            target: target, mrm

        Returns:
            loss value(s), shape [batch] if reduce_batch==False, else a single value
            perms: returned by pit
        """
        losses, perms = pit(preds=preds, target=target, metric_func=self.loss_func, eval_func='min')
        if reduce_batch:
            return losses.mean(), perms
        else:
            return losses, perms
