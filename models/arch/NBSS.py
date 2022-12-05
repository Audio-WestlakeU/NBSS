from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

from models.arch.blstm2_fc1 import BLSTM2_FC1
from models.arch.NBC import NBC
from models.arch.NBC2 import NBC2


def neg_si_sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = target.shape[0]
    si_snr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_snr_val.view(batch_size, -1), dim=1)


class NBSS(nn.Module):
    """Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training.

    A module version of NBSS which takes time domain signal as input, and outputs time domain signal.

    Arch could be NB-BLSTM or NBC
    """

    def __init__(
            self,
            n_channel: int = 8,
            n_speaker: int = 2,
            n_fft: int = 512,
            n_overlap: int = 256,
            ref_channel: int = 0,
            arch: str = "NB_BLSTM",  # could also be NBC, NBC2
            arch_kwargs: Dict[str, Any] = dict(),
    ):
        super().__init__()

        if arch == "NB_BLSTM":
            self.arch: nn.Module = BLSTM2_FC1(input_size=n_channel * 2, output_size=n_speaker * 2, **arch_kwargs)
        elif arch == "NBC":
            self.arch = NBC(input_size=n_channel * 2, output_size=n_speaker * 2, **arch_kwargs)
        elif arch == 'NBC2':
            self.arch = NBC2(input_size=n_channel * 2, output_size=n_speaker * 2, **arch_kwargs)
        else:
            raise Exception(f"Unkown arch={arch}")

        self.register_buffer('window', torch.hann_window(n_fft), False)  # self.window, will be moved to self.device at training time
        self.n_fft = n_fft
        self.n_overlap = n_overlap
        self.ref_channel = ref_channel
        self.n_channel = n_channel
        self.n_speaker = n_speaker

    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x: time domain signal, shape [Batch, Channel, Time]

        Returns:
            y: the predicted time domain signal, shape [Batch, Speaker, Time]
        """

        # STFT
        B, C, T = x.shape
        x = x.reshape((B * C, T))
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_overlap, window=self.window, win_length=self.n_fft, return_complex=True)
        X = X.reshape((B, C, X.shape[-2], X.shape[-1]))  # (batch, channel, freq, time frame)
        X = X.permute(0, 2, 3, 1)  # (batch, freq, time frame, channel)

        # normalization by using ref_channel
        F, TF = X.shape[1], X.shape[2]
        Xr = X[..., self.ref_channel].clone()  # copy
        XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of X
        X[:, :, :, :] /= (XrMM.reshape(B, F, 1, 1) + 1e-8)

        # to real
        X = torch.view_as_real(X)  # [B, F, T, C, 2]
        X = X.reshape(B * F, TF, C * 2)

        # network processing
        output = self.arch(X)

        # to complex
        output = output.reshape(B, F, TF, self.n_speaker, 2)
        output = torch.view_as_complex(output)  # [B, F, TF, S]

        # inverse normalization
        Ys_hat = torch.empty(size=(B, self.n_speaker, F, TF), dtype=torch.complex64, device=output.device)
        XrMM = torch.unsqueeze(XrMM, dim=2).expand(-1, -1, TF)
        for spk in range(self.n_speaker):
            Ys_hat[:, spk, :, :] = output[:, :, :, spk] * XrMM[:, :, :]

        # iSTFT with frequency binding
        ys_hat = torch.istft(Ys_hat.reshape(B * self.n_speaker, F, TF), n_fft=self.n_fft, hop_length=self.n_overlap, window=self.window, win_length=self.n_fft, length=T)
        ys_hat = ys_hat.reshape(B, self.n_speaker, T)
        return ys_hat


if __name__ == '__main__':
    x = torch.randn(size=(10, 8, 16000))
    ys = torch.randn(size=(10, 2, 16000))

    NBSS_with_NB_BLSTM = NBSS(n_channel=8, n_speaker=2, arch="NB_BLSTM")
    ys_hat = NBSS_with_NB_BLSTM(x)
    neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    print(ys_hat.shape, neg_sisdr_loss.mean())

    NBSS_with_NBC = NBSS(n_channel=8, n_speaker=2, arch="NBC")
    ys_hat = NBSS_with_NBC(x)
    neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    print(ys_hat.shape, neg_sisdr_loss.mean())

    NBSS_with_NBC_small = NBSS(n_channel=8,
                               n_speaker=2,
                               arch="NBC2",
                               arch_kwargs={
                                   "n_layers": 8, # 12 for large
                                   "dim_hidden": 96, # 192 for large
                                   "dim_ffn": 192, # 384 for large
                                   "block_kwargs": {
                                       'n_heads': 2,
                                       'dropout': 0,
                                       'conv_kernel_size': 3,
                                       'n_conv_groups': 8,
                                       'norms': ("LN", "GBN", "GBN"),
                                       'group_batch_norm_kwargs': {
                                           'group_size': 257, # 129 for 8k Hz
                                           'share_along_sequence_dim': False,
                                       },
                                   }
                               },)
    ys_hat = NBSS_with_NBC_small(x)
    neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    print(ys_hat.shape, neg_sisdr_loss.mean())
