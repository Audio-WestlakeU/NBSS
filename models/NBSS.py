from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class NBSS(nn.Module):
    """Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training
    """

    def __init__(
            self,
            n_channel: int = 8,
            n_speaker: int = 2,
            hidden_dims: Tuple[int, int] = (256, 128),
            n_fft: int = 512,
            n_overlap: int = 256,
            ref_channel: int = 0,
    ):
        super().__init__()

        self.blstm1 = nn.LSTM(input_size=n_channel * 2, hidden_size=hidden_dims[0], batch_first=True, bidirectional=True)  # type:ignore
        self.blstm2 = nn.LSTM(input_size=hidden_dims[0] * 2, hidden_size=hidden_dims[1], batch_first=True, bidirectional=True)  # type:ignore
        self.linear = nn.Linear(hidden_dims[1] * 2, n_speaker * 2)  # type:ignore

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
        XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of Xm
        X[:, :, :, :] /= (XrMM.reshape(B, F, 1, 1) + 1e-8)

        # to real
        X = torch.view_as_real(X)  # [B, F, T, C, 2]
        X = X.reshape(B * F, TF, C * 2)

        # network processing
        X, _ = self.blstm1(X)
        X, _ = self.blstm2(X)
        output = self.linear(X)

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
    m = NBSS()
    ys_hat = m(x)
    print(ys_hat.shape)
