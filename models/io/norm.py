from torch import nn
from torch import Tensor
from typing import *

import torch


def forgetting_normalization(XrMag: Tensor, sliding_window_len: int = 192) -> Tensor:
    # https://github.com/Audio-WestlakeU/FullSubNet/blob/e97448375cd1e883276ad583317b1828318910dc/audio_zen/model/base_model.py#L103C19-L103C19
    alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
    mu = 0
    mu_list = []
    B, _, F, T = XrMag.shape
    XrMM = XrMag.mean(dim=2, keepdim=True).detach().cpu()  # [B,1,1,T]
    for t in range(T):
        if t < sliding_window_len:
            alpha_this = min((t - 1) / (t + 1), alpha)
        else:
            alpha_this = alpha
        mu = alpha_this * mu + (1 - alpha_this) * XrMM[..., t]
        mu_list.append(mu)

    XrMM = torch.stack(mu_list, dim=-1).to(XrMag.device)
    return XrMM


# 雨洁的实现
# def cumulative_normalization(original_signal_mag: Tensor, sliding_window_len: int = 192) -> Tensor:
#     alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
#     eps = 1e-10
#     mu = 0
#     mu_list = []
#     batch_size, frame_num, freq_num = original_signal_mag.shape
#     for frame_idx in range(frame_num):
#         if frame_idx < sliding_window_len:
#             alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
#             mu = alp * mu + (1 - alp) * torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
#         else:
#             current_frame_mu = torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
#             mu = alpha * mu + (1 - alpha) * current_frame_mu
#         mu_list.append(mu)

#     XrMM = torch.stack(mu_list, dim=-1).permute(0, 2, 1).reshape(batch_size, frame_num, 1, 1)
#     return XrMM


class Norm(nn.Module):

    def __init__(self, mode: Optional[Literal['utterance', 'frequency', 'forgetting', 'none']], online: bool = True) -> None:
        super().__init__()
        self.mode = mode
        self.online = online
        assert mode != 'forgetting' or online == True, 'forgetting is one online normalization'

    def forward(self, X: Tensor, norm_paras: Any = None, inverse: bool = False) -> Any:
        if not inverse:
            return self.norm(X, norm_paras=norm_paras)
        else:
            return self.inorm(X, norm_paras=norm_paras)

    def norm(self, X: Tensor, norm_paras: Any = None, ref_channel: int = None, eps: float = 1e-6) -> Tuple[Tensor, Any]:
        """ normalization
        Args:
            X: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization or for the normalization of other X's
            eps: 1e-6!=0 when dtype=float16

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """
        if self.mode == 'none' or self.mode is None:
            Xr = X[:, [ref_channel], :, :].clone()
            return X, (Xr, None)

        B, C, F, T = X.shape
        if norm_paras is None:
            Xr = X[:, [ref_channel], :, :].clone()  # [B,1,F,T], complex

            if self.mode == 'frequency':
                if self.online:
                    XrMM = torch.abs(Xr) + eps  # [B,1,F,T]
                else:
                    XrMM = torch.abs(Xr).mean(dim=3, keepdim=True) + eps  # Xr_magnitude_mean, [B,1,F,1]
            elif self.mode == 'forgetting':
                XrMM = forgetting_normalization(XrMag=torch.abs(Xr)) + eps  # [B,1,1,T]
            else:
                assert self.mode == 'utterance', self.mode
                if self.online:
                    XrMM = torch.abs(Xr).mean(dim=(2,), keepdim=True) + eps  # Xr_magnitude_mean, [B,1,1,T]
                else:
                    XrMM = torch.abs(Xr).mean(dim=(2, 3), keepdim=True) + eps  # Xr_magnitude_mean, [B,1,1,1]
        else:
            Xr, XrMM = norm_paras
        X[:, :, :, :] /= XrMM
        return X, (Xr, XrMM)

    def inorm(self, X: Tensor, norm_paras: Any) -> Tensor:
        """ inverse normalization
        Args:
            X: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization 

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """

        Xr, XrMM = norm_paras
        return X * XrMM

    def extra_repr(self) -> str:
        return f"{self.mode}, online={self.online}"


if __name__ == '__main__':

    # x = torch.randn((10, 1, 129, 251))
    # y1 = forgetting_normalization(x)
    # y2 = cumulative_normalization(x.squeeze().transpose(-1, -2))
    # print((y1.squeeze() == y2.squeeze()).all())
    # print(torch.allclose(y1.squeeze(), y2.squeeze()))

    x = torch.randn((2, 1, 129, 251), dtype=torch.complex64).cuda()
    # norm = Norm('forgetting')
    norm = Norm('utterance', online=False)
    for i in range(10):
        y = norm.norm(x, ref_channel=0)
    import time
    torch.cuda.synchronize()
    ts = time.time()
    for i in range(1):
        y = norm.norm(x, ref_channel=0)
    torch.cuda.synchronize()
    te = time.time()
    print(te - ts)
