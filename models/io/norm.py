from torch import nn
from torch import Tensor
from typing import *

import torch


class Norm(nn.Module):

    def __init__(self, mode: Literal['utterance', 'frequency', 'none']) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, X: Tensor, norm_paras: Any = None, inverse: bool = False) -> Any:
        if not inverse:
            return self.norm(X, norm_paras=norm_paras)
        else:
            return self.inorm(X, norm_paras=norm_paras)

    def norm(self, X: Tensor, norm_paras: Any = None, ref_channel: int = None) -> Tuple[Tensor, Any]:
        """ normalization
        Args:
            X: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization or for the normalization of other X's

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """
        if self.mode == 'none':
            return X, None

        B, C, F, T = X.shape
        if norm_paras is None:
            Xr = X[:, [ref_channel], :, :].clone()  # [B,1,F,T], complex

            if self.mode == 'frequency':
                XrMM = torch.abs(Xr).mean(dim=2, keepdim=True) + 1e-8  # Xr_magnitude_mean, [B,1,F,1]
            else:
                assert self.mode == 'utterance', self.mode
                XrMM = torch.abs(Xr).mean(dim=(1, 2), keepdim=True) + 1e-8  # Xr_magnitude_mean, [B,1,1,1]
        else:
            Xr, XrMM = norm_paras
        X[:, :, :, :] /= XrMM
        return X, (Xr, XrMM)

    def inorm(self, X: Tensor, norm_paras: Any) -> Tensor:
        """ inverse normalization
        Args:
            x: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization 

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """

        Xr, XrMM = norm_paras
        return X * XrMM

    def extra_repr(self) -> str:
        return f"{self.mode}"
