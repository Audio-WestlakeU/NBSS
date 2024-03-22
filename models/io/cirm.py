import numpy as np
import torch

EPSILON = np.finfo(np.float32).eps


def build_complex_ideal_ratio_mask(noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Build the complex ratio mask.

    Args:
        noisy: [..., F, T], noisy complex-valued stft coefficients
        clean: [..., F, T], clean complex-valued stft coefficients

    References:
        https://ieeexplore.ieee.org/document/7364200

    Returns:
        [..., F, T, 2]
    """
    noisy_real, noisy_imag = noisy.real, noisy.imag
    clean_real, clean_imag = clean.real, clean.imag

    denominator = torch.square(noisy_real) + torch.square(noisy_imag) + EPSILON

    mask_real = (noisy_real * clean_real + noisy_imag * clean_imag) / denominator
    mask_imag = (noisy_real * clean_imag - noisy_imag * clean_real) / denominator

    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=-1)

    cirm = compress_cIRM(complex_ratio_mask, K=10, C=0.1)
    cirm = torch.view_as_complex(cirm)
    return cirm


def compress_cIRM(mask, K=10, C=0.1):
    """Compress the value of cIRM from (-inf, +inf) to [-K ~ K].

    References:
        https://ieeexplore.ieee.org/document/7364200
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask


def decompress_cIRM(mask, K=10, limit=9.9):
    """Decompress cIRM from [-K ~ K] to [-inf, +inf].

    Args:
        mask: cIRM mask
        K: default 10
        limit: default 0.1

    References:
        https://ieeexplore.ieee.org/document/7364200
    """
    mask = torch.view_as_real(mask)
    mask = (limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit))
    mask = -K * torch.log((K - mask) / (K + mask))
    return torch.view_as_complex(mask)


def complex_mul(noisy_r, noisy_i, mask_r, mask_i):
    r = noisy_r * mask_r - noisy_i * mask_i
    i = noisy_r * mask_i + noisy_i * mask_r
    return r, i


def complex_mul_v2(noisy, mask):
    return noisy * mask
