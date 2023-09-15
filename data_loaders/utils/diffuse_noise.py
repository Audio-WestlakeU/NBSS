#############################################################################################
# Translated from https://github.com/ehabets/ANF-Generator. Implementation of the method in
#
# Generating Nonstationary Multisensor Signals under a Spatial Coherence Constraint.
# Habets, Emanuël A. P. and Cohen, Israel and Gannot, Sharon
#
# Note: Though the generated noise is diffuse, but it doesn't simulate the reverberation of rooms
#
# Copyright: Changsheng Quan @ Audio Lab of Westlake University
#############################################################################################

import math

import numpy as np
import scipy
from scipy.signal import stft, istft


def gen_desired_spatial_coherence(pos_mics: np.ndarray, fs: int, noise_field: str = 'spherical', c: float = 343.0, nfft: int = 256) -> np.ndarray:
    """generate desired spatial coherence for one array

    Args:
        pos_mics: microphone positions, shape (num_mics, 3)
        fs: sampling frequency
        noise_field: 'spherical' or 'cylindrical'
        c: sound velocity
        nfft: points of fft

    Raises:
        Exception: Unknown noise field if noise_field != 'spherical' and != 'cylindrical'

    Returns:
        np.ndarray: desired spatial coherence, shape [num_mics, num_mics, num_freqs]
        np.ndarray: desired mixing matrices, shape [num_freqs, num_mics, num_mics]


    Reference:  E. A. P. Habets, “Arbitrary noise field generator.” https://github.com/ehabets/ANF-Generator
    """
    assert pos_mics.shape[1] == 3, pos_mics.shape
    M = pos_mics.shape[0]
    num_freqs = nfft // 2 + 1

    # compute desired spatial coherence matric
    ww = 2 * math.pi * fs * np.array(list(range(num_freqs))) / nfft
    dist = np.linalg.norm(pos_mics[:, np.newaxis, :] - pos_mics[np.newaxis, :, :], axis=-1, keepdims=True)
    if noise_field == 'spherical':
        DSC = np.sinc(ww * dist / (c * math.pi))
    elif noise_field == 'cylindrical':
        DSC = scipy.special(0, ww * dist / c)
    else:
        raise Exception('Unknown noise field')

    # compute mixing matrices of the desired spatial coherence matric
    Cs = np.zeros((num_freqs, M, M), dtype=np.complex128)
    for k in range(1, num_freqs):
        D, V = np.linalg.eig(DSC[:, :, k])
        C = V.T * np.sqrt(D)[:, np.newaxis]
        # C = scipy.linalg.cholesky(DSC[:, :, k])
        Cs[k, ...] = C

    return DSC, Cs


def gen_diffuse_noise(noise: np.ndarray, L: int, Cs: np.ndarray, nfft: int = 256, rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """generate diffuse noise with the mixing matrice of desired spatial coherence

    Args:
        noise: at least `num_mic*L` samples long
        L: the length in samples
        Cs: mixing matrices, shape [num_freqs, num_mics, num_mics]
        nfft: the number of fft points
        rng: random number generator used for reproducibility

    Returns:
        np.ndarray: multi-channel diffuse noise, shape [num_mics, L]
    """

    M = Cs.shape[-1]
    assert noise.shape[-1] >= M * L, noise.shape

    # Generate M mutually 'independent' input signals
    # noise = noise - np.mean(noise)
    assert noise.shape[-1] >= M * L, ("The noise signal should be at least `num_mic*L` samples long", noise.shape, M, L)
    start = rng.integers(low=0, high=noise.shape[-1] - M * L + 1)
    noise = noise[start:start + M * L].reshape(M, L)
    noise = noise - np.mean(noise, axis=-1, keepdims=True)
    f, t, N = stft(noise, window='hann', nperseg=nfft, noverlap=0.75 * nfft, nfft=nfft)  # N: [M,F,T]
    # Generate output in the STFT domain for each frequency bin k
    X = np.einsum('fmn,mft->nft', np.conj(Cs), N)
    # Compute inverse STFT
    F, x = istft(X, window='hann', nperseg=nfft, noverlap=0.75 * nfft, nfft=nfft)
    x = x[:, :L]
    return x  # [M, L]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import soundfile as sf
    from pathlib import Path
    # pos_mic = np.random.randn(8, 3)
    # pos_mic = np.array([[0, 0, 1.5], [0, 0.2, 1.5]])
    nfft = 1024
    num_mics = 3
    pos_mics = [[0, 0, 1.5]]
    for i in range(1, num_mics):
        pos_mics.append([0, 0.3 * i, 1.5])
    pos_mics = np.array(pos_mics)
    DSC, Cs = gen_desired_spatial_coherence(pos_mics=pos_mics, fs=8000, noise_field='spherical', nfft=nfft)

    # wav_files = Path('dataset/datasets/fma_small/000').rglob('*.mp3')
    # noise = np.concatenate([sf.read(wav_file,always_2d=True)[0][:,0] for wav_file in wav_files])
    noise = np.random.randn(8000 * 22 * 8)
    x = gen_diffuse_noise(noise=noise, T=20, fs=8000, Cs=Cs, nfft=nfft)

    f, t, X = stft(x, window='hann', nperseg=nfft, noverlap=0.75 * nfft, nfft=nfft)  # X: [M,F,T]
    cross_psd_0 = np.mean(X[[0], :, :] * np.conj(X[1:, :, :]), axis=-1)
    cross_psd_1 = np.mean(np.abs(X[[0], :, :])**2, axis=-1) * np.mean(np.abs(X[1:, :, :])**2, axis=-1)
    cross_psd = cross_psd_0 / np.sqrt(cross_psd_1)
    sc_generated = np.real(cross_psd)  # 实部是关于f的偶函数。此处只绘制实部是因为虚部的值小？因为前面的spatial conherence是实数？

    ww = 2 * math.pi * 8000 * np.array(list(range(nfft // 2 + 1))) / nfft
    if num_mics > 2:
        dist = np.linalg.norm(pos_mics[1:] - pos_mics[[0], ...], axis=-1, keepdims=True)
    else:
        dist = np.linalg.norm(pos_mics[[1]] - pos_mics[[0], ...], axis=-1, keepdims=True)
    sc_theory = np.sinc(ww * dist / (343 * math.pi))

    for i in range(len(sc_theory)):
        plt.plot(list(range(nfft // 2 + 1)), sc_theory[i])
        plt.plot(list(range(nfft // 2 + 1)), sc_generated[i])
        plt.title(f"Chn {i+2} vs. Chn {1}")
        plt.show()
