import numpy as np


def reverberation_time_shortening_window(rir: np.ndarray, original_T60: float, target_T60: float, sr: int = 8000, time_after_max: float = 0.002, time_before_max: float = None) -> np.ndarray:
    """shorten the T60 of a given rir

    Args:
        rir: the rir array
        original_T60: the T60 of the rir
        target_T60: the target T60
        sr: sample rate
        time_after_max: time in seconds after the maximum value in rir taken as part of the direct path. Defaults to 0.002.
        time_before_max: time in seconds before the maximum value in rir taken as part of the direct path. By default, all the values before the maximum are taken as direct path.

    Returns:
        np.ndarray: the reverberation time shortening window
    """

    if original_T60 <= target_T60:
        return np.ones(shape=rir.shape)
    shape = rir.shape
    rir = rir.reshape(-1, shape[-1])
    win = np.empty(shape=rir.shape, dtype=rir.dtype)
    q = 3 / (target_T60 * sr) - 3 / (original_T60 * sr)
    exps = 10**(-q * np.arange(rir.shape[-1]))
    idx_max_array = np.argmax(np.abs(rir), axis=-1)
    for i, idx_max in enumerate(idx_max_array):
        N1 = idx_max + int(time_after_max * sr)
        win[i, :N1] = 1
        win[i, N1:] = exps[:rir.shape[-1] - N1]
        if time_before_max:
            N0 = int(idx_max - time_before_max * sr)
            if N0 > 0:
                win[i, :N0] = 0
    win = win.reshape(shape)
    return win


def rectangular_window(rir: np.ndarray, sr: int = 8000, time_before_after_max: float = 0.002) -> np.ndarray:
    assert rir.ndim == 1, rir.ndim
    idx = int(np.argmax(np.abs(rir)))
    win = np.zeros(shape=rir.shape)
    N = int(sr * time_before_after_max)
    win[max(0, idx - N):idx + N + 1] = 1
    return win


if __name__ == '__main__':
    rir = np.random.rand(3, 2, 10000)
    rir[..., 1000] = 2
    win = reverberation_time_shortening_window(rir, original_T60=0.8, target_T60=0.1, sr=8000)
    print(win.shape)
