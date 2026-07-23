import numpy as np
from scipy.signal import welch

ALPHA_BAND = (8.0, 12.0)
EPS = 1e-12


def band_power(trials, fs, band, nperseg=256):
    n = trials.shape[-1]
    seg = min(nperseg, n)
    freqs, psd = welch(trials, fs=fs, nperseg=seg, noverlap=seg // 2, axis=-1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        i = np.argmin(np.abs(freqs - np.mean(band)))
        mask = np.zeros_like(freqs, dtype=bool)
        mask[i] = True
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def encode(trials, fs, band=ALPHA_BAND):
    p = band_power(trials, fs, band)
    return np.log(p + EPS).astype(np.float64)
