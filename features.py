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


def encode_pieces(trials, fs, band=ALPHA_BAND, piece_s=1.0):
    """Mean over fixed-length pieces of per-piece log band power, so windows of different length share one estimator."""
    n, ch, T = trials.shape
    p = int(round(piece_s * fs))
    k = T // p
    x = trials[..., :k * p].reshape(n, ch, k, p).transpose(0, 2, 1, 3).reshape(n * k, ch, p)
    lp = np.log(band_power(x, fs, band, nperseg=p) + EPS)
    return lp.reshape(n, k, ch).mean(axis=1).astype(np.float64)
