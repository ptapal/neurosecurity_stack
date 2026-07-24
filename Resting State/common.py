# -*- coding: utf-8 -*-
"""Shared helpers for resting-state alpha injection experiments."""

import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

RNG = np.random.default_rng(42)
ATTACK_FRACTION = 0.30
THRESHOLD_PERCENTILE = 99
EPOCH_SEC = 2.0
INJECTION_FREQ = 10.0
DEFAULT_AMP_UV = 15.0
AMP_SWEEP_UV = [1.0, 3.0, 5.0, 10.0, 15.0, 25.0]


def infer_uv_scale(X):
    """Return native-unit multiplier for a microvolt injection amplitude."""
    return 1.0 if float(np.std(X)) > 1e-2 else 1e-6


def raw_to_epochs_array(raw, epoch_sec=EPOCH_SEC):
    raw = raw.copy()
    raw.pick_types(eeg=True, exclude="bads")
    raw.load_data()
    raw.filter(1.0, 40.0, method="iir", verbose=False)
    data = raw.get_data()
    fs = float(raw.info["sfreq"])
    win = int(epoch_sec * fs)
    n_epochs = data.shape[1] // win
    if n_epochs < 20:
        raise RuntimeError(f"Too few epochs ({n_epochs})")
    X = np.stack([data[:, i * win : (i + 1) * win] for i in range(n_epochs)], axis=0)
    return X, fs


def inject_alpha(X, fs, amp_native, freq=INJECTION_FREQ):
    t = np.arange(X.shape[-1]) / fs
    sine = amp_native * np.sin(2 * np.pi * freq * t)[None, None, :]
    return X + sine


def extract_band_power(X, fs):
    feats = []
    n_fft = min(256, X.shape[-1])
    for trial in X:
        psd, freqs = mne.time_frequency.psd_array_welch(
            trial, sfreq=fs, n_fft=n_fft, verbose=False
        )
        alpha = (freqs >= 8) & (freqs <= 12)
        beta = (freqs >= 13) & (freqs <= 30)
        feats.append(
            np.concatenate(
                [
                    np.log(psd[:, alpha].mean(1) + 1e-10),
                    np.log(psd[:, beta].mean(1) + 1e-10),
                ]
            )
        )
    return np.array(feats)


def score_epochs(X, fs, amp_uv=DEFAULT_AMP_UV, rng=None, min_clean_epochs=None):
    rng = RNG if rng is None else rng
    scale = infer_uv_scale(X)
    amp_native = amp_uv * scale
    attack_mask = rng.random(len(X)) < ATTACK_FRACTION
    if attack_mask.sum() == 0:
        attack_mask[0] = True

    min_clean = min_clean_epochs or max(15, X.shape[1] + 2)
    if (~attack_mask).sum() < min_clean:
        raise RuntimeError("Not enough clean epochs")

    X_atk = X.copy()
    X_atk[attack_mask] = inject_alpha(X[attack_mask], fs, amp_native)

    feat_clean = extract_band_power(X, fs)
    feat_atk = extract_band_power(X_atk, fs)

    scaler = StandardScaler()
    Z_fit = scaler.fit_transform(feat_clean[~attack_mask])
    cov = LedoitWolf().fit(Z_fit)
    thr = np.percentile(np.sqrt(cov.mahalanobis(Z_fit)), THRESHOLD_PERCENTILE)
    scores = np.sqrt(cov.mahalanobis(scaler.transform(feat_atk)))

    return {
        "detection_pct": float((scores[attack_mask] > thr).mean() * 100),
        "far_pct": float((scores[~attack_mask] > thr).mean() * 100),
        "threshold": float(thr),
        "n_epochs": int(len(X)),
        "n_attacked": int(attack_mask.sum()),
        "n_chans": int(X.shape[1]),
        "fs": fs,
        "scale": scale,
        "scores": scores,
        "attack_mask": attack_mask,
        "feat_clean": feat_clean,
        "feat_atk": feat_atk,
        "X": X,
        "X_atk": X_atk,
    }


def score_epochs_cog(X, fs, amp_uv=DEFAULT_AMP_UV):
    """LedoitWolf-safe scorer for COG-BCI when clean trials < n_channels."""
    return score_epochs(X, fs, amp_uv=amp_uv, min_clean_epochs=20)


def mean_psd(X, fs, max_epochs=40):
    psds, freqs = [], None
    for trial in X[:max_epochs]:
        psd, freqs = mne.time_frequency.psd_array_welch(
            trial, sfreq=fs, n_fft=min(256, trial.shape[-1]), verbose=False
        )
        psds.append(psd.mean(0))
    return freqs, np.mean(psds, 0)


def plot_modality(ex, title, output_path=None, show=False):
    fs, n_chans, am = ex["fs"], ex["n_chans"], ex["attack_mask"]
    freqs, psd_c = mean_psd(ex["X"][~am], fs)
    freqs, psd_a = mean_psd(ex["X_atk"][am], fs)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].semilogy(freqs, psd_c, label="Clean")
    axes[0].semilogy(freqs, psd_a, label="Attacked", color="C3")
    axes[0].axvspan(8, 12, color="orange", alpha=0.2)
    axes[0].set_xlim(1, 40)
    axes[0].set_title("PSD")
    axes[0].legend(fontsize=8)

    axes[1].hist(ex["feat_clean"][~am, :n_chans].mean(1), bins=25, alpha=0.7, label="Clean")
    axes[1].hist(
        ex["feat_atk"][am, :n_chans].mean(1),
        bins=25,
        alpha=0.7,
        color="C3",
        label="Attacked",
    )
    axes[1].set_title("Alpha power")
    axes[1].legend()

    n_plot = min(200, len(ex["scores"]))
    idx = np.arange(n_plot)
    axes[2].scatter(
        idx[~am[:n_plot]],
        ex["scores"][:n_plot][~am[:n_plot]],
        s=12,
        label="Clean",
        alpha=0.7,
    )
    axes[2].scatter(
        idx[am[:n_plot]],
        ex["scores"][:n_plot][am[:n_plot]],
        s=12,
        c="C3",
        label="Attacked",
        alpha=0.7,
    )
    axes[2].axhline(ex["threshold"], color="C3", ls="--")
    axes[2].set_title("Mahalanobis")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"{title} | det={ex['detection_pct']:.0f}% FAR={ex['far_pct']:.0f}%")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

