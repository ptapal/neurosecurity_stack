# -*- coding: utf-8 -*-
"""Amplitude sweep for ds004148 sub-01 EO/EC resting-state injection."""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from common import (
    AMP_SWEEP_UV,
    ATTACK_FRACTION,
    THRESHOLD_PERCENTILE,
    extract_band_power,
    infer_uv_scale,
    inject_alpha,
    raw_to_epochs_array,
)
from ds004148_group import DS_DIR, SESSION, TASKS, download_ds004148


def run_amplitude_sweep(save_plots=True, show_plots=False):
    download_ds004148()
    sweep_rows = []
    for task in TASKS:
        vhdr = DS_DIR / f"sub-01/ses-{SESSION}/eeg/sub-01_ses-{SESSION}_task-{task}_eeg.vhdr"
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)
        X, fs = raw_to_epochs_array(raw)

        mask_rng = np.random.default_rng(0)
        base_mask = mask_rng.random(len(X)) < ATTACK_FRACTION
        if base_mask.sum() == 0:
            base_mask[0] = True

        feat_c = extract_band_power(X, fs)
        scaler = StandardScaler()
        Z = scaler.fit_transform(feat_c[~base_mask])
        cov = LedoitWolf().fit(Z)
        thr = np.percentile(np.sqrt(cov.mahalanobis(Z)), THRESHOLD_PERCENTILE)

        for amp in AMP_SWEEP_UV:
            scale = infer_uv_scale(X)
            X_atk = X.copy()
            X_atk[base_mask] = inject_alpha(X[base_mask], fs, amp * scale)
            feat_a = extract_band_power(X_atk, fs)
            scores = np.sqrt(cov.mahalanobis(scaler.transform(feat_a)))
            det = float((scores[base_mask] > thr).mean() * 100)
            far = float((scores[~base_mask] > thr).mean() * 100)
            sweep_rows.append(
                {"modality": task, "amp_uv": amp, "detection_pct": det, "far_pct": far}
            )
            print(f"{task:11s} amp={amp:5.1f} -> det={det:5.1f}% FAR={far:4.1f}%")

    df_sweep = pd.DataFrame(sweep_rows)
    output_dir = Path("results") / "resting_state"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_sweep.to_csv(output_dir / "ds004148_amplitude_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mod, g in df_sweep.groupby("modality"):
        ax.plot(g.amp_uv, g.detection_pct, "o-", label=f"{mod} det")
        ax.plot(g.amp_uv, g.far_pct, "x--", label=f"{mod} FAR")
    ax.set_xlabel("Injection amp (uV)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Stealth curve: EO vs EC")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_plots:
        fig.savefig(output_dir / "ds004148_amplitude_sweep.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    print(df_sweep)
    return df_sweep


if __name__ == "__main__":
    run_amplitude_sweep()

