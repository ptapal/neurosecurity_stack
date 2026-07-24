# -*- coding: utf-8 -*-
"""OpenNeuro ds004148 resting-state EO/EC alpha injection experiment."""

from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from common import DEFAULT_AMP_UV, plot_modality, raw_to_epochs_array, score_epochs

DS_DIR = Path("ds004148")
SUBJECTS = ["01", "02", "03"]
SESSION = "session1"
TASKS = ["eyesopen", "eyesclosed"]
S3 = "https://s3.amazonaws.com/openneuro.org/ds004148"

SUFFIXES = [
    "_eeg.vhdr",
    "_eeg.vmrk",
    "_eeg.eeg",
    "_eeg.json",
    "_channels.tsv",
    "_events.tsv",
]


def download(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print("GET", dest.name)
    urllib.request.urlretrieve(url, dest)


def download_ds004148():
    download(f"{S3}/dataset_description.json", DS_DIR / "dataset_description.json")
    for sid in SUBJECTS:
        eeg_dir = DS_DIR / f"sub-{sid}" / f"ses-{SESSION}" / "eeg"
        elec = f"sub-{sid}_ses-{SESSION}_electrodes.tsv"
        download(f"{S3}/sub-{sid}/ses-{SESSION}/eeg/{elec}", eeg_dir / elec)
        for task in TASKS:
            stem = f"sub-{sid}_ses-{SESSION}_task-{task}"
            for suf in SUFFIXES:
                fname = stem + suf
                download(f"{S3}/sub-{sid}/ses-{SESSION}/eeg/{fname}", eeg_dir / fname)


def run_ds004148(save_plots=True, show_plots=False):
    download_ds004148()
    print("Downloaded vhdr files:")
    for p in sorted(DS_DIR.rglob("*.vhdr")):
        print(" ", p)

    rows, examples = [], {}
    for sid in SUBJECTS:
        for task in TASKS:
            vhdr = (
                DS_DIR
                / f"sub-{sid}/ses-{SESSION}/eeg/sub-{sid}_ses-{SESSION}_task-{task}_eeg.vhdr"
            )
            raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)
            X, fs = raw_to_epochs_array(raw)
            res = score_epochs(X, fs, amp_uv=DEFAULT_AMP_UV)
            rows.append(
                {
                    "dataset": "ds004148",
                    "subject": sid,
                    "modality": task,
                    "n_epochs": res["n_epochs"],
                    "n_attacked": res["n_attacked"],
                    "detection_pct": res["detection_pct"],
                    "far_pct": res["far_pct"],
                    "fs": fs,
                    "n_chans": res["n_chans"],
                    "scale": res["scale"],
                }
            )
            print(
                f"sub-{sid} {task:11s} det={res['detection_pct']:5.1f}% "
                f"FAR={res['far_pct']:4.1f}% epochs={res['n_epochs']} fs={fs}"
            )
            examples[f"{sid}_{task}"] = res

    df_ds = pd.DataFrame(rows)
    print("\n=== SUMMARY by modality @ 15 uV ===")
    print(
        df_ds.groupby("modality")[["detection_pct", "far_pct", "n_epochs"]]
        .agg({"detection_pct": "mean", "far_pct": "mean", "n_epochs": "sum"})
        .round(2)
    )
    print(df_ds)

    output_dir = Path("results") / "resting_state"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_ds.to_csv(output_dir / "ds004148_summary.csv", index=False)

    for task in TASKS:
        output_path = output_dir / f"ds004148_sub-01_{task}.png" if save_plots else None
        plot_modality(
            examples[f"01_{task}"],
            f"ds004148 sub-01 {task}",
            output_path=output_path,
            show=show_plots,
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(SUBJECTS))
    w = 0.35
    for i, mod in enumerate(TASKS):
        g = df_ds[df_ds.modality == mod].set_index("subject").loc[SUBJECTS]
        ax.bar(x + i * w, g["detection_pct"], width=w, label=f"{mod} det")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([f"sub-{s}" for s in SUBJECTS])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Detection %")
    ax.set_title("ds004148: EO vs EC @ 15 uV (never pooled)")
    ax.legend()
    plt.tight_layout()
    if save_plots:
        fig.savefig(output_dir / "ds004148_eo_vs_ec.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    return df_ds, examples


if __name__ == "__main__":
    run_ds004148()

