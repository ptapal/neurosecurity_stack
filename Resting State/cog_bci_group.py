# -*- coding: utf-8 -*-
"""COG-BCI resting-state EO/EC alpha injection experiment."""

from pathlib import Path
import urllib.request
import zipfile

import mne
import numpy as np
import pandas as pd

from common import DEFAULT_AMP_UV, plot_modality, raw_to_epochs_array, score_epochs_cog

COG_DIR = Path("cog_bci")
COG_SUBJECTS = ["01", "02"]


def download_cog(sid):
    COG_DIR.mkdir(exist_ok=True)
    zip_path = COG_DIR / f"sub-{sid}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1e8:
        print("have", zip_path)
        return zip_path
    url = f"https://zenodo.org/records/6874129/files/sub-{sid}.zip?download=1"
    print("Downloading", url)
    urllib.request.urlretrieve(url, zip_path)
    print(f"saved {zip_path.stat().st_size / 1e9:.2f} GB")
    return zip_path


def extract_resting(zip_path, sid):
    out = COG_DIR / f"sub-{sid}_extract"
    out.mkdir(exist_ok=True)
    tags = ("RS_", "EO", "EC", "eyes", "Eyes", "Rest", "rest")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"sub-{sid}: {len(names)} files in zip; sample .set:")
        sets = [n for n in names if n.endswith(".set")]
        for n in sets[:20]:
            print("  ", n)
        keep = [
            n
            for n in names
            if (n.endswith(".set") or n.endswith(".fdt")) and any(t in n for t in tags)
        ]
        if len([k for k in keep if k.endswith(".set")]) < 2:
            keep = [
                n
                for n in names
                if ("RS" in n or "Rest" in n) and (n.endswith(".set") or n.endswith(".fdt"))
            ]
        print(f"extracting {len(keep)} files")
        for n in keep:
            zf.extract(n, out)
    return out


def modality_from_name(name: str):
    s = Path(name).stem.upper().replace("-", "_").replace(" ", "_")
    if s.endswith("_EC") or "_EC_" in s or "CLOSED" in s:
        return "eyesclosed"
    if s.endswith("_EO") or "_EO_" in s or ("OPEN" in s and "CLOSED" not in s):
        return "eyesopen"
    return None


def download_and_extract_cog():
    for sid in COG_SUBJECTS:
        extract_resting(download_cog(sid), sid)

    print("\nExtracted .set files:")
    for p in sorted(COG_DIR.rglob("*.set")):
        print(" ", p)


def run_cog_bci(save_plots=True, show_plots=False):
    download_and_extract_cog()

    grouped = {}
    for set_path in sorted(COG_DIR.rglob("*.set")):
        mod = modality_from_name(set_path.name)
        if mod is None:
            print("skip", set_path.name)
            continue
        sid = next((p.replace("sub-", "")[:2] for p in set_path.parts if p.startswith("sub-")), "?")
        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose=False)
            X, fs = raw_to_epochs_array(raw, epoch_sec=1.0)
            print(f"loaded {set_path.name}: X={X.shape} fs={fs} -> {mod}")
        except Exception as e:
            print("FAIL load", set_path.name, e)
            continue
        grouped.setdefault((sid, mod), []).append((X, fs))

    cog_rows, cog_examples = [], {}
    for (sid, mod), chunks in sorted(grouped.items()):
        Xs = [c[0] for c in chunks]
        fs = chunks[0][1]
        n_ch = min(x.shape[1] for x in Xs)
        X = np.concatenate([x[:, :n_ch, :] for x in Xs], axis=0)
        try:
            res = score_epochs_cog(X, fs, amp_uv=DEFAULT_AMP_UV)
        except Exception as e:
            print(f"FAIL score sub-{sid} {mod}: {e}")
            continue
        cog_rows.append(
            {
                "dataset": "COG-BCI",
                "subject": sid,
                "modality": mod,
                "n_epochs": res["n_epochs"],
                "n_attacked": res["n_attacked"],
                "detection_pct": res["detection_pct"],
                "far_pct": res["far_pct"],
                "n_chans": res["n_chans"],
                "fs": fs,
            }
        )
        print(
            f"COG sub-{sid} {mod:11s} epochs={res['n_epochs']} "
            f"det={res['detection_pct']:5.1f}% FAR={res['far_pct']:4.1f}%"
        )
        cog_examples[f"{sid}_{mod}"] = res

    df_cog = pd.DataFrame(cog_rows)
    if len(df_cog):
        print("\n=== COG-BCI by modality ===")
        print(
            df_cog.groupby("modality")[["detection_pct", "far_pct", "n_epochs"]]
            .agg({"detection_pct": "mean", "far_pct": "mean", "n_epochs": "sum"})
            .round(2)
        )
        print(df_cog)

        output_dir = Path("results") / "resting_state"
        output_dir.mkdir(parents=True, exist_ok=True)
        df_cog.to_csv(output_dir / "cog_bci_summary.csv", index=False)
        for k, ex in list(cog_examples.items())[:2]:
            output_path = output_dir / f"cog_bci_{k}.png" if save_plots else None
            plot_modality(ex, f"COG-BCI {k}", output_path=output_path, show=show_plots)
    else:
        print("No usable COG-BCI groups. Check the extracted .set listing.")

    return df_cog, cog_examples


if __name__ == "__main__":
    run_cog_bci()

