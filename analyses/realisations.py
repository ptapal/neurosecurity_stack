import argparse
import json

import numpy as np

from attack import inject_alpha
from cohort import prep_subject, select_subjects, split_idx
from config import ATTACK_STRENGTH, COHORTS, DATASETS, FS, RESULTS_DIR
from detect import standardize
from features import encode
from run_simulation import run_sigma_sweep

N_SUBJECTS = 15
N_REAL = 3
SEG_S = 0.8
METRICS = ("maha_power", "l2_power", "sig_far", "sig_frr", "epsilon")


def prep_matched(sub, rng):
    n = len(sub.trials)
    calib_i, thresh_i, clean_i, atk_i, test_i = split_idx(n, rng)
    ema_t, tau_t = np.array_split(thresh_i, 2)
    seg = int(round(SEG_S * FS))
    ch, T = sub.trials.shape[1], sub.trials.shape[-1]
    k = T // seg

    def expand(idx):
        return (np.asarray(idx)[:, None] * k + np.arange(k)).ravel()

    segs = sub.trials[:, :, :k * seg].reshape(n, ch, k, seg).transpose(0, 2, 1, 3).reshape(n * k, ch, seg)
    z = encode(segs, FS)
    z_atk = encode(inject_alpha(segs[expand(atk_i)], FS, strength=ATTACK_STRENGTH, rng=rng), FS)

    mu = z[expand(calib_i)].mean(axis=0)
    std = z[expand(calib_i)].std(axis=0) + 1e-8
    return {
        "id": sub.subject_id,
        "z_calib": z[expand(calib_i)],
        "z_thresh_s": standardize(z[expand(thresh_i)], mu, std),
        "z_ema_s": standardize(z[expand(ema_t)], mu, std),
        "z_tau_s": standardize(z[expand(tau_t)], mu, std),
        "z_clean_s": standardize(z[expand(clean_i)], mu, std),
        "z_atk_s": standardize(z_atk, mu, std),
        "offset": mu / std,
    }


def run(name, cache):
    load_fn, list_fn, max_trials = DATASETS[name]
    ids = list_fn()
    out = {"native": [], "matched": []}
    sigma = None
    for r in range(N_REAL):
        chosen = select_subjects(name, ids, N_SUBJECTS, seed=1000 + r)
        for sid in chosen:
            if sid not in cache:
                cache[sid] = load_fn(sid, max_trials=max_trials)
        subs = [cache[sid] for sid in chosen]

        rng = np.random.default_rng(r)
        sweep, _ = run_sigma_sweep([prep_subject(s, rng) for s in subs], rng)
        sigma = [row["sigma"] for row in sweep]
        out["native"].append({m: [row[m] for row in sweep] for m in METRICS})

        rng = np.random.default_rng(r)
        sweep, _ = run_sigma_sweep([prep_matched(s, rng) for s in subs], rng)
        out["matched"].append({m: [row[m] for row in sweep] for m in METRICS})
        print(f"  {name} realisation {r + 1}/{N_REAL}", flush=True)
    return {"sigma": sigma, **out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    args = ap.parse_args()

    out_path = RESULTS_DIR / "realisations.json"
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.cohorts:
        result[name] = run(name, {})
        out_path.write_text(json.dumps(result, indent=2))
