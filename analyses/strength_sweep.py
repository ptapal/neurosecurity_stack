import argparse
import json

import numpy as np

import cohort
from cohort import prep_subject, select_subjects
from config import COHORTS, DATASETS, RESULTS_DIR
from run_simulation import run_sigma_sweep

N_SUBJECTS = 15
N_REAL = 3
STRENGTHS = [0.25, 0.5, 1.0]


def run(name):
    load_fn, list_fn, max_trials = DATASETS[name]
    ids = list_fn()
    cache = {}
    out = {str(s): [] for s in STRENGTHS}
    out_l2 = {str(s): [] for s in STRENGTHS}
    for s in STRENGTHS:
        cohort.ATTACK_STRENGTH = s
        for r in range(N_REAL):
            chosen = select_subjects(name, ids, N_SUBJECTS, seed=1000 + r)
            for sid in chosen:
                if sid not in cache:
                    cache[sid] = load_fn(sid, max_trials=max_trials)
            rng = np.random.default_rng(r)
            sweep, _ = run_sigma_sweep([prep_subject(cache[sid], rng) for sid in chosen], rng)
            i = [row["sigma"] for row in sweep].index(0.01)
            out[str(s)].append(sweep[i]["maha_power"])
            out_l2[str(s)].append(sweep[i]["l2_power"])
        print(f"  {name} strength {s} done", flush=True)
    return {"maha": out, "l2": out_l2}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    args = ap.parse_args()
    out_path = RESULTS_DIR / "strength_sweep.json"
    result = {}
    for name in args.cohorts:
        result[name] = run(name)
        out_path.write_text(json.dumps(result, indent=2))
