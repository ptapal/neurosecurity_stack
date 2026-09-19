"""Sensitivity of the Pillar 2 and 3 sigma sweep to which 15 subjects are drawn.

Each cohort with more than 15 subjects is redrawn n_draws times with seeds 1000 + d, and the sweep is
rerun on every draw. Zhang has exactly 15 subjects, so it has no alternative draw. Needs the full datasets.

Usage: python -m analyses.subsample_sensitivity [--draws 10] [--cohorts cho lee ...]
Writes results/subsample_sensitivity.json.
"""
import argparse
import json

import numpy as np

from cohort import prep_subject, select_subjects
from config import COHORTS, DATASETS, RESULTS_DIR
from run_simulation import run_pillars_2_3

N_SUBJECTS = 15
METRICS = ("epsilon", "maha_power", "l2_power", "sig_far", "sig_frr")


def run(name, n_draws):
    load_fn, list_fn, max_trials = DATASETS[name]
    ids = list_fn()
    if len(ids) <= N_SUBJECTS:
        return None
    cache, draws = {}, []
    for d in range(n_draws):
        chosen = select_subjects(name, ids, N_SUBJECTS, seed=1000 + d)
        for sid in chosen:
            if sid not in cache:
                cache[sid] = load_fn(sid, max_trials=max_trials)
        rng = np.random.default_rng(d)
        prep = [prep_subject(cache[sid], rng) for sid in chosen]
        sweep, _ = run_pillars_2_3(prep, rng)
        draws.append({"subjects": chosen, **{m: [row[m] for row in sweep] for m in METRICS}})
        print(f"  {name} draw {d + 1}/{n_draws}", flush=True)
    return {"sigma": [row["sigma"] for row in sweep], "draws": draws}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    args = ap.parse_args()

    out_path = RESULTS_DIR / "subsample_sensitivity.json"
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.cohorts:
        res = run(name, args.draws)
        if res is None:
            print(f"  {name}: 15 subjects or fewer, no alternative draw")
            continue
        result[name] = res
        out_path.write_text(json.dumps(result, indent=2))
