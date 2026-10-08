import argparse
import json

import numpy as np

from analyses.ablation_fk_sweep import make_gate_fn
from cohort import build_clients, prep_dataset
from config import COHORTS, DELTA, P1_BATCH_SIZE, P1_LOCAL_STEPS, P1_ROUNDS, P1_SIGMA_DEMO, RESULTS_DIR
from fedprox import run_dp_fedprox_eeg
from model import augment
from privacy import clip_l2
from detect import standardize

N_BAD = [4, 10]
N_DRAWS = 3


def test_set(prep, clip):
    X = np.concatenate([augment(clip_l2(standardize(p["z"], p["mu"], p["std"]), clip)[p["test_i"]])
                        for p in prep])
    y = np.concatenate([p["y"][p["test_i"]] for p in prep]).astype(np.float64)
    return X, y


def run(clients, X, y, clip, bad_ids, seed, match_norm, use_gate):
    gate = make_gate_fn() if use_gate else None
    counts = {"bad_drop": 0, "bad_total": 0, "hon_drop": 0, "hon_total": 0}

    def wrapped(client_id, round_i, w_local, w_glob):
        flagged = gate(client_id, round_i, w_local, w_glob) if gate else False
        if round_i > 5:
            key = "bad" if client_id in bad_ids else "hon"
            counts[f"{key}_total"] += 1
            counts[f"{key}_drop"] += int(flagged)
        return flagged

    out = run_dp_fedprox_eeg(clients, X, y, clip_norm_base=clip, sigma=P1_SIGMA_DEMO, delta=DELTA,
                             rounds=P1_ROUNDS, local_steps=P1_LOCAL_STEPS, batch_size=P1_BATCH_SIZE,
                             rng=np.random.default_rng(seed), compromised_ids=bad_ids,
                             gate_fn=wrapped if use_gate else None, poison_mode="gradient",
                             match_norm=match_norm)
    return out["history"][-1].global_acc, counts


def summarise(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    args = ap.parse_args()

    out_path = RESULTS_DIR / "adaptive_poison.json"
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.cohorts:
        prep, clip = prep_dataset(name)
        clients = build_clients(prep, clip)
        X, y = test_set(prep, clip)
        ids = [c.client_id for c in clients]
        clean_acc = [run(clients, X, y, clip, set(), d, False, False)[0] for d in range(N_DRAWS)]
        res = {"clean_acc": summarise(clean_acc), "cells": {}}
        for n_bad in N_BAD:
            for attack, match in [("flip", False), ("flip_matched", True)]:
                for use_gate in [False, True]:
                    gaps, recalls, fprs = [], [], []
                    for d in range(N_DRAWS):
                        rng = np.random.default_rng(100 + d)
                        bad = set(rng.choice(ids, size=n_bad, replace=False))
                        acc, cnt = run(clients, X, y, clip, bad, d, match, use_gate)
                        gaps.append(clean_acc[d] - acc)
                        if use_gate:
                            recalls.append(cnt["bad_drop"] / max(cnt["bad_total"], 1))
                            fprs.append(cnt["hon_drop"] / max(cnt["hon_total"], 1))
                    key = f"n_bad={n_bad}|{attack}|{'gate' if use_gate else 'nogate'}"
                    res["cells"][key] = {"gap": summarise(gaps),
                                         "recall": summarise(recalls) if use_gate else None,
                                         "fpr": summarise(fprs) if use_gate else None}
                    print(name, key, {k: round(v, 3) for k, v in res["cells"][key]["gap"].items()}, flush=True)
        result[name] = res
        out_path.write_text(json.dumps(result, indent=2))
