import argparse
import json

import numpy as np

from analyses.ablation_fk_sweep import make_gate_fn
from analyses.adaptive_poison import N_BAD, N_DRAWS, summarise, test_set
from cohort import build_clients, prep_dataset
from config import COHORTS, DELTA, P1_BATCH_SIZE, P1_LOCAL_STEPS, P1_ROUNDS, P1_SIGMA_DEMO, RESULTS_DIR
from fedprox import run_dp_fedprox_eeg

S_GRID = np.linspace(0.0, 3.0, 61)


def adaptive_attack(gate):
    st = gate.state
    scales = []

    def attack(client_id, round_i, w_local, w_glob):
        if not st["initialized"]:
            return w_local
        d = w_local - w_glob
        mu_k = st["client_mu"].get(client_id, st["global_mu0"])
        best = 0.0
        for s in S_GRID:
            diff = -s * d - mu_k
            if float(diff @ st["cov_inv"] @ diff) <= st["tau"]:
                best = s
        scales.append(best)
        return w_glob - best * d

    attack.scales = scales
    return attack


def run(clients, X, y, clip, bad_ids, seed):
    gate = make_gate_fn()
    attack = adaptive_attack(gate)
    counts = {"bad_drop": 0, "bad_total": 0, "hon_drop": 0, "hon_total": 0}

    def wrapped(client_id, round_i, w_local, w_glob):
        flagged = gate(client_id, round_i, w_local, w_glob)
        if round_i > 5:
            key = "bad" if client_id in bad_ids else "hon"
            counts[f"{key}_total"] += 1
            counts[f"{key}_drop"] += int(flagged)
        return flagged

    out = run_dp_fedprox_eeg(clients, X, y, clip_norm_base=clip, sigma=P1_SIGMA_DEMO, delta=DELTA,
                             rounds=P1_ROUNDS, local_steps=P1_LOCAL_STEPS, batch_size=P1_BATCH_SIZE,
                             rng=np.random.default_rng(seed), compromised_ids=bad_ids,
                             gate_fn=wrapped, poison_mode="none", attack_fn=attack)
    return out["history"][-1].global_acc, counts, attack.scales


def clean_run(clients, X, y, clip, seed):
    out = run_dp_fedprox_eeg(clients, X, y, clip_norm_base=clip, sigma=P1_SIGMA_DEMO, delta=DELTA,
                             rounds=P1_ROUNDS, local_steps=P1_LOCAL_STEPS, batch_size=P1_BATCH_SIZE,
                             rng=np.random.default_rng(seed), compromised_ids=set(), poison_mode="none")
    return out["history"][-1].global_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    args = ap.parse_args()

    out_path = RESULTS_DIR / "adaptive_gate.json"
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.cohorts:
        prep, clip = prep_dataset(name)
        clients = build_clients(prep, clip)
        X, y = test_set(prep, clip)
        ids = [c.client_id for c in clients]
        clean_acc = [clean_run(clients, X, y, clip, d) for d in range(N_DRAWS)]
        res = {"clean_acc": summarise(clean_acc), "cells": {}}
        for n_bad in N_BAD:
            gaps, recalls, fprs, scales = [], [], [], []
            for d in range(N_DRAWS):
                rng = np.random.default_rng(100 + d)
                bad = set(rng.choice(ids, size=n_bad, replace=False))
                acc, cnt, sc = run(clients, X, y, clip, bad, d)
                gaps.append(clean_acc[d] - acc)
                recalls.append(cnt["bad_drop"] / max(cnt["bad_total"], 1))
                fprs.append(cnt["hon_drop"] / max(cnt["hon_total"], 1))
                scales.append(float(np.mean(sc)) if sc else 0.0)
            key = f"n_bad={n_bad}|adaptive|gate"
            res["cells"][key] = {"gap": summarise(gaps), "recall": summarise(recalls),
                                 "fpr": summarise(fprs), "mean_scale": summarise(scales)}
            print(name, key, {k: round(v["mean"], 3) for k, v in res["cells"][key].items()}, flush=True)
        result[name] = res
        out_path.write_text(json.dumps(result, indent=2))
