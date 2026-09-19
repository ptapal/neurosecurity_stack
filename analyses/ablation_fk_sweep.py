"""Sweep the compromised-client fraction f/K for the crafted-gradient poisoning attack and its Pillar-3 gate.

Reports the final-round accuracy gap (clean vs. poisoned-ungated), the gating effect
(ungated vs. gated), and gate precision/recall, over 3 compromised-client draws per point.
"""
import json
import time

import numpy as np

from cohort import build_clients, prep_dataset
from config import (ALPHA0, COHORTS, DELTA, EMA_GAMMA, P1_BATCH_SIZE, P1_LOCAL_STEPS,
                          P1_ROUNDS, P1_SIGMA_DEMO, RESULTS_DIR, RIDGE)
from detect import empirical_threshold, standardize
from fedprox import run_dp_fedprox_eeg
from model import augment
from privacy import clip_l2

N_BAD_LEVELS = [1, 4, 7, 10]
N_REPEATS = 3
GATE_WARMUP_ROUNDS = 5


def make_gate_fn(warmup_rounds=GATE_WARMUP_ROUNDS, alpha0=ALPHA0, ridge=RIDGE, ema_gamma=EMA_GAMMA):
    """Gate scored on the client's submitted update (w_local - w_glob): per-client EMA mean, covariance
    pooled from a warmup period, mean updated only on rounds that pass the check."""
    state = {"warmup_deltas": [], "client_mu": {}, "cov_inv": None, "tau": None,
             "global_mu0": None, "initialized": False}

    def gate_fn(client_id, round_i, w_local, w_glob):
        delta = w_local - w_glob
        if round_i <= warmup_rounds:
            state["warmup_deltas"].append(delta.copy())
            return False

        if not state["initialized"]:
            pool = np.stack(state["warmup_deltas"])
            mu0 = pool.mean(axis=0)
            cov = np.cov(pool, rowvar=False) + ridge * np.eye(pool.shape[1])
            cov_inv = np.linalg.inv(cov)
            diffs = pool - mu0
            calib_scores = np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)
            state["cov_inv"] = cov_inv
            state["tau"] = empirical_threshold(calib_scores, alpha0)
            state["global_mu0"] = mu0
            state["initialized"] = True

        mu_k = state["client_mu"].get(client_id, state["global_mu0"])
        diff = delta - mu_k
        score = float(diff @ state["cov_inv"] @ diff)
        flagged = bool(score > state["tau"])
        if not flagged:
            state["client_mu"][client_id] = (1.0 - ema_gamma) * mu_k + ema_gamma * delta
        return flagged

    return gate_fn


def final_acc(prep, clip, X_test, y_test, bad_ids, seed):
    kw = dict(X_test_aug=X_test, y_test=y_test, clip_norm_base=clip,
              sigma=P1_SIGMA_DEMO, delta=DELTA, rounds=P1_ROUNDS,
              local_steps=P1_LOCAL_STEPS, batch_size=P1_BATCH_SIZE)
    clients = build_clients(prep, clip)
    out_ng = run_dp_fedprox_eeg(clients, rng=np.random.default_rng(seed),
                                compromised_ids=bad_ids, poison_mode="gradient", **kw)

    raw_gate = make_gate_fn()
    decisions = []

    def logging_gate_fn(client_id, round_i, w_local, w_glob):
        flagged = raw_gate(client_id, round_i, w_local, w_glob)
        if round_i > GATE_WARMUP_ROUNDS:
            decisions.append((client_id in bad_ids, flagged))
        return flagged

    out_g = run_dp_fedprox_eeg(clients, rng=np.random.default_rng(seed),
                               compromised_ids=bad_ids, gate_fn=logging_gate_fn, poison_mode="gradient", **kw)
    n_gated_total = sum(h.n_gated_out for h in out_g["history"])

    tp = sum(1 for is_bad, flagged in decisions if is_bad and flagged)
    fp = sum(1 for is_bad, flagged in decisions if not is_bad and flagged)
    n_bad_decisions = sum(1 for is_bad, _ in decisions if is_bad)
    n_flagged = tp + fp
    precision = tp / n_flagged if n_flagged else float("nan")
    recall = tp / n_bad_decisions if n_bad_decisions else float("nan")

    return out_ng["history"][-1].global_acc, out_g["history"][-1].global_acc, n_gated_total, precision, recall


def run_for_dataset(name):
    prep, clip = prep_dataset(name)
    clients = build_clients(prep, clip)
    X_test = np.concatenate([augment(clip_l2(standardize(p["z"], p["mu"], p["std"]), clip)[p["test_i"]])
                             for p in prep])
    y_test = np.concatenate([p["y"][p["test_i"]] for p in prep]).astype(np.float64)

    kw = dict(X_test_aug=X_test, y_test=y_test, clip_norm_base=clip,
              sigma=P1_SIGMA_DEMO, delta=DELTA, rounds=P1_ROUNDS,
              local_steps=P1_LOCAL_STEPS, batch_size=P1_BATCH_SIZE)
    out_clean = run_dp_fedprox_eeg(clients, rng=np.random.default_rng(2), **kw)
    clean_acc = out_clean["history"][-1].global_acc

    k = len(clients)
    points = []
    for n_bad in N_BAD_LEVELS:
        gaps, gate_effects, n_gated_list, precisions, recalls = [], [], [], [], []
        for draw in range(N_REPEATS):
            draw_rng = np.random.default_rng(hash((name, n_bad, draw)) % (2**32))
            bad_ids = set(draw_rng.choice([c.client_id for c in clients], size=n_bad, replace=False))
            acc_ng, acc_g, n_gated, prec, rec = final_acc(prep, clip, X_test, y_test, bad_ids,
                                                          seed=hash((name, n_bad, draw, "train")) % (2**32))
            gaps.append(clean_acc - acc_ng)
            gate_effects.append(acc_ng - acc_g)
            n_gated_list.append(n_gated)
            precisions.append(prec)
            recalls.append(rec)
        points.append({
            "n_bad": n_bad, "frac": round(n_bad / k, 4),
            "gap_mean": float(np.mean(gaps)), "gap_std": float(np.std(gaps)),
            "gate_effect_mean": float(np.mean(gate_effects)), "gate_effect_std": float(np.std(gate_effects)),
            "n_gated_total_mean": float(np.mean(n_gated_list)),
            "gate_precision_mean": float(np.nanmean(precisions)),
            "gate_recall_mean": float(np.nanmean(recalls)),
        })
    return {"dataset": name, "clean_acc": clean_acc, "n_clients": k, "points": points}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}
    for name in COHORTS:
        t0 = time.time()
        print(f"running f/K sweep for {name} ...")
        all_results[name] = run_for_dataset(name)
        print(f"  {name} done in {time.time()-t0:.1f}s")
    out = RESULTS_DIR / "pillar1_fk_sweep_v2.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
