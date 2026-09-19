"""Pillar-1 convergence check for Theorem 6.1, one cohort per invocation.

Trains from scratch per sigma with eta_0 = min(1/(mu+beta), 1/(4*beta*E)) and group-corrected clipping
tilde_C = batch_size * C. Existing results/<cohort>.json runs used the earlier step size and clip norm, so
they are read only for the old_* comparison columns. Also checks Assumption A1 (clipping never binds) at tilde_C.
Writes results/<cohort>_pillar1_v2.json.
"""
import json
import sys
import time

import numpy as np

from cohort import build_clients, prep_dataset
from config import (DELTA, P1_BATCH_SIZE, P1_LOCAL_STEPS, P1_ROUNDS, P1_SIGMA_DEMO,
                          P1_SIGMAS_CONV, RESULTS_DIR)
from detect import standardize
from fedprox import convergence_bound_rhs, estimate_nu2, local_optimum, run_dp_fedprox_eeg
from model import augment, loss as model_loss, per_sample_grad, smoothness_beta


def local_dp_sgd_instrumented(w, w_glob, Xa, y, clip, sigma, mu, lr, steps, batch, rng, stats):
    n = Xa.shape[0]
    for _ in range(steps):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        g = per_sample_grad(w, Xa[idx], y[idx])
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        stats["n_samples"] += norms.shape[0]
        stats["n_clipped"] += int(np.sum(norms.ravel() > clip))
        stats["max_norm"] = max(stats["max_norm"], float(norms.max()))
        g_clip = g * np.minimum(1.0, clip / np.maximum(norms, 1e-12))
        noise = rng.normal(0.0, sigma * clip, size=w.shape) / g_clip.shape[0]
        g_dp = g_clip.mean(axis=0) + noise
        prox = mu * (w - w_glob)
        w = w - lr * (g_dp + prox)
    return w


def check_assumption_a1(clients, tilde_C, sigma=P1_SIGMA_DEMO, rounds=P1_ROUNDS,
                        local_steps=P1_LOCAL_STEPS, batch_size=P1_BATCH_SIZE, seed=1):
    d = clients[0].X_clean.shape[1]
    beta = smoothness_beta(np.concatenate([c.X_clean for c in clients], axis=0))
    mu = beta
    eta0 = min(1.0 / (mu + beta), 1.0 / (4.0 * beta * local_steps))
    rng = np.random.default_rng(seed)
    stats = {"n_samples": 0, "n_clipped": 0, "max_norm": 0.0}
    w = np.zeros(d)
    for t in range(1, rounds + 1):
        updates, weights = [], []
        n_total = sum(c.n_k for c in clients)
        for c in clients:
            w_local = local_dp_sgd_instrumented(
                w.copy(), w, c.X_clean, c.y_clean, tilde_C,
                sigma, mu, eta0, local_steps, batch_size, rng, stats)
            updates.append(w_local)
            weights.append(c.n_k / n_total)
        weights = np.array(weights) / np.sum(weights)
        w = np.sum([a * u for a, u in zip(weights, updates)], axis=0)
    frac_clipped = stats["n_clipped"] / stats["n_samples"]
    return {"tilde_C": tilde_C, "max_grad_norm_seen": stats["max_norm"],
            "n_samples": stats["n_samples"], "n_clipped": stats["n_clipped"],
            "frac_clipped": frac_clipped, "a1_holds": bool(frac_clipped == 0.0)}


def load_old(name):
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return {r["sigma"]: r for r in d["pillar1"]["convergence_checks"]}


def run(name, n_subjects=15):
    prep, clip = prep_dataset(name, n_subjects=n_subjects)
    clients = build_clients(prep, clip)
    d = clients[0].X_clean.shape[1]
    r = d - 1
    K = len(clients)

    X_test = np.concatenate([augment(standardize(p["z"], p["mu"], p["std"])[p["test_i"]]) for p in prep])
    y_test = np.concatenate([p["y"][p["test_i"]] for p in prep]).astype(np.float64)
    F_w0 = model_loss(np.zeros(r + 1), X_test, y_test)

    old = load_old(name)

    rows = []
    for sigma in P1_SIGMAS_CONV:
        out = run_dp_fedprox_eeg(clients, X_test, y_test, clip_norm_base=clip,
                                 sigma=sigma, delta=DELTA, rounds=P1_ROUNDS,
                                 local_steps=P1_LOCAL_STEPS, batch_size=P1_BATCH_SIZE,
                                 rng=np.random.default_rng(1))
        beta = out["beta"]
        mu = out["mu_prox"]
        eta0 = out["eta0"]
        C_tilde = float(np.mean(list(out["clip_norms"].values())))
        w_star, F_star = local_optimum(X_test, y_test, beta, n_steps=400)
        nu2 = estimate_nu2(clients, P1_BATCH_SIZE)

        rhs = convergence_bound_rhs(F_w0, F_star, eta0, beta, mu, sigma, C_tilde, nu2,
                                    r + 1, K, P1_LOCAL_STEPS, P1_ROUNDS)
        lhs = float(np.mean([h.grad_norm_sq for h in out["history"]]))
        ok = bool(rhs["total"] >= lhs - 1e-9)

        old_row = old.get(sigma) if old else None
        rows.append({
            "sigma": sigma, "lhs_empirical": lhs, "rhs_bound": rhs, "holds": ok,
            "beta": beta, "mu": mu, "eta0": eta0, "C_tilde": C_tilde, "nu2": nu2,
            "final_acc": out["history"][-1].global_acc,
            "final_loss": out["history"][-1].global_loss,
            "old_lhs_empirical": old_row["lhs_empirical"] if old_row else None,
            "old_rhs_total": old_row["rhs_bound"]["total"] if old_row else None,
            "old_final_acc": old_row["final_acc"] if old_row else None,
            "old_eta0": (1.0 / (2 * beta)) if old_row else None,
        })

    a1 = check_assumption_a1(clients, rows[0]["C_tilde"])

    result = {"dataset": name, "n_subjects": K, "d": d,
              "convergence_checks": rows, "assumption_a1": a1}

    with open(RESULTS_DIR / f"{name}_pillar1_v2.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {name} ===  K={K}  d={d}")
    print(f"{'sigma':>6} {'lhs_emp':>9} {'opt':>9} {'noise':>9} {'drift':>9} {'total':>9} {'ok?':>4} "
          f"| {'old_lhs':>9} {'old_total':>10} {'old_acc':>8} {'new_acc':>8}")
    for row in rows:
        b = row["rhs_bound"]
        old_lhs = f"{row['old_lhs_empirical']:.5f}" if row["old_lhs_empirical"] is not None else "n/a"
        old_total = f"{row['old_rhs_total']:.4f}" if row["old_rhs_total"] is not None else "n/a"
        old_acc = f"{row['old_final_acc']:.3f}" if row["old_final_acc"] is not None else "n/a"
        print(f"{row['sigma']:6.2f} {row['lhs_empirical']:9.5f} {b['optimization_error']:9.4f} "
              f"{b['dp_noise_error']:9.4f} {b['drift_error']:9.4f} {b['total']:9.4f} "
              f"{'OK' if row['holds'] else 'FAIL':>4} | {old_lhs:>9} {old_total:>10} "
              f"{old_acc:>8} {row['final_acc']:8.3f}")
    print(f"A1 (clipping never binds) under tilde_C={a1['tilde_C']:.3f}: "
          f"frac_clipped={a1['frac_clipped']:.6f}  max_grad_norm_seen={a1['max_grad_norm_seen']:.4f}  "
          f"-> {'HOLDS' if a1['a1_holds'] else 'VIOLATED'}")
    return result


if __name__ == "__main__":
    t0 = time.time()
    run(sys.argv[1] if len(sys.argv) > 1 else "zhang")
    print(f"\ndone in {time.time()-t0:.1f}s")
