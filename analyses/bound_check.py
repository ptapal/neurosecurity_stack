"""Compare Theorem 8.1's detection power (beta_A), the Neyman-Pearson bound (beta*) and the unwhitened
detector (beta_B) against measured Pillar 3 power, per cohort and sigma.

Baseline covariance is fit on pre-noise calibration data, as in detect.fit_baseline.
Usage: python -m analyses.bound_check <cohort>
"""
import json
import sys

import numpy as np
from scipy.stats import norm

from attack import inject_alpha
from cohort import load_subjects, split_idx
from config import ALPHA0, EMA_GAMMA, FS, RESULTS_DIR, RIDGE, SIGMAS
from detect import choose_clip_norm, ema_settle, fit_baseline, standardize
from features import encode
from privacy import clip_l2


def prep_dataset(name, n_subjects=15, seed=0):
    rng = np.random.default_rng(seed)

    prep = []
    for sub in load_subjects(name, n_subjects):
        calib_i, thresh_i, clean_i, atk_i, test_i = split_idx(len(sub.trials), rng)
        z = encode(sub.trials, FS)
        atk_trials = inject_alpha(sub.trials, FS, strength=0.5, rng=rng)
        z_atk = encode(atk_trials, FS)
        mu = z[calib_i].mean(axis=0)
        std = z[calib_i].std(axis=0) + 1e-8
        prep.append(dict(
            z_calib=z[calib_i],
            z_thresh_s=standardize(z[thresh_i], mu, std),
            z_atk_s=standardize(z_atk[atk_i], mu, std),
        ))

    pool = np.concatenate([p["z_thresh_s"] for p in prep], axis=0)
    clip = choose_clip_norm(pool, q=0.95)

    bases, mu1s, calibs = [], [], []
    for p in prep:
        b = fit_baseline(p["z_calib"], clip_norm=clip, ridge=RIDGE)
        b.mu = ema_settle(clip_l2(p["z_thresh_s"], clip), b.mu, gamma=EMA_GAMMA)
        bases.append(b)
        mu1s.append(clip_l2(p["z_atk_s"], clip).mean(axis=0))
        calibs.append(p["z_calib"])
    return bases, mu1s, calibs


def quad_form_moments(M, Sigma, delta):
    """Exact mean/variance of S = w^T M w, w ~ N(delta, Sigma)."""
    MSigma = M @ Sigma
    mean = np.trace(MSigma) + delta @ M @ delta
    var = 2 * np.trace(MSigma @ MSigma) + 4 * delta @ M @ Sigma @ M @ delta
    return mean, var


def per_subject_bounds(mu0, mu1, Sigma_hat, sigma, alpha0=ALPHA0):
    r = len(mu0)
    Sigma_eta = Sigma_hat + sigma**2 * np.eye(r)
    delta = mu1 - mu0

    # beta*: Neyman-Pearson optimal power
    lam = delta @ np.linalg.solve(Sigma_eta, delta)
    beta_star = norm.cdf(np.sqrt(max(lam, 0)) - norm.ppf(1 - alpha0))

    # beta_A: whitened Mahalanobis detector, Theorem 8.1
    M_A = np.linalg.inv(Sigma_hat)
    m0, v0 = quad_form_moments(M_A, Sigma_eta, np.zeros(r))
    m1, v1 = quad_form_moments(M_A, Sigma_eta, delta)
    tau_A = m0 + np.sqrt(max(v0, 1e-12)) * norm.ppf(1 - alpha0)
    beta_A = norm.cdf((m1 - tau_A) / np.sqrt(max(v1, 1e-12)))

    # beta_B: unwhitened L2 detector (M = I)
    M_B = np.eye(r)
    m0b, v0b = quad_form_moments(M_B, Sigma_eta, np.zeros(r))
    m1b, v1b = quad_form_moments(M_B, Sigma_eta, delta)
    tau_B = m0b + np.sqrt(max(v0b, 1e-12)) * norm.ppf(1 - alpha0)
    beta_B = norm.cdf((m1b - tau_B) / np.sqrt(max(v1b, 1e-12)))

    return beta_star, beta_A, beta_B


def run(name):
    bases, mu1s, _ = prep_dataset(name)
    with open(RESULTS_DIR / f"{name}.json") as f:
        measured = {r["sigma"]: r for r in json.load(f)["sigma_sweep"]}

    rows = []
    for sigma in SIGMAS:
        stars, As, Bs = [], [], []
        for b, mu1 in zip(bases, mu1s):
            bs, ba, bb = per_subject_bounds(b.mu, mu1, b.cov, float(sigma))
            stars.append(bs); As.append(ba); Bs.append(bb)
        beta_star_avg = float(np.mean(stars))
        beta_A_avg = float(np.mean(As))
        beta_B_avg = float(np.mean(Bs))
        m = measured[float(sigma)] if float(sigma) in measured else min(measured.values(), key=lambda r: abs(r["sigma"]-sigma))
        rows.append(dict(sigma=float(sigma), beta_star=beta_star_avg, beta_A=beta_A_avg, beta_B=beta_B_avg,
                         measured_maha=m["maha_power"], measured_l2=m["l2_power"], paper_theory=m["theory_power"]))
    return rows


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "cho"
    rows = run(name)
    print(f"=== {name} ===")
    print(f"{'sigma':>8} {'beta*':>7} {'measured_maha':>13} {'ok?':>4} {'betaA':>7} {'betaB':>7} {'measured_l2':>11} {'ok?':>4} {'paper_theory':>12}")
    for r in rows:
        ok_star = "OK" if r["beta_star"] >= r["measured_maha"] - 1e-6 else "FAIL"
        ok_B = "OK" if r["beta_star"] >= r["measured_l2"] - 1e-6 else "FAIL"
        print(f"{r['sigma']:8.3f} {r['beta_star']:7.3f} {r['measured_maha']:13.3f} {ok_star:>4} "
              f"{r['beta_A']:7.3f} {r['beta_B']:7.3f} {r['measured_l2']:11.3f} {ok_B:>4} {r['paper_theory']:12.3f}")
