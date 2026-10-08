import json
import time

import numpy as np
from scipy.optimize import minimize

from attack import inject_alpha
from cohort import load_subjects, split_idx
from config import ALPHA0, FS, LSH_B, LSH_S, RIDGE, TARGET_FRR
from detect import choose_clip_norm, empirical_threshold, fit_baseline, mahalanobis_score, standardize
from features import encode
from privacy import clip_l2, gaussian_mechanism
from signature import (calibrate_threshold, dp_project, enroll_reference, fingerprint,
                             make_lsh, make_projection_matrix, mismatch_frac)

WEAK_SIGMA = 0.01

BOUNDS = [(0.1, 2.0), (8.0, 12.0), (0.2, 1.0)]


def build_clean_state(subjects, rng):
    state = []
    for sub in subjects:
        calib_i, thresh_i, clean_i, atk_i, test_i = split_idx(len(sub.trials), rng)
        z = encode(sub.trials, FS)
        mu = z[calib_i].mean(axis=0)
        std = z[calib_i].std(axis=0) + 1e-8
        state.append(dict(sub=sub, calib_i=calib_i, thresh_i=thresh_i, atk_i=atk_i, z=z, mu=mu, std=std))

    pool_thresh = np.concatenate([standardize(s["z"][s["thresh_i"]], s["mu"], s["std"]) for s in state])
    clip = choose_clip_norm(pool_thresh, q=0.95)

    r = state[0]["z"].shape[1]
    P = make_projection_matrix(r, LSH_S, rng)
    lsh = make_lsh(LSH_S, LSH_B, rng)

    for s in state:
        s["baseline"] = fit_baseline(s["z"][s["calib_i"]], clip_norm=clip, ridge=RIDGE)
        s["offset"] = s["mu"] / s["std"]
        s["dev_ref"] = enroll_reference(dp_project(s["offset"], P, sigma_s=0.05, rng=rng), lsh)
        z_thresh_c = clip_l2(standardize(s["z"][s["thresh_i"]], s["mu"], s["std"]), clip)
        zt = gaussian_mechanism(z_thresh_c, WEAK_SIGMA, rng)
        s["tau_mh"] = empirical_threshold(mahalanobis_score(zt, s["baseline"]), ALPHA0)
        zt_p = dp_project(zt + s["offset"], P, sigma_s=WEAK_SIGMA, rng=rng)
        s["tau_bits"] = calibrate_threshold(mismatch_frac(fingerprint(zt_p, lsh, s["dev_ref"])), TARGET_FRR)
    return state, clip, P, lsh


def evaluate(theta, state, clip, P, lsh, rng, return_detail=False):
    strength, freq, ch_frac = theta
    maha_ratio, mismatches, far_hits, det_hits = [], [], [], []
    for s in state:
        atk_trials = inject_alpha(s["sub"].trials[s["atk_i"]], FS, strength=strength,
                                  freq=freq, ch_frac=ch_frac, rng=rng)
        z_atk = encode(atk_trials, FS)
        za_c = clip_l2(standardize(z_atk, s["mu"], s["std"]), clip)
        za = gaussian_mechanism(za_c, WEAK_SIGMA, rng)

        mh = mahalanobis_score(za, s["baseline"])
        maha_ratio.append(mh / s["tau_mh"])
        det_hits.append(mh > s["tau_mh"])

        za_p = dp_project(za + s["offset"], P, sigma_s=WEAK_SIGMA, rng=rng)
        hd = mismatch_frac(fingerprint(za_p, lsh, s["dev_ref"]))
        mismatches.append(hd)
        far_hits.append(hd <= s["tau_bits"])

    maha_ratio = np.concatenate(maha_ratio)
    mismatches = np.concatenate(mismatches)
    loss = float(np.mean(maha_ratio)) + float(np.mean(mismatches))
    if return_detail:
        maha_power = float(np.mean(np.concatenate(det_hits)))
        far = float(np.mean(np.concatenate(far_hits)))
        return loss, maha_power, far
    return loss


def run(name="zhang"):
    rng = np.random.default_rng(0)
    subjects = load_subjects(name)
    state, clip, P, lsh = build_clean_state(subjects, rng)
    opt_rng = np.random.default_rng(7)

    def obj(theta):
        return evaluate(theta, state, clip, P, lsh, opt_rng)

    starts = [(0.5, 10.0, 1.0), (1.0, 10.0, 1.0), (0.3, 9.0, 0.6), (1.5, 11.0, 0.4)]
    best = None
    for x0 in starts:
        t0 = time.time()
        res = minimize(obj, x0=np.array(x0), method="Nelder-Mead",
                       bounds=BOUNDS, options={"xatol": 1e-2, "fatol": 1e-3, "maxiter": 80})
        print(f"  start={x0} -> theta={res.x.round(3)} loss={res.fun:.4f} ({time.time()-t0:.1f}s, {res.nfev} evals)")
        if best is None or res.fun < best.fun:
            best = res

    final_rng = np.random.default_rng(99)
    loss, maha_power, far = evaluate(best.x, state, clip, P, lsh, final_rng, return_detail=True)
    result = {
        "dataset": name, "theta_strength": float(best.x[0]), "theta_freq": float(best.x[1]),
        "theta_ch_frac": float(best.x[2]), "loss": float(best.fun),
        "maha_power_weak": maha_power, "sig_far_weak": far,
    }
    print("BEST:", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
