import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

from attack import inject_alpha
from cohort import load_subjects
from config import COHORTS, FS, RESULTS_DIR
from detect import mahalanobis_score, standardize
from features import encode
from analyses.joint_optimizer import BOUNDS, build_clean_state, evaluate
from privacy import clip_l2, gaussian_mechanism
from signature import dp_project, fingerprint, mismatch_frac

N_DRAWS = 3
STRENGTH_GRID = np.linspace(0.5, 2.0, 7)
WEAK_SIGMA = 0.01
STARTS = [(0.5, 10.0, 1.0), (1.0, 10.0, 1.0), (0.3, 9.0, 0.6), (1.5, 11.0, 0.4)]


def evaluate_combined(theta, state, clip, P, lsh, rng):
    strength, freq, ch_frac = theta
    caught = []
    for s in state:
        atk_trials = inject_alpha(s["sub"].trials[s["atk_i"]], FS, strength=strength,
                                  freq=freq, ch_frac=ch_frac, rng=rng)
        z_atk = encode(atk_trials, FS)
        za = gaussian_mechanism(clip_l2(standardize(z_atk, s["mu"], s["std"]), clip), WEAK_SIGMA, rng)
        mh_hit = mahalanobis_score(za, s["baseline"]) > s["tau_mh"]
        za_p = dp_project(za + s["offset"], P, sigma_s=WEAK_SIGMA, rng=rng)
        sig_hit = mismatch_frac(fingerprint(za_p, lsh, s["dev_ref"])) > s["tau_bits"]
        caught.append(mh_hit | sig_hit)
    return float(np.mean(np.concatenate(caught)))


def run_cohort(subjects, draw):
    rng = np.random.default_rng(draw)
    state, clip, P, lsh = build_clean_state(subjects, rng)
    opt_rng = np.random.default_rng(7 + draw)

    def obj(theta):
        return evaluate(theta, state, clip, P, lsh, opt_rng)

    best = None
    for x0 in STARTS:
        res = minimize(obj, x0=np.array(x0), method="Nelder-Mead",
                       bounds=BOUNDS, options={"xatol": 1e-2, "fatol": 1e-3, "maxiter": 80})
        if best is None or res.fun < best.fun:
            best = res
    _, freq, ch_frac = best.x

    sweep = []
    for strength in STRENGTH_GRID:
        sweep.append(evaluate_combined((strength, freq, ch_frac), state, clip, P, lsh,
                                       np.random.default_rng(99 + draw)))
    k = int(np.argmin(sweep))
    return {"draw": draw, "theta_freq": float(freq), "theta_ch_frac": float(ch_frac),
            "grid_strength": STRENGTH_GRID.tolist(), "combined": sweep,
            "min_combined": float(sweep[k]), "argmin_strength": float(STRENGTH_GRID[k])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    args = ap.parse_args()

    out_path = RESULTS_DIR / "joint_frontier.json"
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.cohorts:
        t0 = time.time()
        subjects = load_subjects(name)
        draws = [run_cohort(subjects, d) for d in range(N_DRAWS)]
        vals = [d["min_combined"] for d in draws]
        result[name] = {"draws": draws, "mean": float(np.mean(vals)), "std": float(np.std(vals))}
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  {name}: min combined {np.mean(vals):.3f} +- {np.std(vals):.3f} ({time.time()-t0:.0f}s)", flush=True)
