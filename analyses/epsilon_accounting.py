import json
import math

import numpy as np

import run_simulation
from cohort import prep_subject
from config import COHORTS, DATASETS, DELTA, FS, RESULTS_DIR
from run_simulation import run_sigma_sweep

WINDOW_S = {"cho": 4.0, "lee": 4.0, "won": 0.8, "zhang": 0.8,
            "wang_eo": 2.0, "wang_ec": 2.0, "cogbci_eo": 2.0, "cogbci_ec": 2.0}
SESSION_S = 3600.0
FL_T, FL_E = 20, 5
PAPER_SIGMAS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
LARGE_SIGMAS = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]


def eps_from_c(c, delta=DELTA):
    return c + 2.0 * math.sqrt(c * math.log(1.0 / delta))


def c_release(n, C, sigma):
    return n * C ** 2 / (2.0 * sigma ** 2)


def c_fl(sigma, steps=FL_T * FL_E):
    return steps / (2.0 * sigma ** 2)


def sigma_for_eps(eps, n_or_steps_times, sensitivity, delta=DELTA):
    """Smallest sigma with eps_from_c <= eps for c = n_or_steps_times * sensitivity^2 / (2 sigma^2)."""
    L = math.log(1.0 / delta)
    c_target = (math.sqrt(L + eps) - math.sqrt(L)) ** 2
    return sensitivity * math.sqrt(n_or_steps_times / (2.0 * c_target))


def detection_at_large_sigma(name, n_subjects=15):
    run_simulation.SIGMAS = np.array(LARGE_SIGMAS)
    load_fn, list_fn, max_trials = DATASETS[name]
    from cohort import load_subjects
    subs = load_subjects(name, n_subjects)
    rng = np.random.default_rng(0)
    prep = [prep_subject(s, rng) for s in subs]
    sweep, _ = run_sigma_sweep(prep, np.random.default_rng(1))
    run_simulation.SIGMAS = np.array(PAPER_SIGMAS)
    return {row["sigma"]: row["maha_power"] for row in sweep}


if __name__ == "__main__":
    out = {"delta": DELTA, "fl_T": FL_T, "fl_E": FL_E, "session_s": SESSION_S, "cohorts": {}}
    C_by = {}
    for name in COHORTS:
        with open(RESULTS_DIR / f"{name}.json") as f:
            C_by[name] = json.load(f)["clip_norm"]

    for name in COHORTS:
        C = C_by[name]
        n_sess = int(SESSION_S / WINDOW_S[name])
        entry = {
            "C": C, "window_s": WINDOW_S[name], "n_session": n_sess,
            "eps_release_by_sigma": {str(s): eps_from_c(c_release(1, C, s)) for s in PAPER_SIGMAS},
            "eps_session_by_sigma": {str(s): eps_from_c(c_release(n_sess, C, s)) for s in PAPER_SIGMAS},
            "sigma_for_eps1_release": sigma_for_eps(1.0, 1, C),
            "sigma_for_eps8_release": sigma_for_eps(8.0, 1, C),
            "sigma_for_eps1_session": sigma_for_eps(1.0, n_sess, C),
            "sigma_for_eps8_session": sigma_for_eps(8.0, n_sess, C),
        }
        out["cohorts"][name] = entry
        print(f"{name:10s} C={C:6.2f} n_sess={n_sess:5d} "
              f"eps_rel(sigma=.75)={entry['eps_release_by_sigma']['0.75']:.1f} "
              f"eps_sess(sigma=.75)={entry['eps_session_by_sigma']['0.75']:.3g} "
              f"sigma(eps=8,rel)={entry['sigma_for_eps8_release']:.1f} "
              f"sigma(eps=8,sess)={entry['sigma_for_eps8_session']:.1f}")

    out["fl"] = {
        "eps_by_sigma": {str(s): eps_from_c(c_fl(s)) for s in [0.1, 0.5, 2.0]},
        "sigma_for_eps1": sigma_for_eps(1.0, FL_T * FL_E, 1.0),
        "sigma_for_eps8": sigma_for_eps(8.0, FL_T * FL_E, 1.0),
    }
    print("FL eps:", {k: round(v, 1) for k, v in out["fl"]["eps_by_sigma"].items()},
          "sigma(eps=1)", round(out["fl"]["sigma_for_eps1"], 2),
          "sigma(eps=8)", round(out["fl"]["sigma_for_eps8"], 2))

    out["large_sigma_detection"] = {}
    for name in COHORTS:
        det = detection_at_large_sigma(name)
        out["large_sigma_detection"][name] = {str(k): v for k, v in det.items()}
        print(name, {k: round(v, 3) for k, v in det.items()}, flush=True)

    with open(RESULTS_DIR / "epsilon_accounting.json", "w") as f:
        json.dump(out, f, indent=2)
