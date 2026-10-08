import json

import numpy as np

from config import COHORTS, RESULTS_DIR, SIGMAS


def load(name):
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)


def at(sweep, sigma, key):
    return next(r[key] for r in sweep if abs(r["sigma"] - sigma) < 1e-9)


def first_holds(passive):
    ok = [r["sigma"] for r in passive["results"] if r["confidentiality_holds"]]
    return min(ok) if ok else None


def optional(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    out = {"detection": {}, "signature": {}, "passive": {}, "federated": {}}
    for name in COHORTS:
        d = load(name)
        s = d["sigma_sweep"]
        out["detection"][name] = {
            "clip_norm": d["clip_norm"],
            "maha_sigma_0.01": at(s, 0.01, "maha_power"),
            "maha_sigma_0.75": at(s, 0.75, "maha_power"),
            "maha_fpr_sigma_0.75": at(s, 0.75, "maha_fpr"),
            "epsilon_sigma_0.75": at(s, 0.75, "epsilon"),
            "theory_sigma_0.01": at(s, 0.01, "theory_power"),
            "l2_sigma_0.01": at(s, 0.01, "l2_power"),
        }
        out["signature"][name] = {
            "far_sigma_0.01": at(s, 0.01, "sig_far"),
            "far_sigma_0.75": at(s, 0.75, "sig_far"),
            "frr_sigma_0.01": at(s, 0.01, "sig_frr"),
        }
        out["passive"][name] = {
            "first_sigma_confidentiality_holds": first_holds(d["passive_inference"]),
            "trivial_error_sigma_0": d["passive_inference"]["results"][0]["trivial_baseline_error"],
        }
        conv = {c["sigma"]: c for c in d["private_training"]["convergence_checks"]}
        out["federated"][name] = {
            "final_acc_by_sigma": {str(k): round(v["final_acc"], 4) for k, v in conv.items()},
            "lhs_le_rhs_all": all(c["lhs_empirical"] <= c["rhs_bound"]["total"] for c in conv.values()),
            "attack_demo_final_acc": {
                k: round(v[-1]["acc"], 4) for k, v in d["private_training"]["attack_demo"].items()
                if isinstance(v, list) and v and isinstance(v[0], dict)
            },
        }
    for fname in ["joint_frontier.json", "subsample_sensitivity.json", "realisations.json",
                  "adaptive_poison.json", "adaptive_gate.json", "fk_sweep.json",
                  "epsilon_accounting.json", "lee_label_order.json"]:
        data = optional(RESULTS_DIR / fname)
        if data is not None:
            out[fname.replace(".json", "")] = data
    out["sigmas"] = [float(x) for x in SIGMAS]
    with open(RESULTS_DIR / "paper_numbers.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f"wrote {RESULTS_DIR / 'paper_numbers.json'}")


if __name__ == "__main__":
    main()
