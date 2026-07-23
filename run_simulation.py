import argparse
import json
import time
from pathlib import Path

import numpy as np

from attack import inject_alpha
from detect import (choose_clip_norm, ema_settle, empirical_power, empirical_threshold,
                     fit_baseline, l2_score, mahalanobis_score, standardize,
                     theoretical_power_bound)
from features import encode
from fedprox import Client, convergence_bound_rhs, local_optimum, run_dp_fedprox_eeg
from inversion import batch_gradient, reconstruct, reconstruction_error, trivial_baseline_error
from model import augment, loss as model_loss
from privacy import clip_l2, epsilon_of_sigma, gaussian_mechanism
from raw_loader import (SubjectData, list_cho_subjects, list_lee_subjects,
                         list_won_subjects, list_zhang_subjects, load_cho_subject,
                         load_lee_subject, load_won_subject, load_zhang_subject)
from signature import (calibrate_threshold, dp_project, enroll_reference, fingerprint,
                        keygen, make_lsh, make_projection_matrix, mismatch_frac,
                        sign_reading, verify_signature, verify_tamper)

FS = 250.0
ALPHA0 = 0.05
TARGET_FRR = 0.05
DELTA = 1e-5
ATTACK_STRENGTH = 0.5
EMA_GAMMA = 0.1
RIDGE = 1e-2
SIGMAS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])

LSH_S = 8
LSH_B = 64

P1_ROUNDS = 20
P1_LOCAL_STEPS = 5
P1_BATCH_SIZE = 32
P1_SIGMAS_CONV = [0.1, 0.5, 2.0]
P1_SIGMA_DEMO = 0.5
P1_FRAC_BAD = 0.3

DATASETS = {
    "cho": (load_cho_subject, list_cho_subjects, 200),
    "lee": (load_lee_subject, list_lee_subjects, 200),
    "won": (load_won_subject, list_won_subjects, 150),
    "zhang": (load_zhang_subject, list_zhang_subjects, 150),
}


def split_idx(n, rng, fracs=(0.4, 0.15, 0.15, 0.15, 0.15)):
    idx = rng.permutation(n)
    cuts = np.cumsum([int(f * n) for f in fracs])[:-1]
    return np.split(idx, cuts)


def prep_subject(sub, rng):
    calib_i, thresh_i, clean_i, atk_i, test_i = split_idx(len(sub.trials), rng)
    train_i = np.concatenate([calib_i, thresh_i, clean_i])

    z = encode(sub.trials, FS)
    atk_trials = inject_alpha(sub.trials, FS, strength=ATTACK_STRENGTH, rng=rng)
    z_atk = encode(atk_trials, FS)

    mu = z[calib_i].mean(axis=0)
    std = z[calib_i].std(axis=0) + 1e-8

    return {
        "id": sub.subject_id, "y": sub.labels,
        "z": z, "z_atk": z_atk, "mu": mu, "std": std,
        "train_i": train_i, "test_i": test_i,
        "z_calib": z[calib_i],
        "z_thresh_s": standardize(z[thresh_i], mu, std),
        "z_clean_s": standardize(z[clean_i], mu, std),
        "z_atk_s": standardize(z_atk[atk_i], mu, std),
        "offset": mu / std,
    }


def run_pillars_2_3(prep, rng):
    r = prep[0]["z_calib"].shape[1]
    pool = np.concatenate([p["z_thresh_s"] for p in prep], axis=0)
    clip = choose_clip_norm(pool, q=0.95)

    bases = []
    for p in prep:
        b = fit_baseline(p["z_calib"], clip_norm=clip, ridge=RIDGE)
        b.mu = ema_settle(clip_l2(p["z_thresh_s"], clip), b.mu, gamma=EMA_GAMMA)
        bases.append(b)
        p["z_thresh_c"] = clip_l2(p["z_thresh_s"], clip)
        p["z_clean_c"] = clip_l2(p["z_clean_s"], clip)
        p["z_atk_c"] = clip_l2(p["z_atk_s"], clip)

    P = make_projection_matrix(r, LSH_S, rng)
    lsh = make_lsh(LSH_S, LSH_B, rng)
    devs, refs = [], []
    for p in prep:
        dev = keygen(p["id"])
        devs.append(dev)
        ref_proj = dp_project(p["offset"], P, sigma_s=0.05, rng=rng)
        refs.append(enroll_reference(ref_proj, lsh))

    out = []
    for sigma in SIGMAS:
        l2_t, l2_c, l2_a = [], [], []
        mh_t, mh_c, mh_a = [], [], []
        hd_t, hd_c, hd_a = [], [], []
        mu_diffs, nu2s = [], []
        sig_ok_c, sig_ok_a = [], []

        for p, b, dev, ref in zip(prep, bases, devs, refs):
            zt = gaussian_mechanism(p["z_thresh_c"], sigma, rng)
            zc = gaussian_mechanism(p["z_clean_c"], sigma, rng)
            za = gaussian_mechanism(p["z_atk_c"], sigma, rng)

            l2_t.append(l2_score(zt, b))
            l2_c.append(l2_score(zc, b))
            l2_a.append(l2_score(za, b))
            mh_t.append(mahalanobis_score(zt, b))
            mh_c.append(mahalanobis_score(zc, b))
            mh_a.append(mahalanobis_score(za, b))

            off = p["offset"]
            zt_p = dp_project(zt + off, P, sigma_s=float(sigma), rng=rng)
            zc_p = dp_project(zc + off, P, sigma_s=float(sigma), rng=rng)
            za_p = dp_project(za + off, P, sigma_s=float(sigma), rng=rng)

            hd_t.append(mismatch_frac(fingerprint(zt_p, lsh, ref)))
            hd_c.append(mismatch_frac(fingerprint(zc_p, lsh, ref)))
            atk_bits = fingerprint(za_p, lsh, ref)
            hd_a.append(mismatch_frac(atk_bits))

            sig = sign_reading(dev, atk_bits[0], ts=int(sigma * 1000))
            sig_ok_a.append(verify_signature(dev, atk_bits[0], int(sigma * 1000), sig))
            clean_bit0 = fingerprint(zc_p[:1], lsh, ref)[0]
            sig2 = sign_reading(dev, clean_bit0, ts=int(sigma * 1000))
            sig_ok_c.append(verify_signature(dev, clean_bit0, int(sigma * 1000), sig2))

            mu_diffs.append(float(np.linalg.norm(za.mean(axis=0) - b.mu)))
            nu2s.append(float(np.trace(b.cov)))

        l2_t, l2_c, l2_a = map(np.concatenate, (l2_t, l2_c, l2_a))
        mh_t, mh_c, mh_a = map(np.concatenate, (mh_t, mh_c, mh_a))
        hd_t, hd_c, hd_a = map(np.concatenate, (hd_t, hd_c, hd_a))

        tau_l2 = empirical_threshold(l2_t, ALPHA0)
        tau_mh = empirical_threshold(mh_t, ALPHA0)
        tau_bits = calibrate_threshold(hd_t, TARGET_FRR)

        theory = theoretical_power_bound(float(np.mean(mu_diffs)), float(sigma), clip,
                                          r, float(np.mean(nu2s)), ALPHA0)

        out.append({
            "sigma": float(sigma), "epsilon": epsilon_of_sigma(float(sigma), DELTA, clip),
            "l2_power": empirical_power(l2_a, tau_l2),
            "l2_fpr": empirical_power(l2_c, tau_l2),
            "maha_power": empirical_power(mh_a, tau_mh),
            "maha_fpr": empirical_power(mh_c, tau_mh),
            "theory_power": theory,
            "sig_far": float(np.mean(verify_tamper(hd_a, tau_bits))),
            "sig_frr": float(np.mean(~verify_tamper(hd_c, tau_bits))),
            "sig_crypto_valid_clean": float(np.mean(sig_ok_c)),
            "sig_crypto_valid_atk": float(np.mean(sig_ok_a)),
        })

    return out, clip


def build_clients(prep, clip):
    clients = []
    for p in prep:
        z_s = standardize(p["z"], p["mu"], p["std"])
        z_sc = clip_l2(z_s, clip)
        za_s = standardize(p["z_atk"], p["mu"], p["std"])
        za_sc = clip_l2(za_s, clip)

        i = p["train_i"]
        clients.append(Client(
            client_id=p["id"], X_calib=p["z_calib"],
            X_clean=augment(z_sc[i]), y_clean=p["y"][i].astype(np.float64),
            X_attacked=augment(za_sc[i]),
        ))
    return clients


def run_pillar1(prep, clip, rng):
    clients = build_clients(prep, clip)
    r = clients[0].X_clean.shape[1] - 1
    X_test = np.concatenate([augment(clip_l2(standardize(p["z"], p["mu"], p["std"]), clip)[p["test_i"]])
                              for p in prep])
    y_test = np.concatenate([p["y"][p["test_i"]] for p in prep]).astype(np.float64)

    conv = []
    for sigma in P1_SIGMAS_CONV:
        out = run_dp_fedprox_eeg(clients, X_test, y_test, clip_norm_base=clip,
                                  sigma=sigma, delta=DELTA, rounds=P1_ROUNDS,
                                  local_steps=P1_LOCAL_STEPS, batch_size=P1_BATCH_SIZE,
                                  rng=np.random.default_rng(1))
        beta = out["beta"]
        w_star, F_star = local_optimum(X_test, y_test, beta, n_steps=400)
        F_w0 = model_loss(np.zeros(r + 1), X_test, y_test)
        local_stars = [local_optimum(c.X_clean, c.y_clean, beta, n_steps=200)[1] for c in clients]
        n_total = sum(c.n_k for c in clients)
        p_k = np.array([c.n_k / n_total for c in clients])
        gamma = max(F_star - float(np.dot(p_k, local_stars)), 0.0)

        rhs = convergence_bound_rhs(F_w0, F_star, out["eta0"], beta, sigma,
                                     float(np.mean(list(out["clip_norms"].values()))), r + 1,
                                     len(clients), P1_ROUNDS, out["eta_history"], gamma, out["mu_prox"])
        lhs = float(np.mean([h.grad_norm_sq for h in out["history"]]))
        conv.append({
            "sigma": sigma, "lhs_empirical": lhs, "rhs_bound": rhs,
            "F_star": F_star, "beta": beta, "gamma_heterogeneity": gamma,
            "final_acc": out["history"][-1].global_acc, "final_loss": out["history"][-1].global_loss,
            "history": [{"round": h.round, "loss": h.global_loss, "acc": h.global_acc,
                         "grad_norm_sq": h.grad_norm_sq, "epsilon_t": h.epsilon_t} for h in out["history"]],
            "_w": out["w_global"],
        })

    k = len(clients)
    n_bad = max(1, int(round(P1_FRAC_BAD * k)))
    bad_ids = set(rng.choice([c.client_id for c in clients], size=n_bad, replace=False))

    gate_base = fit_baseline(np.concatenate([p["z_calib"] for p in prep]), clip_norm=clip, ridge=RIDGE)

    def gate_fn(client_id, round_i, w_local, w_glob):
        p = next(p for p in prep if p["id"] == client_id)
        z = gaussian_mechanism(p["z_clean_c"][:5], P1_SIGMA_DEMO, rng)
        score = mahalanobis_score(z, gate_base).mean()
        tau = empirical_threshold(
            mahalanobis_score(gaussian_mechanism(p["z_thresh_c"], P1_SIGMA_DEMO, rng), gate_base), ALPHA0)
        return bool(score > tau)

    kw = dict(X_test_aug=X_test, y_test=y_test, clip_norm_base=clip,
              sigma=P1_SIGMA_DEMO, delta=DELTA, rounds=P1_ROUNDS,
              local_steps=P1_LOCAL_STEPS, batch_size=P1_BATCH_SIZE)

    out_clean = run_dp_fedprox_eeg(clients, rng=np.random.default_rng(2), **kw)
    out_data_ng = run_dp_fedprox_eeg(clients, rng=np.random.default_rng(2),
                                      compromised_ids=bad_ids, poison_mode="data", **kw)
    out_data_g = run_dp_fedprox_eeg(clients, rng=np.random.default_rng(2),
                                     compromised_ids=bad_ids, gate_fn=gate_fn, poison_mode="data", **kw)
    out_grad_ng = run_dp_fedprox_eeg(clients, rng=np.random.default_rng(2),
                                      compromised_ids=bad_ids, poison_mode="gradient", **kw)
    out_grad_g = run_dp_fedprox_eeg(clients, rng=np.random.default_rng(2),
                                     compromised_ids=bad_ids, gate_fn=gate_fn, poison_mode="gradient", **kw)

    def summarize(o):
        return [{"round": h.round, "loss": h.global_loss, "acc": h.global_acc,
                  "n_gated_out": h.n_gated_out} for h in o["history"]]

    w_ref = conv[0]["_w"] if conv else None
    for c in conv:
        c.pop("_w", None)

    return {
        "convergence_checks": conv,
        "attack_demo": {
            "sigma": P1_SIGMA_DEMO, "frac_compromised": P1_FRAC_BAD,
            "n_clients": k, "compromised_ids": sorted(bad_ids),
            "clean": summarize(out_clean),
            "data_poison_ungated": summarize(out_data_ng),
            "data_poison_gated": summarize(out_data_g),
            "gradient_poison_ungated": summarize(out_grad_ng),
            "gradient_poison_gated": summarize(out_grad_g),
        },
        "w_ref": w_ref,
    }


def run_passive_inference(prep, clip, w_ref, rng, n_clients=5, n_per_client=2):
    r = prep[0]["z_calib"].shape[1]
    if w_ref is None:
        w_ref = rng.normal(size=r + 1) * 0.1
    pop_mean = np.zeros(r)

    idx = rng.choice(len(prep), size=min(n_clients, len(prep)), replace=False)
    sample = [prep[i] for i in idx]

    out = []
    for sigma in SIGMAS:
        errs, triv = [], []
        for p in sample:
            z = clip_l2(p["z_clean_s"], clip)
            ti = rng.choice(len(z), size=min(n_per_client, len(z)), replace=False)
            for i in ti:
                x_true = z[i:i + 1]
                xa_true = augment(x_true)
                y_true = np.array([float(rng.integers(0, 2))])
                g_clean = batch_gradient(w_ref, xa_true, y_true)
                g_target = gaussian_mechanism(g_clean, sigma, rng) if sigma > 0 else g_clean
                x_hat = reconstruct(w_ref, y_true, g_target, r + 1, rng, n_steps=400, lr=0.05)
                errs.append(reconstruction_error(x_hat[:, :r], x_true))
                triv.append(trivial_baseline_error(x_true, pop_mean))
        out.append({
            "sigma": float(sigma), "epsilon": epsilon_of_sigma(float(sigma), DELTA, clip),
            "mean_reconstruction_error": float(np.mean(errs)),
            "trivial_baseline_error": float(np.mean(triv)),
            "confidentiality_holds": bool(np.mean(errs) >= np.mean(triv)),
        })
    return {"results": out, "n_clients_sampled": len(sample), "n_samples_per_client": n_per_client}


def run_dataset(name, n_subjects, seed=0):
    load_fn, list_fn, max_trials = DATASETS[name]
    rng = np.random.default_rng(seed)
    ids = list_fn()[:n_subjects]
    print(f"  loading {len(ids)} {name} subjects ...")
    subjects = [load_fn(sid, max_trials=max_trials) for sid in ids]
    prep = [prep_subject(s, rng) for s in subjects]

    print(f"  running pillars 2+3 sweep for {name} ...")
    sweep, clip = run_pillars_2_3(prep, rng)

    print(f"  running pillar 1 for {name} ...")
    pillar1 = run_pillar1(prep, clip, rng)
    w_ref = pillar1.pop("w_ref")

    print(f"  running passive inference for {name} ...")
    pi = run_passive_inference(prep, clip, w_ref, rng)

    return {
        "dataset": name, "n_subjects": len(prep), "embedding_dim": prep[0]["z_calib"].shape[1],
        "clip_norm": clip, "attack_strength": ATTACK_STRENGTH,
        "sigma_sweep": sweep, "pillar1": pillar1, "passive_inference": pi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    n_subjects = 4 if args.quick else 16
    if args.quick:
        global P1_ROUNDS
        P1_ROUNDS = 6

    out_dir = Path(__file__).parent / args.out
    out_dir.mkdir(exist_ok=True)

    for name in ("cho", "lee", "won", "zhang"):
        t0 = time.time()
        result = run_dataset(name, n_subjects)
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  {name} done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
