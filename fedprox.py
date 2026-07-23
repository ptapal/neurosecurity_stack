from dataclasses import dataclass, field

import numpy as np

from autocorrelation import corrected_clip_norm, estimate_rho
from model import accuracy, loss, per_sample_grad, smoothness_beta
from privacy import accumulated_epsilon


@dataclass
class Client:
    client_id: str
    X_calib: np.ndarray
    X_clean: np.ndarray
    y_clean: np.ndarray
    X_attacked: np.ndarray
    n_k: int = field(init=False)

    def __post_init__(self):
        self.n_k = len(self.y_clean)


@dataclass
class RoundLog:
    round: int
    epsilon_t: float
    lr_t: float
    global_loss: float
    global_acc: float
    grad_norm_sq: float
    n_gated_out: int


def local_optimum(Xa, y, beta, n_steps=300):
    w = np.zeros(Xa.shape[1])
    lr = 1.0 / max(beta, 1e-6)
    for _ in range(n_steps):
        g = per_sample_grad(w, Xa, y).mean(axis=0)
        w = w - lr * g
    return w, loss(w, Xa, y)


def local_dp_sgd(w, w_glob, Xa, y, clip, sigma, mu, lr, steps, batch, rng):
    n = Xa.shape[0]
    for _ in range(steps):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        g = per_sample_grad(w, Xa[idx], y[idx])
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        g_clip = g * np.minimum(1.0, clip / np.maximum(norms, 1e-12))
        noise = rng.normal(0.0, sigma * clip, size=w.shape) / g_clip.shape[0]
        g_dp = g_clip.mean(axis=0) + noise
        prox = mu * (w - w_glob)
        w = w - lr * (g_dp + prox)
    return w


def run_dp_fedprox_eeg(clients, X_test_aug, y_test, clip_norm_base, sigma, delta,
                        rounds, local_steps, batch_size, rng,
                        compromised_ids=None, gate_fn=None, poison_mode="data"):
    compromised_ids = compromised_ids or set()
    d = clients[0].X_clean.shape[1]

    clips = {}
    for c in clients:
        rho = estimate_rho(c.X_calib, max_lag=5)
        clips[c.client_id] = corrected_clip_norm(clip_norm_base, rho)

    beta = smoothness_beta(np.concatenate([c.X_clean for c in clients], axis=0))
    mu = beta
    eta0 = 1.0 / (mu + beta)

    n_total = sum(c.n_k for c in clients)
    p_k = {c.client_id: c.n_k / n_total for c in clients}

    w = np.zeros(d)
    hist = []
    eta_hist = []

    for t in range(1, rounds + 1):
        eps_t = accumulated_epsilon(t, sigma, delta)
        lr_t = eta0
        eta_hist.append(lr_t)

        updates, weights, n_gated = [], [], 0
        for c in clients:
            bad = c.client_id in compromised_ids
            X_use = c.X_attacked if (bad and poison_mode == "data") else c.X_clean
            w_local = local_dp_sgd(w.copy(), w, X_use, c.y_clean, clips[c.client_id],
                                    sigma, mu, lr_t, local_steps, batch_size, rng)
            if bad and poison_mode == "gradient":
                delta_w = w_local - w
                w_local = w - delta_w
            if gate_fn is not None and gate_fn(c.client_id, t, w_local, w):
                n_gated += 1
                continue
            updates.append(w_local)
            weights.append(p_k[c.client_id])

        if updates:
            weights = np.array(weights) / np.sum(weights)
            w = np.sum([a * u for a, u in zip(weights, updates)], axis=0)

        g = per_sample_grad(w, X_test_aug, y_test).mean(axis=0)
        hist.append(RoundLog(
            round=t, epsilon_t=eps_t, lr_t=lr_t,
            global_loss=loss(w, X_test_aug, y_test),
            global_acc=accuracy(w, X_test_aug, y_test),
            grad_norm_sq=float(np.sum(g ** 2)),
            n_gated_out=n_gated,
        ))

    return {"history": hist, "w_global": w, "beta": beta, "mu_prox": mu,
            "eta0": eta0, "clip_norms": clips, "eta_history": eta_hist}


def convergence_bound_rhs(F_w0, F_star, eta0, beta, sigma, clip_avg, d, m, rounds, eta_hist, gamma, mu):
    opt = 2.0 * (F_w0 - F_star) / (eta0 * rounds)
    dp = (beta * sigma ** 2 * clip_avg ** 2 * d / (m * rounds)) * float(np.sum(np.array(eta_hist) ** 2))
    het = 4.0 * beta ** 2 * gamma ** 2 / mu ** 2
    return {"optimization_error": opt, "dp_noise_error": dp, "heterogeneity_error": het, "total": opt + dp + het}
