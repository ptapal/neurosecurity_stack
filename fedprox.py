from dataclasses import dataclass, field

import numpy as np

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

    # Participant-level group privacy: a client is one subject and each local mini-batch is
    # drawn from that subject alone, so g* = batch_size and tilde_C = g* * C.
    tilde_C = batch_size * clip_norm_base
    clips = {c.client_id: tilde_C for c in clients}

    beta = smoothness_beta(np.concatenate([c.X_clean for c in clients], axis=0))
    mu = beta
    # Theorem 6.1 step-size condition: eta_0 <= min(1/(mu+beta), 1/(4*beta*E)).
    eta0 = min(1.0 / (mu + beta), 1.0 / (4.0 * beta * local_steps))

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


def estimate_nu2(clients, batch_size, w=None):
    """Empirical minibatch-gradient variance (Assumption 6.2's nu^2): Var(per-sample grad) / batch, averaged over clients."""
    per_client_nu2 = []
    for c in clients:
        n = c.X_clean.shape[0]
        w0 = np.zeros(c.X_clean.shape[1]) if w is None else w
        g_full = per_sample_grad(w0, c.X_clean, c.y_clean)
        var_per_sample = np.sum(np.var(g_full, axis=0))
        nu2 = var_per_sample / min(batch_size, n)
        per_client_nu2.append(nu2)
    return float(np.mean(per_client_nu2))


def convergence_bound_rhs(F_w0, F_star, eta0, beta, mu, sigma, C_tilde, nu2, d, m, E, T):
    """Right-hand side of Theorem 6.1: optimization + DP-noise + drift terms with constant eta_0 and no heterogeneity term."""
    opt = 4.0 * (F_w0 - F_star) / (eta0 * E * T)
    noise = 4.0 * beta * E * (nu2 + sigma ** 2 * C_tilde ** 2 * d) / m * eta0
    B_E = eta0 ** 2 * (beta + mu) * C_tilde * (1.0 + sigma * np.sqrt(d)) * E * (E - 1) / 2.0
    drift = 4.0 * B_E ** 2 / (eta0 * E) * (1.0 / (2.0 * eta0 * E) + beta)
    total = opt + noise + drift
    return {"optimization_error": opt, "dp_noise_error": noise, "drift_error": drift,
            "B_E": B_E, "total": total}
