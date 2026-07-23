import numpy as np

from model import sigmoid


def batch_gradient(w, X, y):
    p = sigmoid(X @ w)
    return ((p - y)[:, None] * X).mean(axis=0)


def _inversion_grad(w, Xg, y, g_target):
    n = Xg.shape[0]
    p = sigmoid(Xg @ w)
    g = ((p - y)[:, None] * Xg).mean(axis=0)
    r = 2.0 * (g - g_target)
    dp = p * (1.0 - p)
    rx = Xg @ r
    return (1.0 / n) * (dp[:, None] * rx[:, None] * w[None, :] + (p - y)[:, None] * r[None, :])


def reconstruct(w, y, g_target, d, rng, n_steps=500, lr=0.05):
    n = len(y)
    Xg = rng.normal(0.0, 1.0, size=(n, d))
    m, v = np.zeros_like(Xg), np.zeros_like(Xg)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, n_steps + 1):
        g = _inversion_grad(w, Xg, y, g_target)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        Xg = Xg - lr * mh / (np.sqrt(vh) + eps)
    return Xg


def reconstruction_error(Xh, Xt):
    return float(np.sum((Xh - Xt) ** 2))


def trivial_baseline_error(Xt, pop_mean):
    return float(np.sum((Xt - pop_mean[None, :]) ** 2))
