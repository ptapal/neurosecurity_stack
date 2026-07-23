import numpy as np


def clip_l2(z, c):
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    s = np.minimum(1.0, c / np.maximum(n, 1e-12))
    return z * s


def gaussian_mechanism(z, sigma, rng):
    return z + rng.normal(0.0, sigma, size=z.shape)


def epsilon_of_sigma(sigma, delta=1e-5, c=1.0):
    if sigma <= 0:
        return float("inf")
    return c * np.sqrt(2.0 * np.log(1.25 / delta)) / sigma


def sigma_of_epsilon(eps, delta=1e-5, c=1.0):
    return c * np.sqrt(2.0 * np.log(1.25 / delta)) / eps


_ALPHAS = np.concatenate([np.linspace(1.01, 2.0, 50), np.linspace(2.0, 128.0, 200)])


def rdp_epsilon(t, a, sigma, delta):
    return t * a / (2.0 * sigma ** 2) + np.log(1.0 / delta) / (a - 1.0)


def accumulated_epsilon(t, sigma, delta=1e-5):
    if sigma <= 0 or t <= 0:
        return float("inf")
    return float(np.min(rdp_epsilon(t, _ALPHAS, sigma, delta)))
