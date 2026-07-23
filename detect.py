from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from privacy import clip_l2


@dataclass
class Baseline:
    mu: np.ndarray
    std: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray


def standardize(z, mu, std):
    return (z - mu) / np.maximum(std, 1e-8)


def choose_clip_norm(z, q=0.95):
    return float(np.quantile(np.linalg.norm(z, axis=-1), q))


def fit_baseline(z_calib, clip_norm, ridge=1e-2):
    mu = z_calib.mean(axis=0)
    std = z_calib.std(axis=0) + 1e-8
    zs = clip_l2(standardize(z_calib, mu, std), clip_norm)
    cov = np.cov(zs, rowvar=False)
    cov_reg = cov + ridge * np.eye(cov.shape[0])
    return Baseline(mu=zs.mean(axis=0), std=std, cov=cov_reg, cov_inv=np.linalg.inv(cov_reg))


def ema_settle(z_seq, mu0, gamma=0.1):
    mu = mu0.copy()
    for z in z_seq:
        mu = (1.0 - gamma) * mu + gamma * z
    return mu


def l2_score(z, base):
    diff = z - base.mu
    return np.sum(diff ** 2, axis=-1)


def mahalanobis_score(z, base):
    diff = z - base.mu
    return np.einsum("ni,ij,nj->n", diff, base.cov_inv, diff)


def empirical_threshold(scores_clean, alpha0):
    return float(np.quantile(scores_clean, 1.0 - alpha0))


def empirical_power(scores_atk, tau):
    return float(np.mean(scores_atk > tau))


def theoretical_power_bound(mu_diff, sigma, c, d, nu2, alpha0):
    denom = np.sqrt(sigma ** 2 * c ** 2 * d + nu2)
    z = mu_diff / denom - norm.ppf(1.0 - alpha0)
    return float(norm.cdf(z))
