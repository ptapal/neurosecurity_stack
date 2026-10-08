import sys

import numpy as np

from analyses.bound_check import prep_dataset, quad_form_moments
from config import ALPHA0

C_HW = 1.0 / 8.0

_GAUSSIAN_TRUE_PSI2 = np.sqrt(8.0 / 3.0)


def estimate_subgaussian_norm(w_pooled, ps=(2, 4, 6, 8)):
    vals = [np.mean(np.abs(w_pooled) ** p) ** (1.0 / p) / np.sqrt(p) for p in ps]
    raw = float(max(vals))
    gaussian_ref = np.random.default_rng(0).standard_normal(200_000)
    ref_vals = [np.mean(np.abs(gaussian_ref) ** p) ** (1.0 / p) / np.sqrt(p) for p in ps]
    return raw * _GAUSSIAN_TRUE_PSI2 / max(ref_vals)


def hw_threshold_and_power(M, Sigma_eta, mu0, mu1, K, alpha0=ALPHA0):
    r = len(mu0)
    delta = mu1 - mu0
    A_mat = M @ Sigma_eta
    fro2 = float(np.trace(A_mat @ A_mat.T))
    op = float(np.linalg.norm(A_mat, ord=2))

    m0, _ = quad_form_moments(M, Sigma_eta, np.zeros(r))
    m1, _ = quad_form_moments(M, Sigma_eta, delta)

    A_coef = K**4 * fro2
    B_coef = K**2 * op
    L = np.log(2.0 / alpha0) / C_HW

    t_quad = np.sqrt(A_coef * L) if A_coef > 0 else 0.0
    crossover = A_coef / B_coef if B_coef > 0 else np.inf
    t = t_quad if t_quad <= crossover else B_coef * L
    tau = m0 + t

    gap = m1 - tau
    if gap <= 0:
        beta_hw = 0.0
    else:
        val = min(gap**2 / A_coef if A_coef > 0 else np.inf, gap / B_coef if B_coef > 0 else np.inf)
        beta_hw = max(0.0, 1.0 - 2.0 * np.exp(-C_HW * val))
    return tau, beta_hw, m0, m1


def run(name, sigma):
    bases, mu1s, calibs = prep_dataset(name)
    r = len(bases[0].mu)

    betaHW_list, K_list = [], []
    for b, mu1, z_calib in zip(bases, mu1s, calibs):
        Sigma_hat = b.cov
        Sigma_eta = Sigma_hat + sigma**2 * np.eye(r)
        M = np.linalg.inv(Sigma_hat)

        evals, evecs = np.linalg.eigh(Sigma_hat)
        evals = np.clip(evals, 1e-10, None)
        Sigma_hat_inv_sqrt = evecs @ np.diag(evals**-0.5) @ evecs.T
        resid = z_calib - z_calib.mean(axis=0)
        w = resid @ Sigma_hat_inv_sqrt.T
        K = estimate_subgaussian_norm(w.ravel())
        K_list.append(K)

        tau, beta_hw, m0, m1 = hw_threshold_and_power(M, Sigma_eta, b.mu, mu1, K)
        betaHW_list.append(beta_hw)

    return float(np.mean(betaHW_list)), float(np.mean(K_list))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "cho"
    sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    beta_hw, K = run(name, sigma)
    print(f"{name} sigma={sigma} beta_HW={beta_hw:.4f} K={K:.3f}")
