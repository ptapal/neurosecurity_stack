import numpy as np

def estimate_rho(z_seq, max_lag=5):
    z0 = z_seq - z_seq.mean(axis=0, keepdims=True)
    var = (z0 ** 2).sum(axis=0)
    max_lag = min(max_lag, len(z_seq) - 1)
    rhos = np.zeros(max_lag)
    for tau in range(1, max_lag + 1):
        cov = (z0[:-tau] * z0[tau:]).sum(axis=0)
        rhos[tau - 1] = np.mean(cov / np.maximum(var, 1e-12))
    return rhos


def corrected_clip_norm(c, rho):
    corr = max(1.0 + 2.0 * np.sum(rho), 1e-3)
    return float(c / np.sqrt(corr))
