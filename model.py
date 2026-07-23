import numpy as np


def augment(X):
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.concatenate([X, ones], axis=1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def predict_proba(w, Xa):
    return sigmoid(Xa @ w)


def loss(w, Xa, y, eps=1e-12):
    p = predict_proba(w, Xa)
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def per_sample_grad(w, Xa, y):
    p = predict_proba(w, Xa)
    return (p - y)[:, None] * Xa


def smoothness_beta(Xa):
    n = Xa.shape[0]
    gram = (Xa.T @ Xa) / n
    return 0.25 * float(np.linalg.eigvalsh(gram).max())


def accuracy(w, Xa, y):
    pred = (predict_proba(w, Xa) >= 0.5).astype(np.int64)
    return float(np.mean(pred == y))
