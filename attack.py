import numpy as np


def inject_alpha(trials, fs, strength, freq=10.0, ch_frac=1.0, rng=None):
    rng = rng or np.random.default_rng()
    n, ch, samp = trials.shape

    std = trials.std(axis=-1, keepdims=True)
    t = np.arange(samp) / fs
    phase = rng.uniform(0, 2 * np.pi, size=(n, ch, 1))
    wave = np.sin(2 * np.pi * freq * t[None, None, :] + phase)

    n_hit = max(1, int(round(ch_frac * ch)))
    mask = np.zeros((n, ch, 1), dtype=trials.dtype)
    for i in range(n):
        idx = rng.choice(ch, size=n_hit, replace=False)
        mask[i, idx, 0] = 1.0

    return (trials + strength * std * wave * mask).astype(trials.dtype)
