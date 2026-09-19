import numpy as np

from attack import inject_alpha
from config import ATTACK_STRENGTH, COHORT_SOURCE, DATASETS, FS, SUBJECT_SAMPLE_SEED
from detect import choose_clip_norm, standardize
from features import encode
from fedprox import Client
from model import augment
from privacy import clip_l2


def select_subjects(name, all_ids, n_subjects, seed=None):
    if len(all_ids) <= n_subjects:
        return all_ids
    if seed is None:
        seed = SUBJECT_SAMPLE_SEED[COHORT_SOURCE[name]]
    rng_sub = np.random.default_rng(seed)
    idx = rng_sub.choice(len(all_ids), size=n_subjects, replace=False)
    return sorted(np.array(all_ids)[idx].tolist())


def load_subjects(name, n_subjects=15):
    load_fn, list_fn, max_trials = DATASETS[name]
    ids = select_subjects(name, list_fn(), n_subjects)
    return [load_fn(sid, max_trials=max_trials) for sid in ids]


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


def prep_dataset(name, n_subjects=15, seed=0):
    rng = np.random.default_rng(seed)
    prep = [prep_subject(s, rng) for s in load_subjects(name, n_subjects)]

    pool = np.concatenate([p["z_thresh_s"] for p in prep], axis=0)
    clip = choose_clip_norm(pool, q=0.95)
    for p in prep:
        p["z_thresh_c"] = clip_l2(p["z_thresh_s"], clip)
        p["z_clean_c"] = clip_l2(p["z_clean_s"], clip)
    return prep, clip


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
