"""Lag-1 autocorrelation of the alpha-power embedding z_t over the chronological trial sequence, per cohort.

Conditions of paper Table 11: 15 subjects per cohort, the first max_trials trials per subject (200 for Cho and
Lee, 150 otherwise), mean over channels. Wang and COG-BCI skip their epoch shuffle and Zhang sorts trials by
run and sample position (ordered=True), so the sequence is chronological. Cho was recorded with randomized
left/right instructions but is released as one block per class with no index of the original order, so the
interleaved sequence cannot be recovered; its value is the lag-1 autocorrelation of each class block (trials in
stored order), averaged over the two blocks.

Won and Zhang epochs (0.8 s) are cut around stimuli presented about every 0.1 s, so consecutive epochs share
samples and their alpha power is correlated by construction. non_overlap restricts the statistic to pairs of
consecutive trials whose epochs do not overlap in time (same run, onset gap of at least one epoch length). For the
other cohorts epochs never overlap, so non_overlap equals sequence.
"""
import numpy as np

from cohort import select_subjects
from config import COHORTS, DATASETS, FS
from features import encode

ORDERED = {"zhang", "wang_eo", "wang_ec", "cogbci_eo", "cogbci_ec"}


def lag1_autocorr(z, pair_mask=None):
    """Mean lag-1 autocorrelation across channels, z shape (n_windows, r); pair_mask selects the (n-1) pairs used."""
    if z.shape[0] < 3:
        return np.nan
    if pair_mask is None:
        pair_mask = np.ones(z.shape[0] - 1, dtype=bool)
    if pair_mask.sum() < 5:
        return np.nan
    z_c = z - z.mean(axis=0)
    num = np.sum(z_c[:-1][pair_mask] * z_c[1:][pair_mask], axis=0) * (z.shape[0] - 1) / pair_mask.sum()
    den = np.sum(z_c**2, axis=0)
    den = np.where(den < 1e-12, np.nan, den)
    return np.nanmean(num / den)


def non_overlap_mask(sub, fs):
    if sub.times is None:
        return None
    gap = np.where(sub.runs[:-1] == sub.runs[1:], np.diff(sub.times), np.inf)
    return gap >= sub.trials.shape[-1] / fs


def run(name, n_subjects=15):
    load_fn, list_fn, max_trials = DATASETS[name]
    kwargs = {"ordered": True} if name in ORDERED else {}
    ids = select_subjects(name, list_fn(), n_subjects)
    acs = []
    for sid in ids:
        sub = load_fn(sid, max_trials=max_trials, **kwargs)
        z = encode(sub.trials, FS)
        if name == "cho":
            ac = np.mean([lag1_autocorr(z[sub.labels == c]) for c in (0, 1)])
            acs.append((ac, ac))
        else:
            acs.append((lag1_autocorr(z), lag1_autocorr(z, non_overlap_mask(sub, FS))))
    acs = np.array(acs)
    return acs.mean(axis=0), acs.std(axis=0), len(acs)


if __name__ == "__main__":
    print(f"{'cohort':>10} {'sequence':>9} {'std':>7} {'non_overlap':>12} {'std':>7} {'n_subj':>7}")
    for name in COHORTS:
        m, s, n = run(name)
        print(f"{name:>10} {m[0]:9.3f} {s[0]:7.3f} {m[1]:12.3f} {s[1]:7.3f} {n:7d}")
