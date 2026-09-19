from pathlib import Path

import numpy as np

from raw_loader import (list_cho_subjects, list_cogbci_subjects, list_lee_subjects,
                              list_wang_subjects, list_won_subjects, list_zhang_subjects,
                              load_cho_subject, load_cogbci_ec_subject, load_cogbci_eo_subject,
                              load_lee_subject, load_wang_ec_subject, load_wang_eo_subject,
                              load_won_subject, load_zhang_subject)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

FS = 250.0
ALPHA0 = 0.05
TARGET_FRR = 0.05
DELTA = 1e-5
ATTACK_STRENGTH = 0.5
EMA_GAMMA = 0.1
RIDGE = 1e-2
SIGMAS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])

LSH_S = 8
LSH_B = 64

P1_ROUNDS = 20
P1_LOCAL_STEPS = 5
P1_BATCH_SIZE = 32
P1_SIGMAS_CONV = [0.1, 0.5, 2.0]
P1_SIGMA_DEMO = 0.5
P1_FRAC_BAD = 0.3

DATASETS = {
    "cho": (load_cho_subject, list_cho_subjects, 200),
    "lee": (load_lee_subject, list_lee_subjects, 200),
    "won": (load_won_subject, list_won_subjects, 150),
    "zhang": (load_zhang_subject, list_zhang_subjects, 150),
    "wang_eo": (load_wang_eo_subject, list_wang_subjects, 150),
    "wang_ec": (load_wang_ec_subject, list_wang_subjects, 150),
    "cogbci_eo": (load_cogbci_eo_subject, list_cogbci_subjects, 150),
    "cogbci_ec": (load_cogbci_ec_subject, list_cogbci_subjects, 150),
}
COHORTS = list(DATASETS)

# Fixed per-source seeds make subject subsampling reproducible instead of "first N by ID";
# the eyes-open and eyes-closed arms of one source share a seed so they draw the same subjects.
SUBJECT_SAMPLE_SEED = {"cho": 101, "lee": 102, "won": 103, "zhang": 104,
                       "wang": 105, "cogbci": 106}
COHORT_SOURCE = {"cho": "cho", "lee": "lee", "won": "won", "zhang": "zhang",
                 "wang_eo": "wang", "wang_ec": "wang",
                 "cogbci_eo": "cogbci", "cogbci_ec": "cogbci"}

# Subjects drawn by the seeds above from the full datasets; with only these present, all are used.
PAPER_SUBJECTS = {
    "cho": "s06 s08 s12 s14 s15 s18 s27 s28 s29 s34 s37 s39 s45 s46 s48".split(),
    "lee": [f"sub-{i:02d}" for i in (5, 7, 11, 14, 18, 19, 23, 24, 26, 33, 34, 38, 44, 47, 48)],
    "won": [f"sub-{i:03d}" for i in (4, 6, 9, 10, 14, 16, 17, 23, 26, 33, 42, 43, 49, 53, 55)],
    "zhang": [f"S{i}" for i in range(1, 16)],
    "wang": [f"sub-{i:02d}" for i in (5, 9, 12, 13, 15, 22, 26, 27, 30, 34, 36, 40, 48, 49, 53)],
    "cogbci": [f"sub-{i:02d}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 20, 22, 23, 24)],
}
