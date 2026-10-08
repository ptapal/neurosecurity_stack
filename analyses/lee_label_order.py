import json

import numpy as np

from config import DATASETS, RESULTS_DIR
from cohort import select_subjects
from raw_loader import CLEAN_LOADERS

N_SUBJECTS = 15
N_BLOCKS = 4


def check(s):
    y = s.labels.astype(float)
    order = np.arange(len(y), dtype=float)
    r = float(np.corrcoef(y, order)[0, 1])
    blocks = np.array_split(y, N_BLOCKS)
    imb = max(abs(b.mean() - 0.5) for b in blocks)
    return {"n": int(len(y)), "corr_label_order": r, "max_block_imbalance": float(imb),
            "class_frac_1": float(y.mean())}


if __name__ == "__main__":
    ids = select_subjects("lee", DATASETS["lee"][1](), N_SUBJECTS, seed=None)
    rows = {}
    for sid in ids:
        rows[str(sid)] = check(CLEAN_LOADERS["lee"](sid, max_trials=None))
    r_abs = [abs(v["corr_label_order"]) for v in rows.values()]
    summary = {"max_abs_corr": max(r_abs), "mean_abs_corr": float(np.mean(r_abs)),
               "max_block_imbalance": max(v["max_block_imbalance"] for v in rows.values()),
               "class_frac_range": [min(v["class_frac_1"] for v in rows.values()),
                                    max(v["class_frac_1"] for v in rows.values())]}
    with open(RESULTS_DIR / "lee_label_order.json", "w") as f:
        json.dump({"summary": summary, "subjects": rows}, f, indent=2)
    print(summary)
