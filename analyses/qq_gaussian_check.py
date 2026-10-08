import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cohort import prep_dataset
from config import COHORTS, FIG_DIR, RIDGE
from detect import fit_baseline, mahalanobis_score
from plots.colors import LABELS


def run_cohort(name):
    prep, clip = prep_dataset(name)
    r = prep[0]["z_calib"].shape[1]

    maha2_all = []
    shapiro_p = []
    for p in prep:
        z = p["z_calib"]
        if z.shape[0] < r + 5:
            continue
        base = fit_baseline(z, clip_norm=clip, ridge=RIDGE)
        maha2_all.append(mahalanobis_score(z, base))
        for ch in range(r):
            if z.shape[0] >= 8:
                _, p_val = stats.shapiro(z[:, ch])
                shapiro_p.append(p_val)

    maha2_all = np.concatenate(maha2_all)
    shapiro_p = np.array(shapiro_p)
    return {"name": name, "r": r, "maha2": maha2_all,
            "shapiro_reject_frac": float(np.mean(shapiro_p < 0.05)),
            "n_shapiro_tests": len(shapiro_p)}


def main():
    results = {name: run_cohort(name) for name in COHORTS}

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, name in zip(axes.flat, COHORTS):
        res = results[name]
        maha2 = np.sort(res["maha2"])
        n = len(maha2)
        r = res["r"]
        theo_q = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=r)
        ax.scatter(theo_q, maha2, s=6, alpha=0.4)
        lims = [0, max(theo_q.max(), maha2.max()) * 1.05]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(r"Theoretical $\chi^2_{%d}$ quantile" % r)
        ax.set_ylabel("Empirical squared Mahalanobis dist.")
        ax.set_title(f"{LABELS[name]}\nShapiro reject frac.={res['shapiro_reject_frac']:.2f} (n={res['n_shapiro_tests']})",
                     fontsize=9)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    out = FIG_DIR / "gaussianity_qq_check.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print(f"\n{'cohort':>12} {'r':>4} {'n_maha':>8} {'shapiro_reject_frac':>20}")
    for name in COHORTS:
        res = results[name]
        print(f"{name:>12} {res['r']:>4} {len(res['maha2']):>8} {res['shapiro_reject_frac']:>20.3f}")


if __name__ == "__main__":
    main()
