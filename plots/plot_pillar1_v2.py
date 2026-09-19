"""Convergence figure (empirical vs. Theorem 6.1 bound) from results/<cohort>_pillar1_v2.json."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import COHORTS, FIG_DIR, RESULTS_DIR
from plots.colors import COLOR_THEORY, COLORS, LABELS


def load_v2(name):
    with open(RESULTS_DIR / f"{name}_pillar1_v2.json") as f:
        return json.load(f)


def main():
    data = {name: load_v2(name) for name in COHORTS}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, name in zip(axes.flat, COHORTS):
        d, color = data[name], COLORS[name]
        checks = d["convergence_checks"]
        sigmas = [c["sigma"] for c in checks]
        lhs = [c["lhs_empirical"] for c in checks]
        rhs = [c["rhs_bound"]["total"] for c in checks]
        ax.plot(sigmas, lhs, color=color, lw=2, marker="o", label=r"Empirical $\frac{1}{T}\sum\|\nabla F\|^2$")
        ax.plot(sigmas, rhs, color=COLOR_THEORY, lw=2, ls=":", marker="x", label="Theoretical bound (Eq. convergence\\_bound)")
        ax.set_yscale("log")
        ax.set_xlabel(r"DP noise multiplier $\sigma$")
        ax.set_title(LABELS[name])
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.2)
    for row in axes:
        row[0].set_ylabel("Squared gradient norm (log scale)")
    fig.suptitle("Pillar 1: DP-FedProx-EEG convergence bound, empirical vs. theoretical")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    out = FIG_DIR / "pillar1_convergence_bound_v2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
