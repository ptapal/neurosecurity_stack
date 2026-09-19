"""Pillar 3 detection power under alternative 15-subject draws, from results/subsample_sensitivity.json."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import COHORTS, FIG_DIR, RESULTS_DIR
from plots.colors import COLORS, LABELS


def main():
    with open(RESULTS_DIR / "subsample_sensitivity.json") as f:
        sens = json.load(f)
    names = [n for n in COHORTS if n in sens]

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    for ax, name in zip(axes.flat, names):
        color = COLORS[name]
        for k, d in enumerate(sens[name]["draws"]):
            ax.plot(d["epsilon"], d["maha_power"], color=color, lw=1, alpha=0.35,
                    label="alternative draws" if k == 0 else None)
        with open(RESULTS_DIR / f"{name}.json") as f:
            paper = json.load(f)["sigma_sweep"]
        ax.plot([s["epsilon"] for s in paper], [s["maha_power"] for s in paper],
                color="black", lw=2, ls="--", label="paper subsample")
        ax.set_xscale("log")
        ax.set_xlabel(r"Privacy budget $\epsilon$")
        ax.set_title(f"{LABELS[name]} ({len(sens[name]['draws'])} draws)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(r"Mahalanobis detection power (at $\alpha_0=0.05$)")
    fig.suptitle("Pillar 3 detection power under alternative 15-subject draws")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIG_DIR / "subsample_sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
