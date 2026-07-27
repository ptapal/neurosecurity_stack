import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"cho": "#0072B2", "lee": "#009E73", "won": "#D55E00", "zhang": "#CC79A7",
          "wang_eo": "#E69F00", "wang_ec": "#E69F00",
          "cogbci_eo": "#56B4E9", "cogbci_ec": "#56B4E9"}
COLOR_THEORY = "#666666"
LABELS = {"cho": "Cho (endogenous, MI)", "lee": "Lee (endogenous, MI)",
          "won": "Won (exogenous, RSVP/P300)", "zhang": "Zhang (exogenous, RSVP/face)",
          "wang_eo": "Wang (no driver, eyes-open)",
          "wang_ec": "Wang (no driver, eyes-closed)",
          "cogbci_eo": "COG-BCI (no driver, eyes-open)",
          "cogbci_ec": "COG-BCI (no driver, eyes-closed)"}
DATASETS = ["cho", "lee", "won", "zhang", "wang_eo", "wang_ec", "cogbci_eo", "cogbci_ec"]

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"


def load(name):
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)


def plot_detection_power(data):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    for ax, name in zip(axes.flat, DATASETS):
        d, color = data[name], COLORS[name]
        eps = [s["epsilon"] for s in d["sigma_sweep"]]
        maha = [s["maha_power"] for s in d["sigma_sweep"]]
        l2 = [s["l2_power"] for s in d["sigma_sweep"]]
        theory = [s["theory_power"] for s in d["sigma_sweep"]]
        ax.plot(eps, maha, color=color, lw=2, marker="o", ms=4, label="Mahalanobis (EMA baseline)")
        ax.plot(eps, l2, color=color, lw=2, ls="--", marker="s", ms=4, alpha=0.6, label="L2 (unwhitened)")
        ax.plot(eps, theory, color=COLOR_THEORY, lw=2, ls=":", label="Theoretical bound")
        ax.axhline(0.05, color="#999999", lw=1, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"Privacy budget $\epsilon$")
        ax.set_title(LABELS[name])
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.2)
    for row in axes:
        row[0].set_ylabel(r"Detection power $\beta^*$ (at $\alpha_0=0.05$)")
    fig.suptitle(r"Pillar 3: $\alpha$-band injection detection power vs. privacy budget")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pillar3_detection_power.png", dpi=150)
    plt.close(fig)


def plot_signature(data):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    for ax, name in zip(axes.flat, DATASETS):
        d, color = data[name], COLORS[name]
        eps = [s["epsilon"] for s in d["sigma_sweep"]]
        far = [s["sig_far"] for s in d["sigma_sweep"]]
        frr = [s["sig_frr"] for s in d["sigma_sweep"]]
        ax.plot(eps, far, color=color, lw=2, marker="o", ms=4, label="Fingerprint FAR (attacked accepted)")
        ax.plot(eps, frr, color=color, lw=2, ls="--", marker="^", ms=4, alpha=0.6, label="Fingerprint FRR")
        ax.set_xscale("log")
        ax.set_xlabel(r"Privacy budget $\epsilon$")
        ax.set_title(LABELS[name])
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7.5, loc="center left")
        ax.grid(alpha=0.2)
    for row in axes:
        row[0].set_ylabel("Rate")
    fig.suptitle("Pillar 2: signature fingerprint FAR/FRR vs. privacy budget\n"
                 "(Ed25519 signature itself verifies correctly throughout -- see sig_crypto_valid_*)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pillar2_signature.png", dpi=150)
    plt.close(fig)


def plot_convergence_bound(data):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, name in zip(axes.flat, DATASETS):
        d, color = data[name], COLORS[name]
        checks = d["pillar1"]["convergence_checks"]
        sigmas = [c["sigma"] for c in checks]
        lhs = [c["lhs_empirical"] for c in checks]
        rhs = [c["rhs_bound"]["total"] for c in checks]
        ax.plot(sigmas, lhs, color=color, lw=2, marker="o", label=r"Empirical $\frac{1}{T}\sum\|\nabla F\|^2$")
        ax.plot(sigmas, rhs, color=COLOR_THEORY, lw=2, ls=":", marker="x", label="Theoretical bound (Eq. convergence_bound)")
        ax.set_yscale("log")
        ax.set_xlabel(r"DP noise multiplier $\sigma$")
        ax.set_title(LABELS[name])
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.2)
    for row in axes:
        row[0].set_ylabel("Squared gradient norm (log scale)")
    fig.suptitle("Pillar 1: DP-FedProx-EEG convergence bound, empirical vs. theoretical")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pillar1_convergence_bound.png", dpi=150)
    plt.close(fig)


def plot_training_integrity(data):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    for ax, name in zip(axes.flat, DATASETS):
        d, color = data[name], COLORS[name]
        ad = d["pillar1"]["attack_demo"]
        rounds = [h["round"] for h in ad["clean"]]
        ax.plot(rounds, [h["acc"] for h in ad["clean"]], color="#333333", lw=2.5, label="Clean")
        ax.plot(rounds, [h["acc"] for h in ad["data_poison_ungated"]], color=color, lw=1.5, ls="--",
                 alpha=0.6, label="Data-corrupted, no gating")
        ax.plot(rounds, [h["acc"] for h in ad["data_poison_gated"]], color=color, lw=1.5, ls=":",
                 alpha=0.9, label="Data-corrupted, gated")
        ax.plot(rounds, [h["acc"] for h in ad["gradient_poison_ungated"]], color="#CC0000", lw=1.8, ls="--",
                 alpha=0.7, label="Crafted gradient, no gating")
        ax.plot(rounds, [h["acc"] for h in ad["gradient_poison_gated"]], color="#CC0000", lw=1.8, ls=":",
                 alpha=0.95, label="Crafted gradient, gated")
        ax.set_xlabel("FedAvg round")
        ax.set_title(f"{LABELS[name]}\n({len(ad['compromised_ids'])}/{ad['n_clients']} clients compromised)", fontsize=9)
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.2)
    for row in axes:
        row[0].set_ylabel("Global model test accuracy")
    fig.suptitle("Pillar 1: training-integrity -- incidental data corruption vs. worst-case crafted-gradient attack")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pillar1_training_integrity.png", dpi=150)
    plt.close(fig)


def plot_passive_inference(data):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    for ax, name in zip(axes.flat, DATASETS):
        d, color = data[name], COLORS[name]
        res = d["passive_inference"]["results"]
        eps = [r["epsilon"] for r in res]
        recon = [max(r["mean_reconstruction_error"], 1e-3) for r in res]
        trivial = [r["trivial_baseline_error"] for r in res]
        ax.plot(eps, recon, color=color, lw=2, marker="o", ms=4, label="Reconstruction error")
        ax.plot(eps, trivial, color=COLOR_THEORY, lw=2, ls=":", label="Trivial (guess population mean)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Privacy budget $\epsilon$")
        ax.set_title(LABELS[name])
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.2)
    for row in axes:
        row[0].set_ylabel("Reconstruction error (log scale)")
    fig.suptitle("Passive Inference: gradient-inversion reconstruction vs. privacy budget\n"
                 "(below the dotted line = adversary beats trivial guessing = confidentiality violated)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "passive_inference.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    data = {name: load(name) for name in DATASETS}
    plot_detection_power(data)
    plot_signature(data)
    plot_convergence_bound(data)
    plot_training_integrity(data)
    plot_passive_inference(data)
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
