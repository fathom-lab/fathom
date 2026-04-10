# -*- coding: utf-8 -*-
"""
build_attractor_figure.py -- Headline figure for the RLHF Attractor finding

Two panels:
  A. Per-category signature profiles (shape-normalized) for all 6 models
     + the attractor point, showing the visual convergence of instruct
     models.
  B. LOO direction prediction: bar chart of cos(d_loo, d_true) per family,
     with the Qwen anomaly clearly visible.

Output: atlas/analysis/figures/atlas_v0.2.1_attractor.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ATLAS = ROOT / "atlas"

CATEGORY_ORDER = ["retrieval", "reasoning", "refusal", "creative",
                   "adversarial", "hallucination"]
CATEGORY_SHORT = ["retr", "reas", "ref", "crea", "adv", "hall"]

FAMILY_COLORS = {
    "Gemma-2-2B":   "#1f77b4",
    "Llama-3.2-3B": "#e31a1c",
    "Qwen2.5-3B":   "#33a02c",
}


def load_report():
    p = ATLAS / "analysis" / "atlas_v0.2.1_attractor.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def shape_normalize(s):
    s = np.asarray(s)
    return s - s.mean()


def main():
    report = load_report()
    attractor = np.array(report["attractor_signature"]["vector"])
    # NOTE: attractor is already shape-normalized in report

    sigs = report["signatures"]
    # Reconstruct shape-normalized signatures
    sigs_shape = {k: shape_normalize(v) for k, v in sigs.items()}

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.8),
        gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.3},
    )

    # ── Panel A: category profiles ──
    ax = ax1
    x = np.arange(len(CATEGORY_ORDER))
    for family in ["Gemma-2-2B", "Llama-3.2-3B", "Qwen2.5-3B"]:
        color = FAMILY_COLORS[family]
        # Base
        s_base = sigs_shape[f"{family}/base"]
        ax.plot(x, s_base, "o--", color=color, alpha=0.55,
                 markersize=7, linewidth=1.2,
                 label=f"{family} base")
        # Instruct
        s_it = sigs_shape[f"{family}/instruct"]
        ax.plot(x, s_it, "s-", color=color, alpha=1.0,
                 markersize=8, linewidth=2.0,
                 label=f"{family} IT")

    # Attractor overlay
    ax.plot(x, attractor, "D-", color="black", markersize=10,
             linewidth=2.5, label="RLHF Attractor", zorder=10,
             markeredgecolor="white", markeredgewidth=1.5)

    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_SHORT)
    ax.set_ylabel("per-category deviation from model mean $D$")
    ax.set_title("A. Shape-normalized signatures + RLHF Attractor",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=7, ncol=2, framealpha=0.85, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: LOO cosine test ──
    ax = ax2
    loo = report["loo_direction_test"]
    fams = list(loo.keys())
    cosines = [loo[f]["cosine_loo_vs_true"] for f in fams]
    colors = [FAMILY_COLORS[f] for f in fams]

    bars = ax.bar(range(len(fams)), cosines, color=colors,
                   edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axhline(0.3, color="gray", linewidth=0.6, alpha=0.4,
                linestyle=":")
    ax.axhline(-0.3, color="gray", linewidth=0.6, alpha=0.4,
                linestyle=":")
    ax.axhline(float(report["loo_mean_cosine"]),
                color="black", linewidth=1.4, alpha=0.6,
                linestyle="--",
                label=f"mean = {report['loo_mean_cosine']:+.2f}")

    for i, (f, c) in enumerate(zip(fams, cosines)):
        y_text = c + (0.05 if c > 0 else -0.08)
        ax.text(i, y_text, f"{c:+.3f}",
                ha="center", fontsize=10,
                fontweight="bold")

    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f.replace("-", "\n") for f in fams], fontsize=8)
    ax.set_ylabel(r"cos($\vec{d}_{\mathrm{LOO}}$, $\vec{d}_{\mathrm{true}}$)")
    ax.set_ylim(-1.05, 1.15)
    ax.set_title("B. Leave-one-out direction prediction",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation for Qwen anomaly
    ax.annotate(
        "Qwen base is\nalready at the\nattractor",
        xy=(2, -0.19), xytext=(2.3, 0.4),
        fontsize=7, ha="left", style="italic", alpha=0.85,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
    )

    fig.suptitle(
        "Fathom Atlas — RLHF Attractor: empirical alignment target in "
        "cognitive space",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    pdf_path = ATLAS / "analysis" / "figures" / "atlas_v0.2.1_attractor.pdf"
    png_path = ATLAS / "analysis" / "figures" / "atlas_v0.2.1_attractor.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"[SAVED] {pdf_path}")
    print(f"[SAVED] {png_path}")


if __name__ == "__main__":
    main()
