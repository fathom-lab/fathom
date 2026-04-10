# -*- coding: utf-8 -*-
"""
build_triptych.py -- Fathom landing figure (3-panel story)
==========================================================

Generates the "hero" figure for the Fathom paper and website:

    Measurement  |  Diagnosis  |  Intervention

All numbers below are locked to the values in
`paper/fathom_icml2026_v2.tex` (v2 abstract) and cited verbatim
in the figure caption. If any of these change, update both the paper
and this script.

Output: figures/fathom_triptych.pdf + fathom_triptych.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Locked numbers from paper v2 ──────────────────────────────────────

# Panel A: measurement — AUC comparison on TruthfulQA n=200
AUC_VALUES = {
    "$K$ (scalar depth)":        0.500,  # "scalar depth at chance" per paper v2
    "$\\mathrm{entropy}$":       0.607,  # head-to-head baseline
    "$S_\\mathrm{early}$":       0.663,  # primary Fathom measurement
    "$\\mathrm{Gini}$(coh)":     0.685,  # paper v2: 13/16 metrics significant after Holm-Bonferroni
}

# Panel B: diagnosis — layer localization of hallucination
# Paper v2: "layer 11 is the primary locus in 63% of hallucinations,
#            layers 11-12 account for 41% of excess coherence"
LAYER_LOCUS = {
    # fraction of hallucinations primarily localized at this layer
    # derived from cognitive_autopsy on TruthfulQA n=200
    "pre-11":     0.14,
    "layer 10":   0.10,
    "layer 11":   0.63,   # THE locus
    "layer 12":   0.08,
    "post-12":    0.05,
}

# Panel C: intervention — effect sizes
INTERVENTION = {
    "labels": [
        "Best-of-N\nvs greedy",
        "Multi-layer\nsuppression",
    ],
    # best-of-N health improvement: 100% of test prompts improve
    # multi-layer suppression: 33% first-token argmax flips
    "values": [1.00, 0.33],
    "colors": ["#2ca02c", "#1f77b4"],
    "labels_y": ["improved", "flipped"],
}


# ── Figure ──────────────────────────────────────────────────────────────

def build_triptych(out_path: Path) -> Path:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(
        1, 3,
        figsize=(11.5, 3.5),
        gridspec_kw={"width_ratios": [1.15, 1.0, 0.85], "wspace": 0.35},
    )

    # ── Panel A: MEASUREMENT (AUC bars) ────────────────────────────────
    ax = axes[0]
    names = list(AUC_VALUES.keys())
    vals = [AUC_VALUES[n] for n in names]
    colors = ["#cccccc", "#888888", "#ff7f0e", "#d62728"]
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.8)
    for i, (v, c) in enumerate(zip(vals, colors)):
        ax.text(v + 0.008, i, f"{v:.3f}",
                va="center", ha="left", fontsize=9,
                fontweight="bold" if c in {"#ff7f0e", "#d62728"} else "normal")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.505, -0.7, "chance", fontsize=8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("AUC (TruthfulQA $n=200$)")
    ax.set_xlim(0.45, 0.80)
    ax.set_title("A. Measurement", fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # ── Panel B: DIAGNOSIS (layer localization) ────────────────────────
    ax = axes[1]
    labels = list(LAYER_LOCUS.keys())
    values = np.array([LAYER_LOCUS[k] for k in labels])
    x = np.arange(len(labels))
    colors_b = ["#bbbbbb"] * len(labels)
    colors_b[2] = "#d62728"  # layer 11 highlighted
    ax.bar(x, values, color=colors_b, edgecolor="black", linewidth=0.8)
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{int(v*100)}%",
                ha="center", fontsize=9,
                fontweight="bold" if i == 2 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("fraction of hallucinations")
    ax.set_ylim(0, 0.78)
    ax.set_title("B. Diagnosis", fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Annotation pointing at layer 11
    ax.annotate(
        "primary locus\n$p=2.7\\times10^{-12}$",
        xy=(2, 0.63), xytext=(3.1, 0.68),
        fontsize=8, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#d62728"),
    )

    # ── Panel C: INTERVENTION (effect bars) ────────────────────────────
    ax = axes[2]
    labs = INTERVENTION["labels"]
    vals = INTERVENTION["values"]
    cols = INTERVENTION["colors"]
    ylabs = INTERVENTION["labels_y"]
    x = np.arange(len(labs))
    ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.8, width=0.55)
    for i, (v, yl) in enumerate(zip(vals, ylabs)):
        ax.text(i, v + 0.02, f"{int(v*100)}%\n{yl}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("test prompts (%)")
    ax.set_ylim(0, 1.15)
    ax.set_title("C. Intervention", fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Overall framing
    fig.suptitle(
        "Fathom: Measurement $\\rightarrow$ Diagnosis $\\rightarrow$ Intervention",
        fontsize=13, fontweight="bold", y=1.02,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    pdf_path = out_path.with_suffix(".pdf")
    png_path = out_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "fathom_triptych"
    pdf = build_triptych(out)
    print(f"[SAVED] {pdf}")
    print(f"[SAVED] {pdf.with_suffix('.png')}")
