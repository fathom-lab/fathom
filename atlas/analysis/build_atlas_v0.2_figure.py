# -*- coding: utf-8 -*-
"""
build_atlas_v0.2_figure.py -- Two-panel v0.2 headline figure

Left: cognitive distance matrix heatmap across 6 models
      (base/IT across Gemma, Llama, Qwen)
Right: category-ranking agreement heatmap across the same 6 models

Shows the two headline findings of v0.2:
1. Gemma base is a cognitive outlier
2. RLHF pulls Gemma-IT into the Qwen/Llama-IT cluster
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ATLAS = ROOT / "atlas"


MODEL_ORDER = [
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
]

SHORT_LABELS = {
    "google/gemma-2-2b": "Gemma-2\nbase",
    "google/gemma-2-2b-it": "Gemma-2\nIT",
    "meta-llama/Llama-3.2-3B": "Llama-3.2\nbase",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2\nIT",
    "Qwen/Qwen2.5-3B": "Qwen-2.5\nbase",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen-2.5\nIT",
}


def load_summary():
    with open(ATLAS / "analysis" / "atlas_v0.1_summary.json") as f:
        return json.load(f)


def build():
    summary = load_summary()
    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8),
                                     gridspec_kw={"wspace": 0.45})

    labels = [SHORT_LABELS[m] for m in MODEL_ORDER]
    n = len(MODEL_ORDER)

    # Panel A: distance matrix
    dist_matrix = np.zeros((n, n))
    for i, m1 in enumerate(MODEL_ORDER):
        for j, m2 in enumerate(MODEL_ORDER):
            d = summary["q2_cross_model"]["distances"].get(m1, {}).get(m2)
            dist_matrix[i, j] = d if d is not None else 0.0

    im1 = ax1.imshow(dist_matrix, cmap="RdYlGn_r", vmin=0, vmax=2,
                      aspect="equal")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            color = "white" if dist_matrix[i, j] > 1.0 else "black"
            ax1.text(j, i, f"{dist_matrix[i, j]:.2f}",
                      ha="center", va="center", fontsize=8, color=color)
    ax1.set_title("A. Cognitive distance  (1 − Pearson $r$)",
                   fontweight="bold", loc="left")
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.75)
    cbar1.set_label("distance", fontsize=9)

    # Panel B: category ordering agreement
    agreement_matrix = np.eye(n) * 1.0
    agree_data = summary["q3_ordering"]["cross_model_agreement"]
    for pair, info in agree_data.items():
        m1, m2 = pair.split(" vs ")
        if m1 in MODEL_ORDER and m2 in MODEL_ORDER:
            i, j = MODEL_ORDER.index(m1), MODEL_ORDER.index(m2)
            agreement_matrix[i, j] = info["agreement_fraction"]
            agreement_matrix[j, i] = info["agreement_fraction"]

    im2 = ax2.imshow(agreement_matrix, cmap="RdYlGn", vmin=0, vmax=1,
                      aspect="equal")
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = agreement_matrix[i, j]
            color = "black" if 0.2 < v < 0.8 else "white"
            ax2.text(j, i, f"{v:.0%}",
                      ha="center", va="center", fontsize=8, color=color)
    ax2.set_title("B. Category ranking agreement",
                   fontweight="bold", loc="left")
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.75)
    cbar2.set_label("agreement", fontsize=9)

    fig.suptitle(
        "Fathom Cognitive Atlas v0.2 — 6 models, 3 families, "
        "one universal probe set",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = ATLAS / "analysis" / "figures" / "atlas_v0.2_headline.png"
    out_pdf = ATLAS / "analysis" / "figures" / "atlas_v0.2_headline.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_pdf}")


if __name__ == "__main__":
    build()
