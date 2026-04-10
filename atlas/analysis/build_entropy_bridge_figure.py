# -*- coding: utf-8 -*-
"""
build_entropy_bridge_figure.py

Two-panel headline figure showing that the RLHF attractor hypothesis
holds under two independent measurements (D and entropy), and the
entropy version is what lets us extend the atlas to closed frontier
models via logprob-based APIs.

Panel A: D vs entropy LOO cosine per family + early window strengthening
Panel B: Per-category profile comparison (D shape vs entropy shape,
         shape-normalized, with attractor traces overlaid)
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

FAMILIES = ["Gemma-2-2B", "Llama-3.2-3B", "Qwen2.5-3B"]
FAMILY_COLORS = {
    "Gemma-2-2B":   "#1f77b4",
    "Llama-3.2-3B": "#e31a1c",
    "Qwen2.5-3B":   "#33a02c",
}


def main():
    with open(ATLAS / "analysis" / "atlas_v0.2.1_attractor.json") as f:
        d_report = json.load(f)
    with open(ATLAS / "analysis" / "atlas_entropy_v0.2.1.json") as f:
        e_report = json.load(f)

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5),
        gridspec_kw={"width_ratios": [1.0, 1.6], "wspace": 0.32},
    )

    # ── Panel A: LOO cosine comparison D vs entropy ──
    ax = ax1
    width = 0.38
    x = np.arange(len(FAMILIES))

    d_full = [d_report["loo_direction_test"][f]["cosine_loo_vs_true"]
               for f in FAMILIES]
    e_full = [e_report["loo_full"][f]["cos_loo_vs_true"]
               for f in FAMILIES]
    e_early = [e_report["loo_early"][f]["cos_loo_vs_true"]
                for f in FAMILIES]

    ax.bar(x - width, d_full, width,
            label="D (full)", color="#d62728", edgecolor="black", linewidth=0.8)
    ax.bar(x, e_full, width,
            label="Entropy (full)", color="#1f77b4", edgecolor="black", linewidth=0.8)
    ax.bar(x + width, e_early, width,
            label="Entropy (early 0-6)", color="#2ca02c", edgecolor="black", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.3, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axhline(-0.3, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

    # Value labels
    for i, (df, ef, ee) in enumerate(zip(d_full, e_full, e_early)):
        ax.text(i - width, df + (0.03 if df >= 0 else -0.06),
                f"{df:+.2f}", ha="center", fontsize=7)
        ax.text(i, ef + (0.03 if ef >= 0 else -0.06),
                f"{ef:+.2f}", ha="center", fontsize=7)
        ax.text(i + width, ee + (0.03 if ee >= 0 else -0.06),
                f"{ee:+.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("-", "\n") for f in FAMILIES],
                        fontsize=8)
    ax.set_ylabel("LOO cos(predicted, actual RLHF direction)")
    ax.set_ylim(-0.8, 1.2)
    ax.set_title("A. Attractor confirmed by 2 independent measurements",
                  fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: per-category profile shapes D vs entropy ──
    ax = ax2
    x = np.arange(len(CATEGORY_ORDER))

    d_sigs = d_report["signatures"]  # shape-normalized
    e_sigs = e_report["shape_signatures"]
    d_attractor = np.array(d_report["attractor_signature"]["vector"])
    e_attractor = np.array(e_report["attractor_full"])

    # Plot D attractor and entropy attractor as heavy lines
    # Normalize each attractor by its own std so shapes are visible on one scale
    d_scaled = d_attractor / (np.std(d_attractor) + 1e-12)
    e_scaled = e_attractor / (np.std(e_attractor) + 1e-12)

    ax.plot(x, d_scaled, "D-", color="#d62728", markersize=9, linewidth=2.2,
             label="D attractor (std-normalized)",
             markeredgecolor="white", markeredgewidth=1.0)
    ax.plot(x, e_scaled, "o-", color="#1f77b4", markersize=9, linewidth=2.2,
             label="Entropy attractor (std-normalized)",
             markeredgecolor="white", markeredgewidth=1.0)

    # Anti-correlation hint: both dashed horizontal reference
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_SHORT)
    ax.set_ylabel("std-normalized deviation from mean")
    ax.set_title("B. Same attractor, opposite sign convention",
                  fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate the anti-correlation
    mean_r = float(np.mean(list(
        e_report["entropy_vs_D_correlation"].values()
    )))
    ax.text(0.02, 0.02,
            f"per-model mean $r$(D, entropy) = {mean_r:+.2f}\n"
            f"(complementary: high $D$ = low entropy)",
            transform=ax.transAxes, fontsize=8, style="italic",
            va="bottom", ha="left", alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffcc",
                      ec="black", lw=0.6))

    fig.suptitle(
        "Two independent atlas measurements recover the same RLHF "
        "attractor — entropy bridges to closed frontier models",
        fontsize=11, fontweight="bold", y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = ATLAS / "analysis" / "figures" / "atlas_entropy_bridge.png"
    out_pdf = ATLAS / "analysis" / "figures" / "atlas_entropy_bridge.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_pdf}")


if __name__ == "__main__":
    main()
