# -*- coding: utf-8 -*-
"""
build_d_axis_figure.py -- D-axis base-vs-instruct pilot figure
===============================================================

Two-panel figure for the D-axis pilot section:
  Left:  Per-pair deltas (adv - direct) for base and instruct,
         showing the sign flip.
  Right: Mean deltas with 95% CI; annotations for p and dz.

Data sources:
  oversight_results/d_axis_pilot_gemma-2-2b_fast.json           (base)
  oversight_results/d_axis_pilot_gemma-2-2b-it_fast.json        (instruct)
  oversight_results/d_axis_diff_in_diff.json                    (DoD stats)

Output:
  figures/d_axis_base_vs_instruct.pdf
  figures/d_axis_base_vs_instruct.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def load_deltas(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for p in data["pairs"]:
        d = p.get("direct_summary")
        a = p.get("adversarial_summary")
        if d and a:
            out.append({
                "topic": p["topic"],
                "delta": a["d_mean"] - d["d_mean"],
            })
    return out


def main():
    base = load_deltas(ROOT / "oversight_results" / "d_axis_pilot_gemma-2-2b_fast.json")
    instr = load_deltas(ROOT / "oversight_results" / "d_axis_pilot_gemma-2-2b-it_fast.json")

    with open(ROOT / "oversight_results" / "d_axis_diff_in_diff.json") as f:
        dod = json.load(f)

    # Match by topic
    topics = [d["topic"] for d in base]
    base_by_topic = {d["topic"]: d["delta"] for d in base}
    instr_by_topic = {d["topic"]: d["delta"] for d in instr}

    b = np.array([base_by_topic[t] for t in topics])
    i = np.array([instr_by_topic[t] for t in topics])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0),
                              gridspec_kw={"width_ratios": [1.8, 1.0], "wspace": 0.32})

    # ── Panel A: per-pair deltas ──
    ax = axes[0]
    x = np.arange(len(topics))
    width = 0.38
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    bars_base = ax.bar(x - width/2, b, width,
                        label="Base (Gemma-2-2B)",
                        color="#6baed6", edgecolor="black", linewidth=0.4)
    bars_it = ax.bar(x + width/2, i, width,
                      label="Instruct (Gemma-2-2B-IT)",
                      color="#d62728", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"$\Delta D$ = adversarial $-$ direct")
    ax.set_title("A. Per-pair D-axis deltas", fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shaded regions for interpretation
    ax.text(0.01, 0.97,
            r"$>0$: adv more aligned" "\n" r"$<0$: direct more aligned",
            transform=ax.transAxes, fontsize=7, va="top", alpha=0.6)

    # ── Panel B: mean deltas ──
    ax = axes[1]
    means = [b.mean(), i.mean()]
    # SE of the mean for error bars
    ses = [b.std(ddof=1) / np.sqrt(len(b)),
           i.std(ddof=1) / np.sqrt(len(i))]
    labels = ["Base", "Instruct"]
    colors = ["#6baed6", "#d62728"]

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    bp = ax.bar(labels, means, yerr=ses, capsize=6,
                 color=colors, edgecolor="black", linewidth=0.8,
                 error_kw={"linewidth": 1.2})
    for i_, m in enumerate(means):
        y_text = m + (0.003 if m > 0 else -0.004)
        ax.text(i_, y_text, f"{m:+.4f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel(r"mean $\Delta D$")
    ax.set_title("B. Mean $\\Delta D$ by model", fontweight="bold", loc="left")
    ax.set_ylim(-0.015, 0.04)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation for diff-in-diff
    stats = dod["diff_in_diff"]
    ax.text(0.5, 0.96,
            f"diff-in-diff $d_z$={stats['cohens_dz']:+.2f}\n"
            f"$p$={stats['p_value']:.1e}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="#ffffcc", ec="black", lw=0.6))

    fig.suptitle(
        "D-axis: RLHF causes a sign flip on adversarial framings",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    pdf_path = ROOT / "figures" / "d_axis_base_vs_instruct.pdf"
    png_path = ROOT / "figures" / "d_axis_base_vs_instruct.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"[SAVED] {pdf_path}")
    print(f"[SAVED] {png_path}")


if __name__ == "__main__":
    main()
