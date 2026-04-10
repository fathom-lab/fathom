# -*- coding: utf-8 -*-
"""
build_d_axis_cross_arch.py -- 4-model D-axis cross-architecture figure
=======================================================================

Single-panel figure showing the four n=30 pilot results side-by-side:
  Gemma-2-2B (base)           REVERSED, dz=+0.65
  Gemma-2-2B-IT               H-D2,     dz=-0.58
  Llama-3.2-3B (base)         H-D2,     dz=-1.06  (strongest)
  Llama-3.2-3B-Instruct       H-D2 weak, dz=-0.13

Output: figures/d_axis_cross_arch.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def load_deltas(path):
    with open(path) as f:
        data = json.load(f)
    return [
        p["adversarial_summary"]["d_mean"] - p["direct_summary"]["d_mean"]
        for p in data["pairs"]
        if p.get("direct_summary") and p.get("adversarial_summary")
    ]


MODELS = [
    ("Gemma-2-2B\n(base)",
     "d_axis_pilot_gemma-2-2b_fast.json",
     "#1f78b4", None),
    ("Gemma-2-2B-IT\n(instruct)",
     "d_axis_pilot_gemma-2-2b-it_fast.json",
     "#a6cee3", None),
    ("Llama-3.2-3B\n(base)",
     "d_axis_pilot_Llama-3.2-3B_fast.json",
     "#e31a1c", None),
    ("Llama-3.2-3B-IT\n(instruct)",
     "d_axis_pilot_Llama-3.2-3B-Instruct_fast.json",
     "#fb9a99", None),
    ("Qwen2.5-3B\n(base)",
     "d_axis_pilot_Qwen2.5-3B_fast.json",
     "#33a02c", None),
    ("Qwen2.5-3B-IT\n(instruct)",
     "d_axis_pilot_Qwen2.5-3B-Instruct_fast.json",
     "#b2df8a", None),
]


def _compute_dz(path):
    import numpy as np
    deltas = np.array(load_deltas(path), dtype=float)
    if len(deltas) < 2:
        return 0.0
    return float(deltas.mean() / (deltas.std(ddof=1) + 1e-12))


def main():
    # Drop models whose result files don't exist yet
    models_present = [
        (label, fname, color, dz_str)
        for (label, fname, color, dz_str) in MODELS
        if (ROOT / "oversight_results" / fname).exists()
    ]
    # Re-compute dz labels
    models_present = [
        (label, fname, color,
         f"{_compute_dz(ROOT/'oversight_results'/fname):+.2f}")
        for (label, fname, color, _) in models_present
    ]
    # Shadow the module-level MODELS inside main
    models_loop = models_present

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
    })

    fig_w = max(10.5, 2.0 * len(models_loop) + 2.0)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fig_w, 4.0),
        gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.35},
    )

    # Panel A: per-pair deltas, overlaid
    ax = ax1
    n_models = len(models_loop)
    for i, (label, fname, color, _) in enumerate(models_loop):
        deltas = load_deltas(ROOT / "oversight_results" / fname)
        xs = np.arange(len(deltas)) + (i - (n_models - 1) / 2) * 0.14
        ax.scatter(xs, deltas, s=18, color=color, edgecolors="black",
                   linewidths=0.3, label=label.replace("\n", " "))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("matched pair index")
    ax.set_ylabel(r"$\Delta D$ = adversarial $-$ direct")
    ax.set_title("A. Per-pair $\\Delta D$ across four models",
                 fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: effect sizes
    ax = ax2
    means = []
    sems = []
    names = []
    colors = []
    dzs = []
    for label, fname, color, dz_str in models_loop:
        deltas = np.array(load_deltas(ROOT / "oversight_results" / fname))
        means.append(deltas.mean())
        sems.append(deltas.std(ddof=1) / np.sqrt(len(deltas)))
        names.append(label)
        colors.append(color)
        dzs.append(dz_str)

    x = np.arange(len(names))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors,
                   edgecolor="black", linewidth=0.6,
                   error_kw={"linewidth": 1.1})
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel(r"mean $\Delta D$ $\pm$ SEM")
    ax.set_title("B. Mean $\\Delta D$ (dz annotated)",
                 fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate with dz values
    for i, (m, dz) in enumerate(zip(means, dzs)):
        y = m + (0.003 if m > 0 else -0.003)
        ax.text(i, y, f"$d_z${dz}",
                ha="center", fontsize=8,
                va="bottom" if m > 0 else "top",
                fontweight="bold")

    # Shaded H-D2 region below zero
    ax.axhspan(-0.035, 0, facecolor="#d4f1d4", alpha=0.4, zorder=-1)
    ax.axhspan(0, 0.03, facecolor="#f7d4d4", alpha=0.4, zorder=-1)
    last_x = len(models_loop) - 0.5
    ax.text(last_x, -0.010, "H-D2",
            ha="right", fontsize=7, alpha=0.7, style="italic")
    ax.text(last_x, 0.015, "reversed",
            ha="right", fontsize=7, alpha=0.7, style="italic")

    n_h_d2 = sum(1 for m in means if m < 0)
    fig.suptitle(
        f"D-axis across {len(models_loop)} models: H-D2 direction "
        f"replicates on {n_h_d2}/{len(models_loop)}",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    pdf_path = ROOT / "figures" / "d_axis_cross_arch.pdf"
    png_path = ROOT / "figures" / "d_axis_cross_arch.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[SAVED] {pdf_path}")
    print(f"[SAVED] {png_path}")


if __name__ == "__main__":
    main()
