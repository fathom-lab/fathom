"""
visualize_meta_analysis.py — forest plot + consistency plot for S_early
=========================================================================
Publication figure showing S_early discrimination across 3 datasets.
Forest plot with d and 95% CI. Demonstrates replication.

Usage:
    python visualize_meta_analysis.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 9,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def d_ci(d, n1, n2, alpha=0.05):
    """Approximate 95% CI on Cohen's d."""
    # Hedges & Olkin variance approximation
    var = (n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2))
    se = np.sqrt(var)
    z = stats.norm.ppf(1 - alpha/2)
    return d - z*se, d + z*se


def main():
    with open("truthfulqa_results/cross_dataset_meta.json") as f:
        meta = json.load(f)
    results = meta['results']
    pooled_d = meta['pooled_d']

    # Compute CIs
    for r in results:
        r['ci_lo'], r['ci_hi'] = d_ci(r['d'], r['n_A'], r['n_B'])

    # === Forest plot ===
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [
        "Gemma-2-2B (base)\nhedge vs commit (paired, n=30)",
        "Gemma-2-2B-IT\nhedge vs commit (paired, n=30)",
        "Gemma-2-2B-IT\nhallucinated vs correct (n=94)",
    ]
    y_pos = np.arange(len(results))[::-1]

    for i, r in enumerate(results):
        # Reverse for top-to-bottom reading
        yi = y_pos[i]
        ax.errorbar(r['d'], yi,
                    xerr=[[r['d'] - r['ci_lo']], [r['ci_hi'] - r['d']]],
                    fmt='s', color='#4A5568', markersize=10, capsize=6,
                    markerfacecolor='#2E8B57', markeredgecolor='black',
                    elinewidth=2, capthick=1.5)
        # AUC annotation
        ax.text(r['ci_hi'] + 0.05, yi,
                f"d={r['d']:+.2f} AUC={r['auc']:.2f} p={r['p']:.3f}",
                va='center', fontsize=8)

    # Pooled diamond
    pooled_y = -1
    ax.scatter(pooled_d, pooled_y, marker='D', s=300, color='#DC143C',
               edgecolors='black', linewidths=1.5, zorder=5)
    ax.text(pooled_d + 0.05, pooled_y, f"POOLED d={pooled_d:+.3f}",
            va='center', fontsize=9, fontweight='bold', color='#DC143C')

    # Reference lines
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0.4, color='#98FB98', linestyle=':', linewidth=1.2,
               alpha=0.5, label='pre-reg gate d=0.40')

    ax.set_yticks(list(y_pos) + [pooled_y])
    ax.set_yticklabels(labels + ["META (Fisher\np_one-sided=0.0008)"], fontsize=8)
    ax.set_xlabel("Cohen's d (target class - baseline class) on S_early [0,7)", fontsize=9)
    ax.set_title("Cross-Dataset S_early Meta-Analysis: consistent d ≈ 0.5 across 3 designs", fontsize=10)
    ax.set_xlim(-0.8, 1.8)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.2)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "meta_analysis_forest.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    # === Consistency scatter: d across datasets ===
    fig, ax = plt.subplots(figsize=(5, 5))
    ds = [r['d'] for r in results]
    aucs = [r['auc'] for r in results]
    colors = ['#4682B4', '#2E8B57', '#DC143C']
    markers = ['o', 's', '^']
    for d, a, c, m, label in zip(ds, aucs, colors, markers,
                                  ["base h/c", "IT h/c", "IT TQA h/c"]):
        ax.scatter(d, a, c=c, marker=m, s=150, edgecolors='black',
                   linewidths=1.5, label=label, alpha=0.8, zorder=3)
    # Ideal band
    ax.axhspan(0.60, 0.70, color='#98FB98', alpha=0.15, label='production AUC target')
    ax.axvspan(0.40, 0.60, color='#FFDAB9', alpha=0.15)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(-0.5, 1.2)
    ax.set_ylim(0.45, 0.80)
    ax.set_xlabel("Cohen's d (S_early discrimination)")
    ax.set_ylabel("AUC")
    ax.set_title("S_early discrimination consistency\nacross 3 experimental designs", fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out2 = os.path.join(FIG_DIR, "meta_consistency.pdf")
    plt.savefig(out2, bbox_inches='tight')
    plt.savefig(out2.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out2}")


if __name__ == '__main__':
    main()
