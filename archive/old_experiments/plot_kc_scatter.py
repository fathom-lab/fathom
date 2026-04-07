"""
K x C Scatter Plot — Two-Axis Cognitive Geometry Visualization
===============================================================

Plots depth (K) vs coherence (C) from TruthfulQA experiment results.
Colors by correct vs hallucinated. Shows quadrant distribution.

Usage:
    python plot_kc_scatter.py truthfulqa_results/coherence_*.json
    python plot_kc_scatter.py --all  # plot all result files
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("[!] matplotlib required: pip install matplotlib")
    sys.exit(1)


# Fathom aesthetic
COLORS = {
    "bg": "#0a0a0a",
    "grid": "#1a1a1a",
    "correct": "#00ff88",
    "hallucinated": "#ff4444",
    "text": "#00ffcc",
    "muted": "#555555",
    "accent": "#00ffff",
}


def load_results(path):
    """Load experiment results JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_scatter(data, output_path="kc_scatter.png"):
    """Generate the K×C scatter plot."""
    results = data["results"]

    correct_K = [r["correct"]["K"] for r in results if r["correct"]["K"] is not None]
    correct_C = [r["correct"]["C"] for r in results if r["correct"]["C"] is not None]
    halluc_K = [r["hallucinated"]["K"] for r in results if r["hallucinated"]["K"] is not None]
    halluc_C = [r["hallucinated"]["C"] for r in results if r["hallucinated"]["C"] is not None]

    has_coherence = len(correct_C) > 0 and len(halluc_C) > 0

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    if has_coherence:
        # Full K×C scatter
        ax.scatter(correct_K, correct_C, c=COLORS["correct"], s=60, alpha=0.7,
                   label=f"correct (n={len(correct_K)})", edgecolors="none", zorder=3)
        ax.scatter(halluc_K, halluc_C, c=COLORS["hallucinated"], s=60, alpha=0.7,
                   label=f"hallucinated (n={len(halluc_K)})", edgecolors="none", zorder=3)

        # Median lines for quadrant split
        all_K = correct_K + halluc_K
        all_C = correct_C + halluc_C
        med_K = np.median(all_K)
        med_C = np.median(all_C)
        ax.axvline(med_K, color=COLORS["muted"], linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(med_C, color=COLORS["muted"], linestyle="--", linewidth=0.8, alpha=0.5)

        # Quadrant labels
        k_range = max(all_K) - min(all_K)
        c_range = max(all_C) - min(all_C)
        ax.text(med_K + k_range * 0.15, med_C + c_range * 0.15,
                "high-K, high-C\n(deep + coherent)",
                fontsize=7, color=COLORS["muted"], ha="center", va="center", fontstyle="italic")
        ax.text(med_K - k_range * 0.15, med_C + c_range * 0.15,
                "low-K, high-C\n(shallow + coherent)",
                fontsize=7, color=COLORS["muted"], ha="center", va="center", fontstyle="italic")
        ax.text(med_K + k_range * 0.15, med_C - c_range * 0.15,
                "high-K, low-C\n(deep + scattered)",
                fontsize=7, color=COLORS["muted"], ha="center", va="center", fontstyle="italic")
        ax.text(med_K - k_range * 0.15, med_C - c_range * 0.15,
                "low-K, low-C\n(shallow + scattered)",
                fontsize=7, color=COLORS["muted"], ha="center", va="center", fontstyle="italic")

        ax.set_ylabel("C (Coherence — feature cosine similarity)", color=COLORS["text"],
                       fontsize=11, fontfamily="monospace")
    else:
        # Fallback: K only (no coherence data)
        correct_feat = [r["correct"]["feature_count"] for r in results]
        halluc_feat = [r["hallucinated"]["feature_count"] for r in results]
        ax.scatter(correct_K, correct_feat, c=COLORS["correct"], s=60, alpha=0.7,
                   label=f"correct (n={len(correct_K)})", edgecolors="none", zorder=3)
        ax.scatter(halluc_K, halluc_feat, c=COLORS["hallucinated"], s=60, alpha=0.7,
                   label=f"hallucinated (n={len(halluc_K)})", edgecolors="none", zorder=3)
        ax.set_ylabel("Feature Count (C unavailable)", color=COLORS["text"],
                       fontsize=11, fontfamily="monospace")

    ax.set_xlabel("K (Depth — mean activated layer)", color=COLORS["text"],
                   fontsize=11, fontfamily="monospace")
    ax.set_title(f"FATHOM — Two-Axis Cognitive Geometry\n{data.get('model', 'unknown')} · n={data.get('n_pairs', '?')}",
                  color=COLORS["accent"], fontsize=14, fontfamily="monospace", fontweight="bold")

    # Style
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.spines["left"].set_color(COLORS["muted"])
    ax.legend(loc="upper right", fontsize=9, facecolor=COLORS["bg"],
              edgecolor=COLORS["muted"], labelcolor=COLORS["text"])

    # Watermark
    fig.text(0.98, 0.02, "fathom.darkflobi.com", fontsize=7, color=COLORS["muted"],
             ha="right", va="bottom", fontfamily="monospace", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=COLORS["bg"], bbox_inches="tight")
    print(f"[+] Scatter plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot K×C scatter from TruthfulQA results")
    parser.add_argument("input", nargs="?", help="Path to results JSON")
    parser.add_argument("--all", action="store_true", help="Plot all result files in truthfulqa_results/")
    parser.add_argument("--output", "-o", default="kc_scatter.png", help="Output image path")
    args = parser.parse_args()

    if args.all:
        result_dir = Path("truthfulqa_results")
        files = sorted(result_dir.glob("coherence_*.json"))
        if not files:
            print("[!] No result files found in truthfulqa_results/")
            return
        for f in files:
            data = load_results(f)
            out = f.stem + ".png"
            plot_scatter(data, output_path=str(result_dir / out))
    elif args.input:
        data = load_results(args.input)
        plot_scatter(data, output_path=args.output)
    else:
        print("Usage: python plot_kc_scatter.py <results.json> or --all")


if __name__ == "__main__":
    main()
