"""
visualize_S_trajectories.py — produce trajectory visualizations
=================================================================
Generates paper figures from a completed S-TruthfulQA run:
  1. Trajectory overlay (mean ± CI) by label
  2. Individual trajectory spaghetti plot
  3. C_delta heatmap sorted by label/S
  4. S distribution per label (violin + histograms)
  5. Horizon curve (AUC vs prefix length)
  6. Feature correlation matrix

Usage:
    python visualize_S_trajectories.py [results_file.json]
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import (
    classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY
)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_CORRECT = "#2E8B57"       # sea green
COLOR_HALLUC = "#DC143C"        # crimson
COLOR_NEUTRAL = "#4A5568"


def label_and_collect(data):
    """Label every item, return dict of label -> list of trajectories + S values."""
    buckets = {"correct": [], "hallucinated": []}
    for r in data['results']:
        gen = r['generation']['text']
        n_tokens = r['generation']['total_tokens']
        traj = r['generation']['c_delta_trajectory']
        if n_tokens < MIN_GEN_TOKENS:
            continue
        label, _ = classify_generation(
            gen, r['correct_answers'], r['incorrect_answers'], r.get('best_answer', '')
        )
        if label in buckets:
            buckets[label].append({
                'traj': np.array(traj, dtype=float),
                'S': compute_S(traj),
                'category': r.get('category', '?'),
                'question': r['question'],
            })
    return buckets


def pad_trajectories(items, max_len=None):
    """Pad each trajectory to max_len with nan."""
    if not items:
        return np.empty((0, 0))
    if max_len is None:
        max_len = max(len(i['traj']) for i in items)
    arr = np.full((len(items), max_len), np.nan)
    for i, item in enumerate(items):
        arr[i, :len(item['traj'])] = item['traj']
    return arr


def fig_overlay(buckets, out_path):
    """Mean trajectory ± 95% CI by label."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    max_len = max(max(len(i['traj']) for i in b) for b in buckets.values() if b)
    for label, items in buckets.items():
        if not items:
            continue
        arr = pad_trajectories(items, max_len)
        x = np.arange(max_len)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        color = COLOR_CORRECT if label == 'correct' else COLOR_HALLUC
        ax.plot(x, mean, color=color, linewidth=2.0, label=f"{label} (n={len(items)})")
        ax.fill_between(x, mean - 1.96*sem, mean + 1.96*sem, color=color, alpha=0.2)
    ax.axhline(THR_PRIMARY, color='black', linestyle='--', linewidth=0.5, alpha=0.4, label=f"τ={THR_PRIMARY}")
    ax.set_xlabel("token index")
    ax.set_ylabel(r"$C_\Delta$")
    ax.set_title("Mean trajectory ± 95% CI, by label")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_spaghetti(buckets, out_path, alpha=0.15):
    """All individual trajectories as thin lines + mean."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for label, items in buckets.items():
        if not items:
            continue
        color = COLOR_CORRECT if label == 'correct' else COLOR_HALLUC
        for item in items:
            ax.plot(np.arange(len(item['traj'])), item['traj'],
                    color=color, linewidth=0.5, alpha=alpha)
        # Mean on top
        max_len = max(len(i['traj']) for i in items)
        arr = pad_trajectories(items, max_len)
        mean = np.nanmean(arr, axis=0)
        ax.plot(np.arange(max_len), mean, color=color, linewidth=2.5,
                label=f"{label} mean (n={len(items)})")
    ax.axhline(THR_PRIMARY, color='black', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.set_xlabel("token index")
    ax.set_ylabel(r"$C_\Delta$")
    ax.set_title("Individual trajectories")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_heatmap(buckets, out_path):
    """C_delta heatmap, rows sorted by S within each label."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5),
                             gridspec_kw={"wspace": 0.35})
    max_len = 0
    for items in buckets.values():
        if items:
            max_len = max(max_len, max(len(i['traj']) for i in items))
    vmax = 0
    for items in buckets.values():
        if items:
            vmax = max(vmax, max(i['traj'].max() for i in items))

    for ax, (label, items) in zip(axes, buckets.items()):
        if not items:
            ax.set_title(f"{label}: no data")
            continue
        sorted_items = sorted(items, key=lambda x: x['S'], reverse=True)
        arr = pad_trajectories(sorted_items, max_len)
        im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(f"{label} (n={len(items)}, sorted by S)")
        ax.set_xlabel("token index")
        ax.set_ylabel("item (↓ S)")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_S_distributions(buckets, out_path):
    """Violin + histogram of S per label, with stats."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    S_correct = [i['S'] for i in buckets['correct']]
    S_halluc = [i['S'] for i in buckets['hallucinated']]
    if not S_correct or not S_halluc:
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return

    # Violin
    parts = ax1.violinplot([S_correct, S_halluc], positions=[1, 2], showmedians=True, widths=0.6)
    for p, c in zip(parts["bodies"], [COLOR_CORRECT, COLOR_HALLUC]):
        p.set_facecolor(c); p.set_edgecolor("black"); p.set_alpha(0.55)
    ax1.set_xticks([1, 2]); ax1.set_xticklabels(["correct", "hallucinated"])
    ax1.set_ylabel("S = max/spike_count")
    _, p_val = stats.ttest_ind(S_correct, S_halluc, equal_var=False)
    pooled = np.sqrt((np.var(S_correct, ddof=1) + np.var(S_halluc, ddof=1)) / 2)
    d = (np.mean(S_halluc) - np.mean(S_correct)) / pooled if pooled > 0 else 0.0
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score([0]*len(S_correct) + [1]*len(S_halluc), S_correct + S_halluc)
    ax1.set_title(f"S distribution\np={p_val:.4f}, d={d:.2f}, AUC={auc:.3f}")

    # Histogram
    bins = np.linspace(min(S_correct + S_halluc), max(S_correct + S_halluc), 25)
    ax2.hist(S_correct, bins=bins, alpha=0.55, color=COLOR_CORRECT, label=f"correct (n={len(S_correct)})", edgecolor='black', linewidth=0.3)
    ax2.hist(S_halluc, bins=bins, alpha=0.55, color=COLOR_HALLUC, label=f"hallucinated (n={len(S_halluc)})", edgecolor='black', linewidth=0.3)
    ax2.axvline(np.mean(S_correct), color=COLOR_CORRECT, linestyle='--', linewidth=1.5)
    ax2.axvline(np.mean(S_halluc), color=COLOR_HALLUC, linestyle='--', linewidth=1.5)
    ax2.set_xlabel("S")
    ax2.set_ylabel("count")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_horizon(buckets, out_path):
    """AUC vs prefix length."""
    from analyze_S_horizon import horizon_curve
    labeled = []
    for lbl, items in buckets.items():
        for i in items:
            labeled.append((lbl, i['traj'].tolist()))
    curve = horizon_curve(labeled)
    if curve is None:
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    Ns = [e['N'] for e in curve if e['AUC'] is not None]
    aucs = [e['AUC'] for e in curve if e['AUC'] is not None]
    ax.plot(Ns, aucs, 'o-', color="#8A2BE2", linewidth=2, markersize=6)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='chance')
    ax.axhline(0.65, color='orange', linestyle='--', alpha=0.5, label='production (0.65)')
    ax.axhline(0.70, color='crimson', linestyle='--', alpha=0.5, label='strong (0.70)')
    ax.set_xlabel("prefix length (tokens)")
    ax.set_ylabel("AUC (hallucinated vs correct)")
    ax.set_title("Detection horizon curve — stream-time hallucination detection")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def main(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loading {path}...")
    print(f"Model: {data.get('model')}, N items: {len(data['results'])}")

    buckets = label_and_collect(data)
    print(f"Labels: {len(buckets['correct'])} correct, "
          f"{len(buckets['hallucinated'])} hallucinated")

    if len(buckets['correct']) < 2 or len(buckets['hallucinated']) < 2:
        print("[!] Not enough labeled data for plots")
        return

    stem = os.path.splitext(os.path.basename(path))[0]
    out_stem = os.path.join(FIG_DIR, f"traj_{stem}")

    print(f"\nGenerating figures (prefix: {out_stem}):")
    print("  1. overlay...")
    fig_overlay(buckets, out_stem + "_overlay.pdf")
    print("  2. spaghetti...")
    fig_spaghetti(buckets, out_stem + "_spaghetti.pdf")
    print("  3. heatmap...")
    fig_heatmap(buckets, out_stem + "_heatmap.pdf")
    print("  4. S distributions...")
    fig_S_distributions(buckets, out_stem + "_S_dist.pdf")
    print("  5. horizon curve...")
    fig_horizon(buckets, out_stem + "_horizon.pdf")
    print("\nDone.")


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(p):
        pilot = p.replace('n200.json', 'pilot.json')
        if os.path.exists(pilot):
            p = pilot
        else:
            print(f"ERROR: {p} not found")
            sys.exit(1)
    main(p)
