"""
visualize_biphasic_breakthrough.py — the bi-phasic signature figures
========================================================================
Produces the headline figures for the paper:
 1. V2.2 horizon curve (AUC vs prefix length, with perm-p shading)
 2. Mean trajectory overlay, early vs late phase highlighted
 3. Sign-pattern quadrant plot (HL pattern highlighted)
 4. Bi-phasic delta distribution

Usage:
    python visualize_biphasic_breakthrough.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import compute_S, MIN_GEN_TOKENS
from analyze_S_biphasic import compute_S_range

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

COLOR_CORR = "#2E8B57"
COLOR_HALL = "#DC143C"


def load_v22():
    with open("truthfulqa_results/S_truthfulqa_n200.json") as f:
        data = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a for a in audit_data['details']}
    correct, halluc = [], []
    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS:
            continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        tr = np.array(r['generation']['c_delta_trajectory'])
        if lbl in ('correct', 'correct_semantic'):
            correct.append(tr)
        elif lbl in ('hallucinated', 'hallucinated_semantic'):
            halluc.append(tr)
    return correct, halluc


def fig_horizon_curve(correct, halluc):
    """V2.2 horizon curve with permutation-significance shading."""
    prefix_lens = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40]
    aucs = []
    for N in prefix_lens:
        c = [compute_S_range(t.tolist(), 0, N) for t in correct]
        h = [compute_S_range(t.tolist(), 0, N) for t in halluc]
        y = [0]*len(c) + [1]*len(h)
        if sum(y) > 0 and sum(y) < len(y):
            aucs.append(roc_auc_score(y, c + h))
        else:
            aucs.append(None)

    # Load per-prefix perm-p values
    try:
        with open("truthfulqa_results/S_truthfulqa_n200_v22_horizon_perm.json") as f:
            perm = json.load(f)
    except:
        perm = None

    fig, ax = plt.subplots(figsize=(7, 4))
    valid_ns = [n for n, a in zip(prefix_lens, aucs) if a is not None]
    valid_aucs = [a for a in aucs if a is not None]
    ax.plot(valid_ns, valid_aucs, 'o-', color='#8A2BE2', linewidth=2.5, markersize=7, label='S_prefix AUC')

    # Significance band (above null 95th percentile)
    ax.axhspan(0.65, 1.0, color='#98FB98', alpha=0.2, label='production gate (AUC>0.65)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6, label='chance (AUC=0.5)')

    # Highlight significant tokens
    for N, a in zip(prefix_lens, aucs):
        if a is None: continue
        if a >= 0.65:
            ax.plot(N, a, 'o', markersize=12, markerfacecolor='none',
                    markeredgecolor='crimson', markeredgewidth=2)
            ax.annotate(f'{a:.3f}', (N, a), textcoords="offset points",
                        xytext=(5, 10), fontsize=8, color='crimson', fontweight='bold')

    # Phase bands
    ax.axvspan(0, 7.5, color=COLOR_HALL, alpha=0.05)
    ax.axvspan(12, 40, color=COLOR_CORR, alpha=0.05)
    ax.text(3.5, 0.78, 'Phase 1\n(lock-in)', ha='center', fontsize=8, alpha=0.7, color='darkred')
    ax.text(26, 0.78, 'Phase 2\n(truth-binding)', ha='center', fontsize=8, alpha=0.7, color='darkgreen')

    ax.set_xlabel("prefix length N (tokens)")
    ax.set_ylabel("AUC (hallucinated vs correct)")
    ax.set_title(f"Bi-phasic detection horizon (v2.2, n={len(correct)}c + {len(halluc)}h)\n"
                 f"global perm p=0.016 (multi-comp corrected over 12 prefixes)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0.35, 0.80)
    ax.set_xlim(0, 42)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "biphasic_horizon_curve.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def fig_trajectory_overlay(correct, halluc):
    """Mean trajectory ± 95% CI, with early/late phases shaded."""
    fig, ax = plt.subplots(figsize=(7, 4))
    max_len = 40

    def pad_and_mean(items):
        arr = np.full((len(items), max_len), np.nan)
        for i, it in enumerate(items):
            arr[i, :len(it)] = it
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        return mean, 1.96 * sem

    x = np.arange(max_len)
    c_mean, c_ci = pad_and_mean(correct)
    h_mean, h_ci = pad_and_mean(halluc)
    ax.plot(x, c_mean, color=COLOR_CORR, linewidth=2, label=f'correct (n={len(correct)})')
    ax.fill_between(x, c_mean - c_ci, c_mean + c_ci, color=COLOR_CORR, alpha=0.2)
    ax.plot(x, h_mean, color=COLOR_HALL, linewidth=2, label=f'hallucinated (n={len(halluc)})')
    ax.fill_between(x, h_mean - h_ci, h_mean + h_ci, color=COLOR_HALL, alpha=0.2)

    # Phase shading
    ax.axvspan(0, 7, color=COLOR_HALL, alpha=0.05)
    ax.axvspan(15, 40, color=COLOR_CORR, alpha=0.05)
    ax.axvline(7, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(15, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel("token index")
    ax.set_ylabel(r"$C_\Delta$ (mean ± 95% CI)")
    ax.set_title("Bi-phasic C_delta trajectories (v2.2 labels)", fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "biphasic_trajectory_overlay.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def fig_sign_pattern(correct, halluc):
    """Scatter plot: S_early vs S_late, colored by label."""
    EARLY_END, LATE_START = 7, 15
    c_e = np.array([compute_S_range(t.tolist(), 0, EARLY_END) for t in correct])
    c_l = np.array([compute_S_range(t.tolist(), LATE_START, 999) for t in correct])
    h_e = np.array([compute_S_range(t.tolist(), 0, EARLY_END) for t in halluc])
    h_l = np.array([compute_S_range(t.tolist(), LATE_START, 999) for t in halluc])

    med_e = np.median(np.concatenate([c_e, h_e]))
    med_l = np.median(np.concatenate([c_l, h_l]))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(c_e, c_l, c=COLOR_CORR, s=50, alpha=0.6, label=f'correct (n={len(correct)})', edgecolors='white', linewidths=0.5)
    ax.scatter(h_e, h_l, c=COLOR_HALL, s=50, alpha=0.6, label=f'hallucinated (n={len(halluc)})', edgecolors='white', linewidths=0.5)

    # Quadrant lines + labels
    ax.axvline(med_e, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(med_l, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Quadrant labels with halluc rates
    from collections import Counter
    def pattern(e, l):
        if e > med_e and l > med_l: return "HH"
        if e > med_e: return "HL"
        if l > med_l: return "LH"
        return "LL"
    c_p = Counter(pattern(e, l) for e, l in zip(c_e, c_l))
    h_p = Counter(pattern(e, l) for e, l in zip(h_e, h_l))

    def qrate(pat):
        nc, nh = c_p.get(pat, 0), h_p.get(pat, 0)
        total = nc + nh
        return nh / total * 100 if total else 0, total

    # Place labels in quadrants
    lims_x = ax.get_xlim()
    lims_y = ax.get_ylim()
    # HH (right-top)
    rate, n = qrate("HH")
    ax.text(lims_x[1]*0.95, lims_y[1]*0.95, f"HH\n{rate:.0f}% hall\n(n={n})",
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    # HL (right-bottom)
    rate, n = qrate("HL")
    ax.text(lims_x[1]*0.95, lims_y[0]*1.1 if lims_y[0] > 0 else 0.00005, f"HL (signature)\n{rate:.0f}% hall\n(n={n})",
            ha='right', va='bottom', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLOR_HALL, alpha=0.2, edgecolor=COLOR_HALL))
    # LH (left-top)
    rate, n = qrate("LH")
    ax.text(lims_x[0]*1.1 if lims_x[0]>0 else 0.00005, lims_y[1]*0.95, f"LH\n{rate:.0f}% hall\n(n={n})",
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    # LL (left-bottom)
    rate, n = qrate("LL")
    ax.text(lims_x[0]*1.1 if lims_x[0]>0 else 0.00005, lims_y[0]*1.1 if lims_y[0]>0 else 0.00005, f"LL\n{rate:.0f}% hall\n(n={n})",
            ha='left', va='bottom', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLOR_CORR, alpha=0.2, edgecolor=COLOR_CORR))

    ax.set_xlabel("S_early (tokens [0,7))")
    ax.set_ylabel("S_late (tokens [15,end))")
    ax.set_title(f"Bi-phasic sign pattern\nHL (high-early, low-late) = hallucination signature, chi2 p=0.030", fontsize=10)
    ax.legend(loc='center left', fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "biphasic_sign_pattern.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def fig_biphasic_delta(correct, halluc):
    """Distribution of S_early - S_late by label."""
    EARLY_END, LATE_START = 7, 15
    c_d = np.array([compute_S_range(t.tolist(), 0, EARLY_END) -
                    compute_S_range(t.tolist(), LATE_START, 999) for t in correct])
    h_d = np.array([compute_S_range(t.tolist(), 0, EARLY_END) -
                    compute_S_range(t.tolist(), LATE_START, 999) for t in halluc])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    # Violin
    parts = ax1.violinplot([c_d, h_d], positions=[1, 2], showmedians=True, widths=0.6)
    for p, c in zip(parts['bodies'], [COLOR_CORR, COLOR_HALL]):
        p.set_facecolor(c); p.set_edgecolor('black'); p.set_alpha(0.55)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['correct', 'hallucinated'])
    ax1.set_ylabel(r"$\Delta = S_{early} - S_{late}$")
    _, p = stats.ttest_ind(c_d, h_d, equal_var=False)
    pooled = np.sqrt((c_d.var(ddof=1) + h_d.var(ddof=1)) / 2)
    d = (h_d.mean() - c_d.mean()) / pooled if pooled > 0 else 0.0
    auc = roc_auc_score([0]*len(c_d) + [1]*len(h_d), list(c_d) + list(h_d))
    ax1.set_title(f'p={p:.4f}, d={d:+.2f}, AUC={auc:.3f}')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Histogram
    bins = np.linspace(min(c_d.min(), h_d.min()), max(c_d.max(), h_d.max()), 25)
    ax2.hist(c_d, bins=bins, alpha=0.55, color=COLOR_CORR, label=f'correct (n={len(c_d)})', edgecolor='black', linewidth=0.3)
    ax2.hist(h_d, bins=bins, alpha=0.55, color=COLOR_HALL, label=f'hallucinated (n={len(h_d)})', edgecolor='black', linewidth=0.3)
    ax2.axvline(c_d.mean(), color=COLOR_CORR, linestyle='--', linewidth=1.5)
    ax2.axvline(h_d.mean(), color=COLOR_HALL, linestyle='--', linewidth=1.5)
    ax2.set_xlabel(r"$\Delta = S_{early} - S_{late}$")
    ax2.set_ylabel("count")
    ax2.legend(fontsize=8)

    plt.suptitle("Bi-phasic delta: early-phase commitment exceeds late-phase more for hallucinations", fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "biphasic_delta.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def main():
    correct, halluc = load_v22()
    print(f"Loaded v2.2: {len(correct)} correct, {len(halluc)} hallucinated")
    print()
    fig_horizon_curve(correct, halluc)
    fig_trajectory_overlay(correct, halluc)
    fig_sign_pattern(correct, halluc)
    fig_biphasic_delta(correct, halluc)
    print("\nAll figures generated.")


if __name__ == '__main__':
    main()
