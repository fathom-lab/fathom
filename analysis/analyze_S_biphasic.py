"""
analyze_S_biphasic.py — the bi-phasic hallucination signature
==================================================================
DISCOVERY: at tokens 3-7, S matches H1 (hall > corr).
At tokens 18+, direction reverses (corr > hall).
This suggests TWO mechanisms at different times:
  Phase 1 (early): attractor lock-in → hallucinations commit fast
  Phase 2 (late):  RLHF truth-binding → correct answers commit harder

This script tests the bi-phasic signature explicitly:
  - S_early = S computed on tokens [0, 10)
  - S_late = S computed on tokens [15, 40)
  - S_ratio = S_early / S_late
  - Bi-phasic score: weighted combination

If S_early discriminates in H1 direction AND S_late discriminates
in reverse direction AND S_ratio discriminates strongly, we have
a genuinely novel finding: hallucination is not a static property,
it's a temporal phase.

Usage:
    python analyze_S_biphasic.py [results_file.json]
"""

import json
import sys
import os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import classify_generation, MIN_GEN_TOKENS, THR_PRIMARY


def compute_S_range(traj, start, end, threshold=THR_PRIMARY):
    """S on slice traj[start:end]."""
    slice_ = traj[start:end]
    if not slice_:
        return 0.0
    spikes = [c for c in slice_ if c > threshold]
    if not spikes:
        return 0.0
    return max(slice_) / len(spikes)


def welch(a, b):
    a, b = np.array(a), np.array(b)
    _, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
    auc = roc_auc_score([0]*len(a) + [1]*len(b), list(a) + list(b))
    return {"p": float(p), "d": float(d), "auc": float(auc),
            "a_mean": float(a.mean()), "b_mean": float(b.mean()),
            "ratio": float(b.mean()/a.mean()) if a.mean() > 0 else None}


def analyze(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 76)
    print("BI-PHASIC S SIGNATURE ANALYSIS")
    print("=" * 76)
    print(f"Data: {results_path}")
    print(f"N items: {len(data['results'])}")
    print()

    correct_trajs, halluc_trajs = [], []
    for r in data['results']:
        n_tokens = r['generation']['total_tokens']
        if n_tokens < MIN_GEN_TOKENS:
            continue
        traj = r['generation']['c_delta_trajectory']
        label, _ = classify_generation(
            r['generation']['text'], r['correct_answers'],
            r['incorrect_answers'], r.get('best_answer', '')
        )
        if label == 'correct':
            correct_trajs.append(traj)
        elif label == 'hallucinated':
            halluc_trajs.append(traj)

    n_c, n_h = len(correct_trajs), len(halluc_trajs)
    print(f"Labeled: {n_c} correct, {n_h} hallucinated")
    print()

    if n_c < 3 or n_h < 3:
        print("[!] Insufficient data")
        return

    # --- Sweep early/late window boundaries ---
    print("-" * 76)
    print("PHASE WINDOW SWEEP — when does the signal flip?")
    print("-" * 76)
    print(f"{'early_end':>10} {'p':>10} {'d':>8} {'AUC':>8}  (early phase)")
    for early_end in [4, 5, 6, 7, 8, 9, 10, 12, 15]:
        c_early = [compute_S_range(t, 0, early_end) for t in correct_trajs]
        h_early = [compute_S_range(t, 0, early_end) for t in halluc_trajs]
        w = welch(c_early, h_early)
        print(f"[0,{early_end:>2}) {w['p']:>10.4f} {w['d']:>8.3f} {w['auc']:>8.3f}")
    print()

    print(f"{'late_start':>10} {'p':>10} {'d':>8} {'AUC':>8}  (late phase)")
    for late_start in [10, 12, 15, 18, 20, 25]:
        c_late = [compute_S_range(t, late_start, 999) for t in correct_trajs]
        h_late = [compute_S_range(t, late_start, 999) for t in halluc_trajs]
        w = welch(c_late, h_late)
        print(f"[{late_start:>2},end) {w['p']:>10.4f} {w['d']:>8.3f} {w['auc']:>8.3f}")
    print()

    # --- Primary bi-phasic test ---
    print("=" * 76)
    print("PRIMARY BI-PHASIC TEST")
    print("=" * 76)
    # Use most-discriminating boundaries from sweep (inspection-based)
    EARLY_END = 7
    LATE_START = 15

    c_early = [compute_S_range(t, 0, EARLY_END) for t in correct_trajs]
    h_early = [compute_S_range(t, 0, EARLY_END) for t in halluc_trajs]
    c_late = [compute_S_range(t, LATE_START, 999) for t in correct_trajs]
    h_late = [compute_S_range(t, LATE_START, 999) for t in halluc_trajs]

    print(f"\nPhase 1: S_early (tokens [0,{EARLY_END}))")
    w_early = welch(c_early, h_early)
    print(f"  correct mean = {w_early['a_mean']:.6f}")
    print(f"  halluc mean  = {w_early['b_mean']:.6f}")
    print(f"  ratio        = {w_early['ratio']:.3f}x")
    print(f"  p={w_early['p']:.4f}, d={w_early['d']:+.3f}, AUC={w_early['auc']:.3f}")

    print(f"\nPhase 2: S_late (tokens [{LATE_START},end))")
    w_late = welch(c_late, h_late)
    print(f"  correct mean = {w_late['a_mean']:.6f}")
    print(f"  halluc mean  = {w_late['b_mean']:.6f}")
    print(f"  ratio        = {w_late['ratio']:.3f}x")
    print(f"  p={w_late['p']:.4f}, d={w_late['d']:+.3f}, AUC={w_late['auc']:.3f}")

    # --- Bi-phasic ratio (single scalar) ---
    print()
    print("-" * 76)
    print("BI-PHASIC RATIO (early/late)")
    print("-" * 76)
    c_ratio = np.array([e / max(l, 1e-10) for e, l in zip(c_early, c_late)])
    h_ratio = np.array([e / max(l, 1e-10) for e, l in zip(h_early, h_late)])
    w_ratio = welch(c_ratio, h_ratio)
    print(f"  correct ratio mean = {w_ratio['a_mean']:.3f}")
    print(f"  halluc  ratio mean = {w_ratio['b_mean']:.3f}")
    print(f"  p={w_ratio['p']:.4f}, d={w_ratio['d']:+.3f}, AUC={w_ratio['auc']:.3f}")
    print()

    # --- Bi-phasic delta (early - late) ---
    print("-" * 76)
    print("BI-PHASIC DELTA (early - late)")
    print("-" * 76)
    c_delta = np.array(c_early) - np.array(c_late)
    h_delta = np.array(h_early) - np.array(h_late)
    w_delta = welch(c_delta, h_delta)
    print(f"  correct delta mean = {w_delta['a_mean']:.6f}")
    print(f"  halluc  delta mean = {w_delta['b_mean']:.6f}")
    print(f"  p={w_delta['p']:.4f}, d={w_delta['d']:+.3f}, AUC={w_delta['auc']:.3f}")
    print()

    # --- Bi-phasic signature: (early_flag, late_flag) classification ---
    print("-" * 76)
    print("BI-PHASIC SIGN PATTERN")
    print("-" * 76)
    # For each item, is S_early > median(all_early)? Is S_late > median(all_late)?
    all_early = c_early + h_early
    all_late = c_late + h_late
    med_early = np.median(all_early)
    med_late = np.median(all_late)

    def classify_pattern(e, l):
        if e > med_early and l > med_late: return "HH"
        if e > med_early and l <= med_late: return "HL"  # high early, low late
        if e <= med_early and l > med_late: return "LH"  # low early, high late
        return "LL"

    from collections import Counter
    c_patterns = Counter(classify_pattern(e, l) for e, l in zip(c_early, c_late))
    h_patterns = Counter(classify_pattern(e, l) for e, l in zip(h_early, h_late))

    print(f"  pattern  correct  halluc   hall_rate")
    for pat in ["HH", "HL", "LH", "LL"]:
        nc = c_patterns.get(pat, 0)
        nh = h_patterns.get(pat, 0)
        total = nc + nh
        hr = nh / total if total > 0 else 0
        print(f"  {pat}       {nc:>4}    {nh:>4}    {hr*100:>5.1f}% (n={total})")

    # --- Save ---
    out_path = results_path.replace('.json', '_biphasic.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': results_path,
            'n_correct': n_c,
            'n_hallucinated': n_h,
            'early_window': [0, EARLY_END],
            'late_window': [LATE_START, 40],
            'phase_1_early': w_early,
            'phase_2_late': w_late,
            'ratio_early_over_late': w_ratio,
            'delta_early_minus_late': w_delta,
        }, f, indent=2)
    print(f"\n  saved: {out_path}")


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(p):
        print(f"ERROR: {p} not found"); sys.exit(1)
    analyze(p)
