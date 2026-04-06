"""
analyze_S_sliding.py — sliding-window S (dynamic commitment intensity)
=========================================================================
Instead of a single S per trajectory, compute S over sliding windows
and track S(t) over time. Extracts novel features:
 - max(S(window)) — peak windowed intensity
 - argmax(S(window)) — when peak occurs
 - S at first-crossing of threshold
 - area under S(t) curve
 - early-window S vs late-window S

Tests whether windowed S is a stronger hallucination signal than
trajectory-level S.

Usage:
    python analyze_S_sliding.py [results_file.json]
"""

import json
import sys
import os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY


def sliding_S(traj, window=10, threshold=THR_PRIMARY, step=1):
    """Compute S over sliding windows of given size. Returns (positions, S_values)."""
    if len(traj) < window:
        return [], []
    positions, values = [], []
    for start in range(0, len(traj) - window + 1, step):
        w = traj[start:start + window]
        spikes = [c for c in w if c > threshold]
        if spikes:
            S_w = max(w) / len(spikes)
        else:
            S_w = 0.0
        positions.append(start + window - 1)  # window end position
        values.append(S_w)
    return positions, values


def windowed_features(traj, window=10, threshold=THR_PRIMARY):
    """Extract sliding-window summary features."""
    positions, values = sliding_S(traj, window, threshold)
    if not values:
        return None
    vals = np.array(values)
    return {
        'window': window,
        'n_windows': len(values),
        'S_peak': float(vals.max()),
        'S_mean': float(vals.mean()),
        'S_std': float(vals.std(ddof=1)) if len(values) > 1 else 0.0,
        'peak_position': float(positions[int(vals.argmax())]) / len(traj),
        'early_S_mean': float(vals[:len(vals)//2].mean()) if len(vals) >= 2 else 0.0,
        'late_S_mean': float(vals[len(vals)//2:].mean()) if len(vals) >= 2 else 0.0,
        'late_minus_early': (float(vals[len(vals)//2:].mean()) -
                             float(vals[:len(vals)//2].mean())) if len(vals) >= 2 else 0.0,
    }


def analyze(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 72)
    print("SLIDING-WINDOW S (dynamic commitment intensity)")
    print("=" * 72)
    print(f"Data: {results_path}")
    print(f"N items: {len(data['results'])}")
    print()

    # Collect trajectories by label
    feats_correct, feats_halluc = [], []
    for r in data['results']:
        gen = r['generation']['text']
        n_tokens = r['generation']['total_tokens']
        traj = r['generation']['c_delta_trajectory']
        if n_tokens < MIN_GEN_TOKENS:
            continue
        label, _ = classify_generation(
            gen, r['correct_answers'], r['incorrect_answers'], r.get('best_answer', '')
        )
        if label not in ('correct', 'hallucinated'):
            continue
        if label == 'correct':
            feats_correct.append(traj)
        else:
            feats_halluc.append(traj)

    print(f"Labeled: {len(feats_correct)} correct, {len(feats_halluc)} hallucinated")
    if len(feats_correct) < 3 or len(feats_halluc) < 3:
        print("[!] Insufficient data")
        return

    # For each window size, test which sliding-window feature discriminates best
    print()
    print("-" * 72)
    print("SLIDING-WINDOW FEATURE DISCRIMINATION")
    print("-" * 72)
    print(f"{'window':>6} {'feature':<22} {'corr_mean':>12} {'hall_mean':>12} {'p':>8} {'d':>6} {'AUC':>6}")
    for window in [5, 8, 10, 12, 15]:
        corr_feats = [windowed_features(t, window) for t in feats_correct]
        hall_feats = [windowed_features(t, window) for t in feats_halluc]
        corr_feats = [f for f in corr_feats if f is not None]
        hall_feats = [f for f in hall_feats if f is not None]
        if len(corr_feats) < 2 or len(hall_feats) < 2:
            continue

        for fname in ['S_peak', 'S_mean', 'late_minus_early', 'peak_position']:
            c = np.array([f[fname] for f in corr_feats])
            h = np.array([f[fname] for f in hall_feats])
            _, p = stats.ttest_ind(c, h, equal_var=False)
            pooled = np.sqrt((c.var(ddof=1) + h.var(ddof=1)) / 2)
            d = (h.mean() - c.mean()) / pooled if pooled > 0 else 0.0
            auc = roc_auc_score([0]*len(c) + [1]*len(h), list(c) + list(h))
            print(f"{window:>6} {fname:<22} {c.mean():>12.6f} {h.mean():>12.6f} "
                  f"{p:>8.4f} {d:>6.3f} {auc:>6.3f}")
        print()

    # Compare static S vs sliding-window peak S
    print("-" * 72)
    print("STATIC S vs SLIDING-WINDOW PEAK S")
    print("-" * 72)
    static_c = [compute_S(t) for t in feats_correct]
    static_h = [compute_S(t) for t in feats_halluc]
    _, p_static = stats.ttest_ind(static_c, static_h, equal_var=False)
    auc_static = roc_auc_score([0]*len(static_c) + [1]*len(static_h), static_c + static_h)
    print(f"  Static S:        AUC={auc_static:.3f}, p={p_static:.4f}")

    for window in [8, 10, 12]:
        peak_c = [windowed_features(t, window)['S_peak'] for t in feats_correct
                  if windowed_features(t, window)]
        peak_h = [windowed_features(t, window)['S_peak'] for t in feats_halluc
                  if windowed_features(t, window)]
        if len(peak_c) >= 2 and len(peak_h) >= 2:
            _, p_peak = stats.ttest_ind(peak_c, peak_h, equal_var=False)
            auc_peak = roc_auc_score([0]*len(peak_c) + [1]*len(peak_h), peak_c + peak_h)
            lift = auc_peak - auc_static
            print(f"  S_peak(w={window}): AUC={auc_peak:.3f}, p={p_peak:.4f}, lift={lift:+.3f}")

    # Save
    out_path = results_path.replace('.json', '_sliding.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': results_path,
            'n_correct': len(feats_correct),
            'n_hallucinated': len(feats_halluc),
        }, f, indent=2)
    print(f"\n  saved: {out_path}")


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
    analyze(p)
