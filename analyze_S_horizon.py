"""
analyze_S_horizon.py — predictive horizon analysis for stream-time hallucination detection
============================================================================================
For each trajectory with correct/hallucinated label, compute S using only
the first N tokens (prefix). Sweep N and compute AUC at each depth.

The "detection horizon" = earliest N at which AUC crosses a target threshold
(e.g., 0.65 production, 0.70 strong, 0.80 excellent).

Output: horizon curve — AUC vs prefix length.
Stakes: if AUC > 0.70 at token 10, we have a real-time production detector.

Usage:
    python analyze_S_horizon.py [results_file.json]
"""

import json
import sys
import os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import (
    classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY
)


def compute_S_prefix(traj, prefix_len, threshold=THR_PRIMARY):
    """S computed using only first prefix_len tokens of trajectory."""
    pref = traj[:prefix_len]
    if not pref:
        return 0.0
    spikes = [c for c in pref if c > threshold]
    if len(spikes) == 0:
        return 0.0
    return max(pref) / len(spikes)


def horizon_curve(labels_and_trajs, prefix_lens=None, threshold=THR_PRIMARY):
    """
    Compute AUC at each prefix length.
    labels_and_trajs: list of (label, trajectory) where label in {'correct','hallucinated'}
    """
    if prefix_lens is None:
        prefix_lens = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40]

    # Filter to binary label
    data = [(lbl, tr) for lbl, tr in labels_and_trajs
            if lbl in ('correct', 'hallucinated')]
    if len(data) < 4:
        return None

    y = [1 if lbl == 'hallucinated' else 0 for lbl, _ in data]
    curve = []
    for N in prefix_lens:
        # only keep items with at least N tokens
        usable = [(lbl, tr) for lbl, tr in data if len(tr) >= N]
        if len(usable) < 4:
            curve.append({'N': N, 'AUC': None, 'n_items': len(usable)})
            continue
        y_u = [1 if lbl == 'hallucinated' else 0 for lbl, _ in usable]
        # need both classes present
        if sum(y_u) == 0 or sum(y_u) == len(y_u):
            curve.append({'N': N, 'AUC': None, 'n_items': len(usable)})
            continue
        scores = [compute_S_prefix(tr, N, threshold) for _, tr in usable]
        try:
            auc = roc_auc_score(y_u, scores)
        except Exception:
            auc = None

        # compute effect size at this prefix
        S_correct = [s for s, l in zip(scores, y_u) if l == 0]
        S_halluc = [s for s, l in zip(scores, y_u) if l == 1]
        if len(S_correct) >= 2 and len(S_halluc) >= 2:
            pooled = np.sqrt((np.var(S_correct, ddof=1) + np.var(S_halluc, ddof=1)) / 2)
            d = (np.mean(S_halluc) - np.mean(S_correct)) / pooled if pooled > 0 else 0.0
            _, p = stats.ttest_ind(S_correct, S_halluc, equal_var=False)
        else:
            d, p = 0.0, 1.0
        curve.append({
            'N': N, 'AUC': auc, 'd': float(d), 'p': float(p),
            'n_items': len(usable),
            'n_correct': int(sum(1 for l in y_u if l == 0)),
            'n_hallucinated': int(sum(y_u)),
        })
    return curve


def detection_horizon(curve, auc_targets=(0.60, 0.65, 0.70, 0.75)):
    """Return earliest N at which AUC crosses each target."""
    horizons = {}
    for target in auc_targets:
        hit_N = None
        for entry in curve:
            if entry['AUC'] is not None and entry['AUC'] >= target:
                hit_N = entry['N']
                break
        horizons[target] = hit_N
    return horizons


def analyze(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 72)
    print("S-axis PREDICTIVE HORIZON analysis")
    print("=" * 72)
    print(f"Data:       {results_path}")
    print(f"Model:      {data.get('model')}")
    print(f"N items:    {len(data['results'])}")
    print(f"Mode:       {data.get('mode')}")
    print()

    # Label every item
    labeled = []
    for r in data['results']:
        gen = r['generation']['text']
        n_tokens = r['generation']['total_tokens']
        traj = r['generation']['c_delta_trajectory']
        if n_tokens < MIN_GEN_TOKENS:
            continue
        label, _ = classify_generation(
            gen, r['correct_answers'], r['incorrect_answers'], r.get('best_answer', '')
        )
        labeled.append((label, traj))

    # Counts
    counts = {}
    for lbl, _ in labeled:
        counts[lbl] = counts.get(lbl, 0) + 1
    print("Label counts:", counts)
    print()

    # Horizon curve
    curve = horizon_curve(labeled)
    if curve is None:
        print("[!] Insufficient data for horizon analysis")
        return

    print("-" * 72)
    print("DETECTION HORIZON CURVE")
    print("-" * 72)
    print(f"{'N':>5} {'AUC':>6} {'d':>6} {'p':>8} {'n_corr':>7} {'n_hall':>7}")
    for e in curve:
        if e['AUC'] is None:
            print(f"{e['N']:>5} {'---':>6} {'---':>6} {'---':>8} "
                  f"{'---':>7} {'---':>7} (n={e['n_items']})")
        else:
            print(f"{e['N']:>5} {e['AUC']:>6.3f} {e['d']:>6.3f} {e['p']:>8.4f} "
                  f"{e['n_correct']:>7} {e['n_hallucinated']:>7}")
    print()

    # Horizon thresholds
    print("-" * 72)
    print("EARLIEST TOKEN TO CROSS AUC TARGET")
    print("-" * 72)
    horizons = detection_horizon(curve)
    for target, N in horizons.items():
        if N is None:
            print(f"  AUC >= {target:.2f}: NOT REACHED within max prefix")
        else:
            print(f"  AUC >= {target:.2f}: token {N}")
    print()

    # Verdict
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    final_auc = next((e['AUC'] for e in reversed(curve) if e['AUC'] is not None), None)
    if final_auc is None:
        print("  No valid AUC computed")
        return

    if horizons[0.70] is not None and horizons[0.70] <= 15:
        print(f"  REAL-TIME DETECTOR: AUC>=0.70 reached at token {horizons[0.70]}")
        print(f"  S can flag hallucination before generation completes.")
        print(f"  PRODUCT CLAIM: stream-time Fathom Monitor at ~{horizons[0.70]*50}ms per token.")
    elif horizons[0.65] is not None and horizons[0.65] <= 20:
        print(f"  PRODUCTION DETECTOR: AUC>=0.65 at token {horizons[0.65]}")
        print(f"  Good enough for uncertainty flagging, not strong commit.")
    elif horizons[0.65] is not None:
        print(f"  POST-HOC DETECTOR: AUC>=0.65 at token {horizons[0.65]}")
        print(f"  Only works with most of the generation available.")
    else:
        print(f"  NO DETECTOR: best AUC = {final_auc:.3f}")
        print(f"  S does not discriminate hallucination in streaming mode.")

    # Save curve
    out_path = results_path.replace(".json", "_horizon.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': results_path,
            'label_counts': counts,
            'curve': curve,
            'horizons': {str(k): v for k, v in horizons.items()},
        }, f, indent=2)
    print(f"\n  curve saved: {out_path}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(path):
        pilot_path = path.replace('n200.json', 'pilot.json')
        if os.path.exists(pilot_path):
            print(f"Using pilot: {pilot_path}")
            path = pilot_path
        else:
            print(f"ERROR: results file not found: {path}")
            sys.exit(1)
    analyze(path)
