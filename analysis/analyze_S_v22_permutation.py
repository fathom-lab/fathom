"""
analyze_S_v22_permutation.py — permutation test for v2.2 combined labels
===========================================================================
Tests whether the observed early-token S discrimination survives
multiple-comparisons correction via 1000 label-shuffle permutations.

Usage:
    python analyze_S_v22_permutation.py [--n-perm 1000]
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import compute_S, MIN_GEN_TOKENS, THR_PRIMARY
from analyze_S_biphasic import compute_S_range


def load_v22_data():
    with open("truthfulqa_results/S_truthfulqa_n200.json") as f:
        data = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a for a in audit_data['details']}

    labeled = []
    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS:
            continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        if lbl in ('correct', 'correct_semantic'):
            labeled.append(('correct', r['generation']['c_delta_trajectory']))
        elif lbl in ('hallucinated', 'hallucinated_semantic'):
            labeled.append(('hallucinated', r['generation']['c_delta_trajectory']))
    return labeled


def auc_at_prefix(labeled, N):
    """Compute AUC of prefix-S at length N."""
    usable = [(lbl, tr) for lbl, tr in labeled if len(tr) >= N]
    if len(usable) < 4:
        return None
    y = [1 if l == 'hallucinated' else 0 for l, _ in usable]
    if sum(y) == 0 or sum(y) == len(y):
        return None
    scores = [compute_S_range(tr, 0, N) for _, tr in usable]
    try:
        return roc_auc_score(y, scores)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-perm', type=int, default=1000)
    args = parser.parse_args()

    labeled = load_v22_data()
    print("=" * 72)
    print("V2.2 HORIZON PERMUTATION TEST")
    print("=" * 72)
    counts = {}
    for l, _ in labeled:
        counts[l] = counts.get(l, 0) + 1
    print(f"Labels: {counts}")

    prefix_lens = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40]
    # Observed AUCs
    obs_aucs = [auc_at_prefix(labeled, N) for N in prefix_lens]
    obs_valid = [a for a in obs_aucs if a is not None]
    obs_max = max(obs_valid)

    # Null via label shuffles
    labels = [l for l, _ in labeled]
    trajs = [t for _, t in labeled]
    rng = np.random.RandomState(42)
    null_max_aucs = []
    null_per_prefix = {N: [] for N in prefix_lens}

    print(f"Running {args.n_perm} permutations...")
    for _ in range(args.n_perm):
        perm_labels = rng.permutation(labels)
        perm_data = list(zip(perm_labels, trajs))
        perm_aucs = [auc_at_prefix(perm_data, N) for N in prefix_lens]
        valid = [a for a in perm_aucs if a is not None]
        if valid:
            null_max_aucs.append(max(valid))
        for N, a in zip(prefix_lens, perm_aucs):
            if a is not None:
                null_per_prefix[N].append(a)

    null_max_aucs = np.array(null_max_aucs)
    p_global = float((null_max_aucs >= obs_max).mean())

    print()
    print(f"Observed max AUC: {obs_max:.3f}")
    print(f"Null max AUC mean: {null_max_aucs.mean():.3f}")
    print(f"Null max AUC p95: {np.percentile(null_max_aucs, 95):.3f}")
    print(f"Null max AUC p99: {np.percentile(null_max_aucs, 99):.3f}")
    print(f"GLOBAL permutation p-value: {p_global:.4f}")
    if p_global < 0.05:
        print("=> Observed horizon curve SIGNIFICANT after multi-comparison correction")
    else:
        print("=> Observed horizon curve NOT significant after correction")
    print()

    print("-" * 72)
    print("PER-PREFIX PERMUTATION p-VALUES")
    print("-" * 72)
    print(f"{'N':>5} {'obs AUC':>8} {'perm p':>9}")
    for N, obs in zip(prefix_lens, obs_aucs):
        if obs is None: continue
        null_vals = np.array(null_per_prefix[N])
        # one-sided: prob null >= observed (if observed > 0.5) or <= (if < 0.5)
        if obs >= 0.5:
            p = float((null_vals >= obs).mean())
        else:
            p = float((null_vals <= obs).mean())
        marker = "*" if p < 0.05 else " "
        print(f"{N:>5} {obs:>8.3f} {p:>9.4f} {marker}")

    # Save
    out_path = "truthfulqa_results/S_truthfulqa_n200_v22_horizon_perm.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'n_permutations': args.n_perm,
            'observed_aucs': obs_aucs,
            'observed_max_auc': float(obs_max),
            'null_max_auc_mean': float(null_max_aucs.mean()),
            'global_p_value': p_global,
            'n_correct': counts.get('correct', 0),
            'n_hallucinated': counts.get('hallucinated', 0),
        }, f, indent=2)
    print(f"\n  saved: {out_path}")


if __name__ == '__main__':
    main()
