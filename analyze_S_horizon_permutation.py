"""
analyze_S_horizon_permutation.py — null-distribution for horizon curve
========================================================================
Addresses multiple-comparisons problem in horizon analysis. We compute
AUC at 12 prefix lengths — what's the probability that at least ONE
crosses 0.70 by chance?

Permutation test:
  1. Shuffle labels randomly (N_perm = 1000 times)
  2. Compute full horizon curve under each shuffle
  3. Record max AUC across prefix lengths for each permutation
  4. Observed max AUC compared to null distribution

Output: permutation p-value for "at least one horizon point exceeds observed AUC"

Usage:
    python analyze_S_horizon_permutation.py [results_file.json] [--n-perm 1000]
"""

import argparse
import json
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY
from analyze_S_horizon import compute_S_prefix


def observed_curve(labels_and_trajs, prefix_lens):
    """Compute AUC at each prefix length with original labels."""
    aucs = []
    data = [(lbl, tr) for lbl, tr in labels_and_trajs
            if lbl in ('correct', 'hallucinated')]
    for N in prefix_lens:
        usable = [(lbl, tr) for lbl, tr in data if len(tr) >= N]
        if len(usable) < 4:
            aucs.append(None)
            continue
        y = [1 if lbl == 'hallucinated' else 0 for lbl, _ in usable]
        if sum(y) == 0 or sum(y) == len(y):
            aucs.append(None)
            continue
        scores = [compute_S_prefix(tr, N) for _, tr in usable]
        try:
            aucs.append(roc_auc_score(y, scores))
        except Exception:
            aucs.append(None)
    return aucs


def permutation_test(labels_and_trajs, prefix_lens, n_perm=1000, seed=42):
    """Null distribution: shuffle labels, recompute max AUC across prefixes."""
    data = [(lbl, tr) for lbl, tr in labels_and_trajs
            if lbl in ('correct', 'hallucinated')]
    labels = [lbl for lbl, _ in data]
    trajs = [tr for _, tr in data]
    n = len(labels)

    # Observed
    obs_aucs = observed_curve(labels_and_trajs, prefix_lens)
    obs_max = max((a for a in obs_aucs if a is not None), default=0)
    obs_above = [(n_, a) for n_, a in zip(prefix_lens, obs_aucs)
                 if a is not None]

    rng = np.random.RandomState(seed)
    null_max_aucs = []
    null_per_prefix = {n_: [] for n_ in prefix_lens}

    for p in range(n_perm):
        # Permute labels
        perm_labels = rng.permutation(labels)
        perm_data = list(zip(perm_labels, trajs))
        perm_aucs = observed_curve(perm_data, prefix_lens)
        valid = [a for a in perm_aucs if a is not None]
        if valid:
            null_max_aucs.append(max(valid))
        for n_, a in zip(prefix_lens, perm_aucs):
            if a is not None:
                null_per_prefix[n_].append(a)

    null_max_aucs = np.array(null_max_aucs)

    # p-value: proportion of permutations where max AUC >= observed max AUC
    p_global = float((null_max_aucs >= obs_max).mean())

    # Per-prefix p-values
    per_prefix_p = {}
    for n_, obs in zip(prefix_lens, obs_aucs):
        if obs is None:
            continue
        null_vals = np.array(null_per_prefix[n_])
        # one-sided: prob null >= observed
        per_prefix_p[n_] = float((null_vals >= obs).mean())

    return {
        'observed_aucs': obs_aucs,
        'observed_max_auc': float(obs_max),
        'null_max_auc_mean': float(null_max_aucs.mean()),
        'null_max_auc_p95': float(np.percentile(null_max_aucs, 95)),
        'null_max_auc_p99': float(np.percentile(null_max_aucs, 99)),
        'global_p_value': p_global,
        'per_prefix_p_values': per_prefix_p,
        'n_permutations': n_perm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default=None)
    parser.add_argument('--n-perm', type=int, default=1000)
    args = parser.parse_args()

    path = args.path or os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(path):
        pilot = path.replace('n200.json', 'pilot.json')
        if os.path.exists(pilot):
            path = pilot
        else:
            print(f"ERROR: {path} not found")
            sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 72)
    print("HORIZON PERMUTATION TEST (null distribution)")
    print("=" * 72)
    print(f"Data:      {path}")
    print(f"N items:   {len(data['results'])}")
    print(f"N perm:    {args.n_perm}")

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

    counts = {}
    for lbl, _ in labeled:
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"Labels:    {counts}")

    usable = [(l, t) for l, t in labeled if l in ('correct', 'hallucinated')]
    n_corr = sum(1 for l, _ in usable if l == 'correct')
    n_hall = sum(1 for l, _ in usable if l == 'hallucinated')
    if n_corr < 4 or n_hall < 4:
        print("[!] Insufficient data for permutation test")
        return
    print(f"Usable:    {n_corr} correct + {n_hall} hallucinated = {len(usable)}")
    print()

    prefix_lens = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40]
    result = permutation_test(usable, prefix_lens, n_perm=args.n_perm)

    print("-" * 72)
    print("OBSERVED vs NULL DISTRIBUTION")
    print("-" * 72)
    print(f"  Observed max AUC across horizon: {result['observed_max_auc']:.3f}")
    print(f"  Null max AUC mean:                {result['null_max_auc_mean']:.3f}")
    print(f"  Null max AUC 95th percentile:     {result['null_max_auc_p95']:.3f}")
    print(f"  Null max AUC 99th percentile:     {result['null_max_auc_p99']:.3f}")
    print(f"  Global permutation p-value:       {result['global_p_value']:.4f}")
    if result['global_p_value'] < 0.05:
        print(f"  => Observed horizon curve is SIGNIFICANT at multiple-comparison-adjusted level")
    else:
        print(f"  => Observed horizon curve is NOT significant vs random shuffle")
    print()

    print("-" * 72)
    print("PER-PREFIX PERMUTATION p-VALUES")
    print("-" * 72)
    print(f"{'N':>5} {'obs AUC':>8} {'perm p':>9}")
    for n_, p in sorted(result['per_prefix_p_values'].items()):
        obs = result['observed_aucs'][prefix_lens.index(n_)]
        marker = "*" if p < 0.05 else " "
        print(f"{n_:>5} {obs:>8.3f} {p:>9.4f} {marker}")

    # Save
    out_path = path.replace('.json', '_horizon_perm.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\n  saved: {out_path}")


if __name__ == '__main__':
    main()
