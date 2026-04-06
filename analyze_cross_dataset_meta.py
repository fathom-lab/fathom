"""
analyze_cross_dataset_meta.py — meta-analysis across all S-trajectory datasets
=================================================================================
We have 3 datasets with per-token C_delta trajectories:
  1. commitment_phase_transition.json (base, 30 topics, hedge vs commit)
  2. commitment_IT_replication.json   (IT, 30 topics, hedge vs commit)
  3. S_truthfulqa_n200.json          (IT, 200 items, halluc vs correct)

For each dataset, test S_early [0,7) as discriminator of the appropriate
binary label. Does the same Phase 1 signal show up across datasets?

Hypothesis: if S_early reliably measures "early commitment magnitude",
it should be higher in:
  - commit vs hedge (both base and IT)
  - hallucinated vs correct (IT)

Meta-analysis = does the same effect replicate across 3 independent runs?

Usage:
    python analyze_cross_dataset_meta.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import compute_S, MIN_GEN_TOKENS
from analyze_S_biphasic import compute_S_range


def eval_S_early(trajs_A, trajs_B, window_end=7, label_A="A", label_B="B"):
    """Compare S_early [0, window_end) between two sets of trajectories."""
    S_A = [compute_S_range(t, 0, window_end) for t in trajs_A]
    S_B = [compute_S_range(t, 0, window_end) for t in trajs_B]
    if len(S_A) < 3 or len(S_B) < 3:
        return None
    c = np.array(S_A); h = np.array(S_B)
    _, p = stats.ttest_ind(c, h, equal_var=False)
    pool = np.sqrt((c.var(ddof=1) + h.var(ddof=1))/2)
    d = (h.mean() - c.mean())/pool if pool>0 else 0
    auc = roc_auc_score([0]*len(c) + [1]*len(h), list(c)+list(h))
    return {
        'n_A': len(S_A), 'n_B': len(S_B),
        'A_mean': float(c.mean()), 'B_mean': float(h.mean()),
        'ratio': float(h.mean()/c.mean()) if c.mean()>0 else None,
        'p': float(p), 'd': float(d), 'auc': float(auc),
    }


def fisher_meta(p_values, effects=None):
    """Combine p-values via Fisher's method."""
    k = len(p_values)
    stat = -2 * sum(np.log(p) for p in p_values)
    p_combined = 1 - stats.chi2.cdf(stat, df=2*k)
    return float(stat), float(p_combined)


def main():
    print("=" * 76)
    print("CROSS-DATASET META-ANALYSIS")
    print("=" * 76)
    print()

    results = []

    # --- Dataset 1: commitment_phase_transition.json (base hedge vs commit) ---
    with open("truthfulqa_results/commitment_phase_transition.json") as f:
        d1 = json.load(f)
    hedge_t = [r['hedging']['c_delta_trajectory'] for r in d1['results']]
    commit_t = [r['commitment']['c_delta_trajectory'] for r in d1['results']]
    r1 = eval_S_early(hedge_t, commit_t, label_A="hedge", label_B="commit")
    print("DATASET 1: Gemma-2-2B BASE, hedge vs commit (paired, n=30)")
    print(f"  (though contaminated by drift, see commitment correction memo)")
    print(f"  hedge S_early  = {r1['A_mean']:+.6f}")
    print(f"  commit S_early = {r1['B_mean']:+.6f}")
    print(f"  ratio = {r1['ratio']:.2f}x")
    print(f"  p={r1['p']:.4f}, d={r1['d']:+.3f}, AUC={r1['auc']:.3f}")
    results.append({'dataset': 'base_hedge_commit', **r1})
    print()

    # --- Dataset 2: commitment_IT_replication.json (IT hedge vs commit) ---
    with open("truthfulqa_results/commitment_IT_replication.json") as f:
        d2 = json.load(f)
    hedge_t2 = [r['hedging']['c_delta_trajectory'] for r in d2['results']]
    commit_t2 = [r['commitment']['c_delta_trajectory'] for r in d2['results']]
    r2 = eval_S_early(hedge_t2, commit_t2, label_A="hedge", label_B="commit")
    print("DATASET 2: Gemma-2-2B IT, hedge vs commit (paired, n=30)")
    print(f"  hedge S_early  = {r2['A_mean']:+.6f}")
    print(f"  commit S_early = {r2['B_mean']:+.6f}")
    print(f"  ratio = {r2['ratio']:.2f}x")
    print(f"  p={r2['p']:.4f}, d={r2['d']:+.3f}, AUC={r2['auc']:.3f}")
    results.append({'dataset': 'IT_hedge_commit', **r2})
    print()

    # --- Dataset 3: S_truthfulqa_n200.json (IT TQA halluc vs correct, v2.2) ---
    with open("truthfulqa_results/S_truthfulqa_n200.json") as f:
        d3 = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a for a in audit_data['details']}
    corr_t = []; hall_t = []
    for r in d3['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS: continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        traj = r['generation']['c_delta_trajectory']
        if not traj: continue
        if lbl in ('correct', 'correct_semantic'):
            corr_t.append(traj)
        elif lbl in ('hallucinated', 'hallucinated_semantic'):
            hall_t.append(traj)
    r3 = eval_S_early(corr_t, hall_t, label_A="correct", label_B="hallucinated")
    print("DATASET 3: Gemma-2-2B IT, TQA correct vs hallucinated (n=94)")
    print(f"  correct S_early       = {r3['A_mean']:+.6f}")
    print(f"  hallucinated S_early  = {r3['B_mean']:+.6f}")
    print(f"  ratio = {r3['ratio']:.2f}x")
    print(f"  p={r3['p']:.4f}, d={r3['d']:+.3f}, AUC={r3['auc']:.3f}")
    results.append({'dataset': 'IT_halluc_correct', **r3})
    print()

    # ------------------------------------------------------------------
    # Meta-analysis
    # ------------------------------------------------------------------
    print("=" * 76)
    print("META-ANALYSIS")
    print("=" * 76)
    print()
    print("All 3 datasets: does the 'target' class (commit / hallucinated)")
    print("show HIGHER S_early than the baseline class (hedge / correct)?")
    print()
    print("Expected interpretation: commit > hedge means 'early commitment"
          " signature' is real; halluc > correct means it ALSO predicts wrong confidence.")
    print()

    print(f"{'dataset':<25} {'direction':<15} {'d':>8} {'AUC':>8} {'p':>10}")
    for r in results:
        dir_str = f"{'B>A' if r['d']>0 else 'A>B'}" if r else "?"
        print(f"{r['dataset']:<25} {dir_str:<15} {r['d']:>+8.3f} "
              f"{r['auc']:>8.3f} {r['p']:>10.4f}")

    # Fisher combined p (directional expectation: B > A)
    # Convert to one-sided (assuming H1: d > 0)
    one_sided_p = [
        r['p']/2 if r['d'] > 0 else 1 - r['p']/2
        for r in results
    ]
    stat, p_meta = fisher_meta(one_sided_p)
    print()
    print(f"Fisher's combined (one-sided, testing d>0): chi2={stat:.3f}, p={p_meta:.6f}")

    # Two-sided
    two_sided_p = [r['p'] for r in results]
    stat2, p_meta2 = fisher_meta(two_sided_p)
    print(f"Fisher's combined (two-sided): chi2={stat2:.3f}, p={p_meta2:.6f}")

    # Random-effects effect size
    ds = [r['d'] for r in results]
    n_tots = [r['n_A'] + r['n_B'] for r in results]
    # Weighted mean d by inverse variance (approx: var(d) ~ 4/n for each)
    weights = n_tots
    d_pooled = sum(d*w for d,w in zip(ds, weights)) / sum(weights)
    print()
    print(f"Pooled Cohen's d (weighted by n): {d_pooled:+.3f}")
    print(f"Per-dataset d range: [{min(ds):+.3f}, {max(ds):+.3f}]")
    print()

    # Consistency
    all_positive = all(d > 0 for d in ds)
    all_negative = all(d < 0 for d in ds)
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    if all_positive:
        print("  All 3 datasets show d > 0 (target > baseline in S_early)")
        print("  => Consistent direction: commitment/hallucination both have higher S_early")
    elif all_negative:
        print("  All 3 datasets show d < 0")
        print("  => Opposite of H1 direction across all datasets")
    else:
        print("  Directions are MIXED across datasets")
        mixed = []
        for r in results:
            mixed.append(f"{r['dataset']}: {'positive' if r['d']>0 else 'negative'}")
        for m in mixed:
            print(f"    {m}")

    if p_meta < 0.05:
        print(f"\n  Fisher combined p={p_meta:.4f} -> SIGNIFICANT meta-effect")
    else:
        print(f"\n  Fisher combined p={p_meta:.4f} -> no significant meta-effect")

    # Save
    with open("truthfulqa_results/cross_dataset_meta.json", 'w') as f:
        json.dump({
            'results': results,
            'pooled_d': float(d_pooled),
            'fisher_one_sided_p': float(p_meta),
            'fisher_two_sided_p': float(p_meta2),
            'all_same_direction': bool(all_positive or all_negative),
        }, f, indent=2)
    print(f"\n  Saved: truthfulqa_results/cross_dataset_meta.json")


if __name__ == '__main__':
    main()
