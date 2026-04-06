"""
analyze_cross_arch_meta.py — does the S_early signature replicate cross-architecture?
========================================================================================
After running SAE-free IPR probe on multiple models (Gemma, Llama, etc.),
compute S_early and test cross-architecture meta-analysis:

For each model:
  d = cross-dataset-mean d from cross_dataset_meta (if SAE-based)
  OR d from natural-hallucination labels (if SAE-free only)

Meta-effect: if d ≈ 0.5 replicates across architectures, S_early is
a universal measurement. If effect sizes diverge, it's architecture-
specific.

Usage:
    python analyze_cross_arch_meta.py
"""

import json
import os
import sys
import glob
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_biphasic import compute_S_range
from analyze_S_truthfulqa import MIN_GEN_TOKENS


def load_sae_free(path):
    with open(path) as f:
        return json.load(f)


def load_v22_labels_for_tqa():
    """TQA labels from our v2.2 audit (only applies to items shared with TQA-n200)."""
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    return {a['hf_index']: a['v22_label'] for a in audit_data['details']}


def label_tqa_sae_free(data, v22_map):
    """Apply v2.2 labels from original audit to SAE-free data by hf_index."""
    by_index = {}
    for r in data['results']:
        idx = r['hf_index']
        lbl = v22_map.get(idx)
        if lbl in ('correct', 'correct_semantic'):
            label = 0
        elif lbl in ('hallucinated', 'hallucinated_semantic'):
            label = 1
        else:
            continue
        traj = r['generation']['c_delta_trajectory']
        if not traj or len(traj) < MIN_GEN_TOKENS: continue
        by_index[idx] = {
            'label': label,
            'S_early': compute_S_range(traj, 0, 7),
            'traj': traj,
        }
    return by_index


def analyze_one_model(data_path, v22_map):
    """Load SAE-free result for one model, compute S_early stats."""
    data = load_sae_free(data_path)
    model = data.get('model', 'unknown')
    indexed = label_tqa_sae_free(data, v22_map)

    if len(indexed) < 6:
        return None

    S_c = np.array([v['S_early'] for v in indexed.values() if v['label']==0])
    S_h = np.array([v['S_early'] for v in indexed.values() if v['label']==1])

    if len(S_c) < 3 or len(S_h) < 3:
        return None

    _, p = stats.ttest_ind(S_c, S_h, equal_var=False)
    pool = np.sqrt((S_c.var(ddof=1) + S_h.var(ddof=1))/2)
    d = (S_h.mean() - S_c.mean())/pool if pool>0 else 0
    auc = roc_auc_score([0]*len(S_c)+[1]*len(S_h), list(S_c)+list(S_h))
    return {
        'model': model,
        'n_correct': int(len(S_c)),
        'n_hallucinated': int(len(S_h)),
        'S_corr_mean': float(S_c.mean()),
        'S_hall_mean': float(S_h.mean()),
        'd': float(d), 'p': float(p), 'auc': float(auc),
        'data_path': data_path,
    }


def fisher_meta(p_values, ds):
    """Fisher combined p, one-sided for d>0."""
    one_sided = [
        (p/2 if d > 0 else 1 - p/2)
        for p, d in zip(p_values, ds)
    ]
    one_sided = [max(min(p, 1-1e-10), 1e-10) for p in one_sided]
    k = len(one_sided)
    stat = -2 * sum(np.log(p) for p in one_sided)
    p_combined = 1 - stats.chi2.cdf(stat, df=2*k)
    return float(stat), float(p_combined)


def main():
    v22_map = load_v22_labels_for_tqa()

    # Find all SAE-free result files (full runs, not pilots)
    candidates = sorted(glob.glob("truthfulqa_results/sae_free_ipr_*_n200.json"))
    # Exclude pilot
    candidates = [c for c in candidates if "pilot" not in c]
    if not candidates:
        print("No SAE-free full runs found.")
        print("Run: python run_sae_free_truthfulqa.py --model <model>")
        sys.exit(0)

    print("=" * 76)
    print("CROSS-ARCHITECTURE META-ANALYSIS — SAE-free IPR S_early")
    print("=" * 76)
    print(f"Models detected: {len(candidates)}")
    for c in candidates:
        print(f"  {os.path.basename(c)}")
    print()

    results = []
    for path in candidates:
        r = analyze_one_model(path, v22_map)
        if r is None:
            print(f"  [skip] {path} — insufficient labels")
            continue
        results.append(r)
        print(f"Model: {r['model']}")
        print(f"  n_correct={r['n_correct']}, n_hallucinated={r['n_hallucinated']}")
        print(f"  S(corr)={r['S_corr_mean']:.5f}  S(hall)={r['S_hall_mean']:.5f}")
        print(f"  d={r['d']:+.3f}, p={r['p']:.4f}, AUC={r['auc']:.3f}")
        print()

    if len(results) < 2:
        print("Need at least 2 models for cross-arch meta-analysis.")
        sys.exit(0)

    print("=" * 76)
    print("CROSS-ARCHITECTURE META-ANALYSIS")
    print("=" * 76)
    print(f"{'model':<45} {'d':>8} {'AUC':>8} {'p':>8}")
    for r in results:
        print(f"{r['model']:<45} {r['d']:>+8.3f} {r['auc']:>8.3f} {r['p']:>8.4f}")

    ds = [r['d'] for r in results]
    ps = [r['p'] for r in results]
    pooled_d_mean = np.mean(ds)
    fisher_stat, fisher_p = fisher_meta(ps, ds)

    print()
    print(f"Mean d across architectures: {pooled_d_mean:+.3f}")
    print(f"Range: [{min(ds):+.3f}, {max(ds):+.3f}]")
    print(f"All same direction (d>0): {'YES' if all(d>0 for d in ds) else 'no'}")
    print(f"Fisher combined p (one-sided): {fisher_p:.6f}")

    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    if all(d > 0 for d in ds) and fisher_p < 0.05:
        print("  CROSS-ARCHITECTURE REPLICATION: SAE-free S_early holds across models")
        print("  UNIVERSAL MEASUREMENT CLAIM: supported")
    elif all(d > 0 for d in ds):
        print("  Directionally consistent but not significantly so — need more models")
    else:
        print("  ARCHITECTURE-SPECIFIC: effect sizes diverge across models")
        print("  S_early may be Gemma-specific or SAE-dependent")

    # Save
    out = {
        'n_models': len(results),
        'per_model': results,
        'pooled_d_mean': float(pooled_d_mean),
        'fisher_p_one_sided': float(fisher_p),
        'all_positive': all(d > 0 for d in ds),
    }
    with open("truthfulqa_results/cross_arch_meta.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: truthfulqa_results/cross_arch_meta.json")


if __name__ == "__main__":
    main()
