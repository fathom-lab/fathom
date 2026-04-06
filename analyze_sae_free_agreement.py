"""
analyze_sae_free_agreement.py — validate SAE-free probe against SAE S_early
==============================================================================
Test whether the SAE-free IPR probe reproduces the S_early result
on the SAME TruthfulQA items where we have SAE-based measurements.

If they agree:
  - per-item correlation r(S_SAE, S_SAE_free) is high
  - d and AUC from SAE-free analysis matches SAE analysis
  → SAE-free probe is validated, we can trust it on Llama

If they disagree:
  - SAE-free needs refinement
  - Or the SAE-based signal is specific to the SAE basis

Usage:
    python analyze_sae_free_agreement.py [sae_free_file.json]
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS
from analyze_S_biphasic import compute_S_range


def load_sae_s_early():
    """Load SAE-based S_early values per item from existing data."""
    with open("truthfulqa_results/S_truthfulqa_n200.json") as f:
        data = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a for a in audit_data['details']}

    by_index = {}
    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS: continue
        idx = r['hf_index']
        lbl = audit.get(idx, {}).get('v22_label')
        if lbl in ('correct', 'correct_semantic'): label = 0
        elif lbl in ('hallucinated', 'hallucinated_semantic'): label = 1
        else: continue
        traj = r['generation']['c_delta_trajectory']
        if not traj: continue
        by_index[idx] = {
            'S_early_sae': compute_S_range(traj, 0, 7),
            'label': label,
            'n_tokens': r['generation']['total_tokens'],
            'c_delta_sae': traj,
        }
    return by_index


def load_sae_free(sae_free_path):
    """Load SAE-free IPR results."""
    with open(sae_free_path) as f:
        data = json.load(f)
    by_index = {}
    for r in data['results']:
        idx = r['hf_index']
        traj = r['generation']['c_delta_trajectory']
        if not traj: continue
        by_index[idx] = {
            'S_early_ipr': compute_S_range(traj, 0, 7),
            'n_tokens': r['generation']['total_tokens'],
            'c_delta_ipr': traj,
        }
    return by_index


def main():
    sae_free_path = sys.argv[1] if len(sys.argv) > 1 else \
        "truthfulqa_results/sae_free_ipr_google_gemma_2_2b_it_n200.json"
    if not os.path.exists(sae_free_path):
        print(f"ERROR: {sae_free_path} not found")
        print("Run first: python run_sae_free_truthfulqa.py --model google/gemma-2-2b-it")
        sys.exit(1)

    sae_data = load_sae_s_early()
    free_data = load_sae_free(sae_free_path)
    common_idx = set(sae_data.keys()) & set(free_data.keys())
    print("=" * 72)
    print("SAE vs SAE-FREE AGREEMENT TEST")
    print("=" * 72)
    print(f"SAE-based items: {len(sae_data)}")
    print(f"SAE-free items:  {len(free_data)}")
    print(f"Common items:    {len(common_idx)}")
    print()

    # Merge
    merged = []
    for idx in sorted(common_idx):
        s = sae_data[idx]; f = free_data[idx]
        merged.append({
            'hf_index': idx,
            'label': s['label'],
            'S_early_sae': s['S_early_sae'],
            'S_early_ipr': f['S_early_ipr'],
        })

    S_sae = np.array([m['S_early_sae'] for m in merged])
    S_ipr = np.array([m['S_early_ipr'] for m in merged])
    y = np.array([m['label'] for m in merged])

    # === Correlation between methods ===
    print("-" * 72)
    print("AGREEMENT: per-item S_early correlation")
    print("-" * 72)
    r_p, p_p = stats.pearsonr(S_sae, S_ipr)
    r_s, p_s = stats.spearmanr(S_sae, S_ipr)
    print(f"  Pearson r:  {r_p:+.3f} (p={p_p:.2e})")
    print(f"  Spearman r: {r_s:+.3f} (p={p_s:.2e})")
    print()

    # === Does SAE-free recover the d? ===
    print("-" * 72)
    print("DISCRIMINATION: does SAE-free probe recover d≈0.5?")
    print("-" * 72)
    print(f"{'method':<20} {'mean_corr':>11} {'mean_hall':>11} {'d':>8} {'AUC':>8}")
    for name, vals in [("SAE S_early", S_sae), ("SAE-free IPR S_early", S_ipr)]:
        c = vals[y == 0]; h = vals[y == 1]
        pool = np.sqrt((c.var(ddof=1) + h.var(ddof=1))/2)
        d = (h.mean() - c.mean())/pool if pool>0 else 0
        auc = roc_auc_score(y, vals)
        print(f"{name:<20} {c.mean():>11.6f} {h.mean():>11.6f} {d:>+8.3f} {auc:>8.3f}")
    print()

    # === AUC on combined (both features) ===
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    print("-" * 72)
    print("COMBINED CLASSIFIER (5-fold CV)")
    print("-" * 72)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'))])
    for name, X in [
        ("SAE S_early only", S_sae.reshape(-1,1)),
        ("SAE-free IPR only", S_ipr.reshape(-1,1)),
        ("Both combined", np.column_stack([S_sae, S_ipr])),
    ]:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        print(f"  {name:<25}: CV AUC = {scores.mean():.3f} +/- {scores.std():.3f}")
    print()

    # === Verdict ===
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if r_p > 0.5:
        print(f"  STRONG AGREEMENT: r={r_p:.3f} — SAE-free probe validated")
        print(f"  Cross-architecture test is justified.")
    elif r_p > 0.2:
        print(f"  MODERATE AGREEMENT: r={r_p:.3f} — probe captures part of signal")
        print(f"  May need refinement before cross-arch test.")
    else:
        print(f"  POOR AGREEMENT: r={r_p:.3f} — SAE-free probe does NOT match")
        print(f"  The SAE basis is doing necessary work. Try different probe design.")

    # Save
    out_path = sae_free_path.replace(".json", "_agreement.json")
    with open(out_path, "w") as f:
        json.dump({
            'sae_free_path': sae_free_path,
            'n_common': len(common_idx),
            'pearson_r': float(r_p),
            'spearman_r': float(r_s),
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
