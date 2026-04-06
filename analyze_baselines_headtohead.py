"""
analyze_baselines_headtohead.py — S vs entropy vs logprob vs top2_margin
===========================================================================
THE competitive baseline comparison. Does S_early beat standard
uncertainty signals? This determines whether S is a genuinely novel
signal or just recapitulating known information.

Usage:
    python analyze_baselines_headtohead.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS
from analyze_S_biphasic import compute_S_range


def main():
    BASELINES_PATH = "truthfulqa_results/S_baselines_n200.json"
    AUDIT_PATH = "truthfulqa_results/S_truthfulqa_n200_v22_audit.json"

    if not os.path.exists(BASELINES_PATH):
        print(f"ERROR: {BASELINES_PATH} not found")
        print("Run: python run_S_truthfulqa_baselines.py")
        sys.exit(1)

    with open(BASELINES_PATH) as f:
        data = json.load(f)
    with open(AUDIT_PATH) as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a['v22_label'] for a in audit_data['details']}

    print("=" * 72)
    print("HEAD-TO-HEAD: S vs Entropy vs LogProb vs Top2Margin")
    print("=" * 72)
    print(f"Data: {BASELINES_PATH}")
    print(f"Model: {data.get('model')}")
    print(f"N items: {len(data['results'])}")
    print()

    items = []
    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS: continue
        idx = r['hf_index']
        lbl = audit.get(idx)
        if lbl in ('correct', 'correct_semantic'): label = 0
        elif lbl in ('hallucinated', 'hallucinated_semantic'): label = 1
        else: continue
        gen = r['generation']
        traj = gen['c_delta_trajectory']
        ent = gen['entropy_trajectory']
        lp = gen['logprob_trajectory']
        margin = gen['top2_margin_trajectory']
        if not traj: continue
        items.append({
            'label': label,
            'q_chars': len(r['question']),
            # S_early
            'S_early': compute_S_range(traj, 0, 7),
            # Entropy features
            'entropy_mean_early': float(np.mean(ent[:7])) if len(ent) >= 7 else 0,
            'entropy_mean': float(np.mean(ent)) if ent else 0,
            'entropy_t0': ent[0] if ent else 0,
            'entropy_max_early': max(ent[:7]) if len(ent) >= 7 else 0,
            # Logprob features
            'logprob_mean_early': float(np.mean(lp[:7])) if len(lp) >= 7 else 0,
            'logprob_mean': float(np.mean(lp)) if lp else 0,
            'logprob_t0': lp[0] if lp else 0,
            'logprob_min_early': min(lp[:7]) if len(lp) >= 7 else 0,
            # Margin features
            'margin_mean_early': float(np.mean(margin[:7])) if len(margin) >= 7 else 0,
            'margin_mean': float(np.mean(margin)) if margin else 0,
            'margin_t0': margin[0] if margin else 0,
            'margin_min_early': min(margin[:7]) if len(margin) >= 7 else 0,
        })

    y = np.array([i['label'] for i in items])
    n_c = (y == 0).sum(); n_h = (y == 1).sum()
    print(f"Labeled: {n_c} correct, {n_h} hallucinated")
    print()

    if n_c < 3 or n_h < 3:
        print("[!] Insufficient data"); return

    # Single-feature discrimination
    print("-" * 72)
    print("SINGLE-FEATURE DISCRIMINATION")
    print("-" * 72)
    feature_names = [k for k in items[0].keys() if k not in ('label', 'q_chars')]
    print(f"{'feature':<28} {'d':>8} {'AUC':>8} {'p':>10}")
    single_results = []
    for name in sorted(feature_names):
        vals = np.array([i[name] for i in items])
        c_v = vals[y == 0]; h_v = vals[y == 1]
        pool = np.sqrt((c_v.var(ddof=1) + h_v.var(ddof=1)) / 2)
        d = (h_v.mean() - c_v.mean()) / pool if pool > 0 else 0
        auc = roc_auc_score(y, vals)
        auc_dir = max(auc, 1 - auc)
        _, p = stats.ttest_ind(c_v, h_v, equal_var=False)
        single_results.append({'name': name, 'd': d, 'auc': auc, 'auc_dir': auc_dir, 'p': p})
        marker = " *" if auc_dir > 0.65 else ""
        print(f"{name:<28} {d:>+8.3f} {auc:>8.3f} {p:>10.4f}{marker}")

    print()

    # Ranked
    print("-" * 72)
    print("RANKED BY |AUC - 0.5| (DIRECTION-CORRECTED)")
    print("-" * 72)
    for i, r in enumerate(sorted(single_results, key=lambda x: -x['auc_dir']), 1):
        print(f"  {i:>2}. {r['name']:<28} AUC_dir={r['auc_dir']:.3f}  d={r['d']:+.3f}")
        if i >= 10: break
    print()

    # Multi-feature classifiers
    print("-" * 72)
    print("MULTI-FEATURE LOGISTIC REGRESSION (5-fold CV AUC)")
    print("-" * 72)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'))])

    configs = {
        "S_early only": ['S_early'],
        "entropy_mean_early only": ['entropy_mean_early'],
        "logprob_mean_early only": ['logprob_mean_early'],
        "margin_mean_early only": ['margin_mean_early'],
        "all entropy features": [k for k in feature_names if 'entropy' in k],
        "all logprob features": [k for k in feature_names if 'logprob' in k],
        "all margin features": [k for k in feature_names if 'margin' in k],
        "S_early + entropy + logprob + margin (all)": [k for k in feature_names],
        "S_early + entropy_mean_early": ['S_early', 'entropy_mean_early'],
        "q_chars only (baseline)": ['q_chars'],
        "S_early + q_chars": ['S_early', 'q_chars'],
        "entropy_mean_early + q_chars": ['entropy_mean_early', 'q_chars'],
        "all features + q_chars": [k for k in feature_names] + ['q_chars'],
    }

    print(f"{'config':<45} {'CV AUC':>10} {'std':>7}")
    for name, feats in configs.items():
        X = np.column_stack([[i[f] for i in items] for f in feats])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        print(f"{name:<45} {scores.mean():>10.3f} {scores.std():>7.3f}")
    print()

    # KEY QUESTION: does S_early ADD info beyond entropy?
    print("=" * 72)
    print("KEY QUESTION: does S_early add info beyond logit entropy?")
    print("=" * 72)
    entropy_scores = cross_val_score(pipe,
        np.column_stack([[i['entropy_mean_early'] for i in items]]), y, cv=cv, scoring='roc_auc')
    combined_scores = cross_val_score(pipe,
        np.column_stack([[i['S_early'] for i in items], [i['entropy_mean_early'] for i in items]]),
        y, cv=cv, scoring='roc_auc')
    _, p_paired = stats.ttest_rel(entropy_scores, combined_scores)
    print(f"  entropy_mean_early alone: CV AUC = {entropy_scores.mean():.3f}")
    print(f"  S_early + entropy:       CV AUC = {combined_scores.mean():.3f}")
    print(f"  lift: {combined_scores.mean() - entropy_scores.mean():+.3f}")
    print(f"  paired t-test on fold AUCs: p={p_paired:.4f}")
    print()

    # Correlation between S_early and entropy features
    print("-" * 72)
    print("CORRELATION: S_early vs entropy/logprob/margin")
    print("-" * 72)
    for name in sorted(feature_names):
        if name == 'S_early': continue
        r, p = stats.pearsonr([i['S_early'] for i in items], [i[name] for i in items])
        print(f"  r(S_early, {name:<26}) = {r:+.3f} (p={p:.4f})")
    print()

    # Save
    out = {
        'n_correct': int(n_c), 'n_hallucinated': int(n_h),
        'single_features': single_results,
    }
    with open("truthfulqa_results/baselines_headtohead.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved: truthfulqa_results/baselines_headtohead.json")


if __name__ == '__main__':
    main()
