"""
analyze_cosine_cross_arch.py — cross-architecture test with cosine probe
===========================================================================
Tests whether the cosine probe's C_delta (late_self - early_self layer
similarity) discriminates hallucination across architectures.

Also tests C_align (cross-band similarity) and all 4 trajectories
combined via cognitive fingerprint.

Usage:
    python analyze_cosine_cross_arch.py
"""

import json
import os
import sys
import glob
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS


def compute_S_range(traj, start, end, threshold=None):
    """S = max/spike_count on traj[start:end]. Threshold auto-set if None."""
    sl = traj[start:end] if end < 999 else traj[start:]
    if not sl: return 0.0
    if threshold is None:
        # Auto: median of trajectory as threshold
        threshold = float(np.median(traj))
    spikes = [c for c in sl if c > threshold]
    if not spikes: return 0.0
    return max(sl) / len(spikes)


def load_v22_labels():
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    return {a['hf_index']: a['v22_label'] for a in audit_data['details']}


def analyze_one_model(data, v22_labels):
    """Analyze one model's cosine probe results."""
    model = data.get('model', 'unknown')

    items = []
    for r in data['results']:
        gen = r['generation']
        if gen['total_tokens'] < MIN_GEN_TOKENS: continue
        lbl = v22_labels.get(r['hf_index'])
        if lbl in ('correct', 'correct_semantic'): label = 0
        elif lbl in ('hallucinated', 'hallucinated_semantic'): label = 1
        else: continue
        items.append({'label': label, 'gen': gen, 'hf_index': r['hf_index']})

    n_c = sum(1 for i in items if i['label'] == 0)
    n_h = sum(1 for i in items if i['label'] == 1)
    if n_c < 3 or n_h < 3:
        return None

    y = np.array([i['label'] for i in items])
    result = {'model': model, 'n_correct': n_c, 'n_hallucinated': n_h}

    # Test each trajectory type
    traj_names = ['c_delta_trajectory', 'c_align_trajectory',
                  'c_early_self_trajectory', 'c_late_self_trajectory']
    for tname in traj_names:
        trajs = [i['gen'].get(tname, []) for i in items]
        if not trajs or not trajs[0]: continue

        # Mean of early window [0,7)
        early_means = []
        for t in trajs:
            if len(t) >= 7:
                early_means.append(float(np.mean(t[:7])))
            else:
                early_means.append(float(np.mean(t)))
        early_means = np.array(early_means)

        c_v = early_means[y == 0]; h_v = early_means[y == 1]
        _, p = stats.ttest_ind(c_v, h_v, equal_var=False)
        pool = np.sqrt((c_v.var(ddof=1) + h_v.var(ddof=1)) / 2)
        d = (h_v.mean() - c_v.mean()) / pool if pool > 0 else 0.0
        auc = roc_auc_score(y, early_means)

        short = tname.replace('_trajectory', '')
        result[f'{short}_d'] = float(d)
        result[f'{short}_auc'] = float(auc)
        result[f'{short}_p'] = float(p)

    # Multi-feature fingerprint
    feat_names_all = []
    X_rows = []
    for i in items:
        feats = {}
        for tname in traj_names:
            t = i['gen'].get(tname, [])
            if not t: continue
            prefix = tname.replace('_trajectory', '_')
            arr = np.array(t[:min(40, len(t))])
            if len(arr) < 3: continue
            feats[f'{prefix}mean'] = float(arr.mean())
            feats[f'{prefix}std'] = float(arr.std(ddof=1))
            feats[f'{prefix}max'] = float(arr.max())
            feats[f'{prefix}min'] = float(arr.min())
            if len(arr) >= 7:
                feats[f'{prefix}p1_mean'] = float(arr[:7].mean())
                feats[f'{prefix}p1_std'] = float(arr[:7].std(ddof=1))
            if len(arr) >= 14:
                feats[f'{prefix}late_mean'] = float(arr[14:].mean())
        if not feat_names_all:
            feat_names_all = sorted(feats.keys())
        X_rows.append([feats.get(f, 0.0) for f in feat_names_all])

    if X_rows and len(feat_names_all) > 0:
        X = np.array(X_rows)
        mask = np.all(np.isfinite(X), axis=1)
        X = X[mask]; y_clean = y[mask]
        if len(np.unique(y_clean)) == 2 and min(np.bincount(y_clean)) >= 3:
            cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y_clean))),
                                 shuffle=True, random_state=42)
            pipe = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(C=1.0, max_iter=2000,
                                                       class_weight='balanced'))])
            try:
                scores = cross_val_score(pipe, X, y_clean, cv=cv, scoring='roc_auc')
                result['fingerprint_cv_auc'] = float(scores.mean())
                result['fingerprint_cv_std'] = float(scores.std())
                result['n_fingerprint_features'] = len(feat_names_all)
            except Exception:
                pass

    return result


def main():
    v22 = load_v22_labels()

    # Find cosine probe files
    files_n200 = sorted(glob.glob("truthfulqa_results/cosine_probe_*_n200.json"))
    files_pilot = sorted(glob.glob("truthfulqa_results/cosine_probe_*_pilot.json"))
    # Prefer n200
    seen = set()
    files = []
    for f in files_n200:
        files.append(f)
        seen.add(f.replace("_n200.json", ""))
    for f in files_pilot:
        base = f.replace("_pilot.json", "")
        if base not in seen:
            files.append(f)

    if not files:
        print("No cosine probe results found.")
        sys.exit(1)

    print("=" * 76)
    print("COSINE PROBE CROSS-ARCHITECTURE ANALYSIS")
    print("=" * 76)
    print(f"Files: {len(files)}")
    for f in files:
        print(f"  {os.path.basename(f)}")
    print()

    all_results = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        r = analyze_one_model(data, v22)
        if r is None:
            print(f"  {data.get('model')}: insufficient data")
            continue
        all_results.append(r)

    # Print table
    print("=" * 76)
    print("PER-MODEL RESULTS")
    print("=" * 76)
    print(f"{'model':<40} {'c_delta_d':>10} {'c_align_d':>10} {'c_delta_AUC':>12} "
          f"{'fp_CV_AUC':>10}")
    for r in all_results:
        print(f"{r['model']:<40} "
              f"{r.get('c_delta_d', 0):>+10.3f} "
              f"{r.get('c_align_d', 0):>+10.3f} "
              f"{r.get('c_delta_auc', 0):>12.3f} "
              f"{r.get('fingerprint_cv_auc', 0):>10.3f}")

    # Meta-analysis on c_delta
    print()
    print("=" * 76)
    print("C_DELTA META-ANALYSIS")
    print("=" * 76)
    ds = [r.get('c_delta_d', 0) for r in all_results]
    ps = [r.get('c_delta_p', 1) for r in all_results]
    aucs = [r.get('c_delta_auc', 0.5) for r in all_results]
    print(f"d values: {[f'{d:+.3f}' for d in ds]}")
    print(f"AUC values: {[f'{a:.3f}' for a in aucs]}")
    print(f"All d > 0: {'YES' if all(d > 0 for d in ds) else 'no'}")
    print(f"Mean d: {np.mean(ds):+.3f}")

    # Fisher combined
    one_sided = [max(1e-10, min(1-1e-10, p/2 if d > 0 else 1-p/2))
                 for p, d in zip(ps, ds)]
    stat = -2 * sum(np.log(p) for p in one_sided)
    from scipy.stats import chi2
    p_fisher = 1 - chi2.cdf(stat, df=2*len(one_sided))
    print(f"Fisher combined p (one-sided): {p_fisher:.6f}")

    # C_align meta
    print()
    print("=" * 76)
    print("C_ALIGN META-ANALYSIS")
    print("=" * 76)
    ds_a = [r.get('c_align_d', 0) for r in all_results]
    ps_a = [r.get('c_align_p', 1) for r in all_results]
    print(f"d values: {[f'{d:+.3f}' for d in ds_a]}")
    print(f"All d > 0: {'YES' if all(d > 0 for d in ds_a) else 'no'}")
    one_sided_a = [max(1e-10, min(1-1e-10, p/2 if d > 0 else 1-p/2))
                   for p, d in zip(ps_a, ds_a)]
    stat_a = -2 * sum(np.log(p) for p in one_sided_a)
    p_fisher_a = 1 - chi2.cdf(stat_a, df=2*len(one_sided_a))
    print(f"Fisher combined p (one-sided): {p_fisher_a:.6f}")

    # Fingerprint
    print()
    print("=" * 76)
    print("COGNITIVE FINGERPRINT CV AUC PER MODEL")
    print("=" * 76)
    for r in all_results:
        fp = r.get('fingerprint_cv_auc', None)
        if fp:
            print(f"  {r['model']}: CV AUC = {fp:.3f} +/- {r.get('fingerprint_cv_std',0):.3f} "
                  f"({r.get('n_fingerprint_features',0)} features)")
    fp_aucs = [r['fingerprint_cv_auc'] for r in all_results if 'fingerprint_cv_auc' in r]
    if fp_aucs:
        print(f"\n  Mean fingerprint CV AUC: {np.mean(fp_aucs):.3f}")
        print(f"  All > 0.55: {'YES' if all(a > 0.55 for a in fp_aucs) else 'no'}")

    # Verdict
    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    if all(d > 0 for d in ds) and p_fisher < 0.05:
        print("  CROSS-ARCHITECTURE REPLICATION: cosine C_delta consistent across models")
    elif all(d > 0 for d in ds):
        print("  Directionally consistent but not significant (need more power)")
    else:
        print("  Mixed directions across models")

    # Save
    with open("truthfulqa_results/cosine_cross_arch.json", "w") as f:
        json.dump({
            'results': all_results,
            'c_delta_fisher_p': float(p_fisher),
            'c_align_fisher_p': float(p_fisher_a),
            'all_c_delta_positive': all(d > 0 for d in ds),
        }, f, indent=2)
    print(f"\nSaved: truthfulqa_results/cosine_cross_arch.json")


if __name__ == '__main__':
    main()
