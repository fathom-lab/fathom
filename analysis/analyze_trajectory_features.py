"""
analyze_trajectory_features.py — comprehensive trajectory shape features
==========================================================================
Extracts a full feature vector from each C_delta trajectory and tests
which features (alone or in combination) discriminate correct vs
hallucinated generations.

Tests whether S(max/spike_count) is the best single feature, or whether
a richer feature set (autocorrelation, slope, peak position) beats it.

Usage:
    python analyze_trajectory_features.py [results_file.json]
"""

import json
import sys
import os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import (
    classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY
)


def trajectory_features(traj, thresholds=(0.003, 0.005, 0.008, 0.010)):
    """Extract comprehensive feature vector from a trajectory."""
    if not traj or len(traj) < 3:
        return None
    t = np.array(traj, dtype=float)
    n = len(t)
    features = {}

    # Basic stats
    features['mean'] = float(t.mean())
    features['median'] = float(np.median(t))
    features['std'] = float(t.std(ddof=1)) if n > 1 else 0.0
    features['max'] = float(t.max())
    features['min'] = float(t.min())
    features['range'] = features['max'] - features['min']
    features['cv'] = features['std'] / features['mean'] if features['mean'] > 0 else 0.0

    # Peak position and timing
    features['peak_idx'] = int(t.argmax())
    features['peak_position'] = features['peak_idx'] / n
    features['mean_position'] = float(np.sum(t * np.arange(n)) / t.sum() / n) if t.sum() > 0 else 0.5

    # Spike counts and S values at multiple thresholds
    for thr in thresholds:
        spikes = t > thr
        n_spikes = int(spikes.sum())
        features[f'n_spikes_t{thr}'] = n_spikes
        features[f'S_t{thr}'] = features['max'] / n_spikes if n_spikes > 0 else 0.0
        if n_spikes > 0:
            spike_positions = np.where(spikes)[0]
            features[f'first_spike_t{thr}'] = float(spike_positions.min() / n)
            features[f'last_spike_t{thr}'] = float(spike_positions.max() / n)
            features[f'mean_spike_t{thr}'] = float(t[spikes].mean())
        else:
            features[f'first_spike_t{thr}'] = 1.0
            features[f'last_spike_t{thr}'] = 0.0
            features[f'mean_spike_t{thr}'] = 0.0

    # Slope / trend
    if n >= 2:
        x = np.arange(n)
        slope, _, r_val, _, _ = stats.linregress(x, t)
        features['slope'] = float(slope)
        features['trend_r'] = float(r_val)
    else:
        features['slope'] = 0.0
        features['trend_r'] = 0.0

    # Autocorrelation at lag 1
    if n >= 3:
        centered = t - t.mean()
        denom = (centered * centered).sum()
        if denom > 0:
            features['autocorr_1'] = float((centered[:-1] * centered[1:]).sum() / denom)
        else:
            features['autocorr_1'] = 0.0
    else:
        features['autocorr_1'] = 0.0

    # Early vs late halves
    half = n // 2
    features['early_mean'] = float(t[:half].mean())
    features['late_mean'] = float(t[half:].mean())
    features['late_minus_early'] = features['late_mean'] - features['early_mean']
    features['early_max'] = float(t[:half].max())
    features['late_max'] = float(t[half:].max())

    # First/last third
    third = max(1, n // 3)
    features['first_third_max'] = float(t[:third].max())
    features['last_third_max'] = float(t[-third:].max())

    # Monotonicity
    diffs = np.diff(t)
    features['pct_increasing'] = float((diffs > 0).mean()) if len(diffs) > 0 else 0.5
    features['max_increase'] = float(diffs.max()) if len(diffs) > 0 else 0.0
    features['max_decrease'] = float(diffs.min()) if len(diffs) > 0 else 0.0

    return features


def discriminate(features_list_corr, features_list_hall, feature_name):
    """Single-feature unpaired test + AUC."""
    c = np.array([f[feature_name] for f in features_list_corr])
    h = np.array([f[feature_name] for f in features_list_hall])
    if len(c) < 2 or len(h) < 2:
        return None
    _, p = stats.ttest_ind(c, h, equal_var=False)
    pooled = np.sqrt((c.var(ddof=1) + h.var(ddof=1)) / 2)
    d = (h.mean() - c.mean()) / pooled if pooled > 0 else 0.0
    # AUC using hallucinated=1
    auc = roc_auc_score([0]*len(c) + [1]*len(h), list(c) + list(h))
    # flip to get "better direction" for ranking
    auc_dir = max(auc, 1 - auc)
    return {
        'name': feature_name,
        'correct_mean': float(c.mean()),
        'hallucinated_mean': float(h.mean()),
        'p': float(p),
        'd': float(d),
        'auc': float(auc),
        'auc_directional': float(auc_dir),
    }


def analyze(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 72)
    print("Trajectory SHAPE FEATURES analysis")
    print("=" * 72)
    print(f"Data:       {results_path}")
    print(f"N items:    {len(data['results'])}")
    print()

    # Label + extract features
    correct_feats, halluc_feats = [], []
    for r in data['results']:
        gen = r['generation']['text']
        n_tokens = r['generation']['total_tokens']
        traj = r['generation']['c_delta_trajectory']
        if n_tokens < MIN_GEN_TOKENS:
            continue
        label, _ = classify_generation(
            gen, r['correct_answers'], r['incorrect_answers'], r.get('best_answer', '')
        )
        feats = trajectory_features(traj)
        if feats is None:
            continue
        if label == 'correct':
            correct_feats.append(feats)
        elif label == 'hallucinated':
            halluc_feats.append(feats)

    print(f"Labeled: {len(correct_feats)} correct, {len(halluc_feats)} hallucinated")
    print()

    if len(correct_feats) < 4 or len(halluc_feats) < 4:
        print("[!] Insufficient data for feature analysis.")
        return

    # Single-feature discrimination
    all_names = list(correct_feats[0].keys())
    results_single = []
    for name in all_names:
        r = discriminate(correct_feats, halluc_feats, name)
        if r:
            results_single.append(r)

    # Rank by directional AUC
    results_single.sort(key=lambda x: x['auc_directional'], reverse=True)

    print("-" * 72)
    print("TOP 20 FEATURES (ranked by |AUC - 0.5|)")
    print("-" * 72)
    print(f"{'feature':<25} {'AUC':>6} {'AUC_dir':>9} {'d':>7} {'p':>8} "
          f"{'corr_mean':>10} {'hall_mean':>10}")
    for r in results_single[:20]:
        print(f"{r['name']:<25} {r['auc']:>6.3f} {r['auc_directional']:>9.3f} "
              f"{r['d']:>7.3f} {r['p']:>8.4f} "
              f"{r['correct_mean']:>10.5f} {r['hallucinated_mean']:>10.5f}")
    print()

    # Multi-feature classifier: all features + logreg with CV
    print("-" * 72)
    print("MULTI-FEATURE LOGISTIC REGRESSION (5-fold CV)")
    print("-" * 72)
    X_corr = np.array([[f[n] for n in all_names] for f in correct_feats])
    X_hall = np.array([[f[n] for n in all_names] for f in halluc_feats])
    X = np.vstack([X_corr, X_hall])
    y = np.array([0] * len(X_corr) + [1] * len(X_hall))
    n_min = min(len(X_corr), len(X_hall))
    if n_min < 5:
        print("  Insufficient samples for 5-fold CV")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0, penalty='l2')
            cv = StratifiedKFold(n_splits=min(5, n_min), shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
            print(f"  All {len(all_names)} features -> CV AUC: "
                  f"{scores.mean():.3f} ± {scores.std():.3f}")
            print(f"  (individual fold AUCs: {[f'{s:.3f}' for s in scores]})")
        except Exception as e:
            print(f"  Classifier error: {e}")

        # Top-5 feature set
        top5_names = [r['name'] for r in results_single[:5]]
        X_top5 = np.array([[f[n] for n in top5_names] for f in correct_feats + halluc_feats])
        X_top5_scaled = StandardScaler().fit_transform(X_top5)
        try:
            scores5 = cross_val_score(clf, X_top5_scaled, y, cv=cv, scoring='roc_auc')
            print(f"  Top-5 features -> CV AUC: {scores5.mean():.3f} ± {scores5.std():.3f}")
            print(f"  Top-5 names: {top5_names}")
        except Exception as e:
            print(f"  Classifier error: {e}")

        # S-only baseline
        X_s = np.array([[f[f'S_t{THR_PRIMARY}']] for f in correct_feats + halluc_feats])
        X_s_scaled = StandardScaler().fit_transform(X_s)
        try:
            scores_s = cross_val_score(clf, X_s_scaled, y, cv=cv, scoring='roc_auc')
            print(f"  S alone (t={THR_PRIMARY}) -> CV AUC: {scores_s.mean():.3f} ± {scores_s.std():.3f}")
            print(f"  Multi-feature lift over S: {scores.mean() - scores_s.mean():+.3f}")
        except Exception as e:
            print(f"  Classifier error: {e}")

    # Save feature analysis
    out_path = results_path.replace('.json', '_features.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': results_path,
            'n_correct': len(correct_feats),
            'n_hallucinated': len(halluc_feats),
            'single_feature_ranking': results_single,
        }, f, indent=2)
    print(f"\n  features saved: {out_path}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(path):
        pilot = path.replace('n200.json', 'pilot.json')
        if os.path.exists(pilot):
            path = pilot
        else:
            print(f"ERROR: {path} not found")
            sys.exit(1)
    analyze(path)
