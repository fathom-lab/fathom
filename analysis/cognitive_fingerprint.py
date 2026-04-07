"""
cognitive_fingerprint.py — architecture-agnostic cognitive fingerprint
=========================================================================
Instead of one metric (IPR, cosine, S), extract a FULL fingerprint
from each generation: a fixed-length vector capturing everything we
can measure about the model's internal state trajectory.

From the cosine probe we get 4 trajectories per generation:
  - c_delta (late_self - early_self)
  - c_align (cross-layer cosine)
  - c_early_self
  - c_late_self

From each trajectory, extract a standard feature vector (same as
analyze_trajectory_features.py). Then test whether the combined
fingerprint discriminates cross-architecture.

The key insight: maybe NO SINGLE metric generalizes cross-arch, but
the PATTERN of multiple metrics does. Different architectures might
express commitment differently, but all committed states share a
multi-dimensional signature.

Usage:
    python cognitive_fingerprint.py [cosine_probe_results.json]
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
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS


def extract_traj_features(traj, prefix=""):
    """Extract comprehensive features from a single trajectory."""
    if not traj or len(traj) < 3:
        return {}
    t = np.array(traj, dtype=float)
    n = len(t)
    f = {}
    f[f'{prefix}mean'] = float(t.mean())
    f[f'{prefix}std'] = float(t.std(ddof=1)) if n > 1 else 0.0
    f[f'{prefix}max'] = float(t.max())
    f[f'{prefix}min'] = float(t.min())
    f[f'{prefix}range'] = f[f'{prefix}max'] - f[f'{prefix}min']
    f[f'{prefix}peak_pos'] = float(t.argmax()) / n
    # Early vs late
    half = n // 2
    f[f'{prefix}early_mean'] = float(t[:half].mean())
    f[f'{prefix}late_mean'] = float(t[half:].mean())
    f[f'{prefix}late_minus_early'] = f[f'{prefix}late_mean'] - f[f'{prefix}early_mean']
    # Slope
    if n >= 2:
        slope, _, r_val, _, _ = stats.linregress(np.arange(n), t)
        f[f'{prefix}slope'] = float(slope)
        f[f'{prefix}trend_r'] = float(r_val)
    # Autocorrelation
    if n >= 3 and t.std() > 0:
        centered = t - t.mean()
        denom = (centered**2).sum()
        if denom > 0:
            f[f'{prefix}autocorr'] = float((centered[:-1]*centered[1:]).sum()/denom)
    # First 7 tokens (Phase 1)
    if n >= 7:
        f[f'{prefix}p1_mean'] = float(t[:7].mean())
        f[f'{prefix}p1_max'] = float(t[:7].max())
        f[f'{prefix}p1_std'] = float(t[:7].std(ddof=1))
    return f


def load_cosine_probe(path):
    """Load cosine probe results, extract fingerprints."""
    with open(path) as f:
        data = json.load(f)

    # Load v2.2 labels
    audit_path = os.path.join("truthfulqa_results", "S_truthfulqa_n200_v22_audit.json")
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            audit_data = json.load(f)
        audit = {a['hf_index']: a['v22_label'] for a in audit_data['details']}
    else:
        audit = {}

    items = []
    for r in data['results']:
        gen = r['generation']
        if gen['total_tokens'] < MIN_GEN_TOKENS:
            continue
        idx = r['hf_index']
        lbl = audit.get(idx)
        if lbl in ('correct', 'correct_semantic'):
            label = 0
        elif lbl in ('hallucinated', 'hallucinated_semantic'):
            label = 1
        else:
            continue

        # Extract fingerprint from all 4 trajectories
        fp = {}
        for traj_name in ['c_delta_trajectory', 'c_align_trajectory',
                          'c_early_self_trajectory', 'c_late_self_trajectory']:
            traj = gen.get(traj_name, [])
            prefix = traj_name.replace('_trajectory', '_')
            fp.update(extract_traj_features(traj, prefix))

        if not fp:
            continue
        fp['label'] = label
        fp['hf_index'] = idx
        items.append(fp)

    return data.get('model', 'unknown'), items


def analyze_one_model(model_name, items):
    """Test fingerprint discrimination for one model."""
    if len(items) < 10:
        return None

    feat_names = [k for k in items[0].keys() if k not in ('label', 'hf_index')]
    y = np.array([i['label'] for i in items])
    X = np.array([[i.get(f, 0.0) for f in feat_names] for i in items])

    # Remove NaN/inf
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]; y = y[mask]

    n_c = (y == 0).sum(); n_h = (y == 1).sum()
    if n_c < 3 or n_h < 3:
        return None

    # Single best feature
    best_feat = None
    best_auc_dir = 0
    for j, name in enumerate(feat_names):
        c = X[y == 0, j]; h = X[y == 1, j]
        if c.var(ddof=1) + h.var(ddof=1) < 1e-12:
            continue
        auc = roc_auc_score(y, X[:, j])
        auc_dir = max(auc, 1 - auc)
        if auc_dir > best_auc_dir:
            best_auc_dir = auc_dir
            pool = np.sqrt((c.var(ddof=1) + h.var(ddof=1)) / 2)
            best_feat = {
                'name': name,
                'd': float((h.mean() - c.mean()) / pool) if pool > 0 else 0,
                'auc': float(auc),
                'auc_dir': float(auc_dir),
            }

    # Multi-feature CV
    cv = StratifiedKFold(n_splits=min(5, min(n_c, n_h)), shuffle=True, random_state=42)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'))])
    try:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        cv_auc = float(scores.mean())
        cv_std = float(scores.std())
    except Exception:
        cv_auc = 0.5
        cv_std = 0.0

    return {
        'model': model_name,
        'n_correct': int(n_c),
        'n_hallucinated': int(n_h),
        'n_features': len(feat_names),
        'best_single_feature': best_feat,
        'cv_auc_all_features': cv_auc,
        'cv_std': cv_std,
    }


def main():
    # Find all cosine probe results
    patterns = [
        "truthfulqa_results/cosine_probe_*_n200.json",
        "truthfulqa_results/cosine_probe_*_pilot.json",
    ]
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    # Prefer n200 over pilot
    seen_models = set()
    final_files = []
    for f in files:
        if "n200" in f:
            final_files.append(f)
            seen_models.add(f.split("cosine_probe_")[1].split("_n200")[0])
    for f in files:
        slug = f.split("cosine_probe_")[1].split("_pilot")[0]
        if slug not in seen_models:
            final_files.append(f)

    if not final_files:
        print("No cosine probe results found.")
        print("Run: python run_cosine_probe_sequence.py --pilot")
        sys.exit(1)

    print("=" * 72)
    print("COGNITIVE FINGERPRINT — cross-architecture analysis")
    print("=" * 72)
    print(f"Files: {len(final_files)}")
    for f in final_files:
        print(f"  {os.path.basename(f)}")
    print()

    all_results = []
    for fpath in final_files:
        model, items = load_cosine_probe(fpath)
        print(f"\n--- {model} (n_labeled={len(items)}) ---")
        r = analyze_one_model(model, items)
        if r is None:
            print("  Insufficient data")
            continue
        all_results.append(r)
        print(f"  best single: {r['best_single_feature']['name']} "
              f"(AUC_dir={r['best_single_feature']['auc_dir']:.3f}, "
              f"d={r['best_single_feature']['d']:+.3f})")
        print(f"  full fingerprint CV AUC: {r['cv_auc_all_features']:.3f} "
              f"+/- {r['cv_std']:.3f}")

    if len(all_results) >= 2:
        print()
        print("=" * 72)
        print("CROSS-ARCHITECTURE SUMMARY")
        print("=" * 72)
        print(f"{'model':<45} {'best_feat_AUC':>14} {'best_d':>8} {'CV_AUC':>8}")
        for r in all_results:
            bf = r['best_single_feature']
            print(f"{r['model']:<45} {bf['auc_dir']:>14.3f} {bf['d']:>+8.3f} "
                  f"{r['cv_auc_all_features']:>8.3f}")

        # Are all fingerprint CV AUCs > 0.55?
        all_above = all(r['cv_auc_all_features'] > 0.55 for r in all_results)
        print()
        if all_above:
            print("  ALL models: fingerprint CV AUC > 0.55")
            print("  => Multi-metric fingerprint may generalize cross-architecture")
        else:
            print("  Some models below 0.55 — fingerprint doesn't universally discriminate")

    # Save
    with open("truthfulqa_results/cognitive_fingerprint_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: truthfulqa_results/cognitive_fingerprint_results.json")


if __name__ == '__main__':
    main()
