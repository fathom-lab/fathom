"""
S_calibrated_predictor.py — trained, cross-validated hallucination predictor
==============================================================================
Converts S-axis research into a deployable predictor:
 - trains logistic regression on trajectory features
 - 5-fold stratified CV for honest held-out AUC
 - calibration curve (reliability diagram)
 - bootstrap CI on CV AUC
 - compares S-only vs full-features vs random baseline

Output: deployable classifier + calibration diagnostics + paper tables

Usage:
    python S_calibrated_predictor.py [results_file.json]
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY
from analyze_trajectory_features import trajectory_features


FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def build_dataset(results_path):
    """Label + featurize every item."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X_list, y_list, names = [], [], None
    for r in data['results']:
        gen = r['generation']['text']
        n_tokens = r['generation']['total_tokens']
        traj = r['generation']['c_delta_trajectory']
        if n_tokens < MIN_GEN_TOKENS:
            continue
        label, _ = classify_generation(
            gen, r['correct_answers'], r['incorrect_answers'], r.get('best_answer', '')
        )
        if label not in ('correct', 'hallucinated'):
            continue
        feats = trajectory_features(traj)
        if feats is None:
            continue
        if names is None:
            names = list(feats.keys())
        X_list.append([feats[n] for n in names])
        y_list.append(1 if label == 'hallucinated' else 0)

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, names


def cv_auc_with_bootstrap(X, y, n_folds=5, n_bootstrap=1000, seed=42):
    """5-fold CV with bootstrap resampling of fold AUCs."""
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < n_folds:
        return None
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'))
    ])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    # Bootstrap CI on fold scores
    rng = np.random.RandomState(seed)
    boot_means = [rng.choice(scores, len(scores), replace=True).mean()
                  for _ in range(n_bootstrap)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "fold_scores": [float(s) for s in scores],
        "ci_95": [float(ci_lo), float(ci_hi)],
        "n_folds": n_folds,
    }


def feature_importance(X, y, names, n_folds=5, seed=42):
    """Extract logistic regression coefficients from full-data fit.
    Coefficients trained on z-scored features are directly comparable in magnitude.
    """
    if len(np.unique(y)) < 2:
        return None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
    lr.fit(X_scaled, y)
    coefs = lr.coef_[0]
    # Rank by absolute magnitude
    ranking = sorted(zip(names, coefs), key=lambda x: -abs(x[1]))
    return [{
        'feature': name,
        'coef': float(c),
        'direction': '+halluc' if c > 0 else '-halluc',
    } for name, c in ranking]


def oof_predictions(X, y, n_folds=5, seed=42):
    """Out-of-fold predictions for calibration analysis."""
    oof = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(X, y):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'))
        ])
        pipe.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    return oof


def baseline_comparisons(X, y, names):
    """Compare S-only, top-3, and all-features classifiers."""
    s_idx = names.index(f"S_t{THR_PRIMARY}")
    max_idx = names.index("max")
    mean_idx = names.index("mean")

    comparisons = {}
    comparisons["S_only"] = cv_auc_with_bootstrap(X[:, [s_idx]], y)
    comparisons["max_only"] = cv_auc_with_bootstrap(X[:, [max_idx]], y)
    comparisons["mean_only"] = cv_auc_with_bootstrap(X[:, [mean_idx]], y)
    comparisons["S_max_mean"] = cv_auc_with_bootstrap(X[:, [s_idx, max_idx, mean_idx]], y)
    comparisons["all_features"] = cv_auc_with_bootstrap(X, y)
    return comparisons


def fig_calibration(X, y, out_path):
    """Reliability diagram: predicted prob vs observed rate."""
    oof = oof_predictions(X, y)
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=10, strategy='quantile')

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='perfect calibration')
    ax.plot(mean_pred, frac_pos, 'o-', color='#2E8B57', linewidth=2, markersize=6, label='model')
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed hallucination rate")
    ax.set_title("Reliability diagram (5-fold OOF)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_roc(X, y, out_path):
    """ROC curve from OOF predictions."""
    oof = oof_predictions(X, y)
    fpr, tpr, _ = roc_curve(y, oof)
    auc = roc_auc_score(y, oof)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='chance')
    ax.plot(fpr, tpr, color='#DC143C', linewidth=2, label=f'model (AUC={auc:.3f})')
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve (5-fold OOF)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def analyze(results_path):
    print("=" * 72)
    print("S-axis CALIBRATED PREDICTOR (cross-validated)")
    print("=" * 72)
    print(f"Data: {results_path}")

    X, y, names = build_dataset(results_path)
    print(f"Samples: {len(y)} ({int(y.sum())} hallucinated, {int(len(y) - y.sum())} correct)")
    print(f"Features: {len(names)}")

    if len(y) < 10 or len(np.unique(y)) < 2:
        print("[!] Insufficient data for CV")
        return

    # Main CV result
    print()
    print("-" * 72)
    print("5-FOLD CV AUC (held-out, class-balanced logistic regression)")
    print("-" * 72)
    cv_full = cv_auc_with_bootstrap(X, y)
    print(f"  All {len(names)} features:")
    print(f"    mean AUC: {cv_full['mean']:.3f} ± {cv_full['std']:.3f}")
    print(f"    95% CI:   [{cv_full['ci_95'][0]:.3f}, {cv_full['ci_95'][1]:.3f}]")
    print(f"    folds:    {[f'{s:.3f}' for s in cv_full['fold_scores']]}")

    print()
    print("-" * 72)
    print("BASELINE COMPARISONS")
    print("-" * 72)
    comparisons = baseline_comparisons(X, y, names)
    for name, cv in comparisons.items():
        if cv:
            print(f"  {name:<20}: AUC {cv['mean']:.3f} ± {cv['std']:.3f}  "
                  f"CI[{cv['ci_95'][0]:.3f}, {cv['ci_95'][1]:.3f}]")

    # Feature importance
    print()
    print("-" * 72)
    print("FEATURE IMPORTANCE (standardized logistic regression coefficients)")
    print("-" * 72)
    importance = feature_importance(X, y, names)
    if importance:
        print(f"{'rank':>4} {'feature':<25} {'coef':>8} {'direction':>12}")
        for i, item in enumerate(importance[:15], 1):
            print(f"{i:>4} {item['feature']:<25} {item['coef']:>+8.3f} {item['direction']:>12}")

    # Generate figures
    print()
    print("Generating figures...")
    stem = os.path.splitext(os.path.basename(results_path))[0]
    fig_calibration(X, y, os.path.join(FIG_DIR, f"{stem}_calibration.pdf"))
    fig_roc(X, y, os.path.join(FIG_DIR, f"{stem}_roc.pdf"))
    print(f"  calibration: figures/{stem}_calibration.pdf")
    print(f"  ROC:         figures/{stem}_roc.pdf")

    # Save
    out_path = results_path.replace('.json', '_predictor.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': results_path,
            'n_samples': len(y),
            'n_hallucinated': int(y.sum()),
            'n_correct': int(len(y) - y.sum()),
            'n_features': len(names),
            'cv_full': cv_full,
            'baseline_comparisons': comparisons,
            'feature_importance': importance,
        }, f, indent=2)
    print(f"\n  saved: {out_path}")


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(p):
        print(f"ERROR: {p} not found")
        sys.exit(1)
    analyze(p)
