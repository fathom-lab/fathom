"""
analyze_calibration.py — calibration analysis of S-based predictions
=====================================================================
Reframes S as a CALIBRATION tool, not a hallucination detector.

For each generation, compute P(correct | features) and test:
 1. Expected Calibration Error (ECE)
 2. Reliability diagram
 3. Maximum Calibration Error (MCE)
 4. Brier score
 5. Isotonic calibration correction + post-calibration ECE

Calibration matters for enterprise LLM deployment: "when the model
says it's X% confident, is it actually right X% of the time?"

Usage:
    python analyze_calibration.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS
from analyze_S_biphasic import compute_S_range

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_v22():
    with open("truthfulqa_results/S_truthfulqa_n200.json") as f:
        data = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a for a in audit_data['details']}
    items = []
    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS: continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        if lbl in ('correct', 'correct_semantic'): label = 0
        elif lbl in ('hallucinated', 'hallucinated_semantic'): label = 1
        else: continue
        traj = r['generation']['c_delta_trajectory']
        if not traj: continue
        items.append({
            'S_early': compute_S_range(traj, 0, 7),
            'S_late': compute_S_range(traj, 15, 999),
            'q_chars': len(r['question']),
            'label': label,
        })
    return items


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE)."""
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_bounds[i]) & (y_prob < bin_bounds[i+1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_bounds[i]) & (y_prob <= bin_bounds[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_acc - bin_conf)
    return ece


def compute_mce(y_true, y_prob, n_bins=10):
    """Maximum Calibration Error."""
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_bounds[i]) & (y_prob < bin_bounds[i+1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_bounds[i]) & (y_prob <= bin_bounds[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))
    return mce


def main():
    items = load_v22()
    print("=" * 72)
    print("CALIBRATION ANALYSIS — S as calibration tool")
    print("=" * 72)
    print(f"n={len(items)}")
    print()

    S_early = np.array([r['S_early'] for r in items])
    S_late = np.array([r['S_late'] for r in items])
    q_chars = np.array([r['q_chars'] for r in items])
    y = np.array([r['label'] for r in items])  # 1 = hallucinated

    feature_sets = {
        "S_early only": S_early.reshape(-1, 1),
        "S_early + S_late": np.column_stack([S_early, S_late]),
        "S_early + q_chars": np.column_stack([S_early, q_chars]),
        "S_early + S_late + q_chars": np.column_stack([S_early, S_late, q_chars]),
        "q_chars only (baseline)": q_chars.reshape(-1, 1),
    }

    # Generate OOF predictions for each feature set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'))])

    print("-" * 72)
    print("CALIBRATION METRICS (5-fold OOF, P(hallucinated))")
    print("-" * 72)
    print(f"{'feature set':<32} {'ECE':>7} {'MCE':>7} {'Brier':>7} {'AUC':>7}")

    oof_by_feature = {}
    for name, X in feature_sets.items():
        oof_probs = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]
        oof_by_feature[name] = oof_probs
        ece = compute_ece(y, oof_probs)
        mce = compute_mce(y, oof_probs)
        brier = brier_score_loss(y, oof_probs)
        auc = roc_auc_score(y, oof_probs)
        print(f"{name:<32} {ece:>7.3f} {mce:>7.3f} {brier:>7.3f} {auc:>7.3f}")
    print()

    # Also report on raw S_early as calibration signal (no classifier fit)
    # Rescale to [0, 1]
    S_early_01 = (S_early - S_early.min()) / (S_early.max() - S_early.min() + 1e-9)
    ece_raw = compute_ece(y, S_early_01)
    brier_raw = brier_score_loss(y, S_early_01)
    print(f"  Raw S_early (rescaled to [0,1]): ECE={ece_raw:.3f}, Brier={brier_raw:.3f}")

    # Isotonic calibration on best feature set
    print()
    print("-" * 72)
    print("ISOTONIC CALIBRATION (on best feature set)")
    print("-" * 72)
    # Best = S_early + S_late + q_chars
    best_name = "S_early + S_late + q_chars"
    oof_probs = oof_by_feature[best_name]
    ece_pre = compute_ece(y, oof_probs)
    iso = IsotonicRegression(out_of_bounds='clip').fit(oof_probs, y)
    calibrated = iso.transform(oof_probs)
    ece_post = compute_ece(y, calibrated)
    brier_post = brier_score_loss(y, calibrated)
    print(f"  Pre-isotonic:  ECE={ece_pre:.3f}")
    print(f"  Post-isotonic: ECE={ece_post:.3f}")
    print(f"  Post-isotonic Brier: {brier_post:.3f}")
    print(f"  Improvement: delta-ECE = {ece_pre - ece_post:+.3f}")
    print()

    # Generate reliability diagram (best feature set, pre/post isotonic)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, probs, title in [
        (axes[0], oof_probs, f"Pre-isotonic (ECE={ece_pre:.3f})"),
        (axes[1], calibrated, f"Post-isotonic (ECE={ece_post:.3f})"),
    ]:
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy='quantile')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='perfect calibration')
        ax.plot(mean_pred, frac_pos, 'o-', color='#2E8B57', linewidth=2, markersize=7, label='model')
        ax.set_xlabel("predicted P(hallucinated)")
        ax.set_ylabel("observed hallucination rate")
        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.suptitle(f"Reliability diagram: {best_name}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "calibration_reliability.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "calibration_reliability.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: figures/calibration_reliability.pdf")
    print()

    # Quantile analysis: at each S_early quantile, what is halluc rate?
    print("-" * 72)
    print("S_early DECILE CALIBRATION (n=94, raw values)")
    print("-" * 72)
    n_deciles = 10
    sorted_idx = np.argsort(S_early)
    decile_size = len(S_early) // n_deciles
    print(f"{'decile':>8} {'S range':>25} {'halluc_rate':>12} {'n':>5}")
    for i in range(n_deciles):
        lo = i * decile_size
        hi = (i + 1) * decile_size if i < n_deciles - 1 else len(S_early)
        idx = sorted_idx[lo:hi]
        s_min, s_max = S_early[idx].min(), S_early[idx].max()
        hr = y[idx].mean() * 100
        print(f"  {i+1:>3}0% {s_min:>11.6f}-{s_max:<11.6f} {hr:>11.1f}% {len(idx):>5}")
    print()

    # Save
    out = {
        'n': len(items),
        'calibration_metrics': {
            name: {
                'ECE': float(compute_ece(y, oof_by_feature[name])),
                'MCE': float(compute_mce(y, oof_by_feature[name])),
                'Brier': float(brier_score_loss(y, oof_by_feature[name])),
                'AUC': float(roc_auc_score(y, oof_by_feature[name])),
            }
            for name in feature_sets
        },
        'isotonic_calibration': {
            'best_features': best_name,
            'ece_pre': float(ece_pre),
            'ece_post': float(ece_post),
            'brier_post': float(brier_post),
        }
    }
    with open("truthfulqa_results/calibration_analysis.json", 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved: truthfulqa_results/calibration_analysis.json")


if __name__ == '__main__':
    main()
