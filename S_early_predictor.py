"""
S_early_predictor.py — deployment-ready stream-time hallucination detector
===========================================================================
Takes a C_delta trajectory (partial or complete) and returns a
stream-time hallucination risk score based on the bi-phasic signature.

Usage as library:
    from S_early_predictor import SEarlyPredictor
    predictor = SEarlyPredictor.from_v22_data()
    score, flag = predictor.score(c_delta_trajectory[:7])

Usage as CLI (on saved data):
    python S_early_predictor.py --cv-eval  # evaluate on v2.2 data
    python S_early_predictor.py --calibrate  # fit threshold
"""

import json
import os
import sys
import argparse
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import compute_S, MIN_GEN_TOKENS
from analyze_S_biphasic import compute_S_range

EARLY_END = 7  # pre-registered (for bi-phasic replication)
THRESHOLD_S = 0.005


class SEarlyPredictor:
    """Stream-time hallucination risk scorer using S_early."""

    def __init__(self, flag_threshold, score_calibration=None):
        self.flag_threshold = flag_threshold
        self.score_calibration = score_calibration

    @classmethod
    def from_v22_data(cls, target_fpr=0.30):
        """Fit predictor on v2.2 labeled data, choose threshold at target FPR."""
        from sklearn.metrics import roc_curve
        correct, halluc = _load_v22()
        c_scores = [compute_S_range(t, 0, EARLY_END) for t in correct]
        h_scores = [compute_S_range(t, 0, EARLY_END) for t in halluc]
        y = [0]*len(c_scores) + [1]*len(h_scores)
        scores = c_scores + h_scores
        fpr, tpr, thresholds = roc_curve(y, scores)
        # Find threshold at target FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        return cls(flag_threshold=float(thresholds[idx]))

    def score(self, trajectory_prefix):
        """Compute S_early on a (partial) trajectory."""
        return compute_S_range(list(trajectory_prefix), 0, EARLY_END)

    def flag(self, trajectory_prefix):
        """Return True if risk exceeds threshold."""
        return self.score(trajectory_prefix) > self.flag_threshold


def _load_v22():
    with open("truthfulqa_results/S_truthfulqa_n200.json") as f:
        data = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a for a in audit_data['details']}
    correct, halluc = [], []
    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS:
            continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        tr = r['generation']['c_delta_trajectory']
        if lbl in ('correct', 'correct_semantic'):
            correct.append(tr)
        elif lbl in ('hallucinated', 'hallucinated_semantic'):
            halluc.append(tr)
    return correct, halluc


def cv_evaluate():
    """5-fold CV evaluation of S_early as a single-feature classifier."""
    correct, halluc = _load_v22()
    c_scores = np.array([compute_S_range(t, 0, EARLY_END) for t in correct])
    h_scores = np.array([compute_S_range(t, 0, EARLY_END) for t in halluc])
    X = np.concatenate([c_scores, h_scores]).reshape(-1, 1)
    y = np.array([0]*len(c_scores) + [1]*len(h_scores))

    print("=" * 72)
    print("S_EARLY DEPLOYMENT EVALUATION (5-fold stratified CV)")
    print("=" * 72)
    print(f"n_correct={len(c_scores)}, n_hallucinated={len(h_scores)}")
    print(f"S_early window: [0, {EARLY_END})")
    print(f"S threshold: {THRESHOLD_S}")
    print()

    # ROC analysis
    auc_full = roc_auc_score(y, X.flatten())
    print(f"Full-data ROC AUC: {auc_full:.3f}")
    fpr, tpr, thresholds = roc_curve(y, X.flatten())

    # Operating points
    print()
    print("OPERATING POINTS (hallucinated=positive class):")
    print(f"{'target':>12} {'threshold':>12} {'TPR':>7} {'FPR':>7} {'precision':>10}")
    for target_fpr in [0.10, 0.20, 0.30, 0.40, 0.50]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        t = thresholds[idx]
        # Compute precision at this threshold
        pred = X.flatten() > t
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        prec = tp / max(1, tp + fp)
        print(f"FPR={target_fpr:>6.2f} {t:>12.6f} {tpr[idx]:>7.3f} {fpr[idx]:>7.3f} {prec:>10.3f}")

    # CV AUC
    print()
    print("5-FOLD CV AUC:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_test = X[test_idx].flatten()
        y_test = y[test_idx]
        if len(np.unique(y_test)) == 2:
            cv_aucs.append(roc_auc_score(y_test, X_test))
    print(f"  CV AUC: {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}")
    print(f"  Fold AUCs: {[f'{a:.3f}' for a in cv_aucs]}")

    # Bootstrap 95% CI on AUC
    print()
    print("BOOTSTRAP 95% CI ON AUC (n=1000 resamples):")
    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) == 2:
            boot_aucs.append(roc_auc_score(y[idx], X[idx].flatten()))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Calibration check: is score monotonic in hallucination probability?
    print()
    print("CALIBRATION CURVE (hallucination rate by S_early decile):")
    deciles = np.percentile(X.flatten(), np.arange(10, 101, 10))
    prev_edge = -np.inf
    print(f"  {'decile':>8} {'range':>25} {'hall_rate':>10} {'n':>5}")
    for i, edge in enumerate(deciles, 1):
        mask = (X.flatten() > prev_edge) & (X.flatten() <= edge)
        if mask.any():
            rate = y[mask].mean()
            n = int(mask.sum())
            print(f"  {i:>3}0% {prev_edge:>11.6f}-{edge:<11.6f} {rate*100:>9.1f}% {n:>5}")
        prev_edge = edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-eval', action='store_true', help='5-fold CV evaluation')
    parser.add_argument('--calibrate', action='store_true', help='Fit threshold')
    parser.add_argument('--target-fpr', type=float, default=0.30)
    args = parser.parse_args()

    if args.cv_eval:
        cv_evaluate()
    elif args.calibrate:
        predictor = SEarlyPredictor.from_v22_data(target_fpr=args.target_fpr)
        print(f"Calibrated threshold (target FPR={args.target_fpr}): {predictor.flag_threshold:.6f}")
    else:
        cv_evaluate()


if __name__ == '__main__':
    main()
