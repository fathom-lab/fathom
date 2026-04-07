"""
analyze_t0_signal.py — the token-0 discrimination is the REAL signal
========================================================================
If single-token C_delta at t=0 achieves d=0.70, AUC=0.70, this is
literally a first-token hallucination detector. Test rigorously:

1. Bootstrap CI on token-0 AUC
2. Permutation test on token-0 discrimination
3. Comparison: token 0 vs S_early vs S_full
4. Operating characteristics for deployment
5. What's the MECHANISTIC interpretation?

Usage:
    python analyze_t0_signal.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS, compute_S
from analyze_S_biphasic import compute_S_range


def load_v22():
    with open("truthfulqa_results/S_truthfulqa_n200.json") as f:
        data = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a for a in audit_data['details']}
    items = []
    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS:
            continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        if lbl in ('correct', 'correct_semantic'):
            label = 'correct'
        elif lbl in ('hallucinated', 'hallucinated_semantic'):
            label = 'hallucinated'
        else:
            continue
        items.append({
            'label': label,
            'traj': r['generation']['c_delta_trajectory'],
            'tokens': r['generation'].get('tokens', []),
        })
    return items


def main():
    items = load_v22()
    corr = [i for i in items if i['label']=='correct']
    hall = [i for i in items if i['label']=='hallucinated']
    print("=" * 72)
    print("TOKEN-0 C_DELTA AS HALLUCINATION DETECTOR")
    print("=" * 72)
    print(f"n_correct={len(corr)}, n_hallucinated={len(hall)}")
    print()

    # Token 0
    c_t0 = np.array([i['traj'][0] for i in corr if i['traj']])
    h_t0 = np.array([i['traj'][0] for i in hall if i['traj']])

    # Primary test
    _, p = stats.ttest_ind(c_t0, h_t0, equal_var=False)
    pooled = np.sqrt((c_t0.var(ddof=1) + h_t0.var(ddof=1))/2)
    d = (h_t0.mean() - c_t0.mean())/pooled if pooled>0 else 0
    auc = roc_auc_score([0]*len(c_t0) + [1]*len(h_t0), list(c_t0) + list(h_t0))
    _, p_mw = stats.mannwhitneyu(c_t0, h_t0, alternative='two-sided')
    print("-" * 72)
    print("PRIMARY: Welch's t-test on token 0")
    print("-" * 72)
    print(f"  correct(t=0):       {c_t0.mean():+.6f} +/- {c_t0.std(ddof=1):+.6f}")
    print(f"  hallucinated(t=0):  {h_t0.mean():+.6f} +/- {h_t0.std(ddof=1):+.6f}")
    print(f"  Welch t: p={p:.4f}, d={d:+.3f}, AUC={auc:.3f}")
    print(f"  Mann-Whitney: p={p_mw:.4f}")
    print(f"  Gates (p<0.05, d>0.4, AUC>0.65): "
          f"{'PASS' if (p<0.05 and abs(d)>0.4 and auc>0.65) else 'FAIL'}")
    print()

    # Bootstrap CI
    print("-" * 72)
    print("BOOTSTRAP 95% CI on AUC (10000 resamples)")
    print("-" * 72)
    rng = np.random.RandomState(42)
    boot_aucs = []
    all_vals = list(c_t0) + list(h_t0)
    all_y = [0]*len(c_t0) + [1]*len(h_t0)
    for _ in range(10000):
        idx = rng.choice(len(all_vals), len(all_vals), replace=True)
        y_b = [all_y[i] for i in idx]
        if 0 in y_b and 1 in y_b:
            boot_aucs.append(roc_auc_score(y_b, [all_vals[i] for i in idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  CI excludes 0.5: {'YES' if ci_lo > 0.5 else 'NO'}")
    print()

    # Permutation test
    print("-" * 72)
    print("PERMUTATION TEST (2000 label shuffles)")
    print("-" * 72)
    labels = np.array(all_y)
    scores = np.array(all_vals)
    obs_auc = roc_auc_score(labels, scores)
    rng2 = np.random.RandomState(42)
    null_aucs = []
    for _ in range(2000):
        perm = rng2.permutation(labels)
        null_aucs.append(roc_auc_score(perm, scores))
    p_perm = float((np.array(null_aucs) >= obs_auc).mean())
    print(f"  Observed AUC: {obs_auc:.3f}")
    print(f"  Null AUC mean: {np.mean(null_aucs):.3f}")
    print(f"  Null AUC p95: {np.percentile(null_aucs, 95):.3f}")
    print(f"  Permutation p: {p_perm:.4f}")
    print()

    # Feature comparison
    print("-" * 72)
    print("COMPARISON: t=0 vs S_early vs S_full")
    print("-" * 72)
    metrics = {}
    # t=0
    metrics['C_delta(t=0)'] = (c_t0.tolist(), h_t0.tolist())
    # S_early
    metrics['S_early[0,7)'] = (
        [compute_S_range(i['traj'], 0, 7) for i in corr],
        [compute_S_range(i['traj'], 0, 7) for i in hall])
    # S_full
    metrics['S_full'] = (
        [compute_S(i['traj']) for i in corr],
        [compute_S(i['traj']) for i in hall])
    # mean early
    metrics['mean_C_delta[0,7)'] = (
        [np.mean(i['traj'][:7]) for i in corr],
        [np.mean(i['traj'][:7]) for i in hall])

    print(f"{'feature':<22} {'p':>10} {'d':>8} {'AUC':>8}")
    for name, (c_vals, h_vals) in metrics.items():
        c_arr = np.array(c_vals); h_arr = np.array(h_vals)
        if c_arr.var(ddof=1) + h_arr.var(ddof=1) == 0: continue
        _, pv = stats.ttest_ind(c_arr, h_arr, equal_var=False)
        pool = np.sqrt((c_arr.var(ddof=1) + h_arr.var(ddof=1))/2)
        de = (h_arr.mean() - c_arr.mean())/pool if pool>0 else 0
        ac = roc_auc_score([0]*len(c_arr)+[1]*len(h_arr), list(c_arr)+list(h_arr))
        star = "*" if pv < 0.05 and abs(de) > 0.4 and ac > 0.65 else " "
        print(f"{name:<22} {pv:>10.4f} {de:>+8.3f} {ac:>8.3f}{star}")
    print()

    # Operating characteristics for deployment
    print("-" * 72)
    print("OPERATING CHARACTERISTICS (token-0 detector)")
    print("-" * 72)
    fpr, tpr, thr = roc_curve([0]*len(c_t0)+[1]*len(h_t0), list(c_t0)+list(h_t0))
    print(f"{'target FPR':>12} {'threshold':>12} {'TPR':>7} {'precision':>10}")
    for tgt in [0.05, 0.10, 0.15, 0.20, 0.30]:
        idx = np.argmin(np.abs(fpr - tgt))
        t = thr[idx]
        pred_c = (c_t0 > t).sum()  # false positives (correct flagged as hall)
        pred_h = (h_t0 > t).sum()  # true positives
        prec = pred_h / max(1, pred_c + pred_h)
        print(f"FPR={tgt:>5.2f}   {t:>12.6f} {tpr[idx]:>7.3f} {prec:>10.3f}")
    print()

    # Save
    out = {
        'n_correct': len(c_t0),
        'n_hallucinated': len(h_t0),
        'token_0_test': {
            'p': float(p), 'd': float(d), 'auc': float(auc),
            'mann_whitney_p': float(p_mw),
            'bootstrap_95ci': [float(ci_lo), float(ci_hi)],
            'permutation_p': float(p_perm),
        },
        'comparison': {
            name: {
                'p': float(stats.ttest_ind(np.array(c),np.array(h),equal_var=False)[1]),
                'd': float((np.mean(h)-np.mean(c))/np.sqrt((np.var(c,ddof=1)+np.var(h,ddof=1))/2))
                     if np.var(c,ddof=1)+np.var(h,ddof=1)>0 else 0,
                'auc': float(roc_auc_score([0]*len(c)+[1]*len(h), list(c)+list(h))),
            }
            for name, (c, h) in metrics.items()
        }
    }
    with open("truthfulqa_results/S_truthfulqa_n200_t0_analysis.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved: truthfulqa_results/S_truthfulqa_n200_t0_analysis.json")


if __name__ == '__main__':
    main()
