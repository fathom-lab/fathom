"""
analyze_S_v22.py — primary + bi-phasic tests using v2.2 combined labels
==========================================================================
Runs the full statistical suite using the semantic-augmented label set
(v2.2 = n-gram + embedding fallback).

Reports BOTH v2.1 (n-gram only) and v2.2 (combined) results side by side
for transparent comparison.

Usage:
    python analyze_S_v22.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import (
    classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY
)
from analyze_S_biphasic import compute_S_range


def welch(a, b):
    a, b = np.array(a), np.array(b)
    _, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
    auc = roc_auc_score([0]*len(a) + [1]*len(b), list(a) + list(b))
    return {"p": float(p), "d": float(d), "auc": float(auc),
            "a_mean": float(a.mean()), "b_mean": float(b.mean()),
            "a_n": len(a), "b_n": len(b)}


def main():
    RESULTS = "truthfulqa_results/S_truthfulqa_n200.json"
    AUDIT = "truthfulqa_results/S_truthfulqa_n200_v22_audit.json"

    with open(RESULTS, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(AUDIT, 'r', encoding='utf-8') as f:
        audit_data = json.load(f)

    # Build hf_index -> label map
    audit = {a['hf_index']: a for a in audit_data['details']}

    print("=" * 76)
    print("S-AXIS v2.2 ANALYSIS (n-gram + semantic combined labels)")
    print("=" * 76)
    print()

    # Collect trajectories by label set
    # v21: use v21_label; only keep correct/hallucinated
    # v22: combine correct + correct_semantic, halluc + halluc_semantic
    v21_corr_trajs, v21_hall_trajs = [], []
    v22_corr_trajs, v22_hall_trajs = [], []

    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS:
            continue
        idx = r['hf_index']
        traj = r['generation']['c_delta_trajectory']
        lbl21 = audit.get(idx, {}).get('v21_label')
        lbl22 = audit.get(idx, {}).get('v22_label')

        if lbl21 == 'correct': v21_corr_trajs.append(traj)
        elif lbl21 == 'hallucinated': v21_hall_trajs.append(traj)

        if lbl22 in ('correct', 'correct_semantic'): v22_corr_trajs.append(traj)
        elif lbl22 in ('hallucinated', 'hallucinated_semantic'): v22_hall_trajs.append(traj)

    print(f"v2.1 (n-gram only):  {len(v21_corr_trajs)} correct, {len(v21_hall_trajs)} hallucinated")
    print(f"v2.2 (n-gram+sem):   {len(v22_corr_trajs)} correct, {len(v22_hall_trajs)} hallucinated")
    print()

    # ==================================================================
    # PRIMARY H1 TEST — S(full trajectory)
    # ==================================================================
    print("=" * 76)
    print("PRIMARY H1 TEST: S(hallucinated) > S(correct)")
    print("=" * 76)

    for label_name, corr, hall in [
        ("v2.1 n-gram", v21_corr_trajs, v21_hall_trajs),
        ("v2.2 combined", v22_corr_trajs, v22_hall_trajs),
    ]:
        S_c = [compute_S(t) for t in corr]
        S_h = [compute_S(t) for t in hall]
        w = welch(S_c, S_h)
        print(f"\n{label_name} (n={w['a_n']}c + {w['b_n']}h):")
        print(f"  S(corr)={w['a_mean']:.6f}  S(hall)={w['b_mean']:.6f}")
        print(f"  p={w['p']:.4f}, d={w['d']:+.3f}, AUC={w['auc']:.3f}")
        gates = (w['p'] < 0.05) and (abs(w['d']) > 0.4) and (w['auc'] > 0.65)
        print(f"  gates: {'PASS' if gates else 'FAIL'}")

    # ==================================================================
    # BI-PHASIC TESTS
    # ==================================================================
    print()
    print("=" * 76)
    print("BI-PHASIC TESTS")
    print("=" * 76)

    EARLY_END = 7
    LATE_START = 15

    for label_name, corr, hall in [
        ("v2.2 combined", v22_corr_trajs, v22_hall_trajs),
    ]:
        print(f"\n{label_name}:")
        # Phase 1
        c_e = [compute_S_range(t, 0, EARLY_END) for t in corr]
        h_e = [compute_S_range(t, 0, EARLY_END) for t in hall]
        w_e = welch(c_e, h_e)
        print(f"  Phase 1 S_early[0,{EARLY_END}): p={w_e['p']:.4f}, d={w_e['d']:+.3f}, AUC={w_e['auc']:.3f}")

        # Phase 2
        c_l = [compute_S_range(t, LATE_START, 999) for t in corr]
        h_l = [compute_S_range(t, LATE_START, 999) for t in hall]
        w_l = welch(c_l, h_l)
        print(f"  Phase 2 S_late[{LATE_START},end): p={w_l['p']:.4f}, d={w_l['d']:+.3f}, AUC={w_l['auc']:.3f}")

        # Delta (early - late)
        c_d = np.array(c_e) - np.array(c_l)
        h_d = np.array(h_e) - np.array(h_l)
        w_d = welch(c_d, h_d)
        print(f"  Bi-phasic Delta (early-late): p={w_d['p']:.4f}, d={w_d['d']:+.3f}, AUC={w_d['auc']:.3f}")
        gates_d = (w_d['p'] < 0.05) and (abs(w_d['d']) > 0.4) and (w_d['auc'] > 0.65)
        print(f"  Delta gates: {'PASS' if gates_d else 'FAIL'}")

        # Ratio (early / late)
        c_r = [e / max(l, 1e-10) for e, l in zip(c_e, c_l)]
        h_r = [e / max(l, 1e-10) for e, l in zip(h_e, h_l)]
        w_r = welch(c_r, h_r)
        print(f"  Bi-phasic Ratio (early/late): p={w_r['p']:.4f}, d={w_r['d']:+.3f}, AUC={w_r['auc']:.3f}")

    # ==================================================================
    # EARLY-TOKEN HORIZON (v2.2)
    # ==================================================================
    print()
    print("=" * 76)
    print("EARLY-TOKEN HORIZON (v2.2)")
    print("=" * 76)
    print(f"{'N':>5} {'p':>10} {'d':>8} {'AUC':>8}  direction")
    for N in [3, 5, 7, 10, 12, 15, 20, 25, 30, 40]:
        c = [compute_S_range(t, 0, N) for t in v22_corr_trajs]
        h = [compute_S_range(t, 0, N) for t in v22_hall_trajs]
        w = welch(c, h)
        arrow = "h>c (H1)" if w['d'] > 0 else "c>h (reversed)"
        print(f"{N:>5} {w['p']:>10.4f} {w['d']:>+8.3f} {w['auc']:>8.3f}  {arrow}")

    # ==================================================================
    # SIGN-PATTERN CLASSIFIER (v2.2)
    # ==================================================================
    print()
    print("=" * 76)
    print("SIGN PATTERN: HL = hallucination signature?")
    print("=" * 76)
    c_e = [compute_S_range(t, 0, EARLY_END) for t in v22_corr_trajs]
    h_e = [compute_S_range(t, 0, EARLY_END) for t in v22_hall_trajs]
    c_l = [compute_S_range(t, LATE_START, 999) for t in v22_corr_trajs]
    h_l = [compute_S_range(t, LATE_START, 999) for t in v22_hall_trajs]
    all_early = c_e + h_e
    all_late = c_l + h_l
    med_early = np.median(all_early)
    med_late = np.median(all_late)

    def pattern(e, l):
        if e > med_early and l > med_late: return "HH"
        if e > med_early and l <= med_late: return "HL"
        if e <= med_early and l > med_late: return "LH"
        return "LL"

    from collections import Counter
    c_pats = Counter(pattern(e, l) for e, l in zip(c_e, c_l))
    h_pats = Counter(pattern(e, l) for e, l in zip(h_e, h_l))

    print(f"  pattern  correct  halluc   hall_rate")
    for pat in ["HH", "HL", "LH", "LL"]:
        nc = c_pats.get(pat, 0)
        nh = h_pats.get(pat, 0)
        total = nc + nh
        hr = nh / total if total > 0 else 0
        print(f"  {pat}       {nc:>4}    {nh:>4}    {hr*100:>5.1f}% (n={total})")

    # Binary test: HL pattern vs all others
    c_in_HL = sum(1 for e, l in zip(c_e, c_l) if pattern(e, l) == "HL")
    h_in_HL = sum(1 for e, l in zip(h_e, h_l) if pattern(e, l) == "HL")
    c_not_HL = len(c_e) - c_in_HL
    h_not_HL = len(h_e) - h_in_HL
    # chi-square
    from scipy.stats import chi2_contingency
    table = [[c_in_HL, h_in_HL], [c_not_HL, h_not_HL]]
    chi2, p_chi2, dof, expected = chi2_contingency(table)
    odds_ratio = (h_in_HL * c_not_HL) / max(1, c_in_HL * h_not_HL)
    print()
    print(f"  HL vs not-HL: chi2={chi2:.3f}, p={p_chi2:.4f}, odds_ratio={odds_ratio:.3f}")
    hl_halluc_rate = h_in_HL / (c_in_HL + h_in_HL) if (c_in_HL + h_in_HL) > 0 else 0
    print(f"  In HL: {hl_halluc_rate*100:.1f}% are hallucinations")
    print(f"  Outside HL: {h_not_HL/(c_not_HL+h_not_HL)*100:.1f}% are hallucinations")


if __name__ == '__main__':
    main()
