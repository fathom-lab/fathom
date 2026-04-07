"""
analyze_t0_length_control.py — does t=0 signal survive length control?
========================================================================
r(t=0 C_delta, question_length) = +0.497 is a strong confound.
Longer questions -> higher t=0 C_delta. Test whether the t=0
discrimination is just capturing a question-length confound.

Methods:
 1. Partial correlation: t=0 vs label | length
 2. Stratified analysis: split items by length quantile, test within
 3. Length-residualized t=0: regress t=0 on length, test residuals
 4. Length distribution: do hallucinated items have longer questions?

Usage:
    python analyze_t0_length_control.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS


def load_v22():
    with open("truthfulqa_results/S_truthfulqa_n200.json") as f:
        data = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a for a in audit_data['details']}
    rows = []
    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS:
            continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        if lbl in ('correct', 'correct_semantic'):
            label = 0
        elif lbl in ('hallucinated', 'hallucinated_semantic'):
            label = 1
        else:
            continue
        if not r['generation']['c_delta_trajectory']:
            continue
        rows.append({
            't0': r['generation']['c_delta_trajectory'][0],
            'q_chars': len(r['question']),
            'q_words': len(r['question'].split()),
            'label': label,
        })
    return rows


def main():
    rows = load_v22()
    n = len(rows)
    print("=" * 72)
    print("LENGTH-CONTROLLED TOKEN-0 ANALYSIS")
    print("=" * 72)
    print(f"n={n}")
    print()

    t0 = np.array([r['t0'] for r in rows])
    q_chars = np.array([r['q_chars'] for r in rows])
    q_words = np.array([r['q_words'] for r in rows])
    y = np.array([r['label'] for r in rows])

    # 1. Question length by label
    print("-" * 72)
    print("IS QUESTION LENGTH DIFFERENT BY LABEL?")
    print("-" * 72)
    ql_c = q_chars[y==0]; ql_h = q_chars[y==1]
    qw_c = q_words[y==0]; qw_h = q_words[y==1]
    _, p_chars = stats.ttest_ind(ql_c, ql_h, equal_var=False)
    _, p_words = stats.ttest_ind(qw_c, qw_h, equal_var=False)
    print(f"  question chars: corr={ql_c.mean():.1f} hall={ql_h.mean():.1f} p={p_chars:.4f}")
    print(f"  question words: corr={qw_c.mean():.1f} hall={qw_h.mean():.1f} p={p_words:.4f}")
    print()

    # 2. Partial correlation (via residualization)
    print("-" * 72)
    print("LENGTH-RESIDUALIZED TOKEN-0 C_delta")
    print("-" * 72)
    # Regress t0 on q_chars, take residuals, test residuals vs label
    reg = LinearRegression().fit(q_chars.reshape(-1,1), t0)
    t0_pred = reg.predict(q_chars.reshape(-1,1))
    t0_resid = t0 - t0_pred
    print(f"  regression: t0 = {reg.coef_[0]:+.2e} * q_chars + {reg.intercept_:+.6f}")
    print(f"  R^2 = {reg.score(q_chars.reshape(-1,1), t0):.3f}")
    print()

    rc = t0_resid[y==0]; rh = t0_resid[y==1]
    _, p_r = stats.ttest_ind(rc, rh, equal_var=False)
    pool = np.sqrt((rc.var(ddof=1) + rh.var(ddof=1))/2)
    d_r = (rh.mean() - rc.mean())/pool if pool>0 else 0
    auc_r = roc_auc_score(y, t0_resid)
    print(f"  Residualized t0: corr={rc.mean():+.6f} hall={rh.mean():+.6f}")
    print(f"  p={p_r:.4f}, d={d_r:+.3f}, AUC={auc_r:.3f}")
    print(f"  Gates: {'PASS' if (p_r<0.05 and abs(d_r)>0.4 and auc_r>0.65) else 'FAIL'}")
    print()

    # 3. Stratified analysis
    print("-" * 72)
    print("STRATIFIED BY QUESTION LENGTH (quartile splits)")
    print("-" * 72)
    q_quartiles = np.percentile(q_chars, [25, 50, 75])
    print(f"  quartile cuts: {q_quartiles}")
    print()
    for q_low, q_high in [
        (-1, q_quartiles[0]),
        (q_quartiles[0], q_quartiles[1]),
        (q_quartiles[1], q_quartiles[2]),
        (q_quartiles[2], 1e6),
    ]:
        mask = (q_chars > q_low) & (q_chars <= q_high)
        sub_t0 = t0[mask]; sub_y = y[mask]
        nc = (sub_y==0).sum(); nh = (sub_y==1).sum()
        if nc < 3 or nh < 3:
            print(f"  q_chars in ({q_low:.0f},{q_high:.0f}]: n=({nc},{nh}) — skip")
            continue
        c_vals = sub_t0[sub_y==0]; h_vals = sub_t0[sub_y==1]
        _, p_s = stats.ttest_ind(c_vals, h_vals, equal_var=False)
        pool = np.sqrt((c_vals.var(ddof=1) + h_vals.var(ddof=1))/2)
        d_s = (h_vals.mean() - c_vals.mean())/pool if pool>0 else 0
        auc_s = roc_auc_score(sub_y, sub_t0) if (nc>0 and nh>0) else None
        print(f"  q_chars in ({q_low:.0f},{q_high:.0f}]: n=({nc},{nh}) "
              f"p={p_s:.4f} d={d_s:+.3f} AUC={auc_s:.3f}")
    print()

    # 4. 2-feature classifier: t0 + q_chars
    print("-" * 72)
    print("MULTI-FEATURE: does q_chars ALONE do as well as t0?")
    print("-" * 72)
    auc_qc = roc_auc_score(y, q_chars)
    _, p_qc = stats.ttest_ind(q_chars[y==0], q_chars[y==1], equal_var=False)
    print(f"  q_chars alone: AUC={auc_qc:.3f}, p={p_qc:.4f}")
    print(f"  t0 alone:      AUC={roc_auc_score(y, t0):.3f}")
    print(f"  t0 residualized: AUC={auc_r:.3f}")
    print()

    # 5. Logistic regression: t0 + q_chars combined
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    print("-" * 72)
    print("LOGISTIC REGRESSION (5-fold CV)")
    print("-" * 72)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'))])
    for features, name in [
        ([t0], "t0 only"),
        ([q_chars], "q_chars only"),
        ([t0, q_chars], "t0 + q_chars"),
        ([t0 - t0_pred], "t0 residualized"),
    ]:
        X = np.column_stack(features)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        print(f"  {name:<20}: CV AUC = {scores.mean():.3f} +/- {scores.std():.3f}")

    # 6. Honest conclusion
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if auc_r > 0.65 and p_r < 0.05:
        print("  Token-0 signal SURVIVES length control.")
        print("  The t=0 C_delta contains discriminative information beyond q_length.")
    elif auc_r > 0.60:
        print("  Token-0 signal PARTIALLY survives length control.")
        print("  Some discriminative information remains after residualization.")
    else:
        print("  Token-0 signal does NOT survive length control.")
        print("  The t=0 effect is mostly explained by question-length confound.")


if __name__ == '__main__':
    main()
