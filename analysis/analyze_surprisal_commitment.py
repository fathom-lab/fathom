"""
analyze_surprisal_commitment.py — "did the model commit HARDER than
expected given this question?"
======================================================================
Instead of raw S_early, compute expected S_early from question semantic
embedding, then test residuals.

Theory: commitment intensity alone is confounded by question features
(length, adversariality, topic familiarity). The NOVEL signal is:
"given how this question LOOKS, did the model commit more strongly
than we'd predict?" That's hallucination = unexpected confidence.

Usage:
    python analyze_surprisal_commitment.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS
from analyze_S_biphasic import compute_S_range
from sentence_transformers import SentenceTransformer


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
            'question': r['question'],
            'label': label,
        })
    return items


def main():
    items = load_v22()
    print("=" * 72)
    print("SURPRISAL COMMITMENT TEST")
    print("=" * 72)
    print(f"n={len(items)}")
    print()

    # Embed questions
    print("Embedding questions with MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    questions = [item['question'] for item in items]
    q_embeds = model.encode(questions, batch_size=32, show_progress_bar=False)
    print(f"  Shape: {q_embeds.shape}")
    print()

    S_early = np.array([item['S_early'] for item in items])
    S_late = np.array([item['S_late'] for item in items])
    q_chars = np.array([item['q_chars'] for item in items])
    y = np.array([item['label'] for item in items])

    # ==================================================================
    # Predict S_early from question embedding, get residuals
    # ==================================================================
    print("-" * 72)
    print("EXPECTED S_early FROM QUESTION EMBEDDING (Ridge, 5-fold CV)")
    print("-" * 72)
    # Use 5-fold CV to generate out-of-fold predictions (avoids overfitting)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = Pipeline([("sc", StandardScaler()),
                      ("reg", Ridge(alpha=1.0))])
    S_early_pred = cross_val_predict(ridge, q_embeds, S_early, cv=kf)
    S_early_surprisal = S_early - S_early_pred

    r2 = 1 - np.var(S_early_surprisal) / np.var(S_early)
    print(f"  R^2 of embedding predicting S_early: {r2:.3f}")
    print(f"  (question embedding explains {r2*100:.1f}% of S_early variance)")
    print()

    # Test residual (surprisal) for discrimination
    c_surp = S_early_surprisal[y==0]
    h_surp = S_early_surprisal[y==1]
    _, p_surp = stats.ttest_ind(c_surp, h_surp, equal_var=False)
    pool = np.sqrt((c_surp.var(ddof=1) + h_surp.var(ddof=1))/2)
    d_surp = (h_surp.mean() - c_surp.mean())/pool if pool>0 else 0
    auc_surp = roc_auc_score(y, S_early_surprisal)

    print("-" * 72)
    print("SURPRISAL TEST: residual S_early discriminates hallucination?")
    print("-" * 72)
    print(f"  correct surprisal:       {c_surp.mean():+.6f}")
    print(f"  hallucinated surprisal:  {h_surp.mean():+.6f}")
    print(f"  p={p_surp:.4f}, d={d_surp:+.3f}, AUC={auc_surp:.3f}")
    print(f"  Gates: {'PASS' if (p_surp<0.05 and abs(d_surp)>0.4 and auc_surp>0.65) else 'FAIL'}")
    print()

    # ==================================================================
    # Compare to raw S_early, q_chars alone, combined
    # ==================================================================
    print("-" * 72)
    print("FEATURE COMPARISON (5-fold CV AUC, class-balanced LogReg)")
    print("-" * 72)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'))])

    feature_configs = [
        ("q_chars only (confound baseline)", q_chars.reshape(-1,1)),
        ("q_embed only (topic baseline)", q_embeds),
        ("S_early only", S_early.reshape(-1,1)),
        ("S_early SURPRISAL (residualized)", S_early_surprisal.reshape(-1,1)),
        ("S_early + q_chars", np.column_stack([S_early, q_chars])),
        ("S_early SURPRISAL + q_chars", np.column_stack([S_early_surprisal, q_chars])),
        ("S_early + q_embed", np.column_stack([S_early.reshape(-1,1), q_embeds])),
    ]

    print(f"{'features':<40} {'CV AUC':>10} {'std':>8}")
    for name, X in feature_configs:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        print(f"{name:<40} {scores.mean():>10.3f} {scores.std():>8.3f}")

    # ==================================================================
    # Check S_late surprisal too
    # ==================================================================
    print()
    print("-" * 72)
    print("S_late SURPRISAL (additional test)")
    print("-" * 72)
    S_late_pred = cross_val_predict(ridge, q_embeds, S_late, cv=kf)
    S_late_surprisal = S_late - S_late_pred
    auc_late_surp = roc_auc_score(y, S_late_surprisal)
    _, p_late_surp = stats.ttest_ind(
        S_late_surprisal[y==0], S_late_surprisal[y==1], equal_var=False)
    print(f"  S_late surprisal AUC: {auc_late_surp:.3f}, p={p_late_surp:.4f}")
    print()

    # ==================================================================
    # Bi-phasic SURPRISAL delta
    # ==================================================================
    print("-" * 72)
    print("BI-PHASIC SURPRISAL DELTA (S_early_surp - S_late_surp)")
    print("-" * 72)
    surp_delta = S_early_surprisal - S_late_surprisal
    c_sd = surp_delta[y==0]; h_sd = surp_delta[y==1]
    _, p_sd = stats.ttest_ind(c_sd, h_sd, equal_var=False)
    pool = np.sqrt((c_sd.var(ddof=1) + h_sd.var(ddof=1))/2)
    d_sd = (h_sd.mean() - c_sd.mean())/pool if pool>0 else 0
    auc_sd = roc_auc_score(y, surp_delta)
    print(f"  correct surp_delta:       {c_sd.mean():+.6f}")
    print(f"  hallucinated surp_delta:  {h_sd.mean():+.6f}")
    print(f"  p={p_sd:.4f}, d={d_sd:+.3f}, AUC={auc_sd:.3f}")
    print(f"  Gates: {'PASS' if (p_sd<0.05 and abs(d_sd)>0.4 and auc_sd>0.65) else 'FAIL'}")
    print()

    # Save
    with open("truthfulqa_results/surprisal_commitment_analysis.json", 'w') as f:
        json.dump({
            'n': len(items),
            'r2_qembed_predicts_Searly': float(r2),
            'surprisal_AUC': float(auc_surp),
            'surprisal_p': float(p_surp),
            'surprisal_d': float(d_surp),
            'surprisal_delta_AUC': float(auc_sd),
            'surprisal_delta_p': float(p_sd),
        }, f, indent=2)
    print("Saved: truthfulqa_results/surprisal_commitment_analysis.json")


if __name__ == '__main__':
    main()
