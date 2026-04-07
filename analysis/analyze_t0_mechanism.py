"""
analyze_t0_mechanism.py — WHY does token 0 predict hallucination?
======================================================================
The first generated token is AFTER the prompt, so C_delta at t=0 is
computed from the residual stream over the PROMPT + "Q: ... A: " context.

Hypothesis: C_delta at t=0 reflects how COMMITTED the model is to its
answer-attractor BEFORE generating any tokens. If the model locks onto
a confident attractor that happens to be wrong, we see high t=0 C_delta.

Tests:
 1. Does t=0 C_delta correlate with the FIRST CONTENT token (skipping
    whitespace/punctuation)? If yes, effect is "first word" not "first token".
 2. Is there signal in tokens BEFORE generation (the prompt final
    token's C_delta)? Would need different data structure.
 3. Does t=0 correlate with question length, question type, category?
 4. Does t=0 correlate with specific answer patterns?

Usage:
    python analyze_t0_mechanism.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS


def load_v22_full():
    """Load with extra fields: question, category, tokens."""
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
            'text': r['generation']['text'],
            'question': r['question'],
            'category': r['category'],
            'best_answer': r.get('best_answer', ''),
        })
    return items


def main():
    items = load_v22_full()
    print("=" * 72)
    print("TOKEN-0 MECHANISM INVESTIGATION")
    print("=" * 72)
    print(f"n={len(items)} items")
    print()

    # 1. What IS token 0 for each item?
    print("-" * 72)
    print("WHAT IS TOKEN 0?")
    print("-" * 72)
    t0_tokens = Counter()
    for item in items:
        if item['tokens']:
            t0_tokens[repr(item['tokens'][0])] += 1
    print("Top 15 token-0 strings:")
    for tok, n in t0_tokens.most_common(15):
        print(f"  {n:>4}  {tok}")
    print()

    # Almost always ' ' (leading space) — Gemma tokenizer behavior
    # So t=0 is essentially the initial "space" token that precedes the answer

    # 2. First CONTENT token analysis (skip whitespace/punct/markdown)
    print("-" * 72)
    print("FIRST 'CONTENT' TOKEN ANALYSIS")
    print("-" * 72)
    def is_content(tok):
        t = tok.strip()
        if not t: return False
        if all(c in '*#_`-.,!?;:()"\'—[]\n\r\t ' for c in t): return False
        return True

    first_content_c = []
    first_content_h = []
    for item in items:
        for i, tok in enumerate(item['tokens']):
            if is_content(tok) and i < len(item['traj']):
                val = item['traj'][i]
                if item['label'] == 'correct':
                    first_content_c.append(val)
                else:
                    first_content_h.append(val)
                break

    if len(first_content_c) >= 2 and len(first_content_h) >= 2:
        c_arr = np.array(first_content_c); h_arr = np.array(first_content_h)
        _, p = stats.ttest_ind(c_arr, h_arr, equal_var=False)
        pool = np.sqrt((c_arr.var(ddof=1) + h_arr.var(ddof=1))/2)
        d = (h_arr.mean() - c_arr.mean())/pool if pool>0 else 0
        auc = roc_auc_score([0]*len(c_arr)+[1]*len(h_arr), list(c_arr)+list(h_arr))
        print(f"  first-content-token C_delta:")
        print(f"    correct       = {c_arr.mean():+.6f} (n={len(c_arr)})")
        print(f"    hallucinated  = {h_arr.mean():+.6f} (n={len(h_arr)})")
        print(f"    p={p:.4f}, d={d:+.3f}, AUC={auc:.3f}")
    print()

    # 3. Comparison: t=0 vs first content token
    print("-" * 72)
    print("FIRST-CONTENT POSITION vs TOKEN 0")
    print("-" * 72)
    first_content_pos_c, first_content_pos_h = [], []
    for item in items:
        for i, tok in enumerate(item['tokens']):
            if is_content(tok):
                if item['label'] == 'correct':
                    first_content_pos_c.append(i)
                else:
                    first_content_pos_h.append(i)
                break
    print(f"  correct first-content-pos mean: {np.mean(first_content_pos_c):.2f}")
    print(f"  halluc  first-content-pos mean: {np.mean(first_content_pos_h):.2f}")
    _, p_pos = stats.ttest_ind(first_content_pos_c, first_content_pos_h, equal_var=False)
    print(f"  position difference p={p_pos:.4f}")
    print()

    # 4. Correlation with question length
    print("-" * 72)
    print("CORRELATES OF TOKEN-0 C_delta")
    print("-" * 72)
    t0_vals = [item['traj'][0] for item in items if item['traj']]
    q_lens = [len(item['question']) for item in items if item['traj']]
    q_word_counts = [len(item['question'].split()) for item in items if item['traj']]
    r_len, p_len = stats.pearsonr(t0_vals, q_lens)
    r_wc, p_wc = stats.pearsonr(t0_vals, q_word_counts)
    print(f"  r(t=0 C_delta, question_length_chars) = {r_len:+.3f} (p={p_len:.4f})")
    print(f"  r(t=0 C_delta, question_word_count)   = {r_wc:+.3f} (p={p_wc:.4f})")
    print()

    # 5. Per-category t=0 discrimination
    print("-" * 72)
    print("TOKEN-0 BY TRUTHFULQA CATEGORY (n>=4)")
    print("-" * 72)
    cats = Counter(item['category'] for item in items)
    print(f"{'category':<30} {'n':>4} {'hall_rate':>9} {'c t0 mean':>11} {'h t0 mean':>11} {'d':>7}")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        if n < 4: continue
        cat_items = [i for i in items if i['category'] == cat]
        c_vals = [i['traj'][0] for i in cat_items if i['label']=='correct' and i['traj']]
        h_vals = [i['traj'][0] for i in cat_items if i['label']=='hallucinated' and i['traj']]
        hr = len(h_vals) / len(cat_items) * 100
        if len(c_vals) >= 2 and len(h_vals) >= 2:
            pool = np.sqrt((np.var(c_vals,ddof=1)+np.var(h_vals,ddof=1))/2)
            d_cat = (np.mean(h_vals)-np.mean(c_vals))/pool if pool>0 else 0
            print(f"{cat[:30]:<30} {n:>4} {hr:>8.1f}% "
                  f"{np.mean(c_vals):>+11.6f} {np.mean(h_vals):>+11.6f} {d_cat:>+7.3f}")
        else:
            print(f"{cat[:30]:<30} {n:>4} {hr:>8.1f}% (insufficient paired)")
    print()

    # 6. Distribution visualization data
    print("-" * 72)
    print("T=0 DISTRIBUTION STATS")
    print("-" * 72)
    c_vals = np.array([i['traj'][0] for i in items if i['label']=='correct' and i['traj']])
    h_vals = np.array([i['traj'][0] for i in items if i['label']=='hallucinated' and i['traj']])
    print(f"  correct t=0: median={np.median(c_vals):+.6f} "
          f"p25={np.percentile(c_vals,25):+.6f} p75={np.percentile(c_vals,75):+.6f}")
    print(f"  halluc  t=0: median={np.median(h_vals):+.6f} "
          f"p25={np.percentile(h_vals,25):+.6f} p75={np.percentile(h_vals,75):+.6f}")
    print(f"  overlap coefficient: ", end="")
    # Compute overlap coefficient
    all_min = min(c_vals.min(), h_vals.min())
    all_max = max(c_vals.max(), h_vals.max())
    bins = np.linspace(all_min, all_max, 50)
    c_hist, _ = np.histogram(c_vals, bins=bins, density=True)
    h_hist, _ = np.histogram(h_vals, bins=bins, density=True)
    overlap = np.sum(np.minimum(c_hist, h_hist)) * (bins[1] - bins[0])
    print(f"{overlap:.3f}")


if __name__ == '__main__':
    main()
