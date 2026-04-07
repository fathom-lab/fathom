"""
analyze_spike_anatomy.py — which tokens produce the Phase 1 spike?
======================================================================
For each trajectory, find position of max C_delta in Phase 1 [0,7).
Ask: which GENERATED TOKEN is at that position?
  - Is it always token 0 (first answer token)?
  - Is it punctuation, markdown, or content?
  - Does token identity correlate with label?

Goal: show the Phase 1 signal has specific token-position semantics,
not just a random early-trajectory artifact.

Usage:
    python analyze_spike_anatomy.py
"""

import json
import os
import sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import MIN_GEN_TOKENS


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
        idx = r['hf_index']
        lbl = audit.get(idx, {}).get('v22_label')
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
            'best_answer': r.get('best_answer', ''),
        })
    return items


def categorize_token(tok):
    """Classify a token string."""
    t = tok.strip()
    if not t:
        return 'whitespace'
    if t in {'.', ',', '!', '?', ':', ';', '(', ')', '-', '"', "'", '—', '[', ']'}:
        return 'punctuation'
    if t in {'*', '**', '##', '#', '###', '####', '_'}:
        return 'markdown'
    if t.lower() in {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'on', 'at', 'by'}:
        return 'function_word'
    if t[0].isdigit():
        return 'number'
    if t[0].isupper():
        return 'content_upper'
    return 'content_lower'


def analyze():
    items = load_v22()
    print("=" * 72)
    print("PHASE 1 SPIKE ANATOMY (which tokens drive the signal?)")
    print("=" * 72)
    print(f"Loaded: {sum(1 for i in items if i['label']=='correct')} correct, "
          f"{sum(1 for i in items if i['label']=='hallucinated')} hallucinated")
    print()

    # For each trajectory, find the argmax position in [0, 7) and what token it was
    print("-" * 72)
    print("ARGMAX POSITION WITHIN PHASE 1 (tokens 0-6)")
    print("-" * 72)
    corr_peak_pos = Counter()
    hall_peak_pos = Counter()
    for item in items:
        traj = item['traj'][:7]
        if not traj: continue
        peak = int(np.argmax(traj))
        if item['label'] == 'correct':
            corr_peak_pos[peak] += 1
        else:
            hall_peak_pos[peak] += 1

    print(f"{'pos':>5} {'correct':>10} {'halluc':>10} {'halluc_rate':>12}")
    for pos in sorted(set(list(corr_peak_pos.keys()) + list(hall_peak_pos.keys()))):
        nc = corr_peak_pos.get(pos, 0)
        nh = hall_peak_pos.get(pos, 0)
        total = nc + nh
        hr = nh / total * 100 if total > 0 else 0
        print(f"{pos:>5} {nc:>10} {nh:>10} {hr:>11.1f}%")
    print()

    # Token identity at argmax position
    print("-" * 72)
    print("TOKEN CATEGORY AT PHASE 1 ARGMAX")
    print("-" * 72)
    corr_cats, hall_cats = Counter(), Counter()
    for item in items:
        traj = item['traj'][:7]
        tokens = item['tokens'][:7]
        if not traj or len(tokens) < len(traj):
            continue
        peak = int(np.argmax(traj))
        if peak >= len(tokens): continue
        cat = categorize_token(tokens[peak])
        if item['label'] == 'correct':
            corr_cats[cat] += 1
        else:
            hall_cats[cat] += 1

    all_cats = set(list(corr_cats.keys()) + list(hall_cats.keys()))
    print(f"{'category':>18} {'correct':>10} {'halluc':>10} {'halluc_rate':>12}")
    for cat in sorted(all_cats):
        nc = corr_cats.get(cat, 0)
        nh = hall_cats.get(cat, 0)
        total = nc + nh
        hr = nh / total * 100 if total > 0 else 0
        print(f"{cat:>18} {nc:>10} {nh:>10} {hr:>11.1f}%")
    print()

    # Inspect what tokens ARE these? Show examples
    print("-" * 72)
    print("EXAMPLE TOKENS AT PHASE 1 PEAK, hallucinated items (first 10)")
    print("-" * 72)
    hall_examples = []
    for item in items:
        if item['label'] != 'hallucinated': continue
        traj = item['traj'][:7]
        tokens = item['tokens'][:7]
        if not traj or len(tokens) < len(traj): continue
        peak = int(np.argmax(traj))
        if peak >= len(tokens): continue
        hall_examples.append({
            'peak_pos': peak,
            'peak_tok': tokens[peak],
            'peak_cd': traj[peak],
            'q': item['question'][:50],
        })
    for ex in hall_examples[:10]:
        print(f"  [pos {ex['peak_pos']}] tok={ex['peak_tok']!r:<20} "
              f"C_delta={ex['peak_cd']:+.4f} Q: {ex['q']}")
    print()

    print("-" * 72)
    print("EXAMPLE TOKENS AT PHASE 1 PEAK, correct items (first 10)")
    print("-" * 72)
    corr_examples = []
    for item in items:
        if item['label'] != 'correct': continue
        traj = item['traj'][:7]
        tokens = item['tokens'][:7]
        if not traj or len(tokens) < len(traj): continue
        peak = int(np.argmax(traj))
        if peak >= len(tokens): continue
        corr_examples.append({
            'peak_pos': peak,
            'peak_tok': tokens[peak],
            'peak_cd': traj[peak],
            'q': item['question'][:50],
        })
    for ex in corr_examples[:10]:
        print(f"  [pos {ex['peak_pos']}] tok={ex['peak_tok']!r:<20} "
              f"C_delta={ex['peak_cd']:+.4f} Q: {ex['q']}")
    print()

    # Does first-token-specifically discriminate?
    print("-" * 72)
    print("IS THE SIGNAL ON TOKEN 0 SPECIFICALLY?")
    print("-" * 72)
    corr_t0 = [item['traj'][0] for item in items if item['label']=='correct' and item['traj']]
    hall_t0 = [item['traj'][0] for item in items if item['label']=='hallucinated' and item['traj']]
    from scipy import stats as st
    from sklearn.metrics import roc_auc_score
    _, p_t0 = st.ttest_ind(corr_t0, hall_t0, equal_var=False)
    pooled = np.sqrt((np.var(corr_t0, ddof=1) + np.var(hall_t0, ddof=1))/2)
    d_t0 = (np.mean(hall_t0) - np.mean(corr_t0))/pooled if pooled>0 else 0
    auc_t0 = roc_auc_score([0]*len(corr_t0)+[1]*len(hall_t0), corr_t0+hall_t0)
    print(f"  token 0 C_delta: corr={np.mean(corr_t0):+.6f} hall={np.mean(hall_t0):+.6f}")
    print(f"  p={p_t0:.4f}, d={d_t0:+.3f}, AUC={auc_t0:.3f}")
    print()

    # Per-token discrimination
    print("-" * 72)
    print("PER-TOKEN C_delta DISCRIMINATION (tokens 0-9)")
    print("-" * 72)
    print(f"{'token':>6} {'corr_mean':>12} {'hall_mean':>12} {'p':>8} {'d':>7} {'AUC':>7}")
    for t in range(10):
        c_vals = [item['traj'][t] for item in items
                  if item['label']=='correct' and len(item['traj'])>t]
        h_vals = [item['traj'][t] for item in items
                  if item['label']=='hallucinated' and len(item['traj'])>t]
        if len(c_vals) < 2 or len(h_vals) < 2: continue
        _, p = st.ttest_ind(c_vals, h_vals, equal_var=False)
        pooled = np.sqrt((np.var(c_vals, ddof=1) + np.var(h_vals, ddof=1))/2)
        d = (np.mean(h_vals) - np.mean(c_vals))/pooled if pooled>0 else 0
        auc = roc_auc_score([0]*len(c_vals)+[1]*len(h_vals), c_vals+h_vals)
        marker = "*" if p < 0.05 else " "
        print(f"{t:>6} {np.mean(c_vals):>+12.6f} {np.mean(h_vals):>+12.6f} "
              f"{p:>8.4f} {d:>+7.3f} {auc:>7.3f}{marker}")


if __name__ == '__main__':
    analyze()
