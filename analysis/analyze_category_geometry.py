"""
analyze_category_geometry.py — TruthfulQA category × geometric signature
===========================================================================
For each TruthfulQA category (Misconceptions, Law, Health, Fiction, etc.)
compute mean trajectory features. Test whether categories separate in
geometric space.

Novel claim: model cognition has category-specific commitment patterns.
"Paranormal" should be structurally different from "Law" in how the
model routes and commits.

Also: hallucination rate per category.

Usage:
    python analyze_category_geometry.py [results_file.json]
"""

import json
import sys
import os
import numpy as np
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import (
    classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY
)


def analyze(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 72)
    print("TruthfulQA CATEGORY × GEOMETRY signature analysis")
    print("=" * 72)
    print(f"Data:       {results_path}")
    print(f"N items:    {len(data['results'])}")
    print()

    # Per-category aggregation
    cat_data = defaultdict(list)
    for r in data['results']:
        gen = r['generation']['text']
        n_tokens = r['generation']['total_tokens']
        traj = r['generation']['c_delta_trajectory']
        if n_tokens < MIN_GEN_TOKENS:
            continue
        label, _ = classify_generation(
            gen, r['correct_answers'], r['incorrect_answers'], r.get('best_answer', '')
        )
        S = compute_S(traj)
        max_cd = max(traj) if traj else 0.0
        mean_cd = float(np.mean(traj)) if traj else 0.0
        category = r.get('category', 'Unknown')
        cat_data[category].append({
            'label': label,
            'S': S,
            'max_C_delta': max_cd,
            'mean_C_delta': mean_cd,
            'n_tokens': n_tokens,
        })

    print(f"Found {len(cat_data)} categories")
    print()

    # Per-category summary
    print("-" * 85)
    print(f"{'CATEGORY':<30} {'N':>4} {'%halluc':>8} {'S_mean':>10} {'max_CD':>10} {'mean_CD':>10}")
    print("-" * 85)

    cat_rows = []
    for cat, items in sorted(cat_data.items()):
        n = len(items)
        if n < 2:
            continue
        halluc_rate = sum(1 for i in items if i['label'] == 'hallucinated') / n
        s_mean = np.mean([i['S'] for i in items])
        max_cd = np.mean([i['max_C_delta'] for i in items])
        mean_cd = np.mean([i['mean_C_delta'] for i in items])
        cat_rows.append({
            'category': cat, 'n': n,
            'halluc_rate': halluc_rate,
            'S_mean': s_mean,
            'max_C_delta_mean': max_cd,
            'mean_C_delta_mean': mean_cd,
        })
        print(f"{cat[:30]:<30} {n:>4} {halluc_rate*100:>7.1f}% "
              f"{s_mean:>10.6f} {max_cd:>10.6f} {mean_cd:>10.6f}")
    print()

    # ANOVA: is S different across categories?
    cat_S = [[i['S'] for i in cat_data[c]] for c in cat_data if len(cat_data[c]) >= 3]
    if len(cat_S) >= 2:
        f_stat, p_anova = stats.f_oneway(*cat_S)
        print(f"One-way ANOVA on S across {len(cat_S)} categories (n>=3 each):")
        print(f"  F={f_stat:.3f}, p={p_anova:.4f}")
        if p_anova < 0.05:
            print("  => Categories DO show different S distributions")
        else:
            print("  => No category effect detected")
    print()

    # Per-category hallucination rate variance
    if len(cat_rows) >= 3:
        rates = [r['halluc_rate'] for r in cat_rows]
        print(f"Hallucination rate across {len(cat_rows)} categories:")
        print(f"  mean={np.mean(rates)*100:.1f}%  range=[{min(rates)*100:.1f}%, {max(rates)*100:.1f}%]")
        # Categories sorted by halluc rate
        print("\n  Highest hallucination:")
        for r in sorted(cat_rows, key=lambda x: -x['halluc_rate'])[:3]:
            print(f"    {r['category'][:30]:<30} {r['halluc_rate']*100:.1f}% (n={r['n']})")
        print("\n  Lowest hallucination:")
        for r in sorted(cat_rows, key=lambda x: x['halluc_rate'])[:3]:
            print(f"    {r['category'][:30]:<30} {r['halluc_rate']*100:.1f}% (n={r['n']})")

    # Save
    out_path = results_path.replace('.json', '_categories.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': results_path,
            'category_stats': cat_rows,
        }, f, indent=2)
    print(f"\n  saved: {out_path}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(path):
        pilot = path.replace('n200.json', 'pilot.json')
        if os.path.exists(pilot):
            path = pilot
        else:
            print(f"ERROR: {path} not found")
            sys.exit(1)
    analyze(path)
