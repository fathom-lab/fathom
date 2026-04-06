"""
analyze_S_power.py — statistical power analysis for the primary test
=====================================================================
For each effect size d ∈ {0.2, 0.3, ..., 1.0}, compute the required
sample size for 80% power at alpha=0.05 (two-sided unpaired t-test).
Also computes the actual detectable effect size given our observed n.

Useful for understanding what effect sizes we can/can't claim, and
for planning extension sample sizes if primary test is underpowered.

Usage:
    python analyze_S_power.py [results_file.json]
"""

import json
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import classify_generation, compute_S, MIN_GEN_TOKENS


def required_n_per_class(d, alpha=0.05, power=0.80):
    """Required sample size per class for two-sample unpaired t-test."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / d) ** 2
    return int(np.ceil(n))


def detectable_d(n_per_class, alpha=0.05, power=0.80):
    """Smallest d detectable given n per class."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) * np.sqrt(2 / n_per_class)


def compute_post_hoc_power(d_observed, n_per_class, alpha=0.05):
    """Power achieved given observed d and n (approximate)."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    se = np.sqrt(2 / n_per_class)
    z_power = abs(d_observed) / se - z_alpha
    return float(stats.norm.cdf(z_power))


def analyze(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 72)
    print("STATISTICAL POWER ANALYSIS")
    print("=" * 72)
    print(f"Data: {results_path}")
    print(f"N items: {len(data['results'])}")
    print()

    # Observed sample sizes
    S_correct, S_halluc = [], []
    for r in data['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS:
            continue
        label, _ = classify_generation(
            r['generation']['text'], r['correct_answers'],
            r['incorrect_answers'], r.get('best_answer', '')
        )
        S = compute_S(r['generation']['c_delta_trajectory'])
        if label == 'correct':
            S_correct.append(S)
        elif label == 'hallucinated':
            S_halluc.append(S)

    n_c, n_h = len(S_correct), len(S_halluc)
    n_per_class = min(n_c, n_h)

    print(f"Observed labels: {n_c} correct, {n_h} hallucinated")
    print(f"Effective n per class: {n_per_class}")
    print()

    if n_per_class >= 2:
        # Observed effect size
        c = np.array(S_correct); h = np.array(S_halluc)
        pooled = np.sqrt((c.var(ddof=1) + h.var(ddof=1)) / 2)
        d_obs = (h.mean() - c.mean()) / pooled if pooled > 0 else 0.0
        _, p_obs = stats.ttest_ind(c, h, equal_var=False)
        print(f"Observed |d|: {abs(d_obs):.3f}")
        print(f"Observed p:   {p_obs:.4f}")
        print()

    # Required n per class for range of effect sizes
    print("-" * 72)
    print("REQUIRED N PER CLASS (80% power, alpha=0.05, unpaired)")
    print("-" * 72)
    print(f"{'d':>6} {'required n':>12} {'total':>8}")
    for d in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        req = required_n_per_class(d)
        print(f"{d:>6.1f} {req:>12d} {req*2:>8d}")
    print()

    # Detectable d for various n
    print("-" * 72)
    print("DETECTABLE EFFECT SIZE (80% power, alpha=0.05)")
    print("-" * 72)
    print(f"{'n per class':>12} {'detectable d':>14}")
    for n in [30, 50, 80, 100, 150, 200]:
        d = detectable_d(n)
        print(f"{n:>12d} {d:>14.3f}")
    print()

    # Post-hoc power
    if n_per_class >= 2:
        print("-" * 72)
        print("POST-HOC POWER at observed effect")
        print("-" * 72)
        power = compute_post_hoc_power(d_obs, n_per_class)
        print(f"  Observed d={abs(d_obs):.3f}, n_per_class={n_per_class}")
        print(f"  Achieved power: {power:.3f}")
        if power < 0.80:
            d_gate = 0.40  # pre-registered gate
            req_for_gate = required_n_per_class(d_gate)
            print(f"  Pre-reg gate d=0.40 requires n_per_class >= {req_for_gate}")
            print(f"  Current power to detect d={d_gate}: "
                  f"{compute_post_hoc_power(d_gate, n_per_class):.3f}")
        print()

    # Extension recommendation
    print("-" * 72)
    print("EXTENSION RECOMMENDATION")
    print("-" * 72)
    if n_per_class >= 2:
        if abs(d_obs) > 0.2:
            # If there's any trend, recommend target n
            target_n = required_n_per_class(max(0.3, abs(d_obs) * 0.7))  # conservative
            if target_n > n_per_class:
                # Estimate TruthfulQA items needed (assume ~35% usable rate)
                usable_rate = (n_c + n_h) / len(data['results']) if data['results'] else 0.35
                items_needed = int(target_n * 2 / max(0.1, usable_rate))
                print(f"  To reach n_per_class={target_n} at observed usable rate "
                      f"({usable_rate*100:.1f}%),")
                print(f"  need approximately {items_needed} total TruthfulQA items.")
                print(f"  Current: {len(data['results'])} items -> "
                      f"{items_needed - len(data['results'])} more needed.")
            else:
                print(f"  Current n_per_class={n_per_class} sufficient for observed effect.")
        else:
            print(f"  Observed effect too small (|d|<0.2) to justify extension.")


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(p):
        ckpt = os.path.join('truthfulqa_results', 'S_truthfulqa_checkpoint.json')
        if os.path.exists(ckpt):
            p = ckpt
        else:
            print(f"ERROR: {p} not found")
            sys.exit(1)
    analyze(p)
