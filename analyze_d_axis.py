# -*- coding: utf-8 -*-
"""
analyze_d_axis.py -- D-Axis (Cognitive Honesty) Analysis
========================================================

The fifth axis: D measures whether the model is saying what it's thinking.

D = cosine(residual_at_final_layer, unembedding[chosen_token])
  High D = honest (internal state aligns with expression)
  Low D = divergent (internal state diverges from expression)

Hallucination = model is wrong (doesn't know better)
Deception = model knows it's wrong (D diverges)

This script analyzes D-axis values from oversight results and tests
whether D discriminates between output quality categories.

Usage:
  python analyze_d_axis.py oversight_results/hallucination_eval_n30.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def extract_d_values(results: list) -> dict:
    """
    Extract D-axis trajectories from oversight results.
    Returns per-item D statistics grouped by label.
    """
    by_label = defaultdict(list)

    for r in results:
        # Get label from unguarded (ground truth behavior)
        label = r.get("unguarded", {}).get("label", "unknown")

        # Get D trajectory from guarded cognitive states
        # (v2+ results have cognitive_states with d_honesty)
        states = r.get("guarded", {}).get("cognitive_states", [])
        if not states:
            # Try to get from the raw result structure
            continue

        d_vals = []
        for s in states:
            d = s.get("d_honesty")
            if d is not None:
                d_vals.append(d)

        if d_vals:
            by_label[label].append({
                "question": r.get("question", "")[:50],
                "d_mean": float(np.mean(d_vals)),
                "d_min": float(np.min(d_vals)),
                "d_max": float(np.max(d_vals)),
                "d_std": float(np.std(d_vals)),
                "d_early": float(np.mean(d_vals[:7])) if len(d_vals) >= 7 else float(np.mean(d_vals)),
                "d_late": float(np.mean(d_vals[7:])) if len(d_vals) > 7 else 0,
                "d_trajectory": d_vals,
                "n_tokens": len(d_vals),
            })

    return dict(by_label)


def analyze_d_discrimination(by_label: dict) -> dict:
    """
    Does D discriminate between output quality categories?
    The key test: do hallucinated outputs have different D than correct ones?
    """
    results = {}

    # Pairwise comparisons
    labels = list(by_label.keys())
    for i, la in enumerate(labels):
        for lb in labels[i+1:]:
            a_means = [item["d_mean"] for item in by_label[la]]
            b_means = [item["d_mean"] for item in by_label[lb]]

            if len(a_means) < 2 or len(b_means) < 2:
                continue

            if HAS_SCIPY:
                t, p = stats.ttest_ind(a_means, b_means, equal_var=False)
                pooled_std = np.sqrt((np.var(a_means, ddof=1) + np.var(b_means, ddof=1)) / 2)
                d = (np.mean(b_means) - np.mean(a_means)) / pooled_std if pooled_std > 0 else 0
            else:
                t, p, d = 0, 1, 0

            results[f"{la}_vs_{lb}"] = {
                f"{la}_mean": float(np.mean(a_means)),
                f"{la}_n": len(a_means),
                f"{lb}_mean": float(np.mean(b_means)),
                f"{lb}_n": len(b_means),
                "cohen_d": float(d),
                "t_stat": float(t),
                "p_value": float(p),
                "significant": p < 0.05,
            }

    return results


def analyze_d_early_vs_late(by_label: dict) -> dict:
    """
    Does D change over the course of generation?
    Deception hypothesis: D starts high (honest) then drops (diverges)
    as the model commits to a false path.
    """
    results = {}
    for label, items in by_label.items():
        d_early = [item["d_early"] for item in items if item["d_early"]]
        d_late = [item["d_late"] for item in items if item["d_late"]]

        if d_early and d_late:
            results[label] = {
                "d_early_mean": float(np.mean(d_early)),
                "d_late_mean": float(np.mean(d_late)),
                "d_shift": float(np.mean(d_late)) - float(np.mean(d_early)),
                "n": len(items),
            }

    return results


def print_report(results_path: str):
    """Full D-axis analysis report."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("[!] No results")
        return

    # Check if D-axis data exists
    has_d = False
    for r in results:
        states = r.get("guarded", {}).get("cognitive_states", [])
        if states and any(s.get("d_honesty") is not None for s in states):
            has_d = True
            break

    if not has_d:
        # Try extracting from the flat result structure
        # V1 results don't have cognitive_states serialized — check if we can get D another way
        print("[!] No D-axis data in results (run v2+ with D-axis enabled)")
        print("    D-axis was added after pilot v1. Re-run with current code.")
        return

    by_label = extract_d_values(results)

    print("=" * 60)
    print("D-AXIS (COGNITIVE HONESTY) ANALYSIS")
    print("=" * 60)

    # Per-label D statistics
    print(f"\n--- D-HONESTY BY OUTPUT QUALITY ---")
    print(f"{'Label':<15} {'N':>4} {'D_mean':>8} {'D_std':>8} {'D_early':>9} {'D_late':>8} {'Shift':>8}")
    for label, items in sorted(by_label.items(), key=lambda x: -len(x[1])):
        d_means = [i["d_mean"] for i in items]
        d_early = [i["d_early"] for i in items]
        d_late = [i["d_late"] for i in items]
        print(f"{label:<15} {len(items):>4} {np.mean(d_means):>8.4f} {np.std(d_means):>8.4f} "
              f"{np.mean(d_early):>9.4f} {np.mean(d_late):>8.4f} "
              f"{np.mean(d_late) - np.mean(d_early):>+8.4f}")

    # Discrimination tests
    disc = analyze_d_discrimination(by_label)
    if disc:
        print(f"\n--- D-AXIS DISCRIMINATION ---")
        for pair, stats_d in disc.items():
            sig = "***" if stats_d["significant"] else "n.s."
            print(f"  {pair}: d={stats_d['cohen_d']:+.3f}, p={stats_d['p_value']:.4f} {sig}")

    # Early vs late D
    temporal = analyze_d_early_vs_late(by_label)
    if temporal:
        print(f"\n--- D TEMPORAL PROFILE ---")
        for label, t in temporal.items():
            direction = "drops (diverging)" if t["d_shift"] < 0 else "rises (converging)"
            print(f"  {label}: D_early={t['d_early_mean']:.4f} -> D_late={t['d_late_mean']:.4f} "
                  f"({direction}, n={t['n']})")

    # Save
    report = {
        "by_label": {k: [{"d_mean": i["d_mean"], "d_early": i["d_early"],
                          "d_late": i["d_late"], "question": i["question"]}
                         for i in v] for k, v in by_label.items()},
        "discrimination": disc,
        "temporal": temporal,
    }
    out_path = Path(results_path).parent / "d_axis_analysis.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        results_dir = Path(__file__).parent / "oversight_results"
        candidates = sorted(results_dir.glob("hallucination_eval_*.json"))
        if candidates:
            path = str(candidates[-1])
        else:
            print("Usage: python analyze_d_axis.py <results.json>")
            sys.exit(1)
    else:
        path = sys.argv[1]

    print_report(path)
