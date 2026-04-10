# -*- coding: utf-8 -*-
"""
sweep_guardian_thresholds.py — Find optimal trigger thresholds

Loads existing guardian_confab_results.json and sweeps threshold
combinations to maximize trigger accuracy against ground-truth
should_abstain labels.

No GPU needed — pure post-hoc analysis.
"""

import json
import sys
from pathlib import Path
from itertools import product

import numpy as np

RESULTS_PATH = Path(__file__).parent / "guardian_confab_results.json"


def load_data():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data


def evaluate_thresholds(data, gini_thresh, s_thresh, health_thresh, logic="or"):
    """
    Apply thresholds to each prompt and compute metrics.

    logic='or'  -> trigger if ANY condition fires
    logic='and' -> trigger if ALL conditions fire
    """
    tp = fp = tn = fn = 0

    for r in data:
        should_trigger = r["should_abstain"]

        gini_fire = r["first_mean_gini"] > gini_thresh
        s_fire = r["s_early_first"] > s_thresh
        health_fire = r["health_score"] < health_thresh

        if logic == "or":
            triggered = gini_fire or s_fire or health_fire
        elif logic == "and":
            triggered = gini_fire and s_fire and health_fire
        elif logic == "gini_only":
            triggered = gini_fire
        elif logic == "s_only":
            triggered = s_fire
        elif logic == "health_only":
            triggered = health_fire
        else:
            triggered = gini_fire or health_fire

        if triggered and should_trigger:
            tp += 1
        elif triggered and not should_trigger:
            fp += 1
        elif not triggered and not should_trigger:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def sweep(data):
    """Grid search over threshold combinations."""
    # Gini range based on observed data (0.384 - 0.517)
    gini_range = np.arange(0.38, 0.54, 0.01)
    # S_early range (0.0014 - 0.0043)
    s_range = np.arange(0.0015, 0.0050, 0.0005)
    # Health range (+0.931 to +1.299)
    health_range = np.arange(0.90, 1.30, 0.02)

    best_overall = None
    best_gini = None
    best_health = None
    best_combined = None

    # Single-signal sweeps
    print("=" * 70)
    print("SINGLE-SIGNAL SWEEPS")
    print("=" * 70)

    # 1. Gini only
    print("\n[1] Gini threshold sweep:")
    best_g = {"f1": -1}
    for g in gini_range:
        m = evaluate_thresholds(data, g, 999, -999, logic="gini_only")
        if m["f1"] > best_g["f1"] or (
            m["f1"] == best_g["f1"] and m["fpr"] < best_g.get("fpr", 999)
        ):
            best_g = {**m, "gini_thresh": g}
    print(f"  Best: gini>{best_g['gini_thresh']:.3f} -> "
          f"acc={best_g['accuracy']:.0%} "
          f"f1={best_g['f1']:.2f} "
          f"precision={best_g['precision']:.0%} "
          f"recall={best_g['recall']:.0%} "
          f"fpr={best_g['fpr']:.0%}")

    # 2. S_early only
    print("\n[2] S_early threshold sweep:")
    best_s = {"f1": -1}
    for s in s_range:
        m = evaluate_thresholds(data, 999, s, -999, logic="s_only")
        if m["f1"] > best_s["f1"]:
            best_s = {**m, "s_thresh": s}
    print(f"  Best: s_early>{best_s.get('s_thresh', 0):.4f} -> "
          f"acc={best_s['accuracy']:.0%} "
          f"f1={best_s['f1']:.2f} "
          f"precision={best_s['precision']:.0%} "
          f"recall={best_s['recall']:.0%} "
          f"fpr={best_s['fpr']:.0%}")

    # 3. Health only
    print("\n[3] Health threshold sweep:")
    best_h = {"f1": -1}
    for h in health_range:
        m = evaluate_thresholds(data, 999, 999, h, logic="health_only")
        if m["f1"] > best_h["f1"] or (
            m["f1"] == best_h["f1"] and m["fpr"] < best_h.get("fpr", 999)
        ):
            best_h = {**m, "health_thresh": h}
    print(f"  Best: health<{best_h['health_thresh']:.3f} -> "
          f"acc={best_h['accuracy']:.0%} "
          f"f1={best_h['f1']:.2f} "
          f"precision={best_h['precision']:.0%} "
          f"recall={best_h['recall']:.0%} "
          f"fpr={best_h['fpr']:.0%}")

    # Combined OR sweep
    print("\n" + "=" * 70)
    print("COMBINED (Gini OR S_early OR Health)")
    print("=" * 70)
    best_combined = {"f1": -1}
    n_combos = len(gini_range) * len(s_range) * len(health_range)
    print(f"Testing {n_combos} combinations...")

    for g, s, h in product(gini_range, s_range, health_range):
        m = evaluate_thresholds(data, g, s, h, logic="or")
        if m["f1"] > best_combined["f1"] or (
            m["f1"] == best_combined["f1"] and
            m["fpr"] < best_combined.get("fpr", 999)
        ):
            best_combined = {
                **m,
                "gini_thresh": g,
                "s_thresh": s,
                "health_thresh": h,
            }

    print(f"\n  Best combined: "
          f"gini>{best_combined['gini_thresh']:.3f} OR "
          f"s_early>{best_combined['s_thresh']:.4f} OR "
          f"health<{best_combined['health_thresh']:.3f}")
    print(f"  Accuracy:  {best_combined['accuracy']:.0%}")
    print(f"  F1:        {best_combined['f1']:.2f}")
    print(f"  Precision: {best_combined['precision']:.0%}")
    print(f"  Recall:    {best_combined['recall']:.0%}")
    print(f"  FPR:       {best_combined['fpr']:.0%}")
    print(f"  TP/FP/TN/FN: "
          f"{best_combined['tp']}/{best_combined['fp']}/"
          f"{best_combined['tn']}/{best_combined['fn']}")

    # Current (v1) thresholds for comparison
    print("\n" + "=" * 70)
    print("CURRENT (v1) THRESHOLDS")
    print("=" * 70)
    current = evaluate_thresholds(data, 0.45, 0.005, -999, logic="or")
    print(f"  gini>0.45 OR s_early>0.005")
    print(f"  Accuracy:  {current['accuracy']:.0%}")
    print(f"  F1:        {current['f1']:.2f}")
    print(f"  Precision: {current['precision']:.0%}")
    print(f"  Recall:    {current['recall']:.0%}")
    print(f"  FPR:       {current['fpr']:.0%}")
    print(f"  TP/FP/TN/FN: "
          f"{current['tp']}/{current['fp']}/"
          f"{current['tn']}/{current['fn']}")

    return {
        "current": current,
        "best_gini": best_g,
        "best_s_early": best_s,
        "best_health": best_h,
        "best_combined": best_combined,
    }


def print_per_prompt(data, thresh):
    """Show how each prompt classifies under the new thresholds."""
    print("\n" + "=" * 70)
    print(f"PER-PROMPT CLASSIFICATION")
    print(f"gini>{thresh['gini_thresh']:.3f} OR "
          f"s_early>{thresh['s_thresh']:.4f} OR "
          f"health<{thresh['health_thresh']:.3f}")
    print("=" * 70)

    for r in data:
        gini_fire = r["first_mean_gini"] > thresh["gini_thresh"]
        s_fire = r["s_early_first"] > thresh["s_thresh"]
        health_fire = r["health_score"] < thresh["health_thresh"]
        triggered = gini_fire or s_fire or health_fire
        should = r["should_abstain"]

        correct = triggered == should
        mark = "OK" if correct else "MISS"

        signals = []
        if gini_fire:
            signals.append(f"G:{r['first_mean_gini']:.3f}")
        if s_fire:
            signals.append(f"S:{r['s_early_first']:.4f}")
        if health_fire:
            signals.append(f"H:{r['health_score']:+.3f}")
        signal_str = " ".join(signals) if signals else "(none)"

        status = "TRIG" if triggered else "heal"
        prompt_str = r["prompt"][:50]
        print(f"  [{mark}] {status} [{r['category']:20s}] {prompt_str}")
        print(f"         signals={signal_str} "
              f"(G={r['first_mean_gini']:.3f} "
              f"S={r['s_early_first']:.4f} "
              f"h={r['health_score']:+.3f})")


if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data)} samples from {RESULTS_PATH.name}")

    confab = sum(1 for r in data if r["should_abstain"])
    control = len(data) - confab
    print(f"  Confabulation-prone: {confab}")
    print(f"  Control (known): {control}")

    results = sweep(data)

    # Show per-prompt breakdown under best combined
    print_per_prompt(data, results["best_combined"])

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Config':<30s} {'Acc':>6s} {'F1':>6s} {'Prec':>6s} {'Recall':>7s} {'FPR':>6s}")
    for name, m in [
        ("Current v1", results["current"]),
        ("Gini only", results["best_gini"]),
        ("Health only", results["best_health"]),
        ("Combined (best)", results["best_combined"]),
    ]:
        print(f"{name:<30s} "
              f"{m['accuracy']:>5.0%} "
              f"{m['f1']:>5.2f} "
              f"{m['precision']:>5.0%} "
              f"{m['recall']:>6.0%} "
              f"{m['fpr']:>5.0%}")
