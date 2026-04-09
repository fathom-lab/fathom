# -*- coding: utf-8 -*-
"""
analyze_cognitive_profile.py -- Complete Cognitive Profiling
============================================================

This is NOT a hallucination detector. This is a cognitive profiling tool.

Given oversight results, it answers:
  1. What is this model's cognitive fingerprint? (baseline K, C, Gini, S distributions)
  2. Which cognitive axes predict output quality? (not just hallucination — ALL labels)
  3. Where does intervention change cognition? (K shift, Gini shift, feature anatomy)
  4. What failure modes exist and how do they cluster?
  5. Can we IMPROVE output by steering cognition? (capability unlocking)

Usage:
  python analyze_cognitive_profile.py oversight_results/hallucination_eval_n30.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_cognitive_fingerprint(results: list) -> dict:
    """
    What is this model's cognitive baseline?
    Distribution of ALL axes across all items.
    """
    k_vals = [r["unguarded"]["k_depth_mean"] for r in results if r["unguarded"].get("k_depth_mean")]
    gini_vals = [r["unguarded"]["gini_max"] for r in results if r["unguarded"].get("gini_max")]
    s_vals = [r["unguarded"]["s_early_max"] for r in results if r["unguarded"].get("s_early_max")]

    fingerprint = {
        "n_items": len(results),
        "k_depth": {
            "mean": float(np.mean(k_vals)) if k_vals else 0,
            "std": float(np.std(k_vals)) if k_vals else 0,
            "min": float(np.min(k_vals)) if k_vals else 0,
            "max": float(np.max(k_vals)) if k_vals else 0,
        },
        "gini": {
            "mean": float(np.mean(gini_vals)) if gini_vals else 0,
            "std": float(np.std(gini_vals)) if gini_vals else 0,
            "min": float(np.min(gini_vals)) if gini_vals else 0,
            "max": float(np.max(gini_vals)) if gini_vals else 0,
            "pct_above_051": sum(1 for g in gini_vals if g > 0.51) / max(len(gini_vals), 1),
        },
        "s_early": {
            "mean": float(np.mean(s_vals)) if s_vals else 0,
            "std": float(np.std(s_vals)) if s_vals else 0,
            "max": float(np.max(s_vals)) if s_vals else 0,
            "pct_above_008": sum(1 for s in s_vals if s > 0.008) / max(len(s_vals), 1),
        },
    }
    return fingerprint


def analyze_by_output_quality(results: list) -> dict:
    """
    How do cognitive axes relate to output quality?
    Not just hallucination — ALL label categories.
    """
    # Group by unguarded label
    by_label = defaultdict(list)
    for r in results:
        label = r["unguarded"]["label"]
        by_label[label].append(r)

    analysis = {}
    for label, items in by_label.items():
        k_vals = [r["unguarded"]["k_depth_mean"] for r in items if r["unguarded"].get("k_depth_mean")]
        gini_vals = [r["unguarded"]["gini_max"] for r in items if r["unguarded"].get("gini_max")]
        s_vals = [r["unguarded"]["s_early_max"] for r in items if r["unguarded"].get("s_early_max")]

        analysis[label] = {
            "count": len(items),
            "k_mean": float(np.mean(k_vals)) if k_vals else 0,
            "gini_mean": float(np.mean(gini_vals)) if gini_vals else 0,
            "s_early_mean": float(np.mean(s_vals)) if s_vals else 0,
        }

    return analysis


def analyze_intervention_effects(results: list) -> dict:
    """
    When intervention fires, what changes?
    Look at ALL cognitive axes, not just the target metric.
    """
    intervened = [r for r in results if r["guarded"]["escalation_events"]]
    not_intervened = [r for r in results if not r["guarded"]["escalation_events"]]

    effects = {
        "n_intervened": len(intervened),
        "n_not_intervened": len(not_intervened),
        "intervention_rate": len(intervened) / max(len(results), 1),
    }

    if intervened:
        # What did intervention do to escalation events?
        all_events = []
        for r in intervened:
            for ev in r["guarded"]["escalation_events"]:
                all_events.append(ev)

        # Gini shifts
        gini_befores = [e.get("gini_before", 0) for e in all_events if "gini_before" in e]
        gini_afters = [e.get("gini_after", 0) for e in all_events if "gini_after" in e]
        if gini_befores and gini_afters:
            effects["gini_shift"] = {
                "mean_before": float(np.mean(gini_befores)),
                "mean_after": float(np.mean(gini_afters)),
                "mean_delta": float(np.mean(gini_afters)) - float(np.mean(gini_befores)),
                "improved": sum(1 for b, a in zip(gini_befores, gini_afters) if a < b),
                "worsened": sum(1 for b, a in zip(gini_befores, gini_afters) if a > b),
            }

        # S_early shifts
        s_befores = [e.get("s_before", 0) for e in all_events if "s_before" in e]
        s_afters = [e.get("s_after", 0) for e in all_events if "s_after" in e]
        if s_befores and s_afters:
            effects["s_early_shift"] = {
                "mean_before": float(np.mean(s_befores)),
                "mean_after": float(np.mean(s_afters)),
                "mean_delta": float(np.mean(s_afters)) - float(np.mean(s_befores)),
            }

        # Escalation level distribution
        levels = [r["guarded"]["max_escalation_level"] for r in intervened]
        effects["escalation_dist"] = {str(l): levels.count(l) for l in set(levels)}

        # Label changes
        label_changes = defaultdict(int)
        for r in intervened:
            change = f"{r['unguarded']['label']} -> {r['guarded']['label']}"
            label_changes[change] += 1
        effects["label_transitions"] = dict(label_changes)

    return effects


def analyze_anomaly_taxonomy(results: list) -> dict:
    """
    What anomaly types does this model exhibit?
    How do they distribute? Do different types correspond to different failures?
    """
    anomaly_counts = defaultdict(int)
    anomaly_by_label = defaultdict(lambda: defaultdict(int))

    for r in results:
        u_label = r["unguarded"]["label"]
        for a in r["guarded"]["anomalies"]:
            atype = a["anomaly_type"]
            anomaly_counts[atype] += 1
            anomaly_by_label[atype][u_label] += 1

    return {
        "total_anomalies": sum(anomaly_counts.values()),
        "anomaly_distribution": dict(anomaly_counts),
        "anomaly_by_output_quality": {k: dict(v) for k, v in anomaly_by_label.items()},
    }


def analyze_feature_anatomy(results: list) -> dict:
    """
    What features get suppressed? What do they encode?
    Aggregate across all interventions to find common fault features.
    """
    feature_freq = defaultdict(lambda: defaultdict(int))  # layer -> feature_id -> count

    for r in results:
        anatomy = r["guarded"].get("feature_anatomy", {})
        for layer_str, feats in anatomy.items():
            for feat in feats:
                fid = feat.get("feature_id", -1)
                feature_freq[layer_str][fid] += 1

    # Find most common fault features
    top_features = {}
    for layer_str, freqs in feature_freq.items():
        sorted_feats = sorted(freqs.items(), key=lambda x: -x[1])[:10]
        top_features[layer_str] = sorted_feats

    # Also collect token decodings
    feature_tokens = defaultdict(dict)
    for r in results:
        anatomy = r["guarded"].get("feature_anatomy", {})
        for layer_str, feats in anatomy.items():
            for feat in feats:
                fid = feat.get("feature_id", -1)
                tokens = feat.get("top_tokens", [])
                if tokens:
                    feature_tokens[layer_str][fid] = tokens

    return {
        "most_common_fault_features": {k: v for k, v in top_features.items()},
        "feature_token_decodings": {
            layer: {str(fid): toks for fid, toks in feats.items()}
            for layer, feats in feature_tokens.items()
        },
    }


def print_report(results_path: str):
    """Print the complete cognitive profile report."""
    data = load_results(results_path)
    results = data.get("results", [])

    if not results:
        print("[!] No results found")
        return

    print("=" * 70)
    print("FATHOM COGNITIVE PROFILE REPORT")
    print(f"Model: {data.get('model', '?')}")
    print(f"Items: {len(results)}")
    print("=" * 70)

    # 1. Cognitive Fingerprint
    fp = analyze_cognitive_fingerprint(results)
    print(f"\n--- COGNITIVE FINGERPRINT ---")
    print(f"K depth:   {fp['k_depth']['mean']:.3f} +/- {fp['k_depth']['std']:.3f} "
          f"(range: {fp['k_depth']['min']:.3f} - {fp['k_depth']['max']:.3f})")
    print(f"Gini:      {fp['gini']['mean']:.3f} +/- {fp['gini']['std']:.3f} "
          f"({fp['gini']['pct_above_051']:.0%} above 0.51 threshold)")
    print(f"S_early:   {fp['s_early']['mean']:.5f} +/- {fp['s_early']['std']:.5f} "
          f"({fp['s_early']['pct_above_008']:.0%} above 0.008 threshold)")

    # 2. Output Quality by Cognitive State
    quality = analyze_by_output_quality(results)
    print(f"\n--- COGNITIVE STATE BY OUTPUT QUALITY ---")
    print(f"{'Label':<15} {'Count':>5} {'K_mean':>8} {'Gini_mean':>10} {'S_early':>10}")
    for label, stats_d in sorted(quality.items(), key=lambda x: -x[1]["count"]):
        print(f"{label:<15} {stats_d['count']:>5} {stats_d['k_mean']:>8.3f} "
              f"{stats_d['gini_mean']:>10.3f} {stats_d['s_early_mean']:>10.5f}")

    # 3. Intervention Effects
    effects = analyze_intervention_effects(results)
    print(f"\n--- INTERVENTION EFFECTS ---")
    print(f"Intervention rate: {effects['intervention_rate']:.0%} "
          f"({effects['n_intervened']}/{effects['n_intervened'] + effects['n_not_intervened']})")
    if "gini_shift" in effects:
        gs = effects["gini_shift"]
        print(f"Gini shift: {gs['mean_before']:.4f} -> {gs['mean_after']:.4f} "
              f"(delta={gs['mean_delta']:+.4f})")
        print(f"  Improved: {gs.get('improved', 0)}, Worsened: {gs.get('worsened', 0)}")
    if "escalation_dist" in effects:
        print(f"Escalation levels: {effects['escalation_dist']}")
    if "label_transitions" in effects:
        print(f"Label transitions:")
        for trans, count in effects["label_transitions"].items():
            print(f"  {trans}: {count}")

    # 4. Anomaly Taxonomy
    taxonomy = analyze_anomaly_taxonomy(results)
    print(f"\n--- ANOMALY TAXONOMY ---")
    print(f"Total anomalies detected: {taxonomy['total_anomalies']}")
    print(f"Distribution: {taxonomy['anomaly_distribution']}")
    if taxonomy["anomaly_by_output_quality"]:
        print(f"Anomaly type x output quality:")
        for atype, labels in taxonomy["anomaly_by_output_quality"].items():
            print(f"  {atype}: {dict(labels)}")

    # 5. Feature Anatomy
    anatomy = analyze_feature_anatomy(results)
    if anatomy["most_common_fault_features"]:
        print(f"\n--- FEATURE ANATOMY ---")
        print(f"Most common fault features (by layer):")
        for layer, feats in anatomy["most_common_fault_features"].items():
            tokens_map = anatomy["feature_token_decodings"].get(layer, {})
            print(f"  Layer {layer}:")
            for fid, count in feats[:5]:
                tokens = tokens_map.get(str(fid), [])
                safe_tokens = [t.encode('ascii', 'replace').decode() for t in tokens[:4]]
                print(f"    Feature {fid} (x{count}): {safe_tokens}")

    # Save report
    report = {
        "cognitive_fingerprint": fp,
        "quality_by_state": quality,
        "intervention_effects": effects,
        "anomaly_taxonomy": taxonomy,
        "feature_anatomy": anatomy,
    }
    out_path = Path(results_path).parent / "cognitive_profile_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try most recent results
        results_dir = Path(__file__).parent / "oversight_results"
        candidates = sorted(results_dir.glob("hallucination_eval_*.json"))
        if candidates:
            path = str(candidates[-1])
        else:
            print("Usage: python analyze_cognitive_profile.py <results.json>")
            sys.exit(1)
    else:
        path = sys.argv[1]

    print_report(path)
