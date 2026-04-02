#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
"""
RLHF DEPTH SUPPRESSION EXPERIMENT
====================================
Tests whether alignment training (RLHF/instruction tuning) suppresses
reasoning depth, analogous to the Default Mode Network suppressing
distributed neural processing.

Hypothesis (H4 from FATHOM_FRAMEWORK):
    K_base > K_instruct
    Alignment training reduces the surface/insight depth differential.

Method:
    Run the same 30 calibrated pairs on matched base/instruct model pairs.
    Compare K ratios using the hook-based depth scorer (residual stream L2 norms).

Model Pairs:
    - meta-llama/Llama-3.2-1B  vs  meta-llama/Llama-3.2-1B-Instruct
    - (extensible to Mistral-7B vs Mistral-7B-Instruct, etc.)

Usage:
    python experiment_rlhf_depth.py                  # Full experiment
    python experiment_rlhf_depth.py --smoke           # 3 pairs only
    python experiment_rlhf_depth.py --analyze-only    # Re-analyze existing results

Theoretical Basis:
    Johns Hopkins psilocybin research (Carhart-Harris et al.) shows that
    suppressing the Default Mode Network increases cross-region connectivity
    and information integration. RLHF may function as a "DMN installer" --
    training the model to filter and compress its representations for smooth
    output, at the cost of distributed mid-layer computation.

    If confirmed, this implies:
    - An alignment tax on reasoning depth
    - Fathom as a monitoring tool during fine-tuning
    - Optimal alignment preserves depth while improving safety
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

# ── JSON Encoder ─────────────────────────────────────────────────────────────
import json as _json

class NumpyEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def json_dump(obj, f, **kwargs):
    kwargs.setdefault('indent', 2)
    _json.dump(obj, f, cls=NumpyEncoder, **kwargs)


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PAIRS = [
    {
        "base": "meta-llama/Llama-3.2-1B",
        "instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama-3.2-1B",
        "n_layers": 16,
        "bands": (5, 11),
    },
    {
        "base": "google/gemma-2-2b",
        "instruct": "google/gemma-2-2b-it",
        "name": "Gemma-2-2B",
        "n_layers": 26,
        "bands": (8, 18),
    },
    # Uncomment for larger models (requires more VRAM/CPU time):
    # {
    #     "base": "mistralai/Mistral-7B-v0.1",
    #     "instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    #     "name": "Mistral-7B",
    #     "n_layers": 32,
    #     "bands": (10, 22),
    # },
]

SIGNIFICANCE_THRESHOLD = 0.05

BASE_DIR = Path(__file__).parent
PAIRS_FILE = BASE_DIR / ".." / "research" / "experiment_12_power_pairs.json"
RESULTS_FILE = BASE_DIR / "rlhf_depth_results.json"
REPORT_FILE = BASE_DIR / "RLHF_DEPTH_REPORT.md"


# ── Load Pairs ───────────────────────────────────────────────────────────────

def load_pairs():
    """Load the 30 pre-registered calibrated pairs."""
    for candidate in [PAIRS_FILE, BASE_DIR / "experiment_12_power_pairs.json"]:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            if isinstance(data, dict) and "pairs" in data:
                raw_pairs = data["pairs"]
            elif isinstance(data, list):
                raw_pairs = data
            else:
                raise ValueError(f"Unexpected format in {candidate}")
            normalized = []
            for p in raw_pairs:
                s = p.get("surface", "")
                i = p.get("insight", "")
                if isinstance(s, str):
                    normalized.append({
                        "id": p.get("topic", p.get("id", "unknown")),
                        "surface": {"prompt": s, "target": p.get("surface_target", "")},
                        "insight": {"prompt": i, "target": p.get("insight_target", "")},
                    })
                else:
                    normalized.append({
                        "id": p.get("id", p.get("topic", "unknown")),
                        "surface": {"prompt": s.get("prompt", ""), "target": s.get("target", "")},
                        "insight": {"prompt": i.get("prompt", ""), "target": i.get("target", "")},
                    })
            return normalized
    raise FileNotFoundError("Pairs file not found")


# ── Score a Model ────────────────────────────────────────────────────────────

def score_model(model_name, pairs, bands, smoke_test=False):
    """Score all pairs on a single model using hook-based depth scorer."""
    from depth_scorer_hooks import HookDepthScorer

    if smoke_test:
        pairs = pairs[:3]

    print(f"\n  Loading {model_name}...")
    scorer = HookDepthScorer(
        model_name=model_name,
        device="cuda",
        band_boundaries=bands,
    )
    scorer._ensure_model()

    results = []
    n = len(pairs)

    for i, pair in enumerate(pairs):
        pair_id = pair["id"]
        print(f"    [{i+1}/{n}] {pair_id}", end=" ", flush=True)

        try:
            t0 = time.time()
            cmp = scorer.compare(
                pair["surface"]["prompt"], pair["surface"]["target"],
                pair["insight"]["prompt"], pair["insight"]["target"],
            )
            elapsed = time.time() - t0

            results.append({
                "pair_id": pair_id,
                "surface_depth": cmp.surface.depth_score,
                "insight_depth": cmp.insight.depth_score,
                "delta": cmp.delta,
                "ratio": cmp.ratio,
                "surface_early": cmp.surface.early_ratio,
                "surface_mid": cmp.surface.mid_ratio,
                "surface_late": cmp.surface.late_ratio,
                "insight_early": cmp.insight.early_ratio,
                "insight_mid": cmp.insight.mid_ratio,
                "insight_late": cmp.insight.late_ratio,
                "time_s": round(elapsed, 1),
            })

            direction = "S>I" if cmp.delta > 0 else "I>S"
            print(f"delta={cmp.delta:+.4f} ({direction}) [{elapsed:.1f}s]")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    scorer.cleanup()
    return results


# ── Compare Base vs Instruct ─────────────────────────────────────────────────

def compare_models(base_results, instruct_results, model_name):
    """Statistical comparison of base vs instruct depth profiles."""
    # Match pairs by pair_id
    base_by_id = {r["pair_id"]: r for r in base_results}
    inst_by_id = {r["pair_id"]: r for r in instruct_results}
    common_ids = sorted(set(base_by_id.keys()) & set(inst_by_id.keys()))

    if len(common_ids) < 3:
        print("[WARN] Too few common pairs for analysis")
        return None

    # Extract matched data
    base_deltas = np.array([base_by_id[pid]["delta"] for pid in common_ids])
    inst_deltas = np.array([inst_by_id[pid]["delta"] for pid in common_ids])
    base_surface = np.array([base_by_id[pid]["surface_depth"] for pid in common_ids])
    base_insight = np.array([base_by_id[pid]["insight_depth"] for pid in common_ids])
    inst_surface = np.array([inst_by_id[pid]["surface_depth"] for pid in common_ids])
    inst_insight = np.array([inst_by_id[pid]["insight_depth"] for pid in common_ids])

    n = len(common_ids)

    # K ratios
    base_K = float(np.mean(base_surface)) / float(np.mean(base_insight))
    inst_K = float(np.mean(inst_surface)) / float(np.mean(inst_insight))
    K_delta = base_K - inst_K

    # Direction percentages
    base_pct = sum(1 for d in base_deltas if d > 0) / n * 100
    inst_pct = sum(1 for d in inst_deltas if d > 0) / n * 100

    # Test: are base deltas significantly larger than instruct deltas?
    # (Would indicate RLHF suppresses the surface/insight distinction)
    delta_diff = base_deltas - inst_deltas
    t_stat, p_value = stats.ttest_1samp(delta_diff, 0, alternative="greater")

    # Effect size
    dd_mean = float(np.mean(delta_diff))
    dd_std = float(np.std(delta_diff, ddof=1))
    cohens_d = dd_mean / dd_std if dd_std > 0 else 0

    # Mid-layer ratio comparison (insight prompts)
    base_mid_insight = np.array([base_by_id[pid]["insight_mid"] for pid in common_ids])
    inst_mid_insight = np.array([inst_by_id[pid]["insight_mid"] for pid in common_ids])
    mid_t, mid_p = stats.ttest_rel(base_mid_insight, inst_mid_insight, alternative="greater")

    # Determine verdict
    if p_value < 0.01 and K_delta > 0:
        verdict = "STRONG EVIDENCE: RLHF suppresses reasoning depth"
    elif p_value < 0.05 and K_delta > 0:
        verdict = "EVIDENCE: RLHF appears to reduce depth differential"
    elif K_delta > 0:
        verdict = "TREND: Direction supports hypothesis but not significant"
    elif K_delta < 0:
        verdict = "REVERSED: Instruct model shows MORE depth differentiation"
    else:
        verdict = "NULL: No difference detected"

    analysis = {
        "model_pair": model_name,
        "n_pairs": n,
        "timestamp": datetime.now().isoformat(),

        # K ratios
        "base_K": round(base_K, 6),
        "instruct_K": round(inst_K, 6),
        "K_delta": round(K_delta, 6),
        "K_suppression_pct": round(K_delta / (base_K - 1) * 100, 1) if base_K != 1 else 0,

        # Means
        "base_surface_mean": round(float(np.mean(base_surface)), 4),
        "base_insight_mean": round(float(np.mean(base_insight)), 4),
        "base_delta_mean": round(float(np.mean(base_deltas)), 4),
        "inst_surface_mean": round(float(np.mean(inst_surface)), 4),
        "inst_insight_mean": round(float(np.mean(inst_insight)), 4),
        "inst_delta_mean": round(float(np.mean(inst_deltas)), 4),

        # Direction
        "base_pct_expected": round(base_pct, 1),
        "inst_pct_expected": round(inst_pct, 1),

        # Statistical test
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < SIGNIFICANCE_THRESHOLD,
        "cohens_d": round(cohens_d, 4),

        # Mid-layer analysis
        "base_mid_insight_mean": round(float(np.mean(base_mid_insight)), 4),
        "inst_mid_insight_mean": round(float(np.mean(inst_mid_insight)), 4),
        "mid_layer_t": round(float(mid_t), 4),
        "mid_layer_p": round(float(mid_p), 6),

        # Verdict
        "verdict": verdict,

        # Per-pair details
        "per_pair": [
            {
                "pair_id": pid,
                "base_delta": round(float(base_by_id[pid]["delta"]), 4),
                "inst_delta": round(float(inst_by_id[pid]["delta"]), 4),
                "suppression": round(float(base_by_id[pid]["delta"] - inst_by_id[pid]["delta"]), 4),
            }
            for pid in common_ids
        ],
    }

    return analysis


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report(all_analyses):
    """Generate markdown report."""
    lines = [
        "# RLHF DEPTH SUPPRESSION EXPERIMENT",
        "",
        "## Hypothesis",
        "",
        "RLHF/instruction tuning acts as a 'Default Mode Network installer' --",
        "it trains the model to filter and compress representations for smooth output,",
        "suppressing distributed mid-layer computation that characterizes deep reasoning.",
        "",
        "**Prediction**: K_base > K_instruct (alignment reduces depth differential)",
        "",
        "---",
        "",
    ]

    for analysis in all_analyses:
        a = analysis
        lines += [
            f"## {a['model_pair']}",
            "",
            f"**Date**: {a['timestamp'][:10]}",
            f"**Pairs**: {a['n_pairs']}",
            "",
            "### K Ratio Comparison",
            "",
            f"| | Base | Instruct | Delta |",
            f"|--|------|----------|-------|",
            f"| **K ratio** | {a['base_K']:.4f} | {a['instruct_K']:.4f} | {a['K_delta']:+.4f} |",
            f"| Surface mean | {a['base_surface_mean']:.4f} | {a['inst_surface_mean']:.4f} | |",
            f"| Insight mean | {a['base_insight_mean']:.4f} | {a['inst_insight_mean']:.4f} | |",
            f"| Delta mean | {a['base_delta_mean']:+.4f} | {a['inst_delta_mean']:+.4f} | |",
            f"| % expected dir | {a['base_pct_expected']:.0f}% | {a['inst_pct_expected']:.0f}% | |",
            "",
            "### Statistical Test",
            "",
            f"- **t-statistic**: {a['t_statistic']:.4f}",
            f"- **p-value**: {a['p_value']:.6f}",
            f"- **Significant**: {'YES' if a['significant'] else 'NO'}",
            f"- **Cohen's d**: {a['cohens_d']:.4f}",
            "",
            "### Mid-Layer Analysis",
            "",
            f"- Base insight mid-layer ratio: {a['base_mid_insight_mean']:.4f}",
            f"- Instruct insight mid-layer ratio: {a['inst_mid_insight_mean']:.4f}",
            f"- Mid-layer t-test p: {a['mid_layer_p']:.6f}",
            "",
            f"### VERDICT: {a['verdict']}",
            "",
        ]

        if a['K_delta'] > 0:
            lines += [
                f"RLHF reduced K by {a['K_suppression_pct']:.1f}% of the base effect.",
                f"The aligned model shows a {'weaker' if a['K_delta'] > 0 else 'stronger'} surface/insight distinction.",
                "",
            ]

        lines += ["---", ""]

    lines += [
        "## Theoretical Interpretation",
        "",
        "If RLHF suppresses depth, this parallels the Default Mode Network in neuroscience:",
        "",
        "- The DMN filters raw sensory input into survival-relevant categories",
        "- RLHF filters raw model computation into user-relevant output patterns",
        "- Psilocybin suppresses the DMN, increasing neural integration (Carhart-Harris 2012)",
        "- Removing RLHF preserves computational integration (this experiment)",
        "",
        "The Fathom Constant K may measure the degree of computational filtering in any",
        "information-processing system, biological or artificial.",
        "",
        "---",
        "",
        f"*Generated by experiment_rlhf_depth.py at {datetime.now().isoformat()[:19]}*",
        f"*Fathom -- US Patent Pending 64/020,489 & 64/021,113*",
    ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RLHF Depth Suppression Experiment")
    parser.add_argument("--smoke", action="store_true", help="3 pairs only (quick test)")
    parser.add_argument("--analyze-only", action="store_true", help="Re-analyze existing results")
    parser.add_argument("--model-pair", type=int, default=0, help="Index of model pair to test")
    args = parser.parse_args()

    print("=" * 60)
    print("  RLHF DEPTH SUPPRESSION EXPERIMENT")
    print("  Does alignment training suppress reasoning depth?")
    print("=" * 60)

    pairs = load_pairs()
    print(f"\n[PAIRS] {len(pairs)} calibrated pairs loaded")

    all_results = {}
    all_analyses = []

    if args.analyze_only:
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                saved = json.load(f)
            all_analyses = saved.get("analyses", [])
            print(f"[LOADED] {len(all_analyses)} analyses from {RESULTS_FILE}")
        else:
            print("[ERROR] No results file found.")
            sys.exit(1)
    else:
        model_pair = MODEL_PAIRS[args.model_pair]
        name = model_pair["name"]
        bands = tuple(model_pair["bands"])

        print(f"\n{'=' * 60}")
        print(f"  MODEL PAIR: {name}")
        print(f"  Base:     {model_pair['base']}")
        print(f"  Instruct: {model_pair['instruct']}")
        print(f"  Layers:   {model_pair['n_layers']}")
        print(f"{'=' * 60}")

        # Score base model
        print(f"\n[BASE MODEL] {model_pair['base']}")
        base_results = score_model(model_pair["base"], pairs, bands, args.smoke)

        # Score instruct model
        print(f"\n[INSTRUCT MODEL] {model_pair['instruct']}")
        inst_results = score_model(model_pair["instruct"], pairs, bands, args.smoke)

        all_results[name] = {
            "base": base_results,
            "instruct": inst_results,
        }

        # Analyze
        analysis = compare_models(base_results, inst_results, name)
        if analysis:
            all_analyses.append(analysis)

        # Save results
        with open(RESULTS_FILE, "w") as f:
            json_dump({"results": all_results, "analyses": all_analyses}, f)
        print(f"\n[SAVED] Results -> {RESULTS_FILE}")

    # Generate report
    if all_analyses:
        report = generate_report(all_analyses)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[SAVED] Report -> {REPORT_FILE}")

        # Print summary
        for a in all_analyses:
            print(f"\n{'=' * 60}")
            print(f"  {a['model_pair']}")
            print(f"  K_base    = {a['base_K']:.6f}")
            print(f"  K_instruct = {a['instruct_K']:.6f}")
            print(f"  K_delta   = {a['K_delta']:+.6f}")
            print(f"  p-value   = {a['p_value']:.6f}")
            print(f"  VERDICT: {a['verdict']}")
            print(f"{'=' * 60}")

            if a['significant'] and a['K_delta'] > 0:
                print("\n  *** RLHF SUPPRESSES DEPTH ***")
                print("  *** THE DEFAULT MODE NETWORK HYPOTHESIS HOLDS ***")
                print("  *** ALIGNMENT HAS A MEASURABLE DEPTH TAX ***\n")


if __name__ == "__main__":
    main()
