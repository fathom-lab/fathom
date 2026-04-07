#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
"""
FATHOM CONSTANT REPLICATION — Mistral-7B-v0.1 (Hook-Based)
============================================================
Pre-registered experiment. Identical methodology to Gemma-2-2B / Llama-3.2-1B.
Uses hook-based depth scorer (residual stream L2 norms) instead of SAE transcoders.

Usage:
    python run_mistral_experiment.py                    # Full 30-pair experiment
    python run_mistral_experiment.py --smoke             # Single-pair smoke test
    python run_mistral_experiment.py --analyze-only      # Re-analyze existing results

Prerequisites:
    pip install transformer_lens torch transformers scipy numpy
    GPU with >= 8GB VRAM (will auto-fallback to CPU if insufficient)
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

# ── JSON Encoder (handles numpy types) ───────────────────────────────────────
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

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
N_LAYERS = 32
BAND_BOUNDARIES = (10, 22)  # early/mid/late split for 32-layer model

GEMMA_REFERENCE = {"ratio": 1.0302, "p_value": 0.000051, "n_pairs": 30}
LLAMA_REFERENCE = {"ratio": 1.0242, "p_value": 0.00266, "n_pairs": 19}
CONVERGENCE_WINDOW = 0.15  # |K_mistral - 1.03| < 0.15 = converged
SIGNIFICANCE_THRESHOLD = 0.05

BASE_DIR = Path(__file__).parent
PAIRS_FILE = BASE_DIR / ".." / "research" / "experiment_12_power_pairs.json"
RESULTS_FILE = BASE_DIR / "mistral_experiment_results.json"
ANALYSIS_FILE = BASE_DIR / "mistral_experiment_analysis.json"
REPORT_FILE = BASE_DIR / "MISTRAL_EXPERIMENT_REPORT.md"

# ── Load Pairs ───────────────────────────────────────────────────────────────

def load_pairs():
    """Load the 30 pre-registered calibrated pairs."""
    for candidate in [PAIRS_FILE, BASE_DIR / "experiment_12_power_pairs.json"]:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            # Handle wrapper format: {pre_registered_direction, notes, pairs: [...]}
            if isinstance(data, dict) and "pairs" in data:
                raw_pairs = data["pairs"]
            elif isinstance(data, list):
                raw_pairs = data
            else:
                raise ValueError(f"Unexpected pairs format in {candidate}")
            # Normalize to {id, surface: {prompt, target}, insight: {prompt, target}}
            normalized = []
            for p in raw_pairs:
                s = p.get("surface", "")
                i = p.get("insight", "")
                if isinstance(s, str):
                    normalized.append({
                        "id": p.get("topic", p.get("id", p.get("name", "unknown"))),
                        "surface": {"prompt": s, "target": p.get("surface_target", "")},
                        "insight": {"prompt": i, "target": p.get("insight_target", "")},
                    })
                else:
                    normalized.append({
                        "id": p.get("id", p.get("topic", p.get("name", "unknown"))),
                        "surface": {"prompt": s.get("prompt", ""), "target": s.get("target", "")},
                        "insight": {"prompt": i.get("prompt", ""), "target": i.get("target", "")},
                    })
            return normalized
    raise FileNotFoundError(f"Pairs file not found at {PAIRS_FILE}")


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model():
    """Load Mistral-7B with hook-based depth scorer."""
    from depth_scorer_hooks import HookDepthScorer

    print(f"[MODEL] Loading {MODEL_NAME} via HookDepthScorer...")
    print(f"[MODEL] Layers: {N_LAYERS}, Bands: {BAND_BOUNDARIES}")
    print(f"[MODEL] Method: Residual stream L2 norm (hook-based)")

    scorer = HookDepthScorer(
        model_name=MODEL_NAME,
        device="cuda",
        band_boundaries=BAND_BOUNDARIES,
    )
    # Trigger lazy load
    scorer._ensure_model()

    return scorer


# ── Depth Scoring ────────────────────────────────────────────────────────────

def score_depth(scorer, prompt, target):
    """
    Score depth of a prompt+target using hook-based L2 norms.
    Returns dict compatible with the analysis pipeline.
    """
    try:
        result = scorer.score(prompt, target)

        return {
            "depth": result.depth_score,
            "n_layers": result.n_layers,
            "early_ratio": result.early_ratio,
            "mid_ratio": result.mid_ratio,
            "late_ratio": result.late_ratio,
            "layer_profile": result.layer_profile,
            "compute_time_s": result.compute_time_s,
        }

    except Exception as e:
        print(f"[ERROR] Hook scoring failed: {e}")
        return None


# ── Experiment Runner ────────────────────────────────────────────────────────

def run_experiment(scorer, pairs, smoke_test=False):
    """Run the full 30-pair experiment (or single pair for smoke test)."""
    if smoke_test:
        pairs = pairs[:1]
        print("[SMOKE TEST] Running pair #1 only")

    results = []
    n = len(pairs)

    for i, pair in enumerate(pairs):
        pair_id = pair.get("id", pair.get("name", f"pair_{i}"))
        print(f"\n[PAIR {i+1}/{n}] {pair_id}")

        surface_prompt = pair["surface"]["prompt"]
        surface_target = pair["surface"]["target"]
        insight_prompt = pair["insight"]["prompt"]
        insight_target = pair["insight"]["target"]

        print(f"  Surface: ...{surface_prompt[-40:]} → '{surface_target}'")
        print(f"  Insight: ...{insight_prompt[-40:]} → '{insight_target}'")

        # Score surface
        t0 = time.time()
        surface_result = score_depth(scorer, surface_prompt, surface_target)
        t_surface = time.time() - t0

        # Score insight
        t0 = time.time()
        insight_result = score_depth(scorer, insight_prompt, insight_target)
        t_insight = time.time() - t0

        if surface_result is None or insight_result is None:
            print(f"  [SKIP] Scoring failed for pair {pair_id}")
            continue

        delta = surface_result["depth"] - insight_result["depth"]
        direction = "surface_deeper" if delta > 0 else "insight_deeper"

        result = {
            "pair_id": pair_id,
            "pair_index": i,
            "surface_depth": surface_result["depth"],
            "surface_early": surface_result["early_ratio"],
            "surface_mid": surface_result["mid_ratio"],
            "surface_late": surface_result["late_ratio"],
            "insight_depth": insight_result["depth"],
            "insight_early": insight_result["early_ratio"],
            "insight_mid": insight_result["mid_ratio"],
            "insight_late": insight_result["late_ratio"],
            "delta": delta,
            "direction": direction,
            "time_surface_s": round(t_surface, 1),
            "time_insight_s": round(t_insight, 1),
        }

        results.append(result)
        print(f"  Surface: {surface_result['depth']:.4f} (E:{surface_result['early_ratio']:.2f} M:{surface_result['mid_ratio']:.2f} L:{surface_result['late_ratio']:.2f})")
        print(f"  Insight: {insight_result['depth']:.4f} (E:{insight_result['early_ratio']:.2f} M:{insight_result['mid_ratio']:.2f} L:{insight_result['late_ratio']:.2f})")
        print(f"  Delta: {delta:+.4f} ({direction}) [{t_surface + t_insight:.1f}s]")

    return results


# ── Statistical Analysis ─────────────────────────────────────────────────────

def analyze(results):
    """Run full statistical analysis on experiment results."""
    if len(results) < 3:
        print("[ANALYSIS] Too few results for statistical test")
        return None

    surface = np.array([r["surface_depth"] for r in results])
    insight = np.array([r["insight_depth"] for r in results])
    deltas = surface - insight

    n = len(results)
    surface_mean = float(np.mean(surface))
    insight_mean = float(np.mean(insight))
    delta_mean = float(np.mean(deltas))
    delta_std = float(np.std(deltas, ddof=1))

    # Primary test: one-tailed paired t-test
    t_stat, p_value = stats.ttest_rel(surface, insight, alternative="greater")

    # Effect size
    cohens_d = delta_mean / delta_std if delta_std > 0 else 0

    # Direction count
    n_expected = sum(1 for d in deltas if d > 0)
    pct_expected = n_expected / n * 100

    # Fathom ratio
    mistral_ratio = surface_mean / insight_mean if insight_mean > 0 else 0
    ratio_diff = abs(mistral_ratio - 1.03)
    converges = ratio_diff < CONVERGENCE_WINDOW

    # Leave-one-out robustness
    loo_results = []
    for i in range(n):
        s_loo = np.delete(surface, i)
        i_loo = np.delete(insight, i)
        _, p_loo = stats.ttest_rel(s_loo, i_loo, alternative="greater")
        loo_results.append({
            "excluded_pair": results[i]["pair_id"],
            "p_value": float(p_loo),
            "significant": p_loo < SIGNIFICANCE_THRESHOLD,
        })

    loo_all_significant = all(lr["significant"] for lr in loo_results)
    loo_min_p = max(lr["p_value"] for lr in loo_results)  # worst case

    # Determine tier
    if p_value >= SIGNIFICANCE_THRESHOLD:
        tier = 4  # No replication
        verdict = "NO REPLICATION — effect not significant on Mistral"
    elif not converges:
        tier = 3  # Replicates but diverges
        verdict = f"REPLICATES BUT DIVERGES — ratio {mistral_ratio:.4f} outside convergence window"
    elif not loo_all_significant:
        tier = 2  # Converges but not fully robust
        verdict = f"CONVERGES (ratio {mistral_ratio:.4f}) — LOO not fully robust"
    else:
        tier = 1  # Full convergence
        verdict = f"FULL CONVERGENCE — ratio {mistral_ratio:.4f} ≈ 1.03. FATHOM CONSTANT CONFIRMED."

    analysis = {
        "model": MODEL_NAME,
        "method": "hook_l2_norm",
        "n_pairs": n,
        "timestamp": datetime.now().isoformat(),

        # Descriptive stats
        "surface_mean": surface_mean,
        "insight_mean": insight_mean,
        "delta_mean": delta_mean,
        "delta_std": delta_std,

        # Primary test
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < SIGNIFICANCE_THRESHOLD,
        "cohens_d": cohens_d,

        # Direction
        "n_expected_direction": n_expected,
        "pct_expected_direction": pct_expected,

        # Convergence
        "mistral_ratio": mistral_ratio,
        "gemma_ratio": GEMMA_REFERENCE["ratio"],
        "llama_ratio": LLAMA_REFERENCE["ratio"],
        "ratio_difference": ratio_diff,
        "converges": converges,
        "convergence_window": CONVERGENCE_WINDOW,

        # Robustness
        "loo_all_significant": loo_all_significant,
        "loo_worst_p": loo_min_p,
        "loo_results": loo_results,

        # Verdict
        "tier": tier,
        "verdict": verdict,

        # Cross-architecture comparison
        "gemma_reference": GEMMA_REFERENCE,
        "llama_reference": LLAMA_REFERENCE,
        "fathom_constant_estimate": (
            (mistral_ratio + GEMMA_REFERENCE["ratio"] + LLAMA_REFERENCE["ratio"]) / 3
            if converges else None
        ),
    }

    return analysis


# ── Report Generation ────────────────────────────────────────────────────────

def generate_report(results, analysis):
    """Generate human-readable markdown report."""
    if not analysis:
        return "# Mistral Experiment Report\n\nInsufficient data for analysis."

    lines = [
        "# FATHOM CONSTANT REPLICATION — Mistral-7B-v0.1 Results",
        "",
        f"**Date**: {analysis['timestamp'][:10]}",
        f"**Model**: {analysis['model']}",
        f"**Method**: Hook-based (residual stream L2 norm)",
        f"**Pairs**: {analysis['n_pairs']}",
        f"**Tier**: {analysis['tier']}",
        "",
        f"## VERDICT: {analysis['verdict']}",
        "",
        "---",
        "",
        "## Descriptive Statistics",
        "",
        f"| Metric | Mistral-7B | Gemma-2-2B (ref) | Llama-3.2-1B (ref) |",
        f"|--------|-----------|-----------------|-------------------|",
        f"| Surface mean | {analysis['surface_mean']:.4f} | 8.3592 | — |",
        f"| Insight mean | {analysis['insight_mean']:.4f} | 8.1025 | — |",
        f"| Delta | {analysis['delta_mean']:+.4f} | +0.2568 | — |",
        f"| Ratio | {analysis['mistral_ratio']:.4f} | {GEMMA_REFERENCE['ratio']:.4f} | {LLAMA_REFERENCE['ratio']:.4f} |",
        "",
        "## Statistical Test",
        "",
        f"- t-statistic: {analysis['t_statistic']:.4f}",
        f"- p-value: {analysis['p_value']:.6f}",
        f"- Significant (p < 0.05): {'YES' if analysis['significant'] else 'NO'}",
        f"- Cohen's d: {analysis['cohens_d']:.4f}",
        f"- Pairs in expected direction: {analysis['n_expected_direction']}/{analysis['n_pairs']} ({analysis['pct_expected_direction']:.1f}%)",
        "",
        "## Convergence Analysis",
        "",
        f"- Mistral ratio: {analysis['mistral_ratio']:.4f}",
        f"- Gemma ratio: {analysis['gemma_ratio']:.4f}",
        f"- Llama ratio: {analysis['llama_ratio']:.4f}",
        f"- Difference from 1.03: {analysis['ratio_difference']:.4f}",
        f"- Within window (< {analysis['convergence_window']}): {'YES' if analysis['converges'] else 'NO'}",
    ]

    if analysis['converges'] and analysis['fathom_constant_estimate']:
        lines.append(f"- **Fathom Constant (3-model estimate)**: {analysis['fathom_constant_estimate']:.4f}")

    lines += [
        "",
        "## Leave-One-Out Robustness",
        "",
        f"- All subsets significant: {'YES' if analysis['loo_all_significant'] else 'NO'}",
        f"- Worst-case p-value: {analysis['loo_worst_p']:.6f}",
        "",
        "## Per-Pair Results",
        "",
        "| # | Pair | Surface | Insight | Delta | Direction |",
        "|---|------|---------|---------|-------|-----------|",
    ]

    for r in results:
        lines.append(
            f"| {r['pair_index']+1} | {r['pair_id'][:20]} | {r['surface_depth']:.4f} | "
            f"{r['insight_depth']:.4f} | {r['delta']:+.4f} | {r['direction']} |"
        )

    lines += [
        "",
        "---",
        "",
        f"*Generated by run_mistral_experiment.py at {analysis['timestamp']}*",
    ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fathom Constant Replication — Mistral-7B-v0.1")
    parser.add_argument("--smoke", action="store_true", help="Run single-pair smoke test only")
    parser.add_argument("--analyze-only", action="store_true", help="Re-analyze existing results")
    args = parser.parse_args()

    print("=" * 60)
    print("  FATHOM CONSTANT REPLICATION — Mistral-7B-v0.1")
    print("  Hook-based depth scoring (residual stream L2 norm)")
    print("  Pre-registered experiment protocol")
    print("=" * 60)
    print()

    # Load pairs
    pairs = load_pairs()
    print(f"[PAIRS] Loaded {len(pairs)} calibrated pairs")

    if args.analyze_only:
        if not RESULTS_FILE.exists():
            print("[ERROR] No results file found. Run experiment first.")
            sys.exit(1)
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"[ANALYZE] Loaded {len(results)} existing results")
    else:
        scorer = load_model()
        results = run_experiment(scorer, pairs, smoke_test=args.smoke)

        # Save raw results
        with open(RESULTS_FILE, "w") as f:
            json_dump(results, f)
        print(f"\n[SAVED] Raw results → {RESULTS_FILE}")

        # Cleanup
        scorer.cleanup()

    if args.smoke:
        if results:
            r = results[0]
            print(f"\n{'=' * 60}")
            print(f"  SMOKE TEST RESULT")
            print(f"  Surface: {r['surface_depth']:.4f}")
            print(f"  Insight: {r['insight_depth']:.4f}")
            print(f"  Delta: {r['delta']:+.4f}")
            print(f"  Direction: {r['direction']}")
            print(f"  Status: {'PASS' if r['direction'] == 'surface_deeper' else 'UNEXPECTED (investigate)'}")
            print(f"{'=' * 60}")
        return

    # Full analysis
    print(f"\n{'=' * 60}")
    print("  STATISTICAL ANALYSIS")
    print(f"{'=' * 60}\n")

    analysis = analyze(results)

    if analysis:
        with open(ANALYSIS_FILE, "w") as f:
            json_dump(analysis, f)
        print(f"[SAVED] Analysis → {ANALYSIS_FILE}")

        report = generate_report(results, analysis)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[SAVED] Report → {REPORT_FILE}")

        print(f"\n{'=' * 60}")
        print(f"  TIER {analysis['tier']}: {analysis['verdict']}")
        print(f"{'=' * 60}")
        print(f"  p = {analysis['p_value']:.6f}")
        print(f"  delta = {analysis['delta_mean']:+.4f}")
        print(f"  ratio = {analysis['mistral_ratio']:.4f} (Gemma: {GEMMA_REFERENCE['ratio']:.4f}, Llama: {LLAMA_REFERENCE['ratio']:.4f})")
        if analysis['converges']:
            print(f"  FATHOM CONSTANT (3-model) = {analysis['fathom_constant_estimate']:.4f}")
        print(f"{'=' * 60}")

        if analysis['tier'] == 1:
            print("\n  *** FULL CONVERGENCE ACROSS 3 ARCHITECTURES ***")
            print("  *** THE FATHOM CONSTANT IS REAL. ***\n")


if __name__ == "__main__":
    main()
