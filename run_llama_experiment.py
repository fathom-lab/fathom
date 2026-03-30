#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
"""
FATHOM CONSTANT REPLICATION — Llama-3.2-1B
============================================
Pre-registered experiment. Identical methodology to Gemma-2-2B.
Tests whether depth effect replicates and whether the constant converges.

Usage:
    python run_llama_experiment.py                    # Full 30-pair experiment
    python run_llama_experiment.py --smoke             # Single-pair smoke test
    python run_llama_experiment.py --analyze-only      # Re-analyze existing results

Prerequisites:
    - Meta Llama access approved
    - pip install circuit-tracer torch transformers scipy numpy
    - GPU with >= 4GB VRAM (RTX 4070 Laptop = 8.6GB, plenty)
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

MODEL_NAME = "meta-llama/Llama-3.2-1B"
TRANSCODER_SET = "llama"  # mntss/transcoder-Llama-3.2-1B
N_LAYERS = 16
BAND_BOUNDARIES = (5, 11)  # early/mid/late split points

GEMMA_REFERENCE = {
    "surface_mean": 8.3592,
    "insight_mean": 8.1025,
    "delta": 0.2568,
    "ratio": 1.0302,
    "p_value": 0.000051,
    "n_pairs": 30,
}

CONVERGENCE_WINDOW = 0.15  # |llama_ratio - 1.03| < 0.15
SIGNIFICANCE_THRESHOLD = 0.05

BASE_DIR = Path(__file__).parent
PAIRS_FILE = BASE_DIR / ".." / "research" / "experiment_12_power_pairs.json"
RESULTS_FILE = BASE_DIR / "llama_experiment_results.json"
ANALYSIS_FILE = BASE_DIR / "llama_experiment_analysis.json"
REPORT_FILE = BASE_DIR / "LLAMA_EXPERIMENT_REPORT.md"

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
                # Handle two formats:
                # Format A: {topic, surface: "prompt", surface_target: "target", insight: "prompt", insight_target: "target"}
                # Format B: {id, surface: {prompt, target}, insight: {prompt, target}}
                s = p.get("surface", "")
                i = p.get("insight", "")
                if isinstance(s, str):
                    # Format A
                    normalized.append({
                        "id": p.get("topic", p.get("id", p.get("name", "unknown"))),
                        "surface": {"prompt": s, "target": p.get("surface_target", "")},
                        "insight": {"prompt": i, "target": p.get("insight_target", "")},
                    })
                else:
                    # Format B
                    normalized.append({
                        "id": p.get("id", p.get("topic", p.get("name", "unknown"))),
                        "surface": {"prompt": s.get("prompt", ""), "target": s.get("target", "")},
                        "insight": {"prompt": i.get("prompt", ""), "target": i.get("target", "")},
                    })
            return normalized
    raise FileNotFoundError(f"Pairs file not found at {PAIRS_FILE}")


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model():
    """Load Llama-3.2-1B with circuit-tracer transcoders."""
    import torch
    from circuit_tracer import ReplacementModel

    print(f"[MODEL] Loading {MODEL_NAME}...")
    print(f"[MODEL] Transcoder set: {TRANSCODER_SET}")
    print(f"[MODEL] Layers: {N_LAYERS}, Bands: {BAND_BOUNDARIES}")

    model = ReplacementModel.from_pretrained(
        MODEL_NAME,
        TRANSCODER_SET,
        dtype=torch.bfloat16,
        backend="transformerlens",
    )

    print(f"[MODEL] Loaded successfully. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


# ── Depth Scoring ────────────────────────────────────────────────────────────

def score_depth(model, prompt, target, max_features=500):
    """
    Score depth of a prompt+target using circuit-tracer attribution.
    Returns mean activated layer index and metadata.
    """
    import torch
    from circuit_tracer import attribute

    full_prompt = prompt + target

    try:
        graph = attribute(
            full_prompt,
            model,
            attribution_targets=[target],
            max_feature_nodes=max_features,
            batch_size=16,
        )

        layers = graph.active_features[:, 0].float().cpu().numpy()

        if len(layers) == 0:
            print(f"[WARN] No features activated for: {prompt[:50]}...")
            return None

        depth = float(np.mean(layers))
        n_features = len(layers)

        # Layer distribution
        layer_counts = {}
        for l in layers:
            k = int(l)
            layer_counts[k] = layer_counts.get(k, 0) + 1

        # Shannon entropy
        probs = np.array(list(layer_counts.values())) / n_features
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Band ratios
        early = sum(1 for l in layers if l < BAND_BOUNDARIES[0])
        mid = sum(1 for l in layers if BAND_BOUNDARIES[0] <= l < BAND_BOUNDARIES[1])
        late = sum(1 for l in layers if l >= BAND_BOUNDARIES[1])

        return {
            "depth": depth,
            "n_features": n_features,
            "entropy": entropy,
            "early_ratio": early / n_features,
            "mid_ratio": mid / n_features,
            "late_ratio": late / n_features,
            "layer_profile": layer_counts,
        }

    except Exception as e:
        print(f"[ERROR] Attribution failed: {e}")
        return None


# ── Experiment Runner ────────────────────────────────────────────────────────

def run_experiment(model, pairs, smoke_test=False):
    """Run the full 30-pair experiment (or single pair for smoke test)."""
    if smoke_test:
        pairs = pairs[:1]
        print("[SMOKE TEST] Running pair #1 only")

    results = []
    n = len(pairs)

    for i, pair in enumerate(pairs):
        pair_id = pair.get("id", pair.get("name", f"pair_{i}"))
        print(f"\n[PAIR {i+1}/{n}] {pair_id}")

        # Extract prompts and targets
        surface_prompt = pair["surface"]["prompt"]
        surface_target = pair["surface"]["target"]
        insight_prompt = pair["insight"]["prompt"]
        insight_target = pair["insight"]["target"]

        print(f"  Surface: ...{surface_prompt[-40:]} → '{surface_target}'")
        print(f"  Insight: ...{insight_prompt[-40:]} → '{insight_target}'")

        # Score surface
        t0 = time.time()
        surface_result = score_depth(model, surface_prompt, surface_target)
        t_surface = time.time() - t0

        # Score insight
        t0 = time.time()
        insight_result = score_depth(model, insight_prompt, insight_target)
        t_insight = time.time() - t0

        if surface_result is None or insight_result is None:
            print(f"  [SKIP] Attribution failed for pair {pair_id}")
            continue

        delta = surface_result["depth"] - insight_result["depth"]
        direction = "surface_deeper" if delta > 0 else "insight_deeper"

        result = {
            "pair_id": pair_id,
            "pair_index": i,
            "surface_depth": surface_result["depth"],
            "surface_features": surface_result["n_features"],
            "surface_entropy": surface_result["entropy"],
            "insight_depth": insight_result["depth"],
            "insight_features": insight_result["n_features"],
            "insight_entropy": insight_result["entropy"],
            "delta": delta,
            "direction": direction,
            "time_surface_s": round(t_surface, 1),
            "time_insight_s": round(t_insight, 1),
        }

        results.append(result)
        print(f"  Surface: {surface_result['depth']:.4f} ({surface_result['n_features']} features)")
        print(f"  Insight: {insight_result['depth']:.4f} ({insight_result['n_features']} features)")
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
    llama_ratio = surface_mean / insight_mean if insight_mean > 0 else 0
    ratio_diff = abs(llama_ratio - GEMMA_REFERENCE["ratio"])
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
        verdict = "NO REPLICATION — effect not significant on Llama"
    elif not converges:
        tier = 3  # Replicates but diverges
        verdict = f"REPLICATES BUT DIVERGES — ratio {llama_ratio:.4f} outside convergence window"
    elif not loo_all_significant:
        tier = 2  # Converges but not fully robust
        verdict = f"CONVERGES (ratio {llama_ratio:.4f}) — LOO not fully robust"
    else:
        tier = 2  # Full convergence
        verdict = f"FULL CONVERGENCE — ratio {llama_ratio:.4f} ≈ 1.03. FATHOM CONSTANT CONFIRMED."

    analysis = {
        "model": MODEL_NAME,
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
        "llama_ratio": llama_ratio,
        "gemma_ratio": GEMMA_REFERENCE["ratio"],
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
        "fathom_constant_estimate": (llama_ratio + GEMMA_REFERENCE["ratio"]) / 2 if converges else None,
    }

    return analysis


# ── Report Generation ────────────────────────────────────────────────────────

def generate_report(results, analysis):
    """Generate human-readable markdown report."""
    if not analysis:
        return "# Llama Experiment Report\n\nInsufficient data for analysis."

    lines = [
        "# FATHOM CONSTANT REPLICATION — Llama-3.2-1B Results",
        "",
        f"**Date**: {analysis['timestamp'][:10]}",
        f"**Model**: {analysis['model']}",
        f"**Pairs**: {analysis['n_pairs']}",
        f"**Tier**: {analysis['tier']}",
        "",
        f"## VERDICT: {analysis['verdict']}",
        "",
        "---",
        "",
        "## Descriptive Statistics",
        "",
        f"| Metric | Llama-3.2-1B | Gemma-2-2B (ref) |",
        f"|--------|-------------|-----------------|",
        f"| Surface mean | {analysis['surface_mean']:.4f} | {GEMMA_REFERENCE['surface_mean']:.4f} |",
        f"| Insight mean | {analysis['insight_mean']:.4f} | {GEMMA_REFERENCE['insight_mean']:.4f} |",
        f"| Delta | {analysis['delta_mean']:+.4f} | {GEMMA_REFERENCE['delta']:+.4f} |",
        f"| Ratio | {analysis['llama_ratio']:.4f} | {GEMMA_REFERENCE['ratio']:.4f} |",
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
        f"- Llama ratio: {analysis['llama_ratio']:.4f}",
        f"- Gemma ratio: {analysis['gemma_ratio']:.4f}",
        f"- Difference: {analysis['ratio_difference']:.4f}",
        f"- Within window (< {analysis['convergence_window']}): {'YES' if analysis['converges'] else 'NO'}",
    ]

    if analysis['converges'] and analysis['fathom_constant_estimate']:
        lines.append(f"- **Fathom Constant (2-model estimate)**: {analysis['fathom_constant_estimate']:.4f}")

    lines += [
        "",
        "## Leave-One-Out Robustness",
        "",
        f"- All 30 subsets significant: {'YES' if analysis['loo_all_significant'] else 'NO'}",
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
        f"*Generated by run_llama_experiment.py at {analysis['timestamp']}*",
    ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fathom Constant Replication — Llama-3.2-1B")
    parser.add_argument("--smoke", action="store_true", help="Run single-pair smoke test only")
    parser.add_argument("--analyze-only", action="store_true", help="Re-analyze existing results")
    parser.add_argument("--max-features", type=int, default=500, help="Max feature nodes per attribution")
    args = parser.parse_args()

    print("=" * 60)
    print("  FATHOM CONSTANT REPLICATION — Llama-3.2-1B")
    print("  Pre-registered experiment protocol")
    print("=" * 60)
    print()

    # Load pairs
    pairs = load_pairs()
    print(f"[PAIRS] Loaded {len(pairs)} calibrated pairs")

    if args.analyze_only:
        # Re-analyze existing results
        if not RESULTS_FILE.exists():
            print("[ERROR] No results file found. Run experiment first.")
            sys.exit(1)
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"[ANALYZE] Loaded {len(results)} existing results")
    else:
        # Load model and run experiment
        model = load_model()
        results = run_experiment(model, pairs, smoke_test=args.smoke)

        # Save raw results
        with open(RESULTS_FILE, "w") as f:
            json_dump(results, f)
        print(f"\n[SAVED] Raw results → {RESULTS_FILE}")

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
        # Save analysis
        with open(ANALYSIS_FILE, "w") as f:
            json_dump(analysis, f)
        print(f"[SAVED] Analysis → {ANALYSIS_FILE}")

        # Generate and save report
        report = generate_report(results, analysis)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[SAVED] Report → {REPORT_FILE}")

        # Print verdict
        print(f"\n{'=' * 60}")
        print(f"  TIER {analysis['tier']}: {analysis['verdict']}")
        print(f"{'=' * 60}")
        print(f"  p = {analysis['p_value']:.6f}")
        print(f"  delta = {analysis['delta_mean']:+.4f}")
        print(f"  ratio = {analysis['llama_ratio']:.4f} (Gemma: {GEMMA_REFERENCE['ratio']:.4f})")
        if analysis['converges']:
            print(f"  FATHOM CONSTANT = {analysis['fathom_constant_estimate']:.4f}")
        print(f"{'=' * 60}")

        if analysis['tier'] == 2:
            print("\n  *** CONVERGENCE DETECTED ***")
            print("  *** STOP EVERYTHING. WRITE PAPER 2. ***\n")


if __name__ == "__main__":
    main()
