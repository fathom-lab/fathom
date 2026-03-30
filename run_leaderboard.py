#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
"""
FATHOM LEADERBOARD — Cross-Model Depth Comparison
====================================================

Scores the 30 calibrated pairs across multiple API models using the
black-box depth probe. Generates a leaderboard showing which models
process language most deeply.

Usage:
    python run_leaderboard.py --smoke           # 1 pair, 1 model
    python run_leaderboard.py --models openai   # all pairs, one provider
    python run_leaderboard.py                   # full run

Environment variables:
    OPENAI_API_KEY      — for GPT models
    ANTHROPIC_API_KEY   — for Claude models
    GOOGLE_API_KEY      — for Gemini models

Requirements:
    pip install openai anthropic google-generativeai
"""

import json
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from blackbox_scorer import BlackBoxDepthScorer

BASE_DIR = Path(__file__).parent
PAIRS_FILE = BASE_DIR / ".." / "research" / "experiment_12_power_pairs.json"
RESULTS_FILE = BASE_DIR / "leaderboard_results.json"
REPORT_FILE = BASE_DIR / "LEADERBOARD.md"
CHECKPOINT_FILE = BASE_DIR / "leaderboard_checkpoint.json"

# ── Model Registry ───────────────────────────────────────────────────────────

MODELS = [
    {"provider": "openai", "model": "gpt-4o", "name": "GPT-4o"},
    {"provider": "openai", "model": "gpt-4o-mini", "name": "GPT-4o Mini"},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "name": "Claude Sonnet"},
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "name": "Claude Haiku"},
    {"provider": "google", "model": "gemini-2.0-flash", "name": "Gemini Flash"},
]

SYSTEM_PROMPT = "Continue the following text naturally. Explain your reasoning in a short paragraph (3-5 sentences). Do not add meta-commentary."
RATE_LIMIT_DELAY = 1.5  # seconds between API calls


# ── Load Pairs ───────────────────────────────────────────────────────────────

def load_pairs():
    """Load the 30 calibrated pairs."""
    for candidate in [PAIRS_FILE, BASE_DIR / "experiment_12_power_pairs.json"]:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            if isinstance(data, dict) and "pairs" in data:
                return data["pairs"]
            return data
    raise FileNotFoundError("Pairs file not found")


# ── Checkpoint ───────────────────────────────────────────────────────────────

def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(data: Dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Score a Model ────────────────────────────────────────────────────────────

def score_model(
    scorer: BlackBoxDepthScorer,
    model_config: Dict,
    pairs: List[Dict],
    checkpoint: Dict,
    smoke_test: bool = False,
) -> List[Dict]:
    """Score all pairs for a single model."""
    provider = model_config["provider"]
    model = model_config["model"]
    name = model_config["name"]

    api_key = os.environ.get(f"{provider.upper()}_API_KEY", "")
    if not api_key:
        print(f"[SKIP] {name} — no {provider.upper()}_API_KEY set")
        return []

    if smoke_test:
        pairs = pairs[:1]

    results = []
    ck_key = f"{provider}:{model}"
    existing = checkpoint.get(ck_key, {})

    for i, pair in enumerate(pairs):
        topic = pair.get("topic", pair.get("id", f"pair_{i}"))

        # Check checkpoint
        if topic in existing:
            results.append(existing[topic])
            continue

        surface_text = pair.get("surface", "")
        insight_text = pair.get("insight", "")
        if isinstance(surface_text, dict):
            surface_text = surface_text.get("prompt", "")
        if isinstance(insight_text, dict):
            insight_text = insight_text.get("prompt", "")

        print(f"  [{name}] Pair {i+1}/{len(pairs)}: {topic}")

        try:
            # Get model outputs
            surface_output = scorer._call_api(
                surface_text, provider, model, api_key, SYSTEM_PROMPT
            )
            time.sleep(RATE_LIMIT_DELAY)

            insight_output = scorer._call_api(
                insight_text, provider, model, api_key, SYSTEM_PROMPT
            )
            time.sleep(RATE_LIMIT_DELAY)

            # Score the PROMPT text (calibrated) — this is what the probe was trained on
            surface_score = scorer.score(surface_text)
            insight_score = scorer.score(insight_text)

            # Also score the OUTPUT text (uncalibrated but interesting)
            surface_output_score = scorer.score(surface_output)
            insight_output_score = scorer.score(insight_output)

            result = {
                "topic": topic,
                "model": model,
                "provider": provider,
                # Prompt-based scores (calibrated)
                "prompt_surface_depth": surface_score.predicted_depth,
                "prompt_insight_depth": insight_score.predicted_depth,
                "prompt_delta": round(surface_score.predicted_depth - insight_score.predicted_depth, 4),
                # Output-based scores (estimated)
                "output_surface_depth": surface_output_score.predicted_depth,
                "output_insight_depth": insight_output_score.predicted_depth,
                "output_delta": round(surface_output_score.predicted_depth - insight_output_score.predicted_depth, 4),
                "output_surface_grade": surface_output_score.grade,
                "output_insight_grade": insight_output_score.grade,
                # Raw outputs (truncated)
                "surface_output_preview": surface_output[:100],
                "insight_output_preview": insight_output[:100],
            }

            results.append(result)

            # Save checkpoint
            if ck_key not in checkpoint:
                checkpoint[ck_key] = {}
            checkpoint[ck_key][topic] = result
            save_checkpoint(checkpoint)

        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

    return results


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_model(results: List[Dict]) -> Dict:
    """Compute aggregate stats for one model."""
    if not results:
        return {}

    import numpy as np

    output_deltas = [r["output_delta"] for r in results]
    surface_depths = [r["output_surface_depth"] for r in results]
    insight_depths = [r["output_insight_depth"] for r in results]

    mean_delta = float(np.mean(output_deltas))
    mean_surface = float(np.mean(surface_depths))
    mean_insight = float(np.mean(insight_depths))
    ratio = mean_surface / mean_insight if mean_insight > 0 else 0
    pct_expected = sum(1 for d in output_deltas if d > 0) / len(output_deltas) * 100

    return {
        "model": results[0]["model"],
        "provider": results[0]["provider"],
        "n_pairs": len(results),
        "mean_surface_depth": round(mean_surface, 4),
        "mean_insight_depth": round(mean_insight, 4),
        "mean_delta": round(mean_delta, 4),
        "ratio": round(ratio, 4),
        "pct_expected_direction": round(pct_expected, 1),
    }


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report(all_analyses: List[Dict]) -> str:
    """Generate markdown leaderboard report."""
    lines = [
        "# FATHOM LEADERBOARD — Cross-Model Depth Comparison",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Method**: Black-box depth probe (text features only)",
        f"**Calibration**: Gemma-2-2B + Llama-3.2-1B white-box measurements",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | Mean Delta | Ratio | % Expected | Pairs |",
        "|------|-------|-----------|-------|------------|-------|",
    ]

    # Sort by mean_delta descending (higher = better depth discrimination)
    sorted_analyses = sorted(all_analyses, key=lambda x: x.get("mean_delta", 0), reverse=True)

    for rank, a in enumerate(sorted_analyses, 1):
        lines.append(
            f"| {rank} | {a.get('model', '?')} | {a.get('mean_delta', 0):+.4f} | "
            f"{a.get('ratio', 0):.4f} | {a.get('pct_expected_direction', 0):.0f}% | "
            f"{a.get('n_pairs', 0)} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Mean Delta**: Average (surface_depth - insight_depth). Positive = model's outputs",
        "  are deeper on surface prompts, matching the Fathom Constant pattern.",
        "- **Ratio**: surface_depth / insight_depth. Values near 1.027 match the universal constant.",
        "- **% Expected**: Percentage of pairs where delta is in the expected direction.",
        "",
        "## White-Box Reference Values",
        "",
        "| Model | Ratio | p-value | Method |",
        "|-------|-------|---------|--------|",
        "| Gemma-2-2B | 1.0302 | 0.000051 | SAE attribution |",
        "| Llama-3.2-1B | 1.0242 | 0.00266 | SAE attribution |",
        "",
        "---",
        "",
        "*Generated by run_leaderboard.py | Fathom — US Patent Pending 64/020,489*",
    ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fathom Leaderboard — Cross-Model Depth Comparison")
    parser.add_argument("--smoke", action="store_true", help="1 pair, 1 model")
    parser.add_argument("--models", type=str, help="Filter by provider (openai, anthropic, google)")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and start fresh")
    args = parser.parse_args()

    print("=" * 60)
    print("  FATHOM LEADERBOARD")
    print("  Cross-Model Depth Comparison (Black-Box)")
    print("=" * 60)

    # Load pairs
    pairs = load_pairs()
    print(f"[DATA] {len(pairs)} calibrated pairs")

    # Init scorer
    scorer = BlackBoxDepthScorer()

    # Checkpoint
    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    checkpoint = load_checkpoint()

    # Filter models
    models = MODELS
    if args.models:
        models = [m for m in MODELS if m["provider"] == args.models]
    if args.smoke:
        models = models[:1]

    # Score each model
    all_results = {}
    all_analyses = []

    for model_config in models:
        name = model_config["name"]
        print(f"\n{'─' * 40}")
        print(f"  {name}")
        print(f"{'─' * 40}")

        results = score_model(scorer, model_config, pairs, checkpoint, smoke_test=args.smoke)

        if results:
            all_results[name] = results
            analysis = analyze_model(results)
            all_analyses.append(analysis)

            print(f"  Mean delta: {analysis.get('mean_delta', 0):+.4f}")
            print(f"  Ratio: {analysis.get('ratio', 0):.4f}")
            print(f"  Expected direction: {analysis.get('pct_expected_direction', 0):.0f}%")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "models": all_results,
            "analyses": all_analyses,
        }, f, indent=2)
    print(f"\n[SAVED] Results → {RESULTS_FILE}")

    # Generate report
    if all_analyses:
        report = generate_report(all_analyses)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[SAVED] Report → {REPORT_FILE}")

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  LEADERBOARD SUMMARY")
        print(f"{'=' * 60}")
        sorted_analyses = sorted(all_analyses, key=lambda x: x.get("mean_delta", 0), reverse=True)
        for rank, a in enumerate(sorted_analyses, 1):
            print(f"  #{rank} {a.get('model', '?'):30s} delta={a.get('mean_delta', 0):+.4f} ratio={a.get('ratio', 0):.4f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
