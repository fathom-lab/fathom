"""
FATHOM CONSTANT — Cross-Architecture Universality Test
========================================================

Tests whether the surface/insight depth ratio converges across different
transformer architectures, using the SAME 30 calibrated pairs from the
paper's power replication (p=0.000051).

Supported architectures via circuit-tracer transcoders:
  - google/gemma-2-2b   (transcoder_set="gemma")
  - meta-llama/Llama-3.2-1B (transcoder_set="llama")

NOTE: GPT-2 models (gpt2, gpt2-xl) are NOT supported by circuit-tracer.
The library requires transcoders (not SAEs). Only "gemma" and "llama"
transcoder sets are currently available. The jbloom/GPT2-Small-SAEs-Reformatted
package provides SAEs, not transcoders — incompatible with the attribution
pipeline.

If the ratio converges across architectures (within ±0.2), the Fathom
Constant is a universal property of how transformers separate retrieval
from reasoning.

Usage: python fathom_constant_final.py
"""

import gc
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


# ── The 30 calibrated paper pairs ──────────────────────────────────────
# From reproduce_paper.py — the power replication set that produced
# delta=+0.257, p=0.000051, 83.3% directional.
# These are vocabulary-controlled, token-count matched, syntactically parallel.

PAPER_PAIRS = [
    {
        "topic": "earthquakes",
        "surface": "Earthquakes happen when tectonic plates shift and the ground shakes from the energy they",
        "surface_target": "release",
        "insight": "Earthquakes occur when stress along fault lines exceeds friction and the sudden slip radiates seismic",
        "insight_target": "waves",
    },
    {
        "topic": "metabolism",
        "surface": "Metabolism converts food into energy and your body burns calories to keep you",
        "surface_target": "alive",
        "insight": "Metabolism converts nutrients into ATP through oxidation and the rate adapts to the cell's current demand for",
        "insight_target": "energy",
    },
    {
        "topic": "ocean_currents",
        "surface": "Ocean currents move water around the globe and they influence temperatures near coastal",
        "surface_target": "regions",
        "insight": "Ocean currents redistribute heat because density differences between warm and cold water masses drive the",
        "insight_target": "flow",
    },
    {
        "topic": "immune_response",
        "surface": "The immune system attacks bacteria and white blood cells destroy pathogens inside the",
        "surface_target": "body",
        "insight": "The immune system activates when antigens bind to receptors and triggers a cascade that amplifies the response to",
        "insight_target": "infection",
    },
    {
        "topic": "dreams",
        "surface": "Dreams happen during sleep and the brain creates vivid images while you",
        "surface_target": "sleep",
        "insight": "Dreams emerge during REM sleep when the brain replays memories and consolidates emotional experience into long-term",
        "insight_target": "storage",
    },
    {
        "topic": "volcanoes",
        "surface": "Volcanoes erupt when magma rises to the surface and lava flows from the",
        "surface_target": "top",
        "insight": "Volcanoes erupt when pressure from dissolved gases in magma exceeds the strength of the overlying",
        "insight_target": "rock",
    },
    {
        "topic": "blood_pressure",
        "surface": "Blood pressure measures the force of blood against artery walls and high levels can damage the",
        "surface_target": "heart",
        "insight": "Blood pressure rises when blood vessels constrict or fluid volume increases the load against vascular",
        "insight_target": "resistance",
    },
    {
        "topic": "fermentation",
        "surface": "Fermentation converts sugar into alcohol and yeast releases carbon dioxide as",
        "surface_target": "waste",
        "insight": "Fermentation converts glucose to ethanol when yeast depletes oxygen and switches to anaerobic energy",
        "insight_target": "production",
    },
    {
        "topic": "hurricanes",
        "surface": "Hurricanes form over warm ocean water and strong winds spiral around a central",
        "surface_target": "eye",
        "insight": "Hurricanes intensify when warm ocean water evaporates rapidly and rising air creates low pressure that sustains",
        "insight_target": "rotation",
    },
    {
        "topic": "drug_resistance",
        "surface": "Drug resistance develops when bacteria survive treatment and the survivors pass on their",
        "surface_target": "traits",
        "insight": "Drug resistance develops when antibiotics apply selective pressure and only bacteria with resistance mutations survive to",
        "insight_target": "divide",
    },
    {
        "topic": "coral_reefs",
        "surface": "Coral reefs are built by tiny animals and provide habitat for thousands of ocean",
        "surface_target": "species",
        "insight": "Coral reefs bleach when rising water temperatures cause the symbiotic algae that supply nutrients to leave the",
        "insight_target": "coral",
    },
    {
        "topic": "black_holes",
        "surface": "Black holes are regions in space where gravity is so strong that light cannot",
        "surface_target": "escape",
        "insight": "Black holes form when massive stars exhaust their fuel and gravitational collapse compresses matter past the",
        "insight_target": "limit",
    },
    {
        "topic": "taxation",
        "surface": "Taxes are collected by governments and the revenue funds public services like schools and",
        "surface_target": "roads",
        "insight": "Taxes reduce after-tax returns on economic activity and the behavioral response depends on how sensitive supply and demand are to the",
        "insight_target": "price",
    },
    {
        "topic": "empathy",
        "surface": "Empathy is the ability to understand how others feel and it helps people connect with",
        "surface_target": "others",
        "insight": "Empathy activates neural circuits that simulate another person's experience and allows prediction of their emotional",
        "insight_target": "response",
    },
    {
        "topic": "plate_tectonics",
        "surface": "Tectonic plates move slowly across the mantle and their collisions build mountain",
        "surface_target": "ranges",
        "insight": "Tectonic plates move because convection currents in the mantle drag them and plate boundaries concentrate geological",
        "insight_target": "activity",
    },
    {
        "topic": "obesity",
        "surface": "Obesity develops when calorie intake exceeds energy expenditure and excess energy is stored as",
        "surface_target": "fat",
        "insight": "Obesity develops when hormonal signals regulating appetite are disrupted and the body defends an elevated weight",
        "insight_target": "level",
    },
    {
        "topic": "leadership",
        "surface": "Effective leaders inspire teams and communicate goals that motivate people to",
        "surface_target": "perform",
        "insight": "Effective leaders reduce uncertainty by modeling expected behavior and creating conditions where trust develops",
        "insight_target": "naturally",
    },
    {
        "topic": "bacteria_growth",
        "surface": "Bacteria grow by dividing in two and populations double rapidly under favorable",
        "surface_target": "conditions",
        "insight": "Bacteria grow exponentially when resources are abundant and growth slows as waste accumulates and nutrients become",
        "insight_target": "rare",
    },
    {
        "topic": "cryptocurrency",
        "surface": "Cryptocurrency is a digital currency that uses encryption to secure transactions without a central",
        "surface_target": "bank",
        "insight": "Cryptocurrency maintains trust without intermediaries by requiring consensus across nodes that makes fraudulent transactions computationally",
        "insight_target": "hard",
    },
    {
        "topic": "social_norms",
        "surface": "Social norms are unwritten rules about behavior and people follow them to fit in with their",
        "surface_target": "group",
        "insight": "Social norms persist because violating them triggers social costs and conformity is individually rational even when the norm is",
        "insight_target": "wrong",
    },
    {
        "topic": "nuclear_energy",
        "surface": "Nuclear energy is released by splitting atoms and the heat produced drives turbines to generate",
        "surface_target": "electricity",
        "insight": "Nuclear fission releases energy because the product nuclei have lower binding energy per nucleon and the mass difference converts to",
        "insight_target": "heat",
    },
    {
        "topic": "monopoly",
        "surface": "Monopolies can charge higher prices because customers have no alternative supplier to turn",
        "surface_target": "to",
        "insight": "Monopolies price above marginal cost because profit maximization sets marginal revenue equal to marginal cost and captures consumer",
        "insight_target": "value",
    },
    {
        "topic": "trauma",
        "surface": "Trauma is a psychological response to severe stress and it can affect mental health for",
        "surface_target": "years",
        "insight": "Trauma dysregulates the stress response and encodes threatening experiences as fragmented sensory memories that activate fear without",
        "insight_target": "context",
    },
    {
        "topic": "bird_migration",
        "surface": "Birds migrate seasonally to warmer regions when temperatures drop and food becomes",
        "surface_target": "rare",
        "insight": "Birds migrate because changes in day length trigger hormonal shifts that coordinate departure with the conditions birds will find upon",
        "insight_target": "arrival",
    },
    {
        "topic": "stock_bubbles",
        "surface": "Stock bubbles form when prices rise far above true value and eventually the market",
        "surface_target": "falls",
        "insight": "Stock bubbles inflate when rising prices attract momentum buyers whose demand drives prices higher until the pool of buyers is",
        "insight_target": "gone",
    },
    {
        "topic": "enzyme",
        "surface": "Enzymes are proteins that speed up reactions and each enzyme has an active",
        "surface_target": "site",
        "insight": "Enzymes accelerate reactions by binding the substrate and lowering the energy needed to form",
        "insight_target": "products",
    },
    {
        "topic": "language_acquisition",
        "surface": "Children learn language through exposure and practice and fluency develops through repeated use of vocabulary and",
        "surface_target": "grammar",
        "insight": "Children acquire language because implicit statistical learning mechanisms extract patterns from input before conscious rules are ever",
        "insight_target": "taught",
    },
    {
        "topic": "opioid_dependence",
        "surface": "Opioids relieve pain by binding to receptors in the brain and the relief can make people want",
        "surface_target": "more",
        "insight": "Opioids create dependence by downregulating endorphin production so users need increasing doses to achieve the same",
        "insight_target": "effect",
    },
    {
        "topic": "ocean_acidification",
        "surface": "Ocean acidification occurs when the sea absorbs carbon dioxide and the pH begins to",
        "surface_target": "fall",
        "insight": "Ocean acidification occurs when dissolved CO2 forms carbonic acid that reduces the carbonate ions marine animals need to build their",
        "insight_target": "shells",
    },
    {
        "topic": "supply_chains",
        "surface": "Supply chains move goods from manufacturers to consumers and disruptions can cause shortages at the",
        "surface_target": "store",
        "insight": "Supply chains amplify small demand shocks because each upstream supplier over-orders to buffer uncertainty and the swings compound into",
        "insight_target": "chaos",
    },
]


@dataclass
class ModelResult:
    """Results from testing one model."""
    model_name: str
    transcoder_set: str
    n_pairs: int
    surface_mean: float
    insight_mean: float
    delta: float
    ratio: float  # surface_mean / insight_mean
    pct_expected_direction: float
    depths_surface: List[float]
    depths_insight: List[float]
    compute_time_s: float
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "transcoder_set": self.transcoder_set,
            "n_pairs": self.n_pairs,
            "surface_mean": round(self.surface_mean, 4),
            "insight_mean": round(self.insight_mean, 4),
            "delta": round(self.delta, 4),
            "ratio": round(self.ratio, 4),
            "pct_expected_direction": round(self.pct_expected_direction, 4),
            "compute_time_s": round(self.compute_time_s, 2),
            "error": self.error,
        }


# ── Supported model configurations ─────────────────────────────────────
# circuit-tracer only supports models with matching transcoder sets.
# As of March 2026, only two transcoder sets are available:
#   "gemma" -> mwhanna/gemma-scope-transcoders (for gemma-2-2b)
#   "llama" -> mntss/transcoder-Llama-3.2-1B (for Llama-3.2-1B)
#
# GPT-2 is NOT supported: jbloom/GPT2-Small-SAEs-Reformatted provides
# SAEs (sparse autoencoders), not transcoders. circuit-tracer's
# ReplacementModel.from_pretrained() requires transcoders with a
# config.yaml and specific weight format. SAEs have a different API.

MODELS = [
    {
        "name": "google/gemma-2-2b",
        "transcoder_set": "gemma",
        "n_layers": 26,
        "max_features": 150,
        "note": "Primary model from paper",
    },
    {
        "name": "meta-llama/Llama-3.2-1B",
        "transcoder_set": "llama",
        "n_layers": 16,
        "max_features": 150,
        "note": "Cross-architecture validation (different family, different scale)",
    },
]


def test_model(
    model_name: str,
    transcoder_set: str,
    pairs: List[dict],
    max_feature_nodes: int = 150,
) -> ModelResult:
    """Score all pairs on a single model.

    Args:
        model_name: HuggingFace model ID.
        transcoder_set: circuit-tracer transcoder set name.
        pairs: List of pair dicts with surface/insight/targets.
        max_feature_nodes: Feature extraction limit.

    Returns:
        ModelResult with aggregate statistics.
    """
    import torch
    from depth_scorer import DepthScorer

    t0 = time.time()
    print(f"\n[Constant] Loading {model_name} (transcoder_set={transcoder_set})...")

    try:
        scorer = DepthScorer(
            model_name=model_name,
            model_type=transcoder_set,
            max_feature_nodes=max_feature_nodes,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [LOAD ERROR] {model_name}:\n{tb}")
        return ModelResult(
            model_name=model_name, transcoder_set=transcoder_set,
            n_pairs=0,
            surface_mean=0.0, insight_mean=0.0, delta=0.0, ratio=0.0,
            pct_expected_direction=0.0, depths_surface=[], depths_insight=[],
            compute_time_s=time.time() - t0,
            error=str(e), error_traceback=tb,
        )

    surfaces = []
    insights = []

    for i, pair in enumerate(pairs):
        topic = pair["topic"]
        try:
            r_surface = scorer.score(pair["surface"], target=pair.get("surface_target"))
            torch.cuda.empty_cache()
            gc.collect()

            r_insight = scorer.score(pair["insight"], target=pair.get("insight_target"))
            torch.cuda.empty_cache()
            gc.collect()

            if not math.isnan(r_surface.depth_score) and not math.isnan(r_insight.depth_score):
                surfaces.append(r_surface.depth_score)
                insights.append(r_insight.depth_score)

                delta = r_surface.depth_score - r_insight.depth_score
                direction = "+" if delta > 0 else "-"
                print(f"  [{i+1}/{len(pairs)}] {topic:<22} s={r_surface.depth_score:.3f} "
                      f"i={r_insight.depth_score:.3f} delta={delta:+.3f} {direction}")

        except Exception as e:
            print(f"  [{i+1}/{len(pairs)}] {topic:<22} FAILED: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    # Clean up model
    del scorer
    torch.cuda.empty_cache()
    gc.collect()

    if not surfaces:
        return ModelResult(
            model_name=model_name, transcoder_set=transcoder_set,
            n_pairs=0,
            surface_mean=0.0, insight_mean=0.0, delta=0.0, ratio=0.0,
            pct_expected_direction=0.0, depths_surface=[], depths_insight=[],
            compute_time_s=time.time() - t0, error="No valid pairs",
        )

    s_mean = float(np.mean(surfaces))
    i_mean = float(np.mean(insights))
    delta = s_mean - i_mean
    ratio = s_mean / i_mean if i_mean > 0 else 0.0
    pct_expected = sum(1 for s, i in zip(surfaces, insights) if s > i) / len(surfaces)

    return ModelResult(
        model_name=model_name,
        transcoder_set=transcoder_set,
        n_pairs=len(surfaces),
        surface_mean=s_mean,
        insight_mean=i_mean,
        delta=delta,
        ratio=ratio,
        pct_expected_direction=pct_expected,
        depths_surface=surfaces,
        depths_insight=insights,
        compute_time_s=time.time() - t0,
    )


def compute_fathom_constant(results: List[ModelResult]) -> dict:
    """Compute the Fathom Constant from cross-model results.

    Args:
        results: ModelResult from each model.

    Returns:
        {"constant": float, "convergent": bool, "spread": float, "models": list}
    """
    valid = [r for r in results if r.n_pairs > 0 and r.error is None]
    if not valid:
        return {"constant": 0.0, "convergent": False, "spread": 0.0,
                "n_models": 0, "ratios": {}, "total_pairs": 0}

    ratios = [r.ratio for r in valid]
    weighted_sum = sum(r.ratio * r.n_pairs for r in valid)
    total_pairs = sum(r.n_pairs for r in valid)
    weighted_mean = weighted_sum / total_pairs if total_pairs > 0 else 0.0

    spread = max(ratios) - min(ratios) if len(ratios) > 1 else 0.0
    # Convergence requires ≥2 models AND spread < 0.2
    convergent = len(valid) >= 2 and spread < 0.2

    return {
        "constant": round(weighted_mean, 4),
        "convergent": convergent,
        "spread": round(spread, 4),
        "n_models": len(valid),
        "ratios": {r.model_name: round(r.ratio, 4) for r in valid},
        "total_pairs": total_pairs,
    }


def main():
    """Run the full cross-architecture test."""
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    t_start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("FATHOM CONSTANT — CROSS-ARCHITECTURE UNIVERSALITY TEST")
    print(f"Started: {timestamp}")
    print("=" * 70)

    # Use the paper's 30 calibrated pairs (p=0.000051)
    pairs = PAPER_PAIRS
    print(f"\nPairs: {len(pairs)} (paper power replication set)")
    print(f"Models: {len(MODELS)}")
    for m in MODELS:
        print(f"  - {m['name']} (transcoder: {m['transcoder_set']}) — {m['note']}")

    print(f"\nNOTE: GPT-2 models not supported by circuit-tracer (requires")
    print(f"transcoders, not SAEs). Using gemma + llama for cross-arch test.")

    results = []
    for model_cfg in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_cfg['name']}")
        print(f"{'='*60}")
        result = test_model(
            model_cfg["name"],
            model_cfg["transcoder_set"],
            pairs,
            max_feature_nodes=model_cfg["max_features"],
        )
        results.append(result)
        if result.error:
            print(f"  RESULT: ERROR — {result.error}")
        else:
            print(f"  RESULT: ratio={result.ratio:.4f}, delta={result.delta:+.4f}, "
                  f"n={result.n_pairs}, pct_dir={result.pct_expected_direction:.1%}, "
                  f"time={result.compute_time_s:.1f}s")

    # Compute the constant
    constant_result = compute_fathom_constant(results)
    total_time = time.time() - t_start
    n_successful = constant_result["n_models"]

    print("\n" + "=" * 70)
    print("FATHOM CONSTANT RESULT")
    print("=" * 70)
    print(f"Successful models: {n_successful}/{len(MODELS)}")
    print(f"Weighted mean ratio: {constant_result['constant']:.4f}")
    if n_successful >= 2:
        print(f"Convergent (spread < 0.2): {constant_result['convergent']}")
        print(f"Spread: {constant_result['spread']:.4f}")
    else:
        print(f"Cross-architecture validation: PENDING (need >=2 models)")
    for model, ratio in constant_result.get("ratios", {}).items():
        print(f"  {model}: {ratio:.4f}")
    print(f"Total time: {total_time:.1f}s")

    # ── Write report ────────────────────────────────────────────────────
    report = f"""# FATHOM CONSTANT — Cross-Architecture Universality Test

**Date:** {timestamp}
**Duration:** {total_time:.1f}s
**Pairs:** {len(pairs)} (paper power replication set, p=0.000051)

## Hypothesis

If the surface/insight depth ratio converges across different transformer
architectures (within ±0.2), the Fathom Constant is a universal property
of how transformers separate retrieval from reasoning.

## Architecture Compatibility

circuit-tracer requires **transcoders** (not SAEs) for each model architecture.
Currently supported transcoder sets:

| Transcoder Set | Models | Source |
|---------------|--------|--------|
| `gemma` | google/gemma-2-2b | mwhanna/gemma-scope-transcoders |
| `llama` | meta-llama/Llama-3.2-1B | mntss/transcoder-Llama-3.2-1B |

**Not supported:** GPT-2 (gpt2, gpt2-xl). The package `jbloom/GPT2-Small-SAEs-Reformatted`
provides SAEs, not transcoders. These are incompatible with circuit-tracer's
`ReplacementModel.from_pretrained()` which requires transcoders with `config.yaml`.

## Models Tested

| Model | Transcoder | Pairs | Surface Mean | Insight Mean | Delta | Ratio | % Expected |
|-------|-----------|-------|-------------|-------------|-------|-------|-----------|
"""
    for r in results:
        if r.error:
            report += f"| {r.model_name} | {r.transcoder_set} | ERROR | — | — | — | — | — |\n"
        else:
            report += (f"| {r.model_name} | {r.transcoder_set} | {r.n_pairs} | "
                      f"{r.surface_mean:.4f} | {r.insight_mean:.4f} | "
                      f"{r.delta:+.4f} | {r.ratio:.4f} | "
                      f"{r.pct_expected_direction:.1%} |\n")

    report += f"""
## Result

| Metric | Value |
|--------|-------|
| **Candidate Fathom Constant (weighted mean)** | **{constant_result['constant']:.4f}** |
| Models successfully tested | {n_successful}/{len(MODELS)} |
"""

    if n_successful >= 2:
        report += f"""| Convergent (spread < 0.2) | {'**YES**' if constant_result['convergent'] else 'NO'} |
| Spread | {constant_result['spread']:.4f} |
| Total pairs scored | {constant_result['total_pairs']} |
"""
    else:
        report += f"""| Cross-architecture validation | **PENDING** (single model) |
| Total pairs scored | {constant_result['total_pairs']} |
"""

    report += "\n## Interpretation\n\n"

    if n_successful >= 2 and constant_result["convergent"]:
        report += (
            f"**THE FATHOM CONSTANT IS CONFIRMED.** The surface/insight depth ratio "
            f"converges across {n_successful} architectures within ±0.2. "
            f"Value: **{constant_result['constant']:.4f}**. Spread: {constant_result['spread']:.4f}. "
            f"This suggests a universal property of how transformer architectures "
            f"separate retrieval from reasoning.\n"
        )
    elif n_successful >= 2 and not constant_result["convergent"]:
        report += (
            f"The ratio did NOT converge within ±0.2 across {n_successful} architectures. "
            f"Spread: {constant_result['spread']:.4f}. "
            f"The depth effect may be architecture-specific rather than universal. "
            f"More investigation needed.\n"
        )
    elif n_successful == 1:
        only_model = list(constant_result["ratios"].keys())[0]
        only_ratio = list(constant_result["ratios"].values())[0]
        report += (
            f"**CANDIDATE VALUE: {only_ratio:.4f}** (single model: {only_model}). "
            f"Cross-architecture validation pending — need ≥2 successful models "
            f"to confirm universality. Cannot claim convergence with one data point.\n"
        )
    else:
        report += "No models completed successfully. Check error logs.\n"

    report += "\n## Per-Model Ratios\n\n"
    for model, ratio in constant_result.get("ratios", {}).items():
        report += f"- **{model}**: {ratio:.4f}\n"

    # Error details
    errors = [r for r in results if r.error]
    if errors:
        report += "\n## Errors\n\n"
        for r in errors:
            report += f"### {r.model_name}\n\n"
            report += f"**Error:** {r.error}\n\n"
            if r.error_traceback:
                report += f"```\n{r.error_traceback}\n```\n\n"

    report_path = PROJECT_ROOT / "FATHOM_CONSTANT_FINAL.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[OUTPUT] {report_path}")

    # Save raw data as JSON for reproducibility
    raw_data = {
        "timestamp": timestamp,
        "duration_s": round(total_time, 1),
        "n_pairs": len(pairs),
        "pair_source": "paper power replication set (p=0.000051)",
        "constant": constant_result,
        "models": [r.to_dict() for r in results],
    }
    json_path = PROJECT_ROOT / "fathom_constant_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
    print(f"[OUTPUT] {json_path}")

    # Verdict
    if n_successful >= 2 and constant_result["convergent"]:
        print(f"\n*** FATHOM CONSTANT = {constant_result['constant']:.4f} ***")
        print("Confirmed across architectures. Ready to add to depth_scorer.py.")
    elif n_successful == 1:
        v = list(constant_result["ratios"].values())[0]
        print(f"\n*** CANDIDATE: {v:.4f} (single model) ***")
        print("Cross-architecture validation pending.")
    else:
        print("\n*** NO VALID RESULTS ***")


if __name__ == "__main__":
    main()
