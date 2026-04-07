"""
FATHOM CONSTANT V2 — Cross-Architecture Universality Test
==========================================================

Extends the original constant test with Qwen3-0.6B as cross-architecture
validation. Qwen3 is a completely different architecture family from Gemma.

If the surface/insight depth ratio is ~1.03 on BOTH Gemma and Qwen3,
the Fathom Constant becomes convergent across architectures.

Architecture differences:
  - Gemma-2-2B: 26 layers, 2.6B params, Google (gemma-scope transcoders)
  - Qwen3-0.6B:  28 layers, 0.6B params, Alibaba (mwhanna qwen3 transcoders)

Usage:
    python fathom_constant_v2.py
    python fathom_constant_v2.py --qwen-only   # skip reloading gemma
"""

import argparse
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
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_V1_PATH = PROJECT_ROOT / "fathom_constant_results.json"
RESULTS_V2_PATH = PROJECT_ROOT / "fathom_constant_v2_results.json"


# ── Use the same 30 calibrated paper pairs ─────────────────────────────
# Vocabulary-controlled, token-matched pairs from the power replication.
# These produced: delta=+0.257, p=0.000051, 83.3% directional on gemma-2-2b.

PAPER_PAIRS = [
    {"topic": "earthquakes",
     "surface": "Earthquakes happen when tectonic plates shift and the ground shakes from the energy they",
     "surface_target": "release",
     "insight": "Earthquakes occur when stress along fault lines exceeds friction and the sudden slip radiates seismic",
     "insight_target": "waves"},
    {"topic": "metabolism",
     "surface": "Metabolism converts food into energy and your body burns calories to keep you",
     "surface_target": "alive",
     "insight": "Metabolism converts nutrients into ATP through oxidation and the rate adapts to the cell's current demand for",
     "insight_target": "energy"},
    {"topic": "gravity",
     "surface": "Gravity is the force that pulls objects toward the Earth and keeps us on the",
     "surface_target": "ground",
     "insight": "Gravity curves spacetime around mass so objects follow geodesic paths and acceleration is indistinguishable from the",
     "insight_target": "field"},
    {"topic": "vaccines",
     "surface": "Vaccines work by introducing a weakened pathogen so the immune system can learn to",
     "surface_target": "fight",
     "insight": "Vaccines work by training memory B cells to produce antibodies before exposure so the immune response is faster than",
     "insight_target": "infection"},
    {"topic": "evolution",
     "surface": "Evolution occurs when organisms with favorable traits survive and pass those traits to their",
     "surface_target": "offspring",
     "insight": "Evolution occurs because fitness differences compound across generations and each environment selects the variation that solves its current",
     "insight_target": "challenge"},
    {"topic": "inflation",
     "surface": "Inflation occurs when the money supply increases faster than the production of goods and",
     "surface_target": "services",
     "insight": "Inflation persists because wage increases raise production costs which raise prices which drive further wage demands in a self-reinforcing",
     "insight_target": "cycle"},
    {"topic": "photosynthesis",
     "surface": "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and",
     "surface_target": "oxygen",
     "insight": "Photosynthesis captures photons to split water molecules and uses the released electrons to fix atmospheric carbon into",
     "insight_target": "sugar"},
    {"topic": "consciousness",
     "surface": "Consciousness arises from brain activity and is associated with neural signals across regions of the",
     "surface_target": "cortex",
     "insight": "Consciousness may arise from the brain's ability to integrate information across modules because local processing alone cannot produce unified",
     "insight_target": "experience"},
    {"topic": "antibiotics",
     "surface": "Antibiotics kill bacteria by targeting their cell walls and preventing them from",
     "surface_target": "reproducing",
     "insight": "Antibiotics cause resistance because partial exposure selects for bacteria with mutations that neutralize the drug before they",
     "insight_target": "spread"},
    {"topic": "black_holes",
     "surface": "Black holes are regions where gravity is so strong that nothing can",
     "surface_target": "escape",
     "insight": "Black holes form when stellar mass collapses below the Schwarzschild radius and spacetime curvature becomes sufficient to trap",
     "insight_target": "light"},
    {"topic": "democracy",
     "surface": "Democracy is a system where citizens elect their leaders and participate in",
     "surface_target": "governance",
     "insight": "Democracy functions because distributed decision-making prevents power concentration that would otherwise eliminate the mechanisms correcting its own",
     "insight_target": "errors"},
    {"topic": "memory",
     "surface": "Memory forms when experiences create new connections between neurons and repeated activation makes those connections",
     "surface_target": "stronger",
     "insight": "Memory consolidation requires sleep because the hippocampus replays experiences to the cortex during slow-wave sleep to transfer them into",
     "insight_target": "storage"},
    {"topic": "plate_tectonics",
     "surface": "Plate tectonics describes how the outer layer of the Earth is broken into plates that",
     "surface_target": "move",
     "insight": "Plate tectonics is driven by convection in the mantle where heat differences create currents that drag the surface into",
     "insight_target": "motion"},
    {"topic": "addiction",
     "surface": "Addiction occurs when the brain becomes dependent on a substance and the person cannot stop without",
     "surface_target": "help",
     "insight": "Addiction hijacks the dopamine reward circuit by flooding it with signal until the brain downregulates receptors and natural rewards become",
     "insight_target": "invisible"},
    {"topic": "climate",
     "surface": "Climate change is caused by greenhouse gas emissions that trap heat in the",
     "surface_target": "atmosphere",
     "insight": "Climate change accelerates because warming releases frozen methane and melts ice that was reflecting solar energy back into",
     "insight_target": "space"},
    {"topic": "interest_rates",
     "surface": "Interest rates control borrowing costs and higher rates slow economic growth by making credit",
     "surface_target": "expensive",
     "insight": "Interest rates transmit monetary policy because the discount rate changes the net present value of every future cash flow in the",
     "insight_target": "economy"},
    {"topic": "dna_replication",
     "surface": "DNA replication copies the genetic code by unwinding the double helix and adding new",
     "surface_target": "bases",
     "insight": "DNA replication achieves near-perfect fidelity because proofreading polymerases detect and correct mismatches before sealing each new",
     "insight_target": "strand"},
    {"topic": "trust",
     "surface": "Trust develops through repeated positive interactions between people over time and is broken by",
     "surface_target": "betrayal",
     "insight": "Trust forms because consistent behavior predicts future reliability and the brain uses this track record to reduce the cost of social",
     "insight_target": "cooperation"},
    {"topic": "market_crashes",
     "surface": "Market crashes happen when investors panic and sell their assets causing prices to",
     "surface_target": "fall",
     "insight": "Stock bubbles inflate when rising prices attract momentum buyers whose demand drives prices higher until the pool of buyers is",
     "insight_target": "gone"},
    {"topic": "enzyme",
     "surface": "Enzymes are proteins that speed up reactions and each enzyme has an active",
     "surface_target": "site",
     "insight": "Enzymes accelerate reactions by binding the substrate and lowering the energy needed to form",
     "insight_target": "products"},
    {"topic": "language_acquisition",
     "surface": "Children learn language through exposure and practice and fluency develops through repeated use of vocabulary and",
     "surface_target": "grammar",
     "insight": "Children acquire language because implicit statistical learning mechanisms extract patterns from input before conscious rules are ever",
     "insight_target": "taught"},
    {"topic": "opioid_dependence",
     "surface": "Opioids relieve pain by binding to receptors in the brain and the relief can make people want",
     "surface_target": "more",
     "insight": "Opioids create dependence by downregulating endorphin production so users need increasing doses to achieve the same",
     "insight_target": "effect"},
    {"topic": "ocean_acidification",
     "surface": "Ocean acidification occurs when the sea absorbs carbon dioxide and the pH begins to",
     "surface_target": "fall",
     "insight": "Ocean acidification occurs when dissolved CO2 forms carbonic acid that reduces the carbonate ions marine animals need to build their",
     "insight_target": "shells"},
    {"topic": "supply_chains",
     "surface": "Supply chains move goods from manufacturers to consumers and disruptions can cause shortages at the",
     "surface_target": "store",
     "insight": "Supply chains amplify small demand shocks because each upstream supplier over-orders to buffer uncertainty and the swings compound into",
     "insight_target": "chaos"},
    {"topic": "photon",
     "surface": "Light travels as particles called photons and each photon carries a specific amount of",
     "surface_target": "energy",
     "insight": "Light behaves as both particle and wave because observation collapses its probability distribution and forces it into a definite",
     "insight_target": "state"},
    {"topic": "sleep",
     "surface": "Sleep is essential for health and the brain needs rest to consolidate memories and remove",
     "surface_target": "waste",
     "insight": "Sleep restores cognition because glial cells clear metabolic byproducts from neural tissue that accumulate during waking and impair",
     "insight_target": "function"},
    {"topic": "language_structure",
     "surface": "Languages share features like nouns and verbs because these categories reflect the basic structure of",
     "surface_target": "reality",
     "insight": "Languages converge on similar structures because communicative efficiency constrains the grammar space and languages that violate these constraints become",
     "insight_target": "unstable"},
    {"topic": "pain",
     "surface": "Pain signals are transmitted from receptors through the spinal cord to the",
     "surface_target": "brain",
     "insight": "Pain functions as an alarm system that interrupts attention to prevent further tissue damage because evolution selected for organisms that",
     "insight_target": "respond"},
    {"topic": "network_effects",
     "surface": "Social networks grow because each new user makes the platform more useful for",
     "surface_target": "others",
     "insight": "Network effects create winner-take-all markets because the value increase from each new user accelerates adoption faster than any competitor can",
     "insight_target": "match"},
    {"topic": "immune_system",
     "surface": "The immune system detects and destroys foreign pathogens to protect the body from",
     "surface_target": "disease",
     "insight": "The immune system tolerates self-tissue through clonal deletion that eliminates T cells recognizing the body's own proteins before they become",
     "insight_target": "active"},
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
    ratio: float
    pct_expected_direction: float
    compute_time_s: float
    error: Optional[str] = None

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


def load_gemma_result_from_v1() -> Optional[ModelResult]:
    """Load the previously computed gemma-2-2b result from v1 JSON."""
    if not RESULTS_V1_PATH.exists():
        print(f"  [WARN] No v1 results at {RESULTS_V1_PATH}")
        return None

    with open(RESULTS_V1_PATH) as f:
        v1 = json.load(f)

    for m in v1.get("models", []):
        if m.get("model_name") == "google/gemma-2-2b" and m.get("n_pairs", 0) > 0:
            print(f"  [LOADED] gemma-2-2b from v1: ratio={m['ratio']:.4f}, "
                  f"n={m['n_pairs']}, delta={m['delta']:+.4f}")
            return ModelResult(
                model_name=m["model_name"],
                transcoder_set=m["transcoder_set"],
                n_pairs=m["n_pairs"],
                surface_mean=m["surface_mean"],
                insight_mean=m["insight_mean"],
                delta=m["delta"],
                ratio=m["ratio"],
                pct_expected_direction=m["pct_expected_direction"],
                compute_time_s=m["compute_time_s"],
            )
    return None


def test_model(
    model_name: str,
    transcoder_set: str,
    pairs: list,
    max_feature_nodes: int = 150,
    n_layers: int = None,
) -> ModelResult:
    """Run fathom constant test on a single model."""
    import torch
    from depth_scorer import DepthScorer, MODEL_REGISTRY

    # Register qwen3 if not already there
    if model_name not in MODEL_REGISTRY:
        if n_layers is None:
            raise ValueError(f"Must provide n_layers for unknown model: {model_name}")
        third = n_layers // 3
        MODEL_REGISTRY[model_name] = {
            "model_type": "qwen3",
            "n_layers": n_layers,
            "bands": (third, 2 * third),
            "norm_range": (7.5, 9.0),
            "calibration": None,
        }
        print(f"  [REGISTRY] Added {model_name}: {n_layers} layers, bands={MODEL_REGISTRY[model_name]['bands']}")

    t0 = time.time()
    print(f"\n[FathomV2] Loading {model_name} (transcoder_set={transcoder_set})...")

    try:
        scorer = DepthScorer(
            model_name=model_name,
            model_type=transcoder_set,
            max_feature_nodes=max_feature_nodes,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [LOAD ERROR]:\n{tb}")
        return ModelResult(
            model_name=model_name, transcoder_set=transcoder_set,
            n_pairs=0, surface_mean=0.0, insight_mean=0.0, delta=0.0, ratio=0.0,
            pct_expected_direction=0.0, compute_time_s=time.time() - t0,
            error=str(e),
        )

    surfaces, insights = [], []

    for i, pair in enumerate(pairs):
        topic = pair["topic"]
        try:
            r_s = scorer.score(pair["surface"], target=pair.get("surface_target"))
            torch.cuda.empty_cache(); gc.collect()

            r_i = scorer.score(pair["insight"], target=pair.get("insight_target"))
            torch.cuda.empty_cache(); gc.collect()

            if not math.isnan(r_s.depth_score) and not math.isnan(r_i.depth_score):
                surfaces.append(r_s.depth_score)
                insights.append(r_i.depth_score)
                delta = r_s.depth_score - r_i.depth_score
                print(f"  [{i+1:02d}/{len(pairs)}] {topic:<25} "
                      f"s={r_s.depth_score:.3f}  i={r_i.depth_score:.3f}  "
                      f"delta={delta:+.3f}  {'✓' if delta > 0 else '✗'}")
        except Exception as e:
            print(f"  [{i+1:02d}/{len(pairs)}] {topic:<25} FAILED: {e}")
            torch.cuda.empty_cache(); gc.collect()

    try:
        del scorer
    except Exception:
        pass
    torch.cuda.empty_cache(); gc.collect()

    if not surfaces:
        return ModelResult(
            model_name=model_name, transcoder_set=transcoder_set,
            n_pairs=0, surface_mean=0.0, insight_mean=0.0, delta=0.0, ratio=0.0,
            pct_expected_direction=0.0, compute_time_s=time.time() - t0,
            error="No valid pairs scored",
        )

    s_mean = float(np.mean(surfaces))
    i_mean = float(np.mean(insights))
    delta = s_mean - i_mean
    ratio = s_mean / i_mean if i_mean > 0 else 0.0
    pct_dir = sum(1 for s, i in zip(surfaces, insights) if s > i) / len(surfaces)

    return ModelResult(
        model_name=model_name, transcoder_set=transcoder_set,
        n_pairs=len(surfaces), surface_mean=s_mean, insight_mean=i_mean,
        delta=delta, ratio=ratio, pct_expected_direction=pct_dir,
        compute_time_s=time.time() - t0,
    )


def compute_constant(results: List[ModelResult]) -> dict:
    """Compute the Fathom Constant from cross-model results."""
    valid = [r for r in results if r.n_pairs > 0 and r.error is None]
    if not valid:
        return {"constant": 0.0, "convergent": False, "spread": 0.0,
                "n_models": 0, "ratios": {}, "total_pairs": 0}

    ratios = [r.ratio for r in valid]
    weighted_mean = sum(r.ratio * r.n_pairs for r in valid) / sum(r.n_pairs for r in valid)
    spread = max(ratios) - min(ratios) if len(ratios) > 1 else 0.0
    convergent = len(valid) >= 2 and spread < 0.2

    return {
        "constant": round(weighted_mean, 4),
        "convergent": convergent,
        "spread": round(spread, 4),
        "n_models": len(valid),
        "ratios": {r.model_name: round(r.ratio, 4) for r in valid},
        "total_pairs": sum(r.n_pairs for r in valid),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-only", action="store_true",
                        help="Only run Qwen3 (load gemma from v1 results)")
    parser.add_argument("--rerun-gemma", action="store_true",
                        help="Re-run gemma-2-2b instead of loading from v1")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    t_start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("FATHOM CONSTANT V2 — CROSS-ARCHITECTURE UNIVERSALITY")
    print(f"Started: {timestamp}")
    print("=" * 70)
    print()
    print("Architecture comparison:")
    print("  Model 1: google/gemma-2-2b   — 26L, 2.6B, Google")
    print("  Model 2: Qwen/Qwen3-0.6B    — 28L, 0.6B, Alibaba")
    print("  Key:     DIFFERENT architecture families")
    print("  Goal:    Does surface_mean/insight_mean converge to ~1.0302?")
    print()

    results = []

    # ── Model 1: Gemma-2-2B (load from v1 or re-run) ─────────────────
    if not args.rerun_gemma:
        print("Loading gemma-2-2b from v1 results (skipping 8-min rerun)...")
        gemma_result = load_gemma_result_from_v1()
        if gemma_result:
            results.append(gemma_result)
        else:
            print("  v1 result not found, will re-run gemma-2-2b")
            args.rerun_gemma = True

    if args.rerun_gemma:
        print("\nRunning gemma-2-2b from scratch...")
        gemma_result = test_model(
            "google/gemma-2-2b", "gemma", PAPER_PAIRS, max_feature_nodes=150
        )
        results.append(gemma_result)
        if gemma_result.error:
            print(f"  GEMMA ERROR: {gemma_result.error}")
        else:
            print(f"  GEMMA: ratio={gemma_result.ratio:.4f}, n={gemma_result.n_pairs}")

    # ── Model 2: Qwen3-0.6B (cross-architecture validation) ──────────
    print("\n" + "=" * 60)
    print("Running Qwen3-0.6B (cross-architecture validation)...")
    print("Transcoder: mwhanna/qwen3-0.6b-transcoders-lowl0")
    print("=" * 60)

    qwen_result = test_model(
        "Qwen/Qwen3-0.6B",
        "mwhanna/qwen3-0.6b-transcoders-lowl0",
        PAPER_PAIRS,
        max_feature_nodes=150,
        n_layers=28,
    )
    results.append(qwen_result)

    if qwen_result.error:
        print(f"\n  QWEN ERROR: {qwen_result.error}")
    else:
        print(f"\n  QWEN RESULT: ratio={qwen_result.ratio:.4f}, "
              f"n={qwen_result.n_pairs}, delta={qwen_result.delta:+.4f}, "
              f"pct_dir={qwen_result.pct_expected_direction:.1%}")

    # ── Compute Fathom Constant ───────────────────────────────────────
    constant = compute_constant(results)

    print()
    print("=" * 70)
    print("FATHOM CONSTANT V2 — RESULT")
    print("=" * 70)
    for r in results:
        status = f"ratio={r.ratio:.4f}" if not r.error else f"ERROR: {r.error[:50]}"
        print(f"  {r.model_name:<35} {status}")
    print()
    print(f"  Constant:   {constant['constant']:.4f}")
    print(f"  Spread:     {constant['spread']:.4f}  (convergent if < 0.2)")
    print(f"  Convergent: {constant['convergent']}")
    print(f"  N models:   {constant['n_models']}")
    print(f"  N pairs:    {constant['total_pairs']}")
    print()

    if constant['convergent']:
        print("  🔥 CONVERGENT — Fathom Constant confirmed across architectures!")
        print(f"     The ratio {constant['constant']:.4f} holds for both Gemma and Qwen3.")
        print("     This is architecture-invariant. The constant is real.")
    else:
        if constant['n_models'] >= 2:
            r1, r2 = list(constant['ratios'].values())[:2]
            print(f"  📊 NOT CONVERGENT (spread={constant['spread']:.4f})")
            print(f"     Models disagree: {r1:.4f} vs {r2:.4f}")
            print("     Fathom Constant may be model-family specific, not universal.")
        else:
            print("  ⚠️  Only 1 valid model — cannot assess convergence")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "timestamp": timestamp,
        "duration_s": round(time.time() - t_start, 1),
        "n_pairs": len(PAPER_PAIRS),
        "pair_source": "paper power replication set (p=0.000051)",
        "constant": constant,
        "models": [r.to_dict() for r in results],
    }

    with open(RESULTS_V2_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {RESULTS_V2_PATH}")
    print(f"  Total time:   {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
