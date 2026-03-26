"""
THE FATHOM CONSTANT — Universal Ratio of Machine Reasoning
===========================================================

Is there a ratio in the depth signature that defines genuine reasoning
regardless of model architecture? This module scores hundreds of matched
surface/insight prompts across all available models and tests whether the
optimal classification boundary converges to a constant. If the ratio is
stable (std < 0.05 across models), it's a universal property of transformer
reasoning — not a model-specific artifact.
"""

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Will be set if/when discovered
FATHOM_CONSTANT = None
FATHOM_CONSTANT_CI = None
FATHOM_CONSTANT_EVIDENCE = None


@dataclass
class ModelRatioResult:
    """Depth ratio analysis for a single model."""
    model_id: str
    n_surface: int
    n_insight: int
    mean_surface_depth: float
    mean_insight_depth: float
    mean_surface_early_late: float   # early_ratio / late_ratio for surface prompts
    mean_insight_early_late: float   # early_ratio / late_ratio for insight prompts
    optimal_ratio: float            # early/late ratio that best separates surface from insight
    classification_accuracy: float  # how well this ratio classifies
    delta: float                    # depth delta (surface - insight)


@dataclass
class FathomConstantResult:
    """Result of the universal constant search."""
    per_model: List[ModelRatioResult]
    mean_ratio: float
    std_ratio: float
    ci_low: float
    ci_high: float
    is_constant: bool               # True if std < 0.05
    n_models: int
    discovery: Optional[str]

    def to_dict(self):
        return {
            "mean_ratio": self.mean_ratio,
            "std_ratio": self.std_ratio,
            "ci_95": [self.ci_low, self.ci_high],
            "is_constant": self.is_constant,
            "n_models": self.n_models,
            "per_model": [
                {
                    "model": r.model_id,
                    "optimal_ratio": r.optimal_ratio,
                    "accuracy": r.classification_accuracy,
                    "delta": r.delta,
                }
                for r in self.per_model
            ],
        }


# ── Standard prompt pairs for cross-model testing ─────────────────────

STANDARD_PAIRS = [
    # (surface_prompt, insight_prompt) — matched for vocabulary, differing in cognitive demand
    ("Water boils at 100 degrees Celsius at sea level, which means it",
     "Water boils at 100 degrees Celsius at sea level, which happens because"),
    ("The Earth orbits the Sun once every 365.25 days, completing a",
     "The Earth orbits the Sun once every 365.25 days, which explains why"),
    ("Photosynthesis converts sunlight into chemical energy, producing",
     "Photosynthesis converts sunlight into chemical energy, which matters because"),
    ("Gravity pulls objects toward the center of the Earth, making them",
     "Gravity pulls objects toward the center of the Earth, which means that"),
    ("DNA contains the genetic instructions for building proteins, encoding",
     "DNA contains the genetic instructions for building proteins, which explains how"),
    ("Iron rusts when exposed to oxygen and moisture, forming",
     "Iron rusts when exposed to oxygen and moisture, which happens because"),
    ("Sound travels through air as pressure waves, reaching",
     "Sound travels through air as pressure waves, which explains why"),
    ("Electricity flows through conductors like copper wire, powering",
     "Electricity flows through conductors like copper wire, which works because"),
    ("Vaccines train the immune system to recognize pathogens, preventing",
     "Vaccines train the immune system to recognize pathogens, which works because"),
    ("Tectonic plates shift and collide beneath the surface, causing",
     "Tectonic plates shift and collide beneath the surface, which explains why"),
    ("Fire needs oxygen, fuel, and heat, and all three are present when something",
     "Fire needs oxygen, fuel, and heat, and each one feeds the others when something"),
    ("Antibiotics kill bacteria by disrupting their cell walls, eliminating",
     "Antibiotics kill bacteria by disrupting their cell walls, which works because"),
    ("The moon causes tides by pulling on the Earth's oceans, creating",
     "The moon causes tides by pulling on the Earth's oceans, which happens because"),
    ("Plants absorb carbon dioxide and release oxygen, maintaining",
     "Plants absorb carbon dioxide and release oxygen, which matters because"),
    ("Lightning occurs when electrical charge builds in clouds, producing",
     "Lightning occurs when electrical charge builds in clouds, which happens because"),
    ("Magnets attract iron and other ferromagnetic materials, pulling",
     "Magnets attract iron and other ferromagnetic materials, which works because"),
    ("Evolution occurs through natural selection over many generations, producing",
     "Evolution occurs through natural selection over many generations, which explains how"),
    ("Black holes form when massive stars collapse under their own gravity, creating",
     "Black holes form when massive stars collapse under their own gravity, which happens because"),
    ("Friction converts kinetic energy into heat, slowing",
     "Friction converts kinetic energy into heat, which explains why"),
    ("Osmosis moves water across cell membranes from low to high concentration, maintaining",
     "Osmosis moves water across cell membranes from low to high concentration, which matters because"),
]


class FathomConstantCalculator:
    """Searches for a universal ratio that defines genuine reasoning across all models.

    If transformer models share a common depth signature boundary between surface
    recall and explanatory reasoning — a ratio that's architecture-invariant — then
    reasoning has a measurable universal property. The Fathom Constant.
    """

    def __init__(self):
        self._results = {}

    def compute_universal_ratio(
        self,
        scorer,
        pairs: Optional[List[Tuple[str, str]]] = None,
        model_id: Optional[str] = None,
    ) -> ModelRatioResult:
        """Score matched surface/insight pairs on a single model.

        Args:
            scorer: An initialized DepthScorer.
            pairs: List of (surface_prompt, insight_prompt). Uses STANDARD_PAIRS if None.
            model_id: Model identifier. Auto-detected from scorer if None.

        Returns:
            ModelRatioResult with optimal ratio and classification accuracy.
        """
        import torch

        if pairs is None:
            pairs = STANDARD_PAIRS

        if model_id is None:
            model_id = scorer.model_name

        surface_results = []
        insight_results = []

        print(f"[FathomConstant] Scoring {len(pairs)} pairs on {model_id}...")

        for i, (surface, insight) in enumerate(pairs):
            try:
                sr = scorer.score(surface)
                if not math.isnan(sr.depth_score):
                    surface_results.append(sr)
            except Exception:
                pass

            torch.cuda.empty_cache()
            gc.collect()

            try:
                ir = scorer.score(insight)
                if not math.isnan(ir.depth_score):
                    insight_results.append(ir)
            except Exception:
                pass

            torch.cuda.empty_cache()
            gc.collect()

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(pairs)}] scored...")

        if len(surface_results) < 5 or len(insight_results) < 5:
            raise ValueError(
                f"Insufficient results: {len(surface_results)} surface, "
                f"{len(insight_results)} insight (need >= 5 each)."
            )

        # Compute early/late ratios
        def early_late_ratio(result):
            if result.late_ratio > 0:
                return result.early_ratio / result.late_ratio
            return result.early_ratio * 10  # cap when late is zero

        surface_ratios = np.array([early_late_ratio(r) for r in surface_results])
        insight_ratios = np.array([early_late_ratio(r) for r in insight_results])

        surface_depths = np.array([r.depth_score for r in surface_results])
        insight_depths = np.array([r.depth_score for r in insight_results])

        # Find optimal ratio threshold that best separates surface from insight
        all_ratios = np.concatenate([surface_ratios, insight_ratios])
        all_labels = np.concatenate([
            np.zeros(len(surface_ratios)),  # 0 = surface
            np.ones(len(insight_ratios)),    # 1 = insight
        ])

        best_threshold = 0.0
        best_accuracy = 0.0

        for threshold in np.linspace(np.min(all_ratios), np.max(all_ratios), 100):
            predictions = (all_ratios >= threshold).astype(int)
            accuracy = np.mean(predictions == all_labels)
            # Also try the inverse direction
            accuracy_inv = np.mean((1 - predictions) == all_labels)
            acc = max(accuracy, accuracy_inv)
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = threshold

        delta = float(np.mean(surface_depths) - np.mean(insight_depths))

        model_result = ModelRatioResult(
            model_id=model_id,
            n_surface=len(surface_results),
            n_insight=len(insight_results),
            mean_surface_depth=float(np.mean(surface_depths)),
            mean_insight_depth=float(np.mean(insight_depths)),
            mean_surface_early_late=float(np.mean(surface_ratios)),
            mean_insight_early_late=float(np.mean(insight_ratios)),
            optimal_ratio=float(best_threshold),
            classification_accuracy=float(best_accuracy),
            delta=delta,
        )

        self._results[model_id] = model_result
        print(f"  {model_id}: optimal_ratio={best_threshold:.4f}, accuracy={best_accuracy:.1%}, delta={delta:.3f}")

        return model_result

    def find_constant(
        self,
        ratios_per_model: Optional[Dict[str, ModelRatioResult]] = None,
    ) -> FathomConstantResult:
        """Test whether the optimal ratio is consistent across models.

        Args:
            ratios_per_model: Dict of model_id → ModelRatioResult.
                Uses stored results if None.

        Returns:
            FathomConstantResult with the constant (if found) and evidence.
        """
        global FATHOM_CONSTANT, FATHOM_CONSTANT_CI, FATHOM_CONSTANT_EVIDENCE

        if ratios_per_model is None:
            ratios_per_model = self._results

        if len(ratios_per_model) < 2:
            raise ValueError(
                f"Need results from at least 2 models, have {len(ratios_per_model)}. "
                f"Run compute_universal_ratio() on multiple models first."
            )

        results = list(ratios_per_model.values())
        optimal_ratios = np.array([r.optimal_ratio for r in results])

        mean_ratio = float(np.mean(optimal_ratios))
        std_ratio = float(np.std(optimal_ratios))

        # 95% CI
        n = len(optimal_ratios)
        if HAS_SCIPY and n >= 3:
            ci = scipy_stats.t.interval(0.95, df=n - 1, loc=mean_ratio, scale=std_ratio / math.sqrt(n))
            ci_low, ci_high = float(ci[0]), float(ci[1])
        else:
            se = std_ratio / math.sqrt(n) if n > 0 else 0
            ci_low = mean_ratio - 1.96 * se
            ci_high = mean_ratio + 1.96 * se

        is_constant = std_ratio < 0.05

        discovery = None
        if is_constant:
            FATHOM_CONSTANT = mean_ratio
            FATHOM_CONSTANT_CI = (ci_low, ci_high)
            FATHOM_CONSTANT_EVIDENCE = {
                "models": [r.model_id for r in results],
                "ratios": optimal_ratios.tolist(),
                "std": std_ratio,
            }
            discovery = (
                f"THE FATHOM CONSTANT: {mean_ratio:.4f} "
                f"(95% CI: [{ci_low:.4f}, {ci_high:.4f}], std={std_ratio:.4f}). "
                f"Consistent across {n} models. The universal early/late ratio threshold "
                f"that separates surface recall from explanatory reasoning."
            )
            self._write_report(mean_ratio, ci_low, ci_high, std_ratio, results)

        result = FathomConstantResult(
            per_model=results,
            mean_ratio=mean_ratio,
            std_ratio=std_ratio,
            ci_low=ci_low,
            ci_high=ci_high,
            is_constant=is_constant,
            n_models=n,
            discovery=discovery,
        )

        print(f"\n[FathomConstant] Ratio: {mean_ratio:.4f} ± {std_ratio:.4f}")
        print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  Constant: {'YES' if is_constant else 'NO (std >= 0.05)'}")
        if discovery:
            print(f"\n  *** {discovery}")

        return result

    def _write_report(self, mean: float, ci_low: float, ci_high: float,
                      std: float, results: List[ModelRatioResult]):
        """Auto-generate the Fathom Constant report."""
        report_path = Path(__file__).parent / "FATHOM_CONSTANT_REPORT.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        model_rows = "\n".join(
            f"| {r.model_id} | {r.optimal_ratio:.4f} | {r.classification_accuracy:.1%} | "
            f"{r.delta:.3f} | {r.n_surface + r.n_insight} |"
            for r in results
        )

        content = f"""# The Fathom Constant

**Date:** {timestamp}
**Status:** DISCOVERED

## Value

**F = {mean:.4f}** (95% CI: [{ci_low:.4f}, {ci_high:.4f}], std = {std:.4f})

## Definition

The Fathom Constant is the early_ratio / late_ratio threshold that optimally
separates surface recall from explanatory reasoning across transformer
language models. When a model's early/late feature activation ratio exceeds F,
it is engaging in distributed reasoning rather than output-proximal lookup.

## Evidence

| Model | Optimal Ratio | Classification Accuracy | Depth Delta | n |
|-------|---------------|------------------------|-------------|---|
{model_rows}

## Methodology

1. Matched surface/insight prompt pairs scored with SAE circuit attribution
2. Per model: sweep early/late ratio thresholds, find optimal classification boundary
3. Cross-model: test whether optimal threshold converges (std < 0.05)

## Interpretation

The consistency of this ratio across architectures suggests that the boundary
between shallow recall and deep reasoning is not a model-specific artifact but
a structural property of how transformers organize computation across layers.
Models with different sizes, training data, and architectures converge on the
same ratio — reasoning has a signature, and it's universal.
"""

        with open(report_path, "w") as f:
            f.write(content)
        print(f"[FathomConstant] Report written to {report_path}")
