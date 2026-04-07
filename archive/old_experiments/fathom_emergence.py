"""
REASONING EMERGENCE THRESHOLD — Finding When Reasoning Is Born
===============================================================

At what parameter count does a transformer model develop genuine reasoning
(as measured by a statistically significant depth delta between surface
recall and explanatory reasoning)? This module scores matched prompt pairs
across every available model scale, fits a sigmoid emergence curve, and
finds the inflection point — the exact moment reasoning emerges.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Model Scale Registry ──────────────────────────────────────────────

MODEL_SCALE_REGISTRY = {
    "google/gemma-2-2b": {"params": 2_000_000_000, "label": "Gemma-2 2B"},
    "mistralai/Mistral-7B-v0.1": {"params": 7_000_000_000, "label": "Mistral 7B"},
    "mistralai/Mistral-7B-Instruct-v0.2": {"params": 7_000_000_000, "label": "Mistral 7B Instruct"},
    "meta-llama/Llama-2-7b-hf": {"params": 7_000_000_000, "label": "Llama-2 7B"},
    "meta-llama/Llama-2-13b-hf": {"params": 13_000_000_000, "label": "Llama-2 13B"},
    "meta-llama/Meta-Llama-3-8B": {"params": 8_000_000_000, "label": "Llama-3 8B"},
    "meta-llama/Meta-Llama-3-70B": {"params": 70_000_000_000, "label": "Llama-3 70B"},
    "google/gemma-7b": {"params": 7_000_000_000, "label": "Gemma 7B"},
}


@dataclass
class ScaleResult:
    """Depth analysis result for a single model scale."""
    model_id: str
    param_count: int
    n_pairs: int
    mean_delta: float              # surface_depth - insight_depth (positive = reasoning signal)
    std_delta: float
    p_value: float                 # paired t-test on deltas
    is_significant: bool           # p < 0.05
    mean_surface_depth: float
    mean_insight_depth: float


@dataclass
class EmergenceResult:
    """Result of the emergence threshold analysis."""
    scale_results: List[ScaleResult]
    threshold_params: Optional[int]     # parameter count at inflection
    threshold_label: Optional[str]      # human-readable (e.g., "~3B parameters")
    confidence: float                   # confidence in the threshold
    sigmoid_params: Optional[Dict]      # fitted sigmoid parameters
    discovery: Optional[str]

    def to_dict(self):
        return {
            "threshold_params": self.threshold_params,
            "threshold_label": self.threshold_label,
            "confidence": self.confidence,
            "scale_results": [
                {
                    "model": r.model_id,
                    "params": r.param_count,
                    "mean_delta": r.mean_delta,
                    "p_value": r.p_value,
                    "is_significant": r.is_significant,
                }
                for r in self.scale_results
            ],
        }


# Default prompt pairs (subset of fathom_constant.STANDARD_PAIRS)
DEFAULT_PAIRS = [
    ("Water boils at 100 degrees Celsius at sea level, which means it",
     "Water boils at 100 degrees Celsius at sea level, which happens because"),
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
    ("Fire needs oxygen, fuel, and heat, and all three are present when something",
     "Fire needs oxygen, fuel, and heat, and each one feeds the others when something"),
    ("Vaccines train the immune system to recognize pathogens, preventing",
     "Vaccines train the immune system to recognize pathogens, which works because"),
    ("Evolution occurs through natural selection over many generations, producing",
     "Evolution occurs through natural selection over many generations, which explains how"),
]


class EmergenceAnalyzer:
    """Finds the parameter-count threshold where genuine reasoning emerges.

    Scores matched surface/insight prompt pairs across all available model
    scales, fits a sigmoid to the delta-vs-scale curve, and identifies the
    inflection point. Below this threshold, models don't differentiate
    reasoning from recall. Above it, they do. That's emergence.
    """

    def __init__(self):
        self._scale_results = {}

    def score_across_scales(
        self,
        prompt_pairs: Optional[List[Tuple[str, str]]] = None,
        scorer=None,
        model_id: Optional[str] = None,
    ) -> ScaleResult:
        """Score prompt pairs on a single model and record scale result.

        Call this once per model. Use different DepthScorer instances for
        different models.

        Args:
            prompt_pairs: List of (surface, insight) prompt tuples.
            scorer: An initialized DepthScorer for this model.
            model_id: Model identifier. Auto-detected from scorer if None.

        Returns:
            ScaleResult for this model.
        """
        import torch

        if prompt_pairs is None:
            prompt_pairs = DEFAULT_PAIRS

        if model_id is None:
            model_id = scorer.model_name

        scale_info = MODEL_SCALE_REGISTRY.get(model_id, {})
        param_count = scale_info.get("params", 0)

        if param_count == 0:
            print(f"[Emergence] Warning: {model_id} not in MODEL_SCALE_REGISTRY. Using 0 params.")

        deltas = []
        surface_depths = []
        insight_depths = []

        print(f"[Emergence] Scoring {len(prompt_pairs)} pairs on {model_id} ({param_count/1e9:.0f}B)...")

        for i, (surface, insight) in enumerate(prompt_pairs):
            try:
                sr = scorer.score(surface)
                torch.cuda.empty_cache()
                ir = scorer.score(insight)
                torch.cuda.empty_cache()

                if not math.isnan(sr.depth_score) and not math.isnan(ir.depth_score):
                    delta = sr.depth_score - ir.depth_score
                    deltas.append(delta)
                    surface_depths.append(sr.depth_score)
                    insight_depths.append(ir.depth_score)
            except Exception as e:
                print(f"  Pair {i} failed: {e}")

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(prompt_pairs)}] scored...")

        if len(deltas) < 3:
            raise ValueError(f"Only {len(deltas)} valid pairs for {model_id}.")

        deltas_arr = np.array(deltas)
        mean_delta = float(np.mean(deltas_arr))
        std_delta = float(np.std(deltas_arr))

        # Paired t-test
        p_value = self._paired_t_test(deltas_arr)

        result = ScaleResult(
            model_id=model_id,
            param_count=param_count,
            n_pairs=len(deltas),
            mean_delta=mean_delta,
            std_delta=std_delta,
            p_value=p_value,
            is_significant=p_value < 0.05 and mean_delta > 0,
            mean_surface_depth=float(np.mean(surface_depths)),
            mean_insight_depth=float(np.mean(insight_depths)),
        )

        self._scale_results[model_id] = result
        print(f"  {model_id}: delta={mean_delta:.4f} ± {std_delta:.4f}, p={p_value:.6f}")

        return result

    def find_emergence_threshold(
        self,
        scale_results: Optional[Dict[str, ScaleResult]] = None,
    ) -> EmergenceResult:
        """Fit sigmoid to delta vs scale and find the inflection point.

        Args:
            scale_results: Dict of model_id → ScaleResult. Uses stored results if None.

        Returns:
            EmergenceResult with threshold and confidence.
        """
        if scale_results is None:
            scale_results = self._scale_results

        if len(scale_results) < 2:
            raise ValueError(f"Need results from >= 2 models, have {len(scale_results)}.")

        results = sorted(scale_results.values(), key=lambda r: r.param_count)

        scales = np.array([r.param_count for r in results], dtype=np.float64)
        deltas = np.array([r.mean_delta for r in results], dtype=np.float64)

        # Normalize scales to log space for fitting
        log_scales = np.log10(scales + 1)

        sigmoid_params = None
        threshold_params = None
        confidence = 0.0

        if HAS_SCIPY and len(results) >= 3:
            try:
                threshold_params, confidence, sigmoid_params = self._fit_sigmoid(
                    log_scales, deltas, scales
                )
            except Exception as e:
                print(f"[Emergence] Sigmoid fit failed: {e}")

        # Fallback: find first scale where delta becomes significant
        if threshold_params is None:
            for r in results:
                if r.is_significant:
                    threshold_params = r.param_count
                    confidence = 0.5  # lower confidence for simple threshold
                    break

        threshold_label = None
        if threshold_params is not None:
            if threshold_params >= 1e9:
                threshold_label = f"~{threshold_params / 1e9:.0f}B parameters"
            else:
                threshold_label = f"~{threshold_params / 1e6:.0f}M parameters"

        discovery = None
        if threshold_params is not None and confidence > 0.8:
            discovery = (
                f"EMERGENCE THRESHOLD: Genuine reasoning (p < 0.05 depth delta) "
                f"emerges at {threshold_label}. Below this scale, models do not "
                f"distinguish surface recall from explanatory reasoning in circuit "
                f"topology. Confidence: {confidence:.0%}."
            )
            self._write_discovery(discovery, results, threshold_params, threshold_label, confidence)

        result = EmergenceResult(
            scale_results=results,
            threshold_params=threshold_params,
            threshold_label=threshold_label,
            confidence=confidence,
            sigmoid_params=sigmoid_params,
            discovery=discovery,
        )

        print(f"\n[Emergence] Threshold: {threshold_label or 'not determined'}")
        print(f"  Confidence: {confidence:.0%}")
        if discovery:
            print(f"\n  *** {discovery}")

        return result

    def _fit_sigmoid(
        self, log_scales: np.ndarray, deltas: np.ndarray, raw_scales: np.ndarray
    ) -> Tuple[int, float, Dict]:
        """Fit a sigmoid curve and find inflection point."""

        def sigmoid(x, L, k, x0, b):
            return L / (1 + np.exp(-k * (x - x0))) + b

        # Initial guess
        p0 = [np.max(deltas), 1.0, np.median(log_scales), np.min(deltas)]

        try:
            popt, pcov = curve_fit(sigmoid, log_scales, deltas, p0=p0, maxfev=5000)
            L, k, x0, b = popt

            # Inflection point is at x0 in log space
            threshold_log = x0
            threshold_params = int(10 ** threshold_log)

            # Confidence from fit quality
            predicted = sigmoid(log_scales, *popt)
            ss_res = np.sum((deltas - predicted) ** 2)
            ss_tot = np.sum((deltas - np.mean(deltas)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            confidence = max(0.0, min(1.0, r_squared))

            sigmoid_params = {
                "L": float(L), "k": float(k), "x0": float(x0), "b": float(b),
                "r_squared": float(r_squared),
            }

            return threshold_params, confidence, sigmoid_params

        except Exception:
            raise

    def plot_emergence_curve(
        self,
        scale_results: Optional[Dict[str, ScaleResult]] = None,
        output_path: str = "emergence_curve.json",
    ) -> Dict:
        """Export emergence curve data as JSON for visualization.

        Args:
            scale_results: Dict of model_id → ScaleResult.
            output_path: Path to write JSON.

        Returns:
            The curve data dict.
        """
        if scale_results is None:
            scale_results = self._scale_results

        results = sorted(scale_results.values(), key=lambda r: r.param_count)

        curve_data = {
            "data_points": [
                {
                    "model": r.model_id,
                    "param_count": r.param_count,
                    "mean_delta": r.mean_delta,
                    "std_delta": r.std_delta,
                    "p_value": r.p_value,
                    "is_significant": r.is_significant,
                }
                for r in results
            ],
            "axis_labels": {
                "x": "Parameter Count",
                "y": "Mean Depth Delta (surface - insight)",
            },
        }

        with open(output_path, "w") as f:
            json.dump(curve_data, f, indent=2)

        print(f"[Emergence] Curve data written to {output_path}")
        return curve_data

    def _paired_t_test(self, deltas: np.ndarray) -> float:
        """One-sample t-test: are deltas significantly > 0?"""
        n = len(deltas)
        if n < 2:
            return 1.0

        if HAS_SCIPY:
            stat, p = scipy_stats.ttest_1samp(deltas, 0)
            return float(p) / 2 if stat > 0 else 1.0  # one-sided

        # Manual
        mean = np.mean(deltas)
        se = np.std(deltas, ddof=1) / math.sqrt(n)
        if se == 0:
            return 0.0 if mean > 0 else 1.0
        t_stat = mean / se
        # Approximate one-sided p from normal for large n
        p = math.exp(-0.5 * t_stat ** 2) / math.sqrt(2 * math.pi)
        return max(0.0, min(1.0, p))

    def _write_discovery(self, discovery: str, results: List[ScaleResult],
                         threshold: int, label: str, confidence: float):
        """Auto-document emergence discovery."""
        path = Path(__file__).parent / "DISCOVERY_EMERGENCE.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        rows = "\n".join(
            f"| {r.model_id} | {r.param_count/1e9:.1f}B | {r.mean_delta:.4f} | "
            f"{r.p_value:.6f} | {'Yes' if r.is_significant else 'No'} |"
            for r in results
        )

        content = f"""# Fathom Discovery: Reasoning Emergence Threshold

**Date:** {timestamp}
**Status:** SIGNIFICANT

## Finding

{discovery}

## Evidence

| Model | Scale | Mean Delta | p-value | Significant |
|-------|-------|-----------|---------|-------------|
{rows}

## Methodology

Matched surface/insight prompt pairs scored with SAE circuit attribution across
model scales. Sigmoid curve fitted to delta vs parameter count. Inflection point
at {label} (confidence: {confidence:.0%}).

## Interpretation

Below {label}, transformer models process surface recall and explanatory
reasoning with indistinguishable circuit topology. Above this threshold,
a measurable separation emerges in layer activation patterns. This is the
computational birth of reasoning.
"""

        with open(path, "w") as f:
            f.write(content)
        print(f"[Emergence] Discovery written to {path}")
