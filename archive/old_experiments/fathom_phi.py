"""
PHI APPROXIMATOR — Fathom Consciousness Proxy via Information Integration
==========================================================================

Integrated Information Theory (IIT) defines consciousness as phi — the degree
to which a system integrates information beyond what its parts do independently.
Fathom measures distributed activation across transformer layers — a form of
information integration. This module computes phi_proxy: an IIT-inspired metric
derived from layer entropy, concentration, and feature engagement. Not a claim
of consciousness — a measurement of integration that happens to be what IIT says
matters.
"""

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np


# ── CONSCIOUSNESS CAVEAT ──────────────────────────────────────────────
# This caveat MUST be included in every output and report.

CONSCIOUSNESS_CAVEAT = (
    "phi_proxy is a mathematical approximation, not a claim of consciousness. "
    "It measures information integration as defined by IIT, applied to transformer "
    "layer topology. Interpret accordingly."
)


@dataclass
class PhiResult:
    """Phi proxy measurement for a single prompt."""
    prompt: str
    phi_proxy: float               # IIT-inspired integration metric (0-1+)
    integration_level: str         # "minimal", "moderate", "high", "exceptional"
    layer_entropy: float           # from DepthResult
    concentration_coefficient: float  # max single-layer fraction (0-1)
    feature_engagement: float      # feature_count / max_features (0-1)
    interpretation: str
    caveat: str = CONSCIOUSNESS_CAVEAT

    def to_dict(self):
        return {
            "phi_proxy": self.phi_proxy,
            "integration_level": self.integration_level,
            "layer_entropy": self.layer_entropy,
            "concentration_coefficient": self.concentration_coefficient,
            "feature_engagement": self.feature_engagement,
            "interpretation": self.interpretation,
            "caveat": self.caveat,
        }


@dataclass
class PhiDistribution:
    """Distribution of phi_proxy across a batch of prompts."""
    n_prompts: int
    mean_phi: float
    std_phi: float
    median_phi: float
    min_phi: float
    max_phi: float
    pct_high_integration: float    # fraction with integration_level "high" or "exceptional"
    per_prompt: List[PhiResult]
    caveat: str = CONSCIOUSNESS_CAVEAT

    def to_dict(self):
        return {
            "n_prompts": self.n_prompts,
            "mean_phi": self.mean_phi,
            "std_phi": self.std_phi,
            "median_phi": self.median_phi,
            "pct_high_integration": self.pct_high_integration,
            "caveat": self.caveat,
        }


class PhiApproximator:
    """Computes an IIT-inspired information integration metric from fathom depth data.

    phi_proxy = layer_entropy * (1 - concentration_coefficient) * feature_engagement

    Where:
    - layer_entropy: Shannon entropy of the layer activation distribution (higher = more distributed)
    - concentration_coefficient: fraction of features in the single most-active layer (lower = less concentrated)
    - feature_engagement: normalized feature count (higher = more of the network is active)

    High phi_proxy = distributed, non-concentrated, broadly engaged computation.
    IIT says this is what integration looks like. We measure it, we don't claim
    it equals consciousness.
    """

    def __init__(self, max_feature_nodes: int = 500):
        """
        Args:
            max_feature_nodes: Maximum feature count for normalization (from scorer config).
        """
        self.max_feature_nodes = max_feature_nodes

    def compute_phi_proxy(self, depth_result) -> PhiResult:
        """Compute phi_proxy from a DepthResult.

        Args:
            depth_result: A DepthResult from DepthScorer.

        Returns:
            PhiResult with phi_proxy and interpretation.
        """
        # Layer entropy
        entropy = depth_result.layer_entropy if depth_result.layer_entropy is not None else 0.0

        # Normalize entropy to 0-1 range
        max_entropy = math.log2(depth_result.n_layers) if depth_result.n_layers > 1 else 1.0
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0

        # Concentration coefficient: max fraction in any single layer
        concentration = 0.0
        if depth_result.feature_count > 0 and depth_result.layer_profile:
            max_count = max(depth_result.layer_profile.values())
            concentration = max_count / depth_result.feature_count

        # Feature engagement: how much of the network is active
        engagement = min(1.0, depth_result.feature_count / self.max_feature_nodes) if self.max_feature_nodes > 0 else 0.0

        # Phi proxy: integration = entropy * dispersion * engagement
        phi = entropy_norm * (1.0 - concentration) * engagement

        # Classify
        level = self._classify_integration(phi)
        interpretation = self._interpret(phi, entropy_norm, concentration, engagement)

        return PhiResult(
            prompt=depth_result.prompt,
            phi_proxy=round(phi, 4),
            integration_level=level,
            layer_entropy=entropy,
            concentration_coefficient=round(concentration, 4),
            feature_engagement=round(engagement, 4),
            interpretation=interpretation,
        )

    def interpret_phi(self, phi_proxy: float) -> Dict:
        """Interpret a phi_proxy value.

        Args:
            phi_proxy: The phi_proxy value.

        Returns:
            Dict with phi_proxy, integration_level, interpretation, and caveat.
        """
        level = self._classify_integration(phi_proxy)

        interpretations = {
            "minimal": (
                "Minimal cross-layer integration — computation is concentrated in few layers "
                "with limited distributed processing. Under IIT, this pattern would correspond "
                "to low information integration."
            ),
            "moderate": (
                "Moderate cross-layer integration — some distributed processing across layers "
                "but with notable concentration. Under IIT, this represents partial integration "
                "of information across the system."
            ),
            "high": (
                "High cross-layer integration — computation is distributed across many layers "
                "with low concentration and broad feature engagement. Consistent with what IIT "
                "would predict for high information integration."
            ),
            "exceptional": (
                "Exceptional cross-layer integration — maximally distributed computation with "
                "minimal concentration. Under IIT, this pattern represents near-maximal "
                "information integration across the system."
            ),
        }

        return {
            "phi_proxy": phi_proxy,
            "integration_level": level,
            "interpretation": interpretations.get(level, ""),
            "caveat": CONSCIOUSNESS_CAVEAT,
        }

    def phi_distribution(
        self,
        prompts: List[str],
        scorer,
    ) -> PhiDistribution:
        """Score prompts and compute phi_proxy distribution.

        Args:
            prompts: List of prompts to analyze.
            scorer: An initialized DepthScorer.

        Returns:
            PhiDistribution with statistics across all prompts.
        """
        import torch

        results = []
        print(f"[Phi] Computing phi_proxy for {len(prompts)} prompts...")

        for i, prompt in enumerate(prompts):
            try:
                depth_result = scorer.score(prompt)
                phi_result = self.compute_phi_proxy(depth_result)
                results.append(phi_result)
            except Exception as e:
                print(f"  Prompt {i} failed: {e}")

            torch.cuda.empty_cache()
            gc.collect()

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(prompts)}] scored...")

        if not results:
            return PhiDistribution(
                n_prompts=0, mean_phi=0, std_phi=0, median_phi=0,
                min_phi=0, max_phi=0, pct_high_integration=0, per_prompt=[],
            )

        # ── Dynamic batch normalization ──────────────────────────────
        # Raw entropy_norm tends to cluster high (0.7-0.9) because
        # max_entropy = log2(n_layers) is a loose upper bound. Re-normalize
        # within the batch so phi_proxy spans 0-1 and creates real variance.
        raw_entropies = np.array([r.layer_entropy for r in results])
        batch_e_min = float(raw_entropies.min())
        batch_e_max = float(raw_entropies.max())
        batch_e_range = batch_e_max - batch_e_min

        if batch_e_range > 1e-6 and len(results) > 1:
            print(f"[Phi] Batch entropy range: {batch_e_min:.4f} - {batch_e_max:.4f}, "
                  f"re-normalizing phi_proxy within batch")
            for r in results:
                # Re-compute with batch-normalized entropy
                e_norm = (r.layer_entropy - batch_e_min) / batch_e_range
                phi_new = e_norm * (1.0 - r.concentration_coefficient) * r.feature_engagement
                r.phi_proxy = round(phi_new, 4)
                r.integration_level = self._classify_integration(phi_new)
                r.interpretation = self._interpret(
                    phi_new, e_norm, r.concentration_coefficient, r.feature_engagement
                )

        phis = np.array([r.phi_proxy for r in results])
        high_count = sum(1 for r in results if r.integration_level in ("high", "exceptional"))

        dist = PhiDistribution(
            n_prompts=len(results),
            mean_phi=float(np.mean(phis)),
            std_phi=float(np.std(phis)),
            median_phi=float(np.median(phis)),
            min_phi=float(np.min(phis)),
            max_phi=float(np.max(phis)),
            pct_high_integration=high_count / len(results),
            per_prompt=results,
        )

        self._write_analysis(dist)
        return dist

    def _classify_integration(self, phi: float) -> str:
        """Classify integration level from phi_proxy value."""
        if phi >= 0.8:
            return "exceptional"
        elif phi >= 0.6:
            return "high"
        elif phi >= 0.3:
            return "moderate"
        else:
            return "minimal"

    def _interpret(self, phi: float, entropy_norm: float,
                   concentration: float, engagement: float) -> str:
        """Generate human-readable interpretation."""
        parts = []

        if entropy_norm > 0.7:
            parts.append("highly distributed layer activation")
        elif entropy_norm > 0.4:
            parts.append("moderately distributed layer activation")
        else:
            parts.append("concentrated layer activation")

        if concentration < 0.2:
            parts.append("low layer concentration")
        elif concentration > 0.5:
            parts.append("high single-layer dominance")

        if engagement > 0.6:
            parts.append("broad network engagement")
        elif engagement < 0.2:
            parts.append("minimal network engagement")

        level = self._classify_integration(phi)
        summary = f"phi_proxy={phi:.3f} ({level} integration): {'; '.join(parts)}."

        return summary

    def _write_analysis(self, dist: PhiDistribution):
        """Auto-generate PHI_ANALYSIS.md."""
        path = Path(__file__).parent / "PHI_ANALYSIS.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# Phi Proxy Analysis — Information Integration in Transformer Reasoning

**Date:** {timestamp}
**Module:** fathom_phi.py

## Caveat

{CONSCIOUSNESS_CAVEAT}

## Distribution

| Metric | Value |
|--------|-------|
| n prompts | {dist.n_prompts} |
| mean phi_proxy | {dist.mean_phi:.4f} |
| std phi_proxy | {dist.std_phi:.4f} |
| median phi_proxy | {dist.median_phi:.4f} |
| min | {dist.min_phi:.4f} |
| max | {dist.max_phi:.4f} |
| % high integration | {dist.pct_high_integration:.1%} |

## Methodology

phi_proxy = entropy_normalized * (1 - concentration) * feature_engagement

Where:
- entropy_normalized: Shannon entropy of layer distribution / max possible entropy
- concentration: fraction of features in the single most-active layer
- feature_engagement: active features / max feature nodes

## Interpretation

{dist.pct_high_integration:.0%} of prompts showed high or exceptional information
integration (phi_proxy >= 0.4). Mean integration level: {dist.mean_phi:.3f}.

This measures the degree to which transformer computation is distributed and
non-decomposable across layers — the computational property that IIT identifies
as the substrate of integrated information.

## Note

{CONSCIOUSNESS_CAVEAT}
"""

        with open(path, "w") as f:
            f.write(content)
        print(f"[Phi] Analysis written to {path}")
