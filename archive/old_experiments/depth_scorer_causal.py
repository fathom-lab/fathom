"""
CAUSAL DEPTH ANALYZER — Causal Intervention Engine for Reasoning Circuits
=========================================================================

Goes beyond measurement to causality. Layer ablation reveals which layers
are causally necessary for the model's output — not just where features
activate, but which activations actually matter. The causal_depth_score
weights layers by their causal importance rather than feature count,
producing a more honest measure of where real computation happens.
"""

import gc
import math
import time
from dataclasses import dataclass
from typing import Optional, List

import torch
import numpy as np


@dataclass
class AblationResult:
    """Result of ablating a single layer."""
    layer_idx: int
    baseline_prob: float           # top-token probability before ablation
    ablated_prob: float            # top-token probability after ablation
    prob_delta: float              # baseline - ablated (positive = layer was important)
    prob_delta_pct: float          # percentage drop in probability
    baseline_depth_score: float    # depth_score with all layers
    ablated_depth_score: float     # depth_score excluding this layer's features
    depth_delta: float             # change in depth_score from removing this layer
    causal_importance: float       # combined importance metric (0-1)


@dataclass
class CausalDepthResult:
    """Full causal analysis of a prompt."""
    prompt: str
    causal_depth_score: float      # mean layer weighted by causal importance
    standard_depth_score: float    # original unweighted depth_score for comparison
    critical_layers: List[AblationResult]  # all layers ranked by causal importance
    top_3_layers: List[int]        # indices of 3 most causally important layers
    causal_concentration: float    # what fraction of causal mass is in top 3 layers
    compute_time_s: float

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "causal_depth_score": self.causal_depth_score,
            "standard_depth_score": self.standard_depth_score,
            "critical_layers": [
                {
                    "layer": r.layer_idx,
                    "prob_delta": r.prob_delta,
                    "prob_delta_pct": r.prob_delta_pct,
                    "causal_importance": r.causal_importance,
                }
                for r in self.critical_layers[:10]
            ],
            "top_3_layers": self.top_3_layers,
            "causal_concentration": self.causal_concentration,
            "compute_time_s": self.compute_time_s,
        }


class CausalDepthAnalyzer:
    """Causal intervention engine that reveals which layers are necessary for reasoning.

    Uses activation ablation (zeroing layer residual streams via TransformerLens hooks)
    to measure each layer's causal contribution to the model's output. Produces a
    causal_depth_score that weights layers by their actual importance rather than
    mere feature count — a fundamentally more honest depth metric.
    """

    def __init__(self, scorer):
        """
        Args:
            scorer: An initialized DepthScorer instance (model already loaded).
        """
        self.scorer = scorer
        self._inner = scorer._get_inner_model()
        if self._inner is None:
            raise ValueError(
                "CausalDepthAnalyzer requires access to the inner HookedTransformer. "
                "Ensure the scorer's model exposes a .model with .to_tokens()."
            )

    def _get_baseline(self, prompt: str, target: Optional[str] = None):
        """Get baseline output probability and depth result."""
        if target is None:
            target = self.scorer._get_single_token_target(prompt)

        tokens = self._inner.to_tokens(prompt + target)
        target_ids = self._inner.tokenizer.encode(target, add_special_tokens=False)
        target_id = target_ids[0]

        # Baseline logits (on prompt only, predicting target position)
        prompt_tokens = self._inner.to_tokens(prompt)
        with torch.no_grad():
            logits = self._inner(prompt_tokens)
        probs = torch.softmax(logits[0, -1], dim=-1)
        baseline_prob = probs[target_id].item()

        # Baseline depth
        baseline_result = self.scorer.score(prompt, target)

        return target, target_id, baseline_prob, baseline_result

    def ablate_layer(
        self,
        prompt: str,
        layer_idx: int,
        target: Optional[str] = None,
    ) -> AblationResult:
        """Zero out all activations at a given layer and measure the effect.

        Hooks into the layer's residual stream post-attention+MLP and zeros it,
        then measures: (1) change in output probability for the target token,
        (2) change in depth_score when that layer's features are excluded.

        Args:
            prompt: The text to analyze.
            layer_idx: Which layer to ablate (0-indexed).
            target: Optional single-token target.

        Returns:
            AblationResult with probability and depth deltas.
        """
        target, target_id, baseline_prob, baseline_result = self._get_baseline(prompt, target)

        # Ablated forward pass — zero this layer's residual stream output
        prompt_tokens = self._inner.to_tokens(prompt)

        def zero_hook(act, hook):
            return torch.zeros_like(act)

        hook_name = f"blocks.{layer_idx}.hook_resid_post"

        with torch.no_grad():
            ablated_logits = self._inner.run_with_hooks(
                prompt_tokens,
                fwd_hooks=[(hook_name, zero_hook)],
            )
        ablated_probs = torch.softmax(ablated_logits[0, -1], dim=-1)
        ablated_prob = ablated_probs[target_id].item()

        prob_delta = baseline_prob - ablated_prob
        prob_delta_pct = (prob_delta / baseline_prob * 100) if baseline_prob > 0 else 0.0

        # Depth delta: recompute depth_score excluding this layer's features
        ablated_depth = self._depth_without_layer(baseline_result, layer_idx)
        depth_delta = baseline_result.depth_score - ablated_depth

        # Combined importance: geometric mean of probability impact and normalized prob delta
        prob_importance = min(1.0, abs(prob_delta_pct) / 100.0)
        causal_importance = prob_importance

        torch.cuda.empty_cache()
        gc.collect()

        return AblationResult(
            layer_idx=layer_idx,
            baseline_prob=baseline_prob,
            ablated_prob=ablated_prob,
            prob_delta=prob_delta,
            prob_delta_pct=prob_delta_pct,
            baseline_depth_score=baseline_result.depth_score,
            ablated_depth_score=ablated_depth,
            depth_delta=depth_delta,
            causal_importance=causal_importance,
        )

    def _depth_without_layer(self, result, exclude_layer: int) -> float:
        """Recompute depth_score excluding features from one layer."""
        if not result.layer_profile:
            return float("nan")

        layers_weighted = []
        for layer_idx, count in result.layer_profile.items():
            if int(layer_idx) != exclude_layer:
                layers_weighted.extend([int(layer_idx)] * count)

        if not layers_weighted:
            return float("nan")

        return float(np.mean(layers_weighted))

    def find_critical_layers(
        self,
        prompt: str,
        target: Optional[str] = None,
    ) -> List[AblationResult]:
        """Ablate every layer and rank by causal importance.

        This is the skeleton of the reasoning circuit — the ordered list of
        which layers the model's output actually depends on.

        Args:
            prompt: The text to analyze.
            target: Optional single-token target.

        Returns:
            List of AblationResult sorted by causal_importance (most important first).
        """
        if target is None:
            target = self.scorer._get_single_token_target(prompt)

        n_layers = self.scorer._n_layers
        results = []

        print(f"[CausalDepth] Ablating {n_layers} layers...")
        for layer_idx in range(n_layers):
            try:
                r = self.ablate_layer(prompt, layer_idx, target)
                results.append(r)
                if r.prob_delta_pct > 10:
                    print(f"  L{layer_idx}: prob delta {r.prob_delta_pct:+.1f}% — causally significant")
            except Exception as e:
                print(f"  L{layer_idx}: ablation failed — {e}")

        results.sort(key=lambda r: r.causal_importance, reverse=True)
        return results

    def causal_depth_score(
        self,
        prompt: str,
        target: Optional[str] = None,
    ) -> CausalDepthResult:
        """Compute depth score weighted by causal importance per layer.

        Unlike standard depth_score (mean layer weighted by feature count), this
        weights each layer by how much ablating it actually affects the output.
        A layer with many features but zero causal impact contributes nothing.

        Args:
            prompt: The text to analyze.
            target: Optional single-token target.

        Returns:
            CausalDepthResult with causal_depth_score and full ablation analysis.
        """
        t0 = time.time()

        if target is None:
            target = self.scorer._get_single_token_target(prompt)

        # Get baseline depth
        baseline_result = self.scorer.score(prompt, target)

        # Ablate all layers
        ablation_results = self.find_critical_layers(prompt, target)

        # Compute causal depth score: weighted mean layer where weight = causal importance
        total_importance = sum(r.causal_importance for r in ablation_results)

        if total_importance > 0:
            causal_score = sum(
                r.layer_idx * r.causal_importance for r in ablation_results
            ) / total_importance
        else:
            causal_score = baseline_result.depth_score

        # Top 3 critical layers
        top_3 = [r.layer_idx for r in ablation_results[:3]]
        top_3_mass = (
            sum(r.causal_importance for r in ablation_results[:3]) / total_importance
            if total_importance > 0
            else 0.0
        )

        compute_time = time.time() - t0

        return CausalDepthResult(
            prompt=prompt,
            causal_depth_score=causal_score,
            standard_depth_score=baseline_result.depth_score,
            critical_layers=ablation_results,
            top_3_layers=top_3,
            causal_concentration=top_3_mass,
            compute_time_s=compute_time,
        )
