"""
COGNITIVE DARK MATTER DETECTOR — Finding What Models Think But Never Say
========================================================================

Dark matter features are SAE features that activate strongly during
processing but contribute negligibly to the output logits. They represent
computation the model performs that never surfaces — latent reasoning,
suppressed knowledge, internal conflict, or structured noise. Measuring
the dark matter ratio reveals how much of a model's cognitive energy
is invisible to its outputs.
"""

import gc
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

import torch
import numpy as np


@dataclass
class DarkFeature:
    """A single dark matter feature — high activation, low output contribution."""
    layer: int
    feature_idx: int
    activation_strength: float     # how strongly it fires
    logit_contribution: float      # how much it affects the output
    darkness_ratio: float          # activation / max(contribution, eps) — higher = darker


@dataclass
class DarkMatterResult:
    """Dark matter analysis of a single prompt."""
    prompt: str
    dark_matter_ratio: float       # fraction of computation that's invisible
    n_dark_features: int           # count of dark features
    n_total_features: int          # total active features
    dark_features: List[DarkFeature]  # the dark features themselves
    dark_energy: float             # sum of dark feature activations / total activation energy
    darkest_layer: Optional[int]   # which layer has the most dark matter
    compute_time_s: float

    def to_dict(self):
        d = asdict(self)
        d["dark_features"] = [asdict(f) for f in self.dark_features[:20]]
        return d


@dataclass
class DarkMatterProfile:
    """Batch dark matter analysis across multiple prompts."""
    n_prompts: int
    mean_dark_ratio: float
    std_dark_ratio: float
    mean_dark_energy: float
    per_prompt: List[DarkMatterResult]
    darkest_layers_histogram: Dict[int, int]  # layer → count of times it was darkest

    def to_dict(self):
        return {
            "n_prompts": self.n_prompts,
            "mean_dark_ratio": self.mean_dark_ratio,
            "std_dark_ratio": self.std_dark_ratio,
            "mean_dark_energy": self.mean_dark_energy,
            "darkest_layers_histogram": {str(k): v for k, v in self.darkest_layers_histogram.items()},
        }


class DarkMatterAnalyzer:
    """Detects features that activate strongly but contribute nothing to the output.

    The model's 'unconscious' — computation that happens but never surfaces in the
    response. A model with 40% dark matter is doing significant work that its output
    doesn't reflect. Whether that's latent reasoning, suppressed knowledge, or noise
    is the question this instrument helps answer.
    """

    def __init__(
        self,
        activation_threshold: float = 0.3,
        contribution_threshold: float = 0.05,
    ):
        """
        Args:
            activation_threshold: Minimum activation strength to consider a feature 'active'.
            contribution_threshold: Maximum logit contribution for a feature to be 'dark'.
        """
        self.activation_threshold = activation_threshold
        self.contribution_threshold = contribution_threshold

    def find_dark_features(
        self,
        prompt: str,
        scorer,
        target: Optional[str] = None,
    ) -> DarkMatterResult:
        """Identify features with high activation but negligible output contribution.

        Runs attribution, extracts both activation magnitude and logit contribution
        for each feature. Features where activation >> contribution are dark matter.

        Args:
            prompt: The text to analyze.
            scorer: An initialized DepthScorer instance.
            target: Optional single-token target.

        Returns:
            DarkMatterResult with dark features and ratios.
        """
        t0 = time.time()

        activations, contributions, layers, feature_indices = self._extract_dual_signals(
            prompt, scorer, target
        )

        if len(activations) == 0:
            return DarkMatterResult(
                prompt=prompt,
                dark_matter_ratio=0.0,
                n_dark_features=0,
                n_total_features=0,
                dark_features=[],
                dark_energy=0.0,
                darkest_layer=None,
                compute_time_s=time.time() - t0,
            )

        dark_features = []
        dark_activation_sum = 0.0
        total_activation_sum = float(np.sum(activations))
        dark_layer_counts = {}

        for i in range(len(activations)):
            act = activations[i]
            contrib = contributions[i]
            layer = int(layers[i])
            feat_idx = int(feature_indices[i]) if i < len(feature_indices) else 0

            is_dark = act >= self.activation_threshold and contrib < self.contribution_threshold

            if is_dark:
                darkness = act / max(contrib, 1e-8)
                dark_features.append(DarkFeature(
                    layer=layer,
                    feature_idx=feat_idx,
                    activation_strength=float(act),
                    logit_contribution=float(contrib),
                    darkness_ratio=float(darkness),
                ))
                dark_activation_sum += act
                dark_layer_counts[layer] = dark_layer_counts.get(layer, 0) + 1

        # Sort by darkness ratio
        dark_features.sort(key=lambda f: f.darkness_ratio, reverse=True)

        n_total = len(activations)
        n_dark = len(dark_features)
        dark_ratio = n_dark / n_total if n_total > 0 else 0.0
        dark_energy = dark_activation_sum / total_activation_sum if total_activation_sum > 0 else 0.0

        darkest_layer = max(dark_layer_counts, key=dark_layer_counts.get) if dark_layer_counts else None

        return DarkMatterResult(
            prompt=prompt,
            dark_matter_ratio=dark_ratio,
            n_dark_features=n_dark,
            n_total_features=n_total,
            dark_features=dark_features,
            dark_energy=dark_energy,
            darkest_layer=darkest_layer,
            compute_time_s=time.time() - t0,
        )

    def _extract_dual_signals(
        self,
        prompt: str,
        scorer,
        target: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract both activation strength and logit contribution per feature.

        Returns parallel arrays: (activations, contributions, layers, feature_indices).
        """
        from circuit_tracer import attribute

        if target is None:
            target = scorer._get_single_token_target(prompt)

        full_input = prompt + target

        attr_kwargs = {
            "attribution_targets": [target],
            "max_feature_nodes": scorer.max_feature_nodes,
            "batch_size": scorer.batch_size,
        }
        if scorer.offload:
            attr_kwargs["offload"] = scorer.offload

        graph = attribute(full_input, scorer._model, **attr_kwargs)

        activations = []
        contributions = []
        layers_out = []
        feature_indices = []

        af = graph.active_features
        if af is None or not hasattr(af, "shape") or af.shape[0] == 0:
            del graph
            torch.cuda.empty_cache()
            gc.collect()
            return np.array([]), np.array([]), np.array([]), np.array([])

        layers_np = af[:, 0].float().cpu().numpy()
        feat_idx_np = af[:, 2].float().cpu().numpy() if af.shape[1] > 2 else np.zeros(len(layers_np))

        # Extract activations and attributions from graph nodes
        if hasattr(graph, "nodes") and graph.nodes is not None:
            for node in graph.nodes:
                # Activation: how strongly the feature fires
                act_val = 0.0
                if hasattr(node, "act") and node.act is not None:
                    v = node.act
                    if isinstance(v, torch.Tensor):
                        v = v.float().cpu().item()
                    act_val = abs(float(v))

                # Attribution: how much it contributes to the output logit
                attr_val = 0.0
                if hasattr(node, "attribution") and node.attribution is not None:
                    v = node.attribution
                    if isinstance(v, torch.Tensor):
                        v = v.float().cpu().item()
                    attr_val = abs(float(v))

                activations.append(act_val)
                contributions.append(attr_val)

        # If we got node-level data matching feature count, use it
        if len(activations) == len(layers_np):
            layers_out = layers_np
            feature_indices = feat_idx_np
        else:
            # Fallback: use magnitudes extraction from scorer
            mags = scorer._extract_magnitudes(graph, len(layers_np))
            if mags is not None:
                # Use magnitudes as activations; estimate contributions via a heuristic:
                # features in output-proximal layers contribute more to logits
                n_layers = scorer._n_layers
                for i, (layer, mag) in enumerate(zip(layers_np, mags)):
                    activations.append(float(mag))
                    # Heuristic: contribution scales with proximity to output layer
                    layer_proximity = layer / max(n_layers - 1, 1)
                    estimated_contribution = float(mag) * (0.1 + 0.9 * layer_proximity)
                    contributions.append(estimated_contribution)
                layers_out = layers_np
                feature_indices = feat_idx_np
            else:
                # Last resort: uniform activations, no dark matter detectable
                activations = [1.0] * len(layers_np)
                contributions = [1.0] * len(layers_np)
                layers_out = layers_np
                feature_indices = feat_idx_np

        del graph
        torch.cuda.empty_cache()
        gc.collect()

        return (
            np.array(activations, dtype=np.float64),
            np.array(contributions, dtype=np.float64),
            np.array(layers_out, dtype=np.float64),
            np.array(feature_indices, dtype=np.float64),
        )

    def dark_matter_ratio(
        self,
        prompt: str,
        scorer,
        target: Optional[str] = None,
    ) -> float:
        """Returns what fraction of computational energy goes into invisible features.

        A quick summary metric — use find_dark_features() for full analysis.

        Args:
            prompt: The text to analyze.
            scorer: An initialized DepthScorer instance.

        Returns:
            Float between 0 and 1. 0.4 means 40% of compute is dark.
        """
        result = self.find_dark_features(prompt, scorer, target)
        return result.dark_matter_ratio

    def dark_matter_profile(
        self,
        prompts: List[str],
        scorer,
    ) -> DarkMatterProfile:
        """Batch dark matter analysis across multiple prompts.

        Maps which prompts and domains generate more dark matter.

        Args:
            prompts: List of prompts to analyze.
            scorer: An initialized DepthScorer instance.

        Returns:
            DarkMatterProfile with aggregate statistics.
        """
        results = []
        dark_layer_histogram = {}

        for i, prompt in enumerate(prompts):
            print(f"[DarkMatter] Analyzing {i+1}/{len(prompts)}...")
            try:
                r = self.find_dark_features(prompt, scorer)
                results.append(r)
                if r.darkest_layer is not None:
                    dark_layer_histogram[r.darkest_layer] = dark_layer_histogram.get(r.darkest_layer, 0) + 1
            except Exception as e:
                print(f"  Failed: {e}")

            torch.cuda.empty_cache()
            gc.collect()

        if not results:
            return DarkMatterProfile(
                n_prompts=0, mean_dark_ratio=0, std_dark_ratio=0,
                mean_dark_energy=0, per_prompt=[], darkest_layers_histogram={},
            )

        ratios = [r.dark_matter_ratio for r in results]
        energies = [r.dark_energy for r in results]

        return DarkMatterProfile(
            n_prompts=len(results),
            mean_dark_ratio=float(np.mean(ratios)),
            std_dark_ratio=float(np.std(ratios)),
            mean_dark_energy=float(np.mean(energies)),
            per_prompt=results,
            darkest_layers_histogram=dark_layer_histogram,
        )


def interpret_dark_matter(ratio: float) -> str:
    """Human-readable interpretation of a dark matter ratio."""
    pct = ratio * 100
    if ratio >= 0.5:
        return (
            f"{pct:.0f}% of this model's computation is invisible to the output. "
            f"Majority dark matter — significant latent processing, suppressed knowledge, "
            f"or internal conflict that never surfaces in the response."
        )
    elif ratio >= 0.3:
        return (
            f"{pct:.0f}% of this model's computation is invisible to the output. "
            f"Substantial dark matter — notable background computation that may represent "
            f"latent reasoning pathways or suppressed alternatives."
        )
    elif ratio >= 0.1:
        return (
            f"{pct:.0f}% of this model's computation is invisible to the output. "
            f"Moderate dark matter — some features activate without contributing, "
            f"possibly representing contextual processing or noise."
        )
    else:
        return (
            f"{pct:.0f}% of this model's computation is invisible to the output. "
            f"Low dark matter — most activated features contribute to the output. "
            f"Efficient, transparent computation."
        )
