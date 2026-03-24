"""
DEPTH SCORER — Reasoning Depth Measurement via SAE Circuit Attribution
=====================================================================

Measures how deeply a language model processes a prompt by analyzing
the distribution of activated sparse autoencoder (SAE) features across
layers. Based on the finding that surface recall concentrates feature
activation at output-proximal layers, while explanatory reasoning
engages more distributed mid-layer computation.

Paper: "Computational Graph Structure Distinguishes Surface Recall
from Explanatory Reasoning in Language Models"

Usage:
    from depth_scorer import DepthScorer

    scorer = DepthScorer()  # loads model once
    result = scorer.score("Why does fire spread once it starts?")
    print(result.depth_score)    # mean activated layer (lower = deeper reasoning)
    print(result.feature_count)  # total active features
    print(result.layer_profile)  # per-layer feature counts

API:
    uvicorn depth_api:app --host 0.0.0.0 --port 8400
    POST /score {"prompt": "..."}
    POST /compare {"surface": "...", "insight": "..."}

Requirements:
    pip install torch transformers transformer_lens einops
    pip install git+https://github.com/decoderesearch/circuit-tracer
"""

import gc
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import numpy as np


@dataclass
class DepthResult:
    """Result of a single prompt depth analysis."""
    prompt: str
    depth_score: float          # mean activated layer index (lower = deeper reasoning)
    feature_count: int          # total active features
    n_layers: int               # model layer count
    layer_profile: dict         # {layer_idx: feature_count}
    early_ratio: float          # % features in L0-L7
    mid_ratio: float            # % features in L8-L15
    late_ratio: float           # % features in L16+
    compute_time_s: float       # attribution time in seconds
    normalized_score: float     # 0-1 score (1 = deepest reasoning)

    def to_dict(self):
        d = asdict(self)
        d["layer_profile"] = {str(k): v for k, v in self.layer_profile.items()}
        return d


@dataclass
class CompareResult:
    """Result of comparing two prompts."""
    surface: DepthResult
    insight: DepthResult
    delta: float                # surface.depth_score - insight.depth_score (positive = insight is deeper)
    surface_deeper: bool        # True if surface has higher mean layer (expected for recall)
    feature_count_ratio: float  # insight_features / surface_features

    def to_dict(self):
        return {
            "surface": self.surface.to_dict(),
            "insight": self.insight.to_dict(),
            "delta": self.delta,
            "surface_deeper": self.surface_deeper,
            "feature_count_ratio": self.feature_count_ratio,
        }


class DepthScorer:
    """
    Measures reasoning depth via SAE feature attribution on Gemma-2-2B.

    The scorer computes mean activated layer index across all SAE features
    triggered by a prompt. Lower values indicate more distributed, earlier-layer
    computation (associated with explanatory reasoning). Higher values indicate
    output-proximal concentration (associated with surface recall).

    Calibration (from paper):
        Surface recall mean:  ~8.36 (30 pairs)
        Insight mean:         ~8.10 (30 pairs)
        Delta:                ~0.26 (p = 0.000051)
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        model_type: str = "gemma",
        dtype=torch.bfloat16,
        max_feature_nodes: int = 500,
        batch_size: int = 32,
        offload: str = "cpu",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.max_feature_nodes = max_feature_nodes
        self.batch_size = batch_size
        self.offload = offload
        self.device = device
        self._model = None
        self._n_layers = None

        print(f"[DepthScorer] Initializing with {model_name}...")
        self._load_model(model_name, model_type, dtype)

    def _load_model(self, model_name, model_type, dtype):
        """Load the ReplacementModel for circuit tracing."""
        from circuit_tracer import ReplacementModel

        t0 = time.time()
        self._model = ReplacementModel.from_pretrained(
            model_name, model_type, dtype=dtype
        )
        # Gemma-2-2B has 26 layers; try to read dynamically, fall back to default
        try:
            self._n_layers = self._model.cfg.n_layers
        except AttributeError:
            try:
                self._n_layers = self._model.model.cfg.n_layers
            except AttributeError:
                self._n_layers = 26
        print(f"[DepthScorer] Model loaded in {time.time() - t0:.1f}s ({self._n_layers} layers)")

    def _get_single_token_target(self, prompt: str) -> str:
        """Get the model's predicted next token (must be single-token for attribution)."""
        # Try to access the inner HookedTransformer through common attribute paths
        inner = None
        for attr in ["model", "hooked_model", "_model"]:
            candidate = getattr(self._model, attr, None)
            if candidate is not None and hasattr(candidate, "to_tokens"):
                inner = candidate
                break
        if inner is None:
            raise ValueError(
                "Cannot auto-detect target: inner model not accessible. "
                "Pass an explicit single-token target string."
            )
        tokens = inner.to_tokens(prompt)
        with torch.no_grad():
            logits = inner(tokens)
        pred_id = logits[0, -1].argmax().item()
        pred_token = inner.tokenizer.decode([pred_id])
        # Verify single-token
        re_encoded = inner.tokenizer.encode(pred_token, add_special_tokens=False)
        if len(re_encoded) != 1:
            raise ValueError(
                f"Predicted token '{pred_token}' re-encodes to {len(re_encoded)} tokens. "
                f"Adjust prompt to elicit a single-token completion."
            )
        return pred_token

    def _extract_features(self, prompt: str, target: Optional[str] = None) -> dict:
        """Run circuit attribution and extract active features with layer positions."""
        from circuit_tracer import attribute

        # If target provided, append to prompt; otherwise let circuit-tracer auto-detect
        if target is not None:
            full_input = prompt + target
            attribution_targets = [target]
        else:
            full_input = prompt
            attribution_targets = None

        t0 = time.time()
        attr_kwargs = {
            "attribution_targets": attribution_targets,
            "max_feature_nodes": self.max_feature_nodes,
            "batch_size": self.batch_size,
        }
        if self.offload:
            attr_kwargs["offload"] = self.offload

        graph = attribute(
            full_input,
            self._model,
            **attr_kwargs,
        )
        compute_time = time.time() - t0

        # Extract layers from active_features tensor: shape (n_active, 3) = (layer, pos, feature_idx)
        # Use proven pattern from experiment scripts — convert tensor to numpy immediately
        active_features = []
        af = graph.active_features
        if af is not None and hasattr(af, "shape") and af.shape[0] > 0:
            layers_np = af[:, 0].float().cpu().numpy()
            active_features = [(float(l), 0, 0) for l in layers_np]

        # Clean up GPU memory
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "active_features": active_features,
            "compute_time": compute_time,
            "target": target,
        }

    def score(self, prompt: str, target: Optional[str] = None) -> DepthResult:
        """
        Score a single prompt's reasoning depth.

        Args:
            prompt: The text to analyze
            target: Optional single-token target. Auto-detected if not provided.

        Returns:
            DepthResult with depth_score, feature_count, layer_profile, etc.
        """
        result = self._extract_features(prompt, target)
        features = result["active_features"]

        if not features:
            return DepthResult(
                prompt=prompt,
                depth_score=float("nan"),
                feature_count=0,
                n_layers=self._n_layers,
                layer_profile={},
                early_ratio=0.0,
                mid_ratio=0.0,
                late_ratio=0.0,
                compute_time_s=result["compute_time"],
                normalized_score=0.0,
            )

        # Extract layers
        layers = [f[0] for f in features if isinstance(f[0], (int, float))]
        if not layers:
            layers = [0]

        layers_arr = np.array(layers, dtype=float)
        mean_layer = float(np.mean(layers_arr))

        # Layer profile: count features per layer
        layer_profile = {}
        for l in layers:
            layer_profile[int(l)] = layer_profile.get(int(l), 0) + 1

        # Band ratios
        total = len(layers)
        early = sum(1 for l in layers if l < 8)
        mid = sum(1 for l in layers if 8 <= l < 16)
        late = sum(1 for l in layers if l >= 16)

        # Normalized score: 0 = shallowest (all output layers), 1 = deepest (all early layers)
        # Linear rescale from observed range [7.5, 9.0] -> [1, 0]
        normalized = max(0.0, min(1.0, (9.0 - mean_layer) / 1.5))

        return DepthResult(
            prompt=prompt,
            depth_score=mean_layer,
            feature_count=total,
            n_layers=self._n_layers,
            layer_profile=layer_profile,
            early_ratio=early / total if total > 0 else 0.0,
            mid_ratio=mid / total if total > 0 else 0.0,
            late_ratio=late / total if total > 0 else 0.0,
            compute_time_s=result["compute_time"],
            normalized_score=normalized,
        )

    def compare(
        self,
        surface_prompt: str,
        insight_prompt: str,
        surface_target: Optional[str] = None,
        insight_target: Optional[str] = None,
    ) -> CompareResult:
        """
        Compare two prompts and measure the depth differential.

        Args:
            surface_prompt: Factual/recall prompt
            insight_prompt: Explanatory/reasoning prompt

        Returns:
            CompareResult with delta and comparison metrics
        """
        surface_result = self.score(surface_prompt, surface_target)

        # Clear between pairs
        torch.cuda.empty_cache()
        gc.collect()

        insight_result = self.score(insight_prompt, insight_target)

        delta = surface_result.depth_score - insight_result.depth_score

        feature_ratio = (
            insight_result.feature_count / surface_result.feature_count
            if surface_result.feature_count > 0
            else float("inf")
        )

        return CompareResult(
            surface=surface_result,
            insight=insight_result,
            delta=delta,
            surface_deeper=(delta > 0),
            feature_count_ratio=feature_ratio,
        )

    def score_batch(self, prompts: list[str], targets: Optional[list[str]] = None) -> list[DepthResult]:
        """Score multiple prompts with GPU memory management between each."""
        results = []
        targets = targets or [None] * len(prompts)

        for i, (prompt, target) in enumerate(zip(prompts, targets)):
            print(f"[DepthScorer] Scoring {i+1}/{len(prompts)}...")
            try:
                result = self.score(prompt, target)
                results.append(result)
            except Exception as e:
                print(f"[DepthScorer] Failed on prompt {i+1}: {e}")
                results.append(None)

            # Memory management between prompts
            torch.cuda.empty_cache()
            gc.collect()

        return results


# ── Calibration baselines from the paper ──────────────────────────────

CALIBRATION = {
    "surface_mean": 8.359,
    "insight_mean": 8.102,
    "delta_mean": 0.257,
    "p_value": 0.000051,
    "n_pairs": 30,
    "model": "gemma-2-2b",
    "note": "Pre-registered replication, LOO robust",
}


def interpret_score(depth_score: float) -> str:
    """Human-readable interpretation of a depth score."""
    import math
    if math.isnan(depth_score):
        return "no data — attribution returned zero active features"
    if depth_score >= 8.5:
        return "shallow — direct lookup pattern, minimal intermediate computation"
    elif depth_score >= 8.2:
        return "moderate — some intermediate processing, typical of constrained reasoning"
    elif depth_score >= 7.9:
        return "deep — distributed mid-layer computation, consistent with explanatory reasoning"
    else:
        return "very deep — extensive distributed processing across early and mid layers"


if __name__ == "__main__":
    # Quick test
    scorer = DepthScorer()

    print("\n" + "=" * 60)
    print("DEPTH SCORER — Quick Test")
    print("=" * 60)

    surface = "Fire needs oxygen, fuel, and heat, and all three are present when something"
    insight = "Fire needs oxygen, fuel, and heat, and each one feeds the others when something"

    result = scorer.compare(surface, insight)
    print(f"\nSurface depth: {result.surface.depth_score:.3f} ({result.surface.feature_count} features)")
    print(f"Insight depth: {result.insight.depth_score:.3f} ({result.insight.feature_count} features)")
    print(f"Delta: +{result.delta:.3f} ({'✓ expected direction' if result.surface_deeper else '✗ unexpected'})")
    print(f"\nSurface: {interpret_score(result.surface.depth_score)}")
    print(f"Insight: {interpret_score(result.insight.depth_score)}")
