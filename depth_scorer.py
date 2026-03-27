"""
DEPTH SCORER — Cognitive Fingerprinting Engine via SAE Circuit Attribution
=========================================================================

Measures how deeply a language model processes a prompt by analyzing
the distribution of activated sparse autoencoder (SAE) features across
layers. Based on the finding that surface recall concentrates feature
activation at output-proximal layers, while explanatory reasoning
engages more distributed mid-layer computation.

Paper: "Computational Graph Structure Distinguishes Surface Recall
from Explanatory Reasoning in Language Models"

Features:
    - Weighted attribution magnitude scoring
    - Shannon entropy of layer distributions
    - Token-level depth trajectory analysis
    - Cognitive fingerprint vectors for clustering
    - Multi-model support with auto-calibration
    - Bootstrap confidence intervals
    - Anomaly detection on results
    - Streaming score mode for real-time APIs

Usage:
    from depth_scorer import DepthScorer

    scorer = DepthScorer()  # loads model once
    result = scorer.score("Why does fire spread once it starts?")
    print(result.depth_score)           # mean activated layer
    print(result.weighted_depth_score)  # attribution-weighted mean layer
    print(result.layer_entropy)         # Shannon entropy of layer distribution
    print(result.is_anomalous)          # anomaly flag

    # Cognitive fingerprint
    fp = scorer.fingerprint(result)     # numpy vector for clustering/ML

    # Depth trajectory across generated tokens
    trajectory = scorer.score_trajectory("Why does fire spread?", max_tokens=10)
    for token, depth, entropy in trajectory:
        print(f"{token}: depth={depth:.3f}, entropy={entropy:.3f}")

    # Bootstrap confidence intervals
    result = scorer.score("prompt", n_bootstrap=20)
    print(f"depth={result.depth_score:.3f} CI=[{result.depth_score_ci_low:.3f}, {result.depth_score_ci_high:.3f}]")

API:
    uvicorn depth_api:app --host 0.0.0.0 --port 8400
    POST /score {"prompt": "..."}
    POST /compare {"surface": "...", "insight": "..."}

Requirements:
    pip install torch transformers transformer_lens einops
    pip install git+https://github.com/decoderesearch/circuit-tracer
"""

import gc
import math
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, Generator, List, Tuple

import torch
import numpy as np


# ── Model Registry ─────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "google/gemma-2-2b": {
        "model_type": "gemma",
        "n_layers": 26,
        "bands": (8, 16),
        "norm_range": (7.5, 9.0),
        "calibration": {
            "surface_mean": 8.359,
            "insight_mean": 8.102,
            "delta_mean": 0.257,
            "p_value": 0.000051,
            "n_pairs": 30,
        },
    },
    "mistralai/Mistral-7B-v0.1": {
        "model_type": "mistral",
        "n_layers": 32,
        "bands": (11, 22),
        "norm_range": (7.5, 9.0),
        "calibration": None,
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "model_type": "mistral",
        "n_layers": 32,
        "bands": (11, 22),
        "norm_range": (7.5, 9.0),
        "calibration": None,
    },
    "meta-llama/Llama-2-7b-hf": {
        "model_type": "llama",
        "n_layers": 32,
        "bands": (11, 22),
        "norm_range": (7.5, 9.0),
        "calibration": None,
    },
}


@dataclass
class DepthResult:
    """Result of a single prompt depth analysis."""
    prompt: str
    depth_score: float          # mean activated layer index (lower = deeper reasoning)
    feature_count: int          # total active features
    n_layers: int               # model layer count
    layer_profile: dict         # {layer_idx: feature_count}
    early_ratio: float          # % features in early band
    mid_ratio: float            # % features in mid band
    late_ratio: float           # % features in late band
    compute_time_s: float       # attribution time in seconds
    normalized_score: float     # 0-1 score (1 = deepest reasoning)
    # ── Weighted attribution (§1) ──
    weighted_depth_score: Optional[float] = None
    # ── Layer entropy (§2) ──
    layer_entropy: Optional[float] = None
    # ── Model tag (§5) ──
    model_id: Optional[str] = None
    # ── Anomaly detection (§7) ──
    is_anomalous: Optional[bool] = None
    anomaly_reasons: Optional[list] = None
    # ── Bootstrap confidence (§6) ──
    depth_score_ci_low: Optional[float] = None
    depth_score_ci_high: Optional[float] = None
    depth_score_std: Optional[float] = None

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
    # ── Cross-model delta (§5) ──
    cross_model: Optional[dict] = None

    def to_dict(self):
        d = {
            "surface": self.surface.to_dict(),
            "insight": self.insight.to_dict(),
            "delta": self.delta,
            "surface_deeper": self.surface_deeper,
            "feature_count_ratio": self.feature_count_ratio,
        }
        if self.cross_model is not None:
            d["cross_model"] = self.cross_model
        return d


class DepthScorer:
    """
    Cognitive fingerprinting engine via SAE feature attribution.

    Measures reasoning depth, layer entropy, attribution-weighted depth,
    and produces cognitive fingerprint vectors for any supported model.

    Calibration (from paper, gemma-2-2b):
        Surface recall mean:  ~8.36 (30 pairs)
        Insight mean:         ~8.10 (30 pairs)
        Delta:                ~0.26 (p = 0.000051)
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        model_type: Optional[str] = None,
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
        self._compute_times: List[float] = []

        # Auto-detect model_type from registry if not specified
        registry_entry = MODEL_REGISTRY.get(model_name, {})
        if model_type is None:
            model_type = registry_entry.get("model_type", "gemma")
        self._model_type = model_type

        # Layer band boundaries
        self._bands = registry_entry.get("bands")
        self._norm_range = registry_entry.get("norm_range", (7.5, 9.0))

        print(f"[DepthScorer] Initializing with {model_name}...")
        self._load_model(model_name, model_type, dtype)

        # Set default bands if not in registry
        if self._bands is None:
            third = self._n_layers // 3
            self._bands = (third, 2 * third)

    def _load_model(self, model_name, model_type, dtype):
        """Load the ReplacementModel for circuit tracing."""
        from circuit_tracer import ReplacementModel

        t0 = time.time()
        # Some models (Qwen3, etc.) require trust_remote_code=True for TransformerLens
        extra_kwargs = {}
        if "qwen" in model_name.lower() or "qwen" in model_type.lower():
            extra_kwargs["trust_remote_code"] = True
        self._model = ReplacementModel.from_pretrained(
            model_name, model_type, dtype=dtype, **extra_kwargs
        )
        # Try to read layer count dynamically, fall back to registry, then default
        try:
            self._n_layers = self._model.cfg.n_layers
        except AttributeError:
            try:
                self._n_layers = self._model.model.cfg.n_layers
            except AttributeError:
                registry_entry = MODEL_REGISTRY.get(model_name, {})
                self._n_layers = registry_entry.get("n_layers", 26)
        print(f"[DepthScorer] Model loaded in {time.time() - t0:.1f}s ({self._n_layers} layers)")

    def _get_inner_model(self):
        """Access the inner HookedTransformer through common attribute paths."""
        for attr in ["model", "hooked_model", "_model"]:
            candidate = getattr(self._model, attr, None)
            if candidate is not None and hasattr(candidate, "to_tokens"):
                return candidate
        return None

    def _get_single_token_target(self, prompt: str) -> str:
        """Get the model's predicted next token (must be single-token for attribution)."""
        inner = self._get_inner_model()
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
        """Run circuit attribution and extract active features with layer positions and magnitudes."""
        from circuit_tracer import attribute

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

        # Extract layers and magnitudes from active_features tensor
        active_features = []
        magnitudes = None
        af = graph.active_features
        if af is not None and hasattr(af, "shape") and af.shape[0] > 0:
            layers_np = af[:, 0].float().cpu().numpy()
            active_features = [(float(l), 0, 0) for l in layers_np]

            # Extract attribution magnitudes from graph nodes
            magnitudes = self._extract_magnitudes(graph, len(layers_np))

        # Clean up GPU memory
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "active_features": active_features,
            "magnitudes": magnitudes,
            "compute_time": compute_time,
            "target": target,
        }

    def _extract_magnitudes(self, graph, n_features: int) -> Optional[np.ndarray]:
        """Extract attribution magnitudes from the computational graph.

        Tries multiple API paths for robustness across circuit-tracer versions.
        Returns absolute magnitudes as a numpy array, or None if unavailable.
        """
        # Strategy 1: graph.nodes with .attribution or .act
        if hasattr(graph, "nodes") and graph.nodes is not None:
            try:
                mags = []
                for node in graph.nodes:
                    if hasattr(node, "attribution") and node.attribution is not None:
                        val = node.attribution
                        if isinstance(val, torch.Tensor):
                            val = val.float().cpu().item()
                        mags.append(abs(float(val)))
                    elif hasattr(node, "act") and node.act is not None:
                        val = node.act
                        if isinstance(val, torch.Tensor):
                            val = val.float().cpu().item()
                        mags.append(abs(float(val)))
                if len(mags) == n_features:
                    return np.array(mags, dtype=np.float64)
            except Exception:
                pass

        # Strategy 2: feature_attributions tensor
        if hasattr(graph, "feature_attributions") and graph.feature_attributions is not None:
            try:
                fa = graph.feature_attributions
                if isinstance(fa, torch.Tensor):
                    fa = fa.float().cpu().numpy()
                fa = np.abs(fa).flatten()
                if len(fa) == n_features:
                    return fa.astype(np.float64)
            except Exception:
                pass

        # Strategy 3: node_acts tensor
        if hasattr(graph, "node_acts") and graph.node_acts is not None:
            try:
                na = graph.node_acts
                if isinstance(na, torch.Tensor):
                    na = na.float().cpu().numpy()
                na = np.abs(na).flatten()
                if len(na) == n_features:
                    return na.astype(np.float64)
            except Exception:
                pass

        return None

    def _compute_layer_entropy(self, layer_profile: dict, total: int) -> float:
        """Shannon entropy of the layer distribution (bits).

        High entropy = features spread across many layers = deep distributed reasoning.
        Low entropy = features concentrated in few layers = surface recall pattern.
        """
        if total <= 0:
            return 0.0
        probs = np.array([count / total for count in layer_profile.values()], dtype=np.float64)
        probs = probs[probs > 0]
        if len(probs) <= 1:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def _detect_anomalies(self, feature_count: int, layer_profile: dict,
                          compute_time: float) -> Tuple[bool, List[str]]:
        """Check result for anomalous patterns."""
        reasons = []

        # Low feature count
        if feature_count < 10:
            reasons.append(f"suspiciously low feature count ({feature_count} < 10)")

        # Layer concentration: >80% in a single layer
        if feature_count > 0 and layer_profile:
            max_layer_count = max(layer_profile.values())
            if max_layer_count / feature_count > 0.8:
                dominant_layer = max(layer_profile, key=layer_profile.get)
                reasons.append(
                    f"layer concentration: {max_layer_count / feature_count:.0%} of features "
                    f"in layer {dominant_layer}"
                )

        # Compute time anomaly: 10x rolling average
        if len(self._compute_times) >= 3:
            rolling_avg = np.mean(self._compute_times[-20:])
            if compute_time > 10 * rolling_avg:
                reasons.append(
                    f"compute time {compute_time:.1f}s is {compute_time / rolling_avg:.1f}x "
                    f"the rolling average ({rolling_avg:.1f}s)"
                )

        return (len(reasons) > 0, reasons)

    def score(
        self,
        prompt: str,
        target: Optional[str] = None,
        n_bootstrap: Optional[int] = None,
    ) -> DepthResult:
        """
        Score a single prompt's reasoning depth.

        Args:
            prompt: The text to analyze.
            target: Optional single-token target. Auto-detected if not provided.
            n_bootstrap: If set, run n perturbation passes and return confidence intervals.

        Returns:
            DepthResult with depth metrics, entropy, anomaly flags, and optional CI.
        """
        if n_bootstrap is not None and n_bootstrap > 1:
            return self._score_with_bootstrap(prompt, target, n_bootstrap)

        result = self._extract_features(prompt, target)
        return self._build_depth_result(prompt, result)

    def _build_depth_result(self, prompt: str, extraction: dict) -> DepthResult:
        """Build a DepthResult from raw extraction output."""
        features = extraction["active_features"]
        magnitudes = extraction["magnitudes"]
        compute_time = extraction["compute_time"]

        early_bound, mid_bound = self._bands

        if not features:
            is_anom, anom_reasons = self._detect_anomalies(0, {}, compute_time)
            self._compute_times.append(compute_time)
            return DepthResult(
                prompt=prompt,
                depth_score=float("nan"),
                feature_count=0,
                n_layers=self._n_layers,
                layer_profile={},
                early_ratio=0.0,
                mid_ratio=0.0,
                late_ratio=0.0,
                compute_time_s=compute_time,
                normalized_score=0.0,
                weighted_depth_score=float("nan"),
                layer_entropy=0.0,
                model_id=self.model_name,
                is_anomalous=is_anom,
                anomaly_reasons=anom_reasons if anom_reasons else None,
            )

        layers = [f[0] for f in features if isinstance(f[0], (int, float))]
        if not layers:
            layers = [0]

        layers_arr = np.array(layers, dtype=np.float64)
        mean_layer = float(np.mean(layers_arr))
        total = len(layers)

        # Layer profile
        layer_profile = {}
        for l in layers:
            layer_profile[int(l)] = layer_profile.get(int(l), 0) + 1

        # Band ratios (dynamic boundaries)
        early = sum(1 for l in layers if l < early_bound)
        mid = sum(1 for l in layers if early_bound <= l < mid_bound)
        late = sum(1 for l in layers if l >= mid_bound)

        # Normalized score
        norm_low, norm_high = self._norm_range
        norm_span = norm_high - norm_low
        normalized = max(0.0, min(1.0, (norm_high - mean_layer) / norm_span)) if norm_span > 0 else 0.5

        # Weighted depth score (§1)
        weighted_depth = self._compute_weighted_depth(layers_arr, magnitudes)

        # Layer entropy (§2)
        entropy = self._compute_layer_entropy(layer_profile, total)

        # Anomaly detection (§7)
        is_anom, anom_reasons = self._detect_anomalies(total, layer_profile, compute_time)
        self._compute_times.append(compute_time)

        return DepthResult(
            prompt=prompt,
            depth_score=mean_layer,
            feature_count=total,
            n_layers=self._n_layers,
            layer_profile=layer_profile,
            early_ratio=early / total if total > 0 else 0.0,
            mid_ratio=mid / total if total > 0 else 0.0,
            late_ratio=late / total if total > 0 else 0.0,
            compute_time_s=compute_time,
            normalized_score=normalized,
            weighted_depth_score=weighted_depth,
            layer_entropy=entropy,
            model_id=self.model_name,
            is_anomalous=is_anom,
            anomaly_reasons=anom_reasons if anom_reasons else None,
        )

    def _compute_weighted_depth(
        self, layers: np.ndarray, magnitudes: Optional[np.ndarray]
    ) -> float:
        """Compute attribution-magnitude-weighted mean layer index.

        If magnitudes are unavailable, falls back to unweighted mean.
        """
        if magnitudes is None or len(magnitudes) != len(layers):
            return float(np.mean(layers))

        total_mag = np.sum(magnitudes)
        if total_mag == 0:
            return float(np.mean(layers))

        return float(np.sum(layers * magnitudes) / total_mag)

    def _score_with_bootstrap(
        self, prompt: str, target: Optional[str], n_bootstrap: int
    ) -> DepthResult:
        """Run attribution with prompt perturbations to compute confidence intervals."""
        perturbations = self._generate_perturbations(prompt, n_bootstrap)

        scores = []
        # Score the original prompt first
        base_result = self.score(prompt, target)
        scores.append(base_result.depth_score)

        # Score perturbations
        for perturbed in perturbations:
            try:
                torch.cuda.empty_cache()
                gc.collect()
                r = self.score(perturbed, target)
                if not math.isnan(r.depth_score):
                    scores.append(r.depth_score)
            except Exception:
                continue

        scores_arr = np.array(scores, dtype=np.float64)

        if len(scores_arr) >= 2:
            base_result.depth_score_ci_low = float(np.percentile(scores_arr, 2.5))
            base_result.depth_score_ci_high = float(np.percentile(scores_arr, 97.5))
            base_result.depth_score_std = float(np.std(scores_arr))
        else:
            base_result.depth_score_ci_low = base_result.depth_score
            base_result.depth_score_ci_high = base_result.depth_score
            base_result.depth_score_std = 0.0

        return base_result

    def _generate_perturbations(self, prompt: str, n: int) -> List[str]:
        """Generate slight prompt perturbations for bootstrap scoring."""
        perturbations = []
        variants = [
            prompt + " ",
            prompt.rstrip() + ".",
            prompt.rstrip() + ",",
            prompt + "  ",
            " " + prompt,
            prompt.rstrip(),
            prompt.rstrip() + " —",
            prompt.rstrip() + ";",
            prompt.rstrip() + " ...",
            prompt + "\n",
        ]
        # Cycle through variants up to n-1 (first run is the original)
        for i in range(min(n - 1, len(variants))):
            perturbations.append(variants[i])
        # If n > len(variants)+1, add duplicates with double-spaces
        for i in range(max(0, n - 1 - len(variants))):
            perturbations.append(variants[i % len(variants)] + " ")
        return perturbations

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

    def compare_cross_model(
        self, result_a: DepthResult, result_b: DepthResult
    ) -> CompareResult:
        """Compare the same prompt scored on two different models.

        Takes two DepthResult objects (typically from different DepthScorer instances)
        and computes cross-model deltas on all cognitive dimensions.

        Args:
            result_a: DepthResult from model A (treated as 'surface' slot)
            result_b: DepthResult from model B (treated as 'insight' slot)

        Returns:
            CompareResult with cross_model dict containing per-dimension deltas.
        """
        delta = result_a.depth_score - result_b.depth_score

        feature_ratio = (
            result_b.feature_count / result_a.feature_count
            if result_a.feature_count > 0
            else float("inf")
        )

        cross_model = {
            "model_a": result_a.model_id,
            "model_b": result_b.model_id,
            "depth_score_delta": delta,
            "weighted_depth_delta": (
                (result_a.weighted_depth_score or 0) - (result_b.weighted_depth_score or 0)
            ),
            "entropy_delta": (
                (result_a.layer_entropy or 0) - (result_b.layer_entropy or 0)
            ),
            "early_ratio_delta": result_a.early_ratio - result_b.early_ratio,
            "mid_ratio_delta": result_a.mid_ratio - result_b.mid_ratio,
            "late_ratio_delta": result_a.late_ratio - result_b.late_ratio,
        }

        return CompareResult(
            surface=result_a,
            insight=result_b,
            delta=delta,
            surface_deeper=(delta > 0),
            feature_count_ratio=feature_ratio,
            cross_model=cross_model,
        )

    def score_batch(self, prompts: list, targets: Optional[list] = None) -> list:
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

            torch.cuda.empty_cache()
            gc.collect()

        return results

    def score_trajectory(
        self,
        prompt: str,
        max_tokens: int = 20,
    ) -> List[Tuple[str, float, float]]:
        """Score depth at each token position in the completion.

        Generates tokens autoregressively and runs attribution after each,
        revealing how depth evolves across the model's completion.

        Args:
            prompt: The initial prompt.
            max_tokens: Maximum tokens to generate and score.

        Returns:
            List of (token, depth_score, layer_entropy) tuples.
        """
        inner = self._get_inner_model()
        if inner is None:
            raise ValueError(
                "Cannot run trajectory: inner model not accessible for generation."
            )

        trajectory = []
        current_prompt = prompt

        for step in range(max_tokens):
            # Predict next token
            tokens = inner.to_tokens(current_prompt)
            with torch.no_grad():
                logits = inner(tokens)
            pred_id = logits[0, -1].argmax().item()
            pred_token = inner.tokenizer.decode([pred_id])

            # Verify single-token for attribution
            re_encoded = inner.tokenizer.encode(pred_token, add_special_tokens=False)
            if len(re_encoded) != 1:
                break  # Can't attribute multi-token prediction

            # Check for EOS
            eos_id = getattr(inner.tokenizer, "eos_token_id", None)
            if pred_id == eos_id:
                break

            # Run attribution at this position
            try:
                result = self.score(current_prompt, target=pred_token)
                trajectory.append((
                    pred_token,
                    result.depth_score,
                    result.layer_entropy if result.layer_entropy is not None else 0.0,
                ))
            except Exception as e:
                print(f"[DepthScorer] Trajectory step {step} failed: {e}")
                trajectory.append((pred_token, float("nan"), 0.0))

            current_prompt += pred_token

            torch.cuda.empty_cache()
            gc.collect()

        return trajectory

    def fingerprint(self, result: DepthResult) -> np.ndarray:
        """Compute a fixed-length cognitive fingerprint vector.

        Returns a 7-element numpy array:
            [0] weighted_depth_score  — attribution-weighted mean layer
            [1] layer_entropy         — distributional spread
            [2] early_ratio           — early-band feature fraction
            [3] mid_ratio             — mid-band feature fraction
            [4] late_ratio            — late-band feature fraction
            [5] feature_count_norm    — feature count / max_feature_nodes (0-1)
            [6] compute_efficiency    — features per second of compute

        Enables clustering, similarity comparison, and downstream ML tasks.
        """
        weighted = result.weighted_depth_score if result.weighted_depth_score is not None else result.depth_score
        entropy = result.layer_entropy if result.layer_entropy is not None else 0.0

        # Normalize feature count against the configured max
        fc_norm = min(1.0, result.feature_count / self.max_feature_nodes) if self.max_feature_nodes > 0 else 0.0

        # Compute efficiency: features extracted per second
        compute_eff = (
            result.feature_count / result.compute_time_s
            if result.compute_time_s > 0
            else 0.0
        )
        # Normalize efficiency to ~0-1 range (empirical: 100 features/sec is typical)
        compute_eff_norm = min(1.0, compute_eff / 100.0)

        return np.array([
            weighted if not math.isnan(weighted) else 0.0,
            entropy,
            result.early_ratio,
            result.mid_ratio,
            result.late_ratio,
            fc_norm,
            compute_eff_norm,
        ], dtype=np.float64)

    # ── Phase 2: Adversarial Depth Gradient (§2) ─────────────────────

    def depth_gradient(
        self,
        prompt: str,
        n_candidates: int = 20,
        target: Optional[str] = None,
    ) -> List[Tuple[str, str, str, float]]:
        """Find minimal token edits that maximally increase reasoning depth.

        For each token in the prompt, finds semantically similar alternatives
        using cosine similarity in embedding space, scores each variant, and
        returns the top edits ranked by depth improvement.

        Args:
            prompt: The text to optimize.
            n_candidates: Max replacement candidates per token position.
            target: Optional single-token target.

        Returns:
            List of (original_token, replacement_token, modified_prompt, depth_delta)
            tuples, sorted by depth_delta (most improvement first). Negative delta
            means the edit pushed toward deeper reasoning.
        """
        inner = self._get_inner_model()
        if inner is None:
            raise ValueError("Cannot compute depth gradient: inner model not accessible.")

        # Baseline score
        baseline = self.score(prompt, target)
        baseline_depth = baseline.depth_score

        # Tokenize the prompt
        token_ids = inner.tokenizer.encode(prompt, add_special_tokens=False)
        tokens_str = [inner.tokenizer.decode([tid]) for tid in token_ids]

        # Get embedding matrix
        embed_matrix = None
        for attr in ["embed", "W_E", "embed_tokens"]:
            candidate = getattr(inner, attr, None)
            if candidate is not None:
                if hasattr(candidate, "W_E"):
                    embed_matrix = candidate.W_E.data
                elif isinstance(candidate, torch.nn.Embedding):
                    embed_matrix = candidate.weight.data
                elif isinstance(candidate, torch.Tensor):
                    embed_matrix = candidate
                break

        if embed_matrix is None:
            try:
                embed_matrix = inner.W_E
            except AttributeError:
                raise ValueError("Cannot access embedding matrix for gradient computation.")

        edits = []

        for pos, (tid, tok_str) in enumerate(zip(token_ids, tokens_str)):
            # Skip very short tokens (punctuation, spaces)
            if len(tok_str.strip()) <= 1:
                continue

            # Find nearest neighbors in embedding space
            with torch.no_grad():
                tok_embed = embed_matrix[tid]
                # Cosine similarity against full vocabulary
                norms = embed_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                tok_norm = tok_embed.norm().clamp(min=1e-8)
                similarities = (embed_matrix @ tok_embed) / (norms.squeeze() * tok_norm)
                # Get top candidates (exclude self)
                topk = torch.topk(similarities, k=min(n_candidates + 1, len(similarities)))

            for sim_score, cand_id in zip(topk.values, topk.indices):
                cand_id = cand_id.item()
                if cand_id == tid:
                    continue

                cand_token = inner.tokenizer.decode([cand_id])
                # Skip if replacement is empty or just whitespace
                if not cand_token.strip():
                    continue

                # Build modified prompt
                modified_ids = list(token_ids)
                modified_ids[pos] = cand_id
                modified_prompt = inner.tokenizer.decode(modified_ids)

                # Score modified prompt
                try:
                    mod_result = self.score(modified_prompt, target)
                    depth_delta = mod_result.depth_score - baseline_depth
                    edits.append((tok_str, cand_token, modified_prompt, depth_delta))
                except Exception:
                    continue

                torch.cuda.empty_cache()
                gc.collect()

            # Early termination: only check first 10 token positions
            if pos >= 10:
                break

        # Sort by depth delta (most negative = most improvement toward depth)
        edits.sort(key=lambda e: e[3])
        return edits[:5]

    def deepen_prompt(
        self,
        prompt: str,
        target: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Apply the highest-scoring depth edit and return the modified prompt.

        Args:
            prompt: The text to deepen.
            target: Optional single-token target.

        Returns:
            Tuple of (modified_prompt, depth_delta). Negative delta = deeper.
        """
        edits = self.depth_gradient(prompt, target=target)
        if not edits:
            return prompt, 0.0

        best = edits[0]
        original_tok, replacement_tok, modified_prompt, delta = best
        print(
            f"[DepthScorer] Deepened: '{original_tok}' → '{replacement_tok}' "
            f"(depth delta: {delta:+.4f})"
        )
        return modified_prompt, delta

    # ── Phase 2: Reasoning Phase Detector (§3) ────────────────────

    def detect_phase_transitions(
        self,
        trajectory: List[Tuple[str, float, float]],
        window_size: int = 3,
        threshold: float = 0.15,
    ) -> List[Tuple[int, str, str]]:
        """Detect cognitive mode shifts in a depth trajectory.

        Applies change-point detection to identify where the model transitions
        between surface recall and deep reasoning during generation.

        Uses the ruptures library if available for optimal change-point detection,
        otherwise falls back to a sliding-window variance method.

        Args:
            trajectory: Output from score_trajectory() — list of (token, depth, entropy).
            window_size: Sliding window size for variance-based detection.
            threshold: Minimum depth change to qualify as a transition.

        Returns:
            List of (token_idx, token, transition_type) where transition_type is
            one of: "recall→reasoning", "reasoning→recall", "stabilize".
        """
        if len(trajectory) < window_size + 1:
            return []

        depths = np.array([t[1] for t in trajectory], dtype=np.float64)
        # Replace NaN with interpolation
        valid_mask = ~np.isnan(depths)
        if valid_mask.sum() < 3:
            return []
        if not valid_mask.all():
            valid_indices = np.where(valid_mask)[0]
            depths = np.interp(np.arange(len(depths)), valid_indices, depths[valid_mask])

        transitions = []

        # Try ruptures first
        try:
            import ruptures as rpt
            algo = rpt.Pelt(model="rbf", min_size=2).fit(depths.reshape(-1, 1))
            change_points = algo.predict(pen=1.0)
            # Remove the last element (always == len)
            change_points = [cp for cp in change_points if cp < len(depths)]

            for cp in change_points:
                if cp == 0:
                    continue
                before = np.mean(depths[max(0, cp - window_size):cp])
                after = np.mean(depths[cp:min(len(depths), cp + window_size)])
                delta = after - before

                if abs(delta) < threshold:
                    transition_type = "stabilize"
                elif delta < 0:
                    transition_type = "recall→reasoning"
                else:
                    transition_type = "reasoning→recall"

                transitions.append((cp, trajectory[cp][0], transition_type))

            return transitions

        except ImportError:
            pass

        # Fallback: sliding window variance detection
        for i in range(window_size, len(depths) - 1):
            before = depths[max(0, i - window_size):i]
            after = depths[i:min(len(depths), i + window_size)]

            mean_before = np.mean(before)
            mean_after = np.mean(after)
            delta = mean_after - mean_before

            if abs(delta) >= threshold:
                if delta < 0:
                    transition_type = "recall→reasoning"
                else:
                    transition_type = "reasoning→recall"
                transitions.append((i, trajectory[i][0], transition_type))

        # Deduplicate: keep only the strongest transition in each cluster
        if len(transitions) > 1:
            filtered = [transitions[0]]
            for t in transitions[1:]:
                if t[0] - filtered[-1][0] >= window_size:
                    filtered.append(t)
            transitions = filtered

        return transitions

    def reasoning_onset(
        self,
        prompt: str,
        max_tokens: int = 20,
        depth_threshold: Optional[float] = None,
    ) -> Optional[int]:
        """Find the first token where the model commits to deep processing.

        The earlier this happens, the more the model 'understood' the question
        from the start. Late onset suggests the model only realized it needed
        to reason after several tokens of shallow generation.

        Args:
            prompt: The initial prompt.
            max_tokens: Maximum tokens to generate.
            depth_threshold: Depth score below which processing counts as 'deep'.
                Defaults to the calibrated insight mean minus one std.

        Returns:
            Token index of reasoning onset, or None if the model never goes deep.
        """
        if depth_threshold is None:
            registry = MODEL_REGISTRY.get(self.model_name, {})
            cal = registry.get("calibration")
            if cal:
                depth_threshold = cal["insight_mean"] - 0.1
            else:
                depth_threshold = 8.0

        trajectory = self.score_trajectory(prompt, max_tokens=max_tokens)

        for i, (token, depth, entropy) in enumerate(trajectory):
            if not math.isnan(depth) and depth < depth_threshold:
                return i

        return None

    # ── Phase 2: Depth-Guided Generation Hook (§6) ────────────────

    def generate_deep(
        self,
        prompt: str,
        target_depth: float = 7.5,
        max_tokens: int = 50,
        beam_width: int = 5,
        check_interval: int = 3,
    ) -> Tuple[str, List[Tuple[str, float, float]]]:
        """Generate tokens steered toward deeper reasoning.

        EXPERIMENTAL. Generates tokens one at a time, periodically checking depth.
        When depth rises above the target (shallower), re-ranks the top-k candidate
        tokens by their depth impact, selecting the one that steers deepest.

        Args:
            prompt: The initial prompt.
            target_depth: Target depth_score to maintain (lower = deeper).
            max_tokens: Maximum tokens to generate.
            beam_width: Number of candidate tokens to evaluate at steering points.
            check_interval: Check depth every N tokens (attribution is expensive).

        Returns:
            Tuple of (generated_text, trajectory) where trajectory is a list
            of (token, depth_score, entropy) tuples for each generated token.
        """
        inner = self._get_inner_model()
        if inner is None:
            raise ValueError("Cannot generate: inner model not accessible.")

        generated_tokens = []
        trajectory = []
        current_prompt = prompt
        last_depth = target_depth

        for step in range(max_tokens):
            tokens = inner.to_tokens(current_prompt)

            with torch.no_grad():
                logits = inner(tokens)

            # Check for depth steering at intervals
            should_steer = (step % check_interval == 0) and step > 0 and last_depth > target_depth

            if should_steer:
                # Get top-k candidates
                top_logits, top_ids = torch.topk(logits[0, -1], k=beam_width)
                best_token_id = None
                best_depth = float("inf")

                for cand_id in top_ids:
                    cand_id = cand_id.item()
                    cand_token = inner.tokenizer.decode([cand_id])

                    # Skip multi-token or empty
                    re_enc = inner.tokenizer.encode(cand_token, add_special_tokens=False)
                    if len(re_enc) != 1 or not cand_token.strip():
                        continue

                    try:
                        r = self.score(current_prompt, target=cand_token)
                        if r.depth_score < best_depth:
                            best_depth = r.depth_score
                            best_token_id = cand_id
                    except Exception:
                        continue

                    torch.cuda.empty_cache()
                    gc.collect()

                if best_token_id is not None:
                    pred_id = best_token_id
                    last_depth = best_depth
                else:
                    pred_id = logits[0, -1].argmax().item()
            else:
                pred_id = logits[0, -1].argmax().item()

            pred_token = inner.tokenizer.decode([pred_id])

            # Verify single-token
            re_encoded = inner.tokenizer.encode(pred_token, add_special_tokens=False)
            if len(re_encoded) != 1:
                break

            # Check EOS
            eos_id = getattr(inner.tokenizer, "eos_token_id", None)
            if pred_id == eos_id:
                break

            # Score at check intervals for the trajectory record
            if step % check_interval == 0:
                try:
                    r = self.score(current_prompt, target=pred_token)
                    last_depth = r.depth_score
                    trajectory.append((
                        pred_token,
                        r.depth_score,
                        r.layer_entropy if r.layer_entropy is not None else 0.0,
                    ))
                except Exception:
                    trajectory.append((pred_token, float("nan"), 0.0))
            else:
                trajectory.append((pred_token, last_depth, 0.0))

            generated_tokens.append(pred_token)
            current_prompt += pred_token

            torch.cuda.empty_cache()
            gc.collect()

        generated_text = "".join(generated_tokens)
        return generated_text, trajectory

    def score_stream(
        self, prompt: str, target: Optional[str] = None
    ) -> Generator[DepthResult, None, None]:
        """Streaming score mode — yields partial DepthResult objects as attribution refines.

        Runs attribution at progressively higher resolution (increasing max_feature_nodes)
        and yields intermediate results. For real-time APIs where you want to show
        depth updating live as computation runs.

        Args:
            prompt: The text to analyze.
            target: Optional single-token target.

        Yields:
            DepthResult objects with increasing precision.
        """
        from circuit_tracer import attribute

        if target is None:
            target = self._get_single_token_target(prompt)

        full_input = prompt + target

        # Progressive resolution: start coarse, refine
        stages = [
            max(50, self.max_feature_nodes // 8),
            max(100, self.max_feature_nodes // 4),
            max(200, self.max_feature_nodes // 2),
            self.max_feature_nodes,
        ]
        # Deduplicate while preserving order
        seen = set()
        unique_stages = []
        for s in stages:
            if s not in seen:
                seen.add(s)
                unique_stages.append(s)

        for stage_nodes in unique_stages:
            t0 = time.time()
            attr_kwargs = {
                "attribution_targets": [target],
                "max_feature_nodes": stage_nodes,
                "batch_size": self.batch_size,
            }
            if self.offload:
                attr_kwargs["offload"] = self.offload

            graph = attribute(full_input, self._model, **attr_kwargs)
            compute_time = time.time() - t0

            active_features = []
            magnitudes = None
            af = graph.active_features
            if af is not None and hasattr(af, "shape") and af.shape[0] > 0:
                layers_np = af[:, 0].float().cpu().numpy()
                active_features = [(float(l), 0, 0) for l in layers_np]
                magnitudes = self._extract_magnitudes(graph, len(layers_np))

            del graph
            torch.cuda.empty_cache()
            gc.collect()

            extraction = {
                "active_features": active_features,
                "magnitudes": magnitudes,
                "compute_time": compute_time,
                "target": target,
            }

            yield self._build_depth_result(prompt, extraction)


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


def interpret_entropy(entropy: float, n_layers: int = 26) -> str:
    """Human-readable interpretation of a layer entropy value.

    Args:
        entropy: Shannon entropy in bits.
        n_layers: Number of model layers (for computing max possible entropy).
    """
    max_entropy = math.log2(n_layers) if n_layers > 1 else 1.0
    ratio = entropy / max_entropy if max_entropy > 0 else 0.0

    if ratio >= 0.75:
        return "highly distributed — features spread broadly across layers, deep multi-stage computation"
    elif ratio >= 0.50:
        return "moderately distributed — features engage multiple layer bands"
    elif ratio >= 0.25:
        return "concentrated — features cluster in a narrow layer range"
    else:
        return "highly concentrated — features spike at very few layers, surface-level processing"


if __name__ == "__main__":
    # Quick test
    scorer = DepthScorer()

    print("\n" + "=" * 60)
    print("DEPTH SCORER — Cognitive Fingerprinting Test")
    print("=" * 60)

    surface = "Fire needs oxygen, fuel, and heat, and all three are present when something"
    insight = "Fire needs oxygen, fuel, and heat, and each one feeds the others when something"

    result = scorer.compare(surface, insight)
    sr = result.surface
    ir = result.insight

    print(f"\nSurface depth:    {sr.depth_score:.3f} (weighted: {sr.weighted_depth_score:.3f})")
    print(f"  entropy: {sr.layer_entropy:.3f}, features: {sr.feature_count}")
    print(f"  anomalous: {sr.is_anomalous}", end="")
    if sr.anomaly_reasons:
        print(f" — {', '.join(sr.anomaly_reasons)}")
    else:
        print()

    print(f"\nInsight depth:    {ir.depth_score:.3f} (weighted: {ir.weighted_depth_score:.3f})")
    print(f"  entropy: {ir.layer_entropy:.3f}, features: {ir.feature_count}")
    print(f"  anomalous: {ir.is_anomalous}", end="")
    if ir.anomaly_reasons:
        print(f" — {', '.join(ir.anomaly_reasons)}")
    else:
        print()

    print(f"\nDelta: +{result.delta:.3f} ({'expected direction' if result.surface_deeper else 'unexpected'})")
    print(f"\nSurface: {interpret_score(sr.depth_score)}")
    print(f"Insight: {interpret_score(ir.depth_score)}")
    print(f"\nSurface entropy: {interpret_entropy(sr.layer_entropy, sr.n_layers)}")
    print(f"Insight entropy: {interpret_entropy(ir.layer_entropy, ir.n_layers)}")

    # Fingerprint
    fp_s = scorer.fingerprint(sr)
    fp_i = scorer.fingerprint(ir)
    print(f"\nSurface fingerprint: {np.array2string(fp_s, precision=3)}")
    print(f"Insight fingerprint: {np.array2string(fp_i, precision=3)}")
    print(f"Fingerprint cosine similarity: {float(np.dot(fp_s, fp_i) / (np.linalg.norm(fp_s) * np.linalg.norm(fp_i))):.4f}")
