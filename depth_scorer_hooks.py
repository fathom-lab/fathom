"""
Hook-based depth scorer. Works on any HuggingFace transformer model.
Uses residual stream L2 norm per layer as computation depth proxy.

Unlike the SAE-based depth_scorer.py (which requires circuit-tracer transcoders),
this scorer attaches hooks to residual stream outputs and measures where the model
concentrates its computation via L2 norm magnitude.

Usage:
    from depth_scorer_hooks import HookDepthScorer

    scorer = HookDepthScorer("mistralai/Mistral-7B-v0.1")
    result = scorer.score("Why does fire spread once it starts?", "fuel")
    print(result.depth_score)

    cmp = scorer.compare(
        "The patient has a fever", "touch",
        "Fever triggers prostaglandin synthesis", "raise"
    )
    print(f"delta={cmp.delta:.4f}, ratio={cmp.ratio:.4f}")
"""

import gc
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import torch
import numpy as np


@dataclass
class DepthResult:
    """Result of a single prompt depth analysis via hooks."""
    depth_score: float          # weighted mean layer index
    n_layers: int               # model layer count
    layer_profile: Dict[int, float]  # {layer_idx: L2 norm}
    early_ratio: float          # fraction of total norm in early band
    mid_ratio: float            # fraction of total norm in mid band
    late_ratio: float           # fraction of total norm in late band
    compute_time_s: float       # wall clock seconds

    def to_dict(self):
        return asdict(self)


@dataclass
class CompareResult:
    """Result of comparing surface vs insight prompts."""
    surface: DepthResult
    insight: DepthResult
    delta: float                # surface.depth_score - insight.depth_score
    ratio: float                # surface.depth_score / insight.depth_score

    def to_dict(self):
        return {
            "surface": self.surface.to_dict(),
            "insight": self.insight.to_dict(),
            "delta": self.delta,
            "ratio": self.ratio,
        }


class HookDepthScorer:
    """
    Measures reasoning depth using residual stream L2 norms.

    For each layer, collects the L2 norm of the residual stream at the
    final token position. The "depth score" is the norm-weighted mean
    layer index: sum(layer_i * norm_i) / sum(norm_i).

    Works with any model supported by TransformerLens.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype=torch.bfloat16,
        band_boundaries: Optional[tuple] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model = None
        self._n_layers = None
        self._band_boundaries = band_boundaries

    def _ensure_model(self):
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        import transformer_lens

        print(f"[HOOK SCORER] Loading {self.model_name}...")

        # Check VRAM availability
        if self.device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[HOOK SCORER] VRAM available: {vram_gb:.1f} GB")
            # Mistral-7B at bf16 ~ 14GB. If VRAM < 10GB, fall back to CPU.
            if vram_gb < 10 and "7b" in self.model_name.lower():
                print(f"[HOOK SCORER] VRAM insufficient for 7B model, falling back to CPU")
                self.device = "cpu"
                self.dtype = torch.float32
        elif self.device == "cuda" and not torch.cuda.is_available():
            print("[HOOK SCORER] CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.dtype = torch.float32

        self._model = transformer_lens.HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
        )

        self._n_layers = self._model.cfg.n_layers
        print(f"[HOOK SCORER] Loaded. Layers: {self._n_layers}, Device: {self.device}")

        # Set default band boundaries if not specified
        if self._band_boundaries is None:
            third = self._n_layers // 3
            self._band_boundaries = (third, 2 * third)

    def score(self, prompt: str, target: str) -> DepthResult:
        """
        Score depth of a prompt+target using residual stream L2 norms.

        Returns a DepthResult with the weighted mean layer and band ratios.
        """
        self._ensure_model()

        t0 = time.time()
        full_text = prompt + target

        # Storage for hook activations
        layer_norms = {}

        def make_hook(layer_idx):
            def hook_fn(activation, hook):
                # activation shape: [batch, seq_len, d_model]
                # Take L2 norm at the final token position
                final_token = activation[0, -1, :]  # [d_model]
                layer_norms[layer_idx] = float(final_token.norm().item())
            return hook_fn

        # Register hooks on all layers' residual stream post
        hook_points = []
        for layer in range(self._n_layers):
            hook_points.append(
                (f"blocks.{layer}.hook_resid_post", make_hook(layer))
            )

        # Run with hooks
        with torch.no_grad():
            self._model.run_with_hooks(
                full_text,
                fwd_hooks=hook_points,
            )

        # Compute weighted mean layer
        layers = np.array(list(layer_norms.keys()))
        norms = np.array(list(layer_norms.values()))
        total_norm = norms.sum()

        if total_norm > 0:
            depth_score = float(np.sum(layers * norms) / total_norm)
        else:
            depth_score = float(self._n_layers / 2)

        # Band ratios
        early_bound, mid_bound = self._band_boundaries
        early_norm = sum(n for l, n in layer_norms.items() if l < early_bound)
        mid_norm = sum(n for l, n in layer_norms.items() if early_bound <= l < mid_bound)
        late_norm = sum(n for l, n in layer_norms.items() if l >= mid_bound)

        early_ratio = early_norm / total_norm if total_norm > 0 else 0.0
        mid_ratio = mid_norm / total_norm if total_norm > 0 else 0.0
        late_ratio = late_norm / total_norm if total_norm > 0 else 0.0

        compute_time = time.time() - t0

        # Cleanup
        torch.cuda.empty_cache() if self.device == "cuda" else None

        return DepthResult(
            depth_score=depth_score,
            n_layers=self._n_layers,
            layer_profile=dict(layer_norms),
            early_ratio=early_ratio,
            mid_ratio=mid_ratio,
            late_ratio=late_ratio,
            compute_time_s=round(compute_time, 3),
        )

    def compare(
        self,
        surface_prompt: str,
        surface_target: str,
        insight_prompt: str,
        insight_target: str,
    ) -> CompareResult:
        """Compare surface vs insight prompt depth."""
        surface = self.score(surface_prompt, surface_target)
        insight = self.score(insight_prompt, insight_target)

        delta = surface.depth_score - insight.depth_score
        ratio = surface.depth_score / insight.depth_score if insight.depth_score > 0 else 0.0

        return CompareResult(
            surface=surface,
            insight=insight,
            delta=delta,
            ratio=ratio,
        )

    def cleanup(self):
        """Free model memory."""
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

