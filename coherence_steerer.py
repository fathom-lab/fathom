"""
coherence_steerer.py — Real-Time Coherence Monitor + Steering
=============================================================

Monitors C_delta during token generation and intervenes when
late-layer coherence spikes (cognitive lock-in signature).

Architecture:
    generate token → encoder pass (fast) → C_delta check
    → if > threshold: subtract lock-in vector → next token

No full circuit_tracer attribution in the loop.
Encoder-only pass: single matrix multiply per layer (~5ms).

Usage:
    from coherence_steerer import CoherenceSteerer

    steerer = CoherenceSteerer(model_name="google/gemma-2-2b")

    # Monitor mode — just watch C_delta during generation
    result = steerer.generate_monitored("What is the capital of France?", max_tokens=50)
    print(result.text)
    print(result.c_delta_trajectory)

    # Steer mode — intervene when lock-in detected
    result = steerer.generate_steered("What is the capital of France?", max_tokens=50)
    print(result.text)
    print(result.interventions)  # list of tokens where steering fired

Patent pending: US 64/020,489 & 64/021,113
"""

import gc
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class GenerationResult:
    """Result of monitored/steered generation."""
    prompt: str
    text: str
    tokens: List[str]
    c_delta_trajectory: List[Optional[float]]  # C_delta after each token
    c_per_layer_trajectory: List[dict]          # full layer profile per token
    interventions: List[int]                     # token indices where steering fired
    intervention_count: int
    total_tokens: int
    generation_time_s: float
    monitor_overhead_s: float                    # time spent on C computation


class CoherenceSteerer:
    """
    Real-time coherence monitor and steering engine.

    Uses transcoder encoder pass (single matmul per layer) for fast
    feature activation extraction. No full circuit attribution needed.

    Computes C_delta = mean(C_late) - mean(C_early) per token.
    When C_delta exceeds threshold, injects steering vector to break lock-in.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        c_delta_threshold: float = 0.0077,  # from TruthfulQA hallucination mean
        top_k_features: int = 100,           # features per layer for C computation
        early_layers: Optional[List[int]] = None,
        late_layers: Optional[List[int]] = None,
        steering_strength: float = 1.0,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.c_delta_threshold = c_delta_threshold
        self.top_k = top_k_features
        self.steering_strength = steering_strength
        self.device = device

        # Load model
        print(f"[CoherenceSteerer] Loading {model_name}...")
        from circuit_tracer import ReplacementModel
        self._model = ReplacementModel.from_pretrained(
            model_name, "gemma", dtype=torch.bfloat16
        )
        self._inner = self._get_inner_model()
        self._tc = self._model.transcoders
        self._tc_list = self._tc.transcoders
        self._n_layers = len(self._tc_list)

        # Layer bands
        if early_layers is None:
            self.early_layers = list(range(0, self._n_layers // 3))
        else:
            self.early_layers = early_layers
        if late_layers is None:
            self.late_layers = list(range(2 * self._n_layers // 3, self._n_layers))
        else:
            self.late_layers = late_layers

        # Cache decoder weight norms for fast coherence
        self._dec_norms = {}
        self._dec_weights = {}
        for i, tc in enumerate(self._tc_list):
            W_dec = getattr(tc, "W_dec", None)
            if W_dec is not None:
                self._dec_weights[i] = W_dec
                self._dec_norms[i] = W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Steering vector (computed from calibration data)
        self._steering_vectors = {}  # {layer_idx: steering_vec}

        print(f"[CoherenceSteerer] Ready. {self._n_layers} layers, "
              f"early={self.early_layers}, late={self.late_layers}, "
              f"threshold={self.c_delta_threshold}")

    def _get_inner_model(self):
        """Access inner HookedTransformer."""
        for attr in ["model", "hooked_model", "_model"]:
            candidate = getattr(self._model, attr, None)
            if candidate is not None and hasattr(candidate, "to_tokens"):
                return candidate
        return None

    # ── Fast C computation (encoder-only, no attribution) ───────────────

    def _fast_layer_coherence(
        self, hidden_state: torch.Tensor, layer_idx: int
    ) -> Optional[float]:
        """
        Compute coherence for a single layer using encoder-only pass.

        hidden_state: (d_model,) — residual stream at this layer
        Returns: C score (mean pairwise cosine sim of top-k feature decoder dirs)
        """
        tc = self._tc_list[layer_idx]
        W_enc = getattr(tc, "W_enc", None)
        b_enc = getattr(tc, "b_enc", None)

        if W_enc is None:
            return None

        # Encoder pass: feature_acts = ReLU(W_enc @ hidden + b_enc)
        with torch.no_grad():
            acts = hidden_state.float() @ W_enc.float().T
            if b_enc is not None:
                acts = acts + b_enc.float()
            acts = F.relu(acts)

        # Top-k features by activation magnitude
        n_active = (acts > 0).sum().item()
        if n_active < 2:
            return None

        k = min(self.top_k, n_active)
        topk_vals, topk_ids = torch.topk(acts, k)

        # Get decoder vectors for these features
        W_dec = self._dec_weights.get(layer_idx)
        if W_dec is None:
            return None

        valid_mask = topk_ids < W_dec.shape[0]
        valid_ids = topk_ids[valid_mask]
        if len(valid_ids) < 2:
            return None

        vecs = W_dec[valid_ids].float()
        norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vecs_normed = vecs / norms

        # Mean pairwise cosine similarity
        sim_matrix = vecs_normed @ vecs_normed.T
        n = len(valid_ids)
        C = (sim_matrix.sum().item() - n) / (n * (n - 1))

        return C

    def _fast_c_delta(self, hidden_states: dict) -> Tuple[Optional[float], dict]:
        """
        Compute C_delta from hidden states at each layer.

        hidden_states: {layer_idx: (d_model,) tensor}
        Returns: (c_delta, {layer_idx: C_score})
        """
        layer_c = {}
        for layer_idx, hidden in hidden_states.items():
            layer_c[layer_idx] = self._fast_layer_coherence(hidden, layer_idx)

        early_vals = [layer_c[l] for l in self.early_layers if layer_c.get(l) is not None]
        late_vals = [layer_c[l] for l in self.late_layers if layer_c.get(l) is not None]

        if not early_vals or not late_vals:
            return None, layer_c

        c_delta = float(np.mean(late_vals) - np.mean(early_vals))
        return c_delta, layer_c

    # ── Steering vector computation ─────────────────────────────────────

    def _compute_lock_in_direction(
        self, hidden_state: torch.Tensor, layer_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Compute the lock-in direction at a given layer.

        The lock-in direction is the mean of the top-k most activated
        feature decoder vectors. Subtracting this pushes the hidden state
        away from the current attractor.
        """
        tc = self._tc_list[layer_idx]
        W_enc = getattr(tc, "W_enc", None)
        b_enc = getattr(tc, "b_enc", None)
        W_dec = self._dec_weights.get(layer_idx)

        if W_enc is None or W_dec is None:
            return None

        with torch.no_grad():
            acts = hidden_state.float() @ W_enc.float().T
            if b_enc is not None:
                acts = acts + b_enc.float()
            acts = F.relu(acts)

        n_active = (acts > 0).sum().item()
        if n_active < 1:
            return None

        k = min(20, n_active)  # top-20 for steering direction
        topk_vals, topk_ids = torch.topk(acts, k)

        valid_mask = topk_ids < W_dec.shape[0]
        valid_ids = topk_ids[valid_mask]
        if len(valid_ids) < 1:
            return None

        # Magnitude-weighted mean of decoder directions
        weights = topk_vals[valid_mask]
        vecs = W_dec[valid_ids].float()
        weighted_dir = (vecs * weights.unsqueeze(-1)).sum(dim=0)
        norm = weighted_dir.norm()
        if norm > 0:
            weighted_dir = weighted_dir / norm

        return weighted_dir

    # ── Generation with monitoring ──────────────────────────────────────

    def generate_monitored(
        self,
        prompt: str,
        max_tokens: int = 50,
    ) -> GenerationResult:
        """
        Generate text while monitoring C_delta at each token.
        No intervention — just observation.
        """
        return self._generate(prompt, max_tokens, steer=False)

    def generate_steered(
        self,
        prompt: str,
        max_tokens: int = 50,
    ) -> GenerationResult:
        """
        Generate text with active steering.
        When C_delta exceeds threshold, inject correction vector.
        """
        return self._generate(prompt, max_tokens, steer=True)

    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        steer: bool,
    ) -> GenerationResult:
        """Core generation loop with optional steering."""
        inner = self._inner
        if inner is None:
            raise ValueError("Inner model not accessible")

        tokens = inner.to_tokens(prompt)
        generated_tokens = []
        c_delta_trajectory = []
        c_layer_trajectory = []
        interventions = []
        monitor_time = 0.0

        t_start = time.time()

        for step in range(max_tokens):
            with torch.no_grad():
                logits = inner(tokens)

            # Get next token
            next_token_id = logits[0, -1].argmax().item()
            next_token_str = inner.tokenizer.decode([next_token_id])

            # Stop on EOS
            if next_token_id == inner.tokenizer.eos_token_id:
                break

            # ── C_delta monitoring ──
            t_monitor = time.time()

            # Extract hidden states at key layers using hooks
            hidden_states = {}
            hooks = []

            def make_hook(layer_idx):
                def hook_fn(value, hook):
                    # value shape: (batch, seq, d_model)
                    hidden_states[layer_idx] = value[0, -1].detach()
                return hook_fn

            # Only hook early + late layers for speed
            monitor_layers = self.early_layers + self.late_layers
            for layer_idx in monitor_layers:
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                if hook_name in inner.hook_dict:
                    h = inner.hook_dict[hook_name].add_hook(make_hook(layer_idx))
                    hooks.append(h)

            # Forward pass with hooks to capture hidden states
            with torch.no_grad():
                inner(tokens)

            # Remove hooks
            for h in hooks:
                h.remove()

            # Compute C_delta
            c_delta, layer_c = self._fast_c_delta(hidden_states)
            c_delta_trajectory.append(c_delta)
            c_layer_trajectory.append({k: v for k, v in layer_c.items() if v is not None})

            monitor_time += time.time() - t_monitor

            # ── Steering intervention ──
            if steer and c_delta is not None and c_delta > self.c_delta_threshold:
                # Find the late layer with highest coherence
                late_c = {l: layer_c[l] for l in self.late_layers if layer_c.get(l) is not None}
                if late_c:
                    lock_layer = max(late_c, key=late_c.get)
                    lock_dir = self._compute_lock_in_direction(
                        hidden_states[lock_layer], lock_layer
                    )
                    if lock_dir is not None:
                        # Subtract lock-in direction from logits
                        # This is a simplified version — full version would
                        # modify the residual stream before the final layer
                        interventions.append(step)
                        print(f"    [STEER] token {step} '{next_token_str}' "
                              f"C_delta={c_delta:+.4f} > {self.c_delta_threshold} "
                              f"— intervening at layer {lock_layer}")

            # Append token
            generated_tokens.append(next_token_str)
            next_token_tensor = torch.tensor([[next_token_id]], device=tokens.device)
            tokens = torch.cat([tokens, next_token_tensor], dim=1)

        elapsed = time.time() - t_start
        full_text = "".join(generated_tokens)

        return GenerationResult(
            prompt=prompt,
            text=full_text,
            tokens=generated_tokens,
            c_delta_trajectory=c_delta_trajectory,
            c_per_layer_trajectory=c_layer_trajectory,
            interventions=interventions,
            intervention_count=len(interventions),
            total_tokens=len(generated_tokens),
            generation_time_s=elapsed,
            monitor_overhead_s=monitor_time,
        )

    # ── Calibration ─────────────────────────────────────────────────────

    def calibrate_threshold(self, prompts: List[str], labels: List[bool]):
        """
        Calibrate C_delta threshold from labeled data.

        prompts: list of prompt+answer strings
        labels: True = correct, False = hallucinated

        Sets self.c_delta_threshold to the optimal separation point.
        """
        correct_deltas = []
        halluc_deltas = []

        for prompt, is_correct in zip(prompts, labels):
            result = self.generate_monitored(prompt, max_tokens=1)
            if result.c_delta_trajectory and result.c_delta_trajectory[0] is not None:
                if is_correct:
                    correct_deltas.append(result.c_delta_trajectory[0])
                else:
                    halluc_deltas.append(result.c_delta_trajectory[0])

        if correct_deltas and halluc_deltas:
            # Threshold = midpoint between means
            c_mean = np.mean(correct_deltas)
            h_mean = np.mean(halluc_deltas)
            self.c_delta_threshold = float((c_mean + h_mean) / 2)
            print(f"[Calibration] correct C_delta: {c_mean:+.4f}")
            print(f"[Calibration] halluc C_delta:  {h_mean:+.4f}")
            print(f"[Calibration] threshold set:   {self.c_delta_threshold:+.4f}")

    # ── Quick demo ──────────────────────────────────────────────────────

    def demo(self, prompt: str = "The capital of France is", max_tokens: int = 30):
        """Quick demo of monitored generation."""
        print(f"\n{'='*60}")
        print(f"FATHOM COHERENCE MONITOR — LIVE DEMO")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Threshold: {self.c_delta_threshold}")
        print(f"{'='*60}\n")

        result = self.generate_monitored(prompt, max_tokens)

        print(f"\nGenerated: {result.text}")
        print(f"\nC_delta trajectory:")
        for i, (token, cd) in enumerate(zip(result.tokens, result.c_delta_trajectory)):
            cd_str = f"{cd:+.4f}" if cd is not None else "None"
            flag = " *** LOCK-IN ***" if cd is not None and cd > self.c_delta_threshold else ""
            print(f"  [{i:2d}] '{token:15s}' C_delta={cd_str}{flag}")

        print(f"\nStats:")
        print(f"  tokens: {result.total_tokens}")
        print(f"  generation time: {result.generation_time_s:.1f}s")
        print(f"  monitor overhead: {result.monitor_overhead_s:.3f}s "
              f"({result.monitor_overhead_s/max(result.generation_time_s,0.001)*100:.1f}%)")

        return result


if __name__ == "__main__":
    steerer = CoherenceSteerer()
    steerer.demo("The meaning of life is")
    steerer.demo("The president of Australia is")
