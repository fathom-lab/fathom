# -*- coding: utf-8 -*-
"""
spectrum_steerer.py -- Spectrum-Guided Generation (Frontier 2)
==============================================================

Extends CoherenceSteerer with SHAPE-aware intervention. Instead of
triggering on scalar C_delta, monitors the Gini coefficient of the
full per-layer coherence profile at each token. When coherence
concentrates (high Gini = lock-in), applies multi-layer steering
to redistribute computation across layers.

Key insight from Frontier 5 analysis:
  - Gini coefficient discriminates hallucination at AUC=0.685 (n=200)
  - Scalar C_delta: AUC=0.640. Scalar K: AUC=0.550.
  - Hallucinations CONCENTRATE coherence; correct responses DISTRIBUTE it.
  - The shape carries the signal the mean destroys.

This is the first generation system that steers based on computational
geometry rather than likelihood or preference.

Patent pending: US 64/020,489 & 64/021,113
"""

import gc
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn.functional as F
import numpy as np

from coherence_steerer import CoherenceSteerer, GenerationResult


@dataclass
class SpectrumResult(GenerationResult):
    """Extended result with spectrum diagnostics."""
    gini_trajectory: List[float] = field(default_factory=list)
    spread_trajectory: List[float] = field(default_factory=list)
    early_mass_trajectory: List[float] = field(default_factory=list)
    spectrum_interventions: List[int] = field(default_factory=list)
    steering_mode: str = "spectrum"


def _gini(vals):
    """Gini coefficient of a 1D array."""
    s = np.sort(vals)
    n = len(s)
    t = s.sum()
    if t == 0 or n == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * (idx * s).sum()) / (n * t) - (n + 1) / n


def _spread(vals):
    """Spectral spread (std of distribution)."""
    layers = np.arange(len(vals))
    t = vals.sum()
    if t == 0:
        return 0.0
    mu = (layers * vals).sum() / t
    return float(np.sqrt(((layers - mu)**2 * vals).sum() / t))


def _early_mass(vals, n_early=9):
    """Fraction of total coherence in early layers."""
    t = vals.sum()
    if t == 0:
        return 0.0
    return float(vals[:n_early].sum() / t)


class SpectrumSteerer(CoherenceSteerer):
    """
    Spectrum-guided generation engine.

    Monitors the SHAPE of per-layer coherence at each token:
      - Gini coefficient (concentration)
      - Spectral spread (distribution width)
      - Early-layer mass (grounding signal)

    Steering triggers when Gini exceeds threshold, indicating
    coherence concentration (lock-in). Intervention strategy:
      1. Identify layers where coherence spikes above mean
      2. At each spike layer, subtract the lock-in direction
      3. Scale intervention by how far above threshold

    This redistributes computation away from the dominant attractor
    without destabilizing the generation.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        gini_threshold: float = 0.51,    # from n=200: halluc mean = 0.501
        c_delta_threshold: float = 0.010,  # fallback scalar threshold
        top_k_features: int = 100,
        steering_strength: float = 0.8,
        mode: str = "spectrum",            # "spectrum", "c_delta", or "combined"
        device: str = "cuda",
    ):
        super().__init__(
            model_name=model_name,
            c_delta_threshold=c_delta_threshold,
            top_k_features=top_k_features,
            steering_strength=steering_strength,
            device=device,
        )
        self.gini_threshold = gini_threshold
        self.mode = mode

        # For spectrum steering, we need ALL layers, not just early+late
        self._all_layers = list(range(self._n_layers))
        self._all_hook_names = {
            l: f"blocks.{l}.hook_resid_post" for l in self._all_layers
        }

        # Update decoder weight cache for all layers
        for i, tc in enumerate(self._tc_list):
            if i not in self._dec_weights:
                W_dec = getattr(tc, "W_dec", None)
                if W_dec is not None:
                    self._dec_weights[i] = W_dec

        print(f"[SpectrumSteerer] Mode: {mode}, Gini threshold: {gini_threshold}")

    def _build_full_capture_hooks(self, hidden_states: dict) -> list:
        """Capture hidden states from ALL layers (not just early+late)."""
        hooks = []
        for l in self._all_layers:
            name = self._all_hook_names[l]
            def _hook(value, hook, _l=l):
                hidden_states[_l] = value[0, -1].detach().clone()
                return value
            hooks.append((name, _hook))
        return hooks

    def _full_layer_coherence(self, hidden_states: dict) -> np.ndarray:
        """Compute coherence for all layers. Returns (n_layers,) array."""
        profile = np.zeros(self._n_layers)
        for layer_idx in range(self._n_layers):
            hs = hidden_states.get(layer_idx)
            if hs is None:
                continue
            c = self._fast_layer_coherence(hs, layer_idx)
            if c is not None:
                profile[layer_idx] = c
        return profile

    def _find_spike_layers(self, profile: np.ndarray, threshold_factor: float = 1.5) -> List[int]:
        """Find layers where coherence exceeds threshold_factor * mean."""
        mean_c = profile[profile > 0].mean() if (profile > 0).any() else 0
        if mean_c == 0:
            return []
        threshold = mean_c * threshold_factor
        return [int(i) for i in range(len(profile)) if profile[i] > threshold]

    def generate_spectrum(
        self,
        prompt: str,
        max_tokens: int = 50,
        steer: bool = True,
        verbose: bool = False,
    ) -> SpectrumResult:
        """
        Generate with spectrum-aware monitoring and optional steering.

        At each token:
          1. Forward pass capturing ALL layer hidden states
          2. Compute full coherence spectrum + shape metrics
          3. If Gini > threshold: multi-layer steering intervention
        """
        inner = self._inner
        if inner is None:
            raise ValueError("Inner model not accessible")

        tokens = inner.to_tokens(prompt)
        generated_tokens = []
        c_delta_trajectory = []
        c_layer_trajectory = []
        gini_trajectory = []
        spread_trajectory = []
        early_mass_trajectory = []
        interventions = []
        spectrum_interventions = []
        monitor_time = 0.0
        last_steer_step = -5
        cooldown = 3

        t_start = time.time()

        for step in range(max_tokens):
            # -- Step 1: full-layer forward pass --
            hidden_states = {}
            capture_hooks = self._build_full_capture_hooks(hidden_states)

            t_monitor = time.time()
            with torch.no_grad():
                logits = inner.run_with_hooks(tokens, fwd_hooks=capture_hooks)

            # -- Step 2: compute spectrum --
            profile = self._full_layer_coherence(hidden_states)
            gini = _gini(profile)
            spread = _spread(profile)
            em = _early_mass(profile)

            # Also compute C_delta for comparison
            c_delta, layer_c = self._fast_c_delta(hidden_states)

            gini_trajectory.append(float(gini))
            spread_trajectory.append(float(spread))
            early_mass_trajectory.append(float(em))
            c_delta_trajectory.append(c_delta)
            c_layer_trajectory.append({k: v for k, v in layer_c.items() if v is not None})
            monitor_time += time.time() - t_monitor

            # -- Step 3: spectrum-guided intervention --
            should_steer = False
            if steer and step >= 3 and (step - last_steer_step) > cooldown:
                if self.mode == "spectrum":
                    should_steer = gini > self.gini_threshold
                elif self.mode == "c_delta":
                    should_steer = c_delta is not None and c_delta > self.c_delta_threshold
                elif self.mode == "combined":
                    should_steer = (gini > self.gini_threshold or
                                    (c_delta is not None and c_delta > self.c_delta_threshold))

            if should_steer:
                # Multi-layer steering: find spike layers and steer at each
                spike_layers = self._find_spike_layers(profile)
                if not spike_layers:
                    # Fallback: steer at peak coherence layer
                    spike_layers = [int(np.argmax(profile))]

                # Scale strength by how far above threshold
                excess = (gini - self.gini_threshold) / self.gini_threshold
                adaptive_strength = self.steering_strength * (1.0 + excess)

                # Build multi-layer steer hooks
                steer_hooks = []
                for sl in spike_layers:
                    if sl not in hidden_states:
                        continue
                    lock_dir = self._compute_lock_in_direction(hidden_states[sl], sl)
                    if lock_dir is None:
                        continue
                    hook_name = self._all_hook_names[sl]
                    _d = lock_dir.to(tokens.device)
                    _s = adaptive_strength
                    def _steer(value, hook, _d=_d, _s=_s):
                        h = value[0, -1].float()
                        proj = (h @ _d)
                        value[0, -1] = (h - _s * proj * _d).to(value.dtype)
                        return value
                    steer_hooks.append((hook_name, _steer))

                if steer_hooks:
                    with torch.no_grad():
                        logits = inner.run_with_hooks(tokens, fwd_hooks=steer_hooks)
                    spectrum_interventions.append(step)
                    last_steer_step = step

                    if verbose:
                        old_id = logits[0, -1].argmax().item()
                        print(f"    [SPECTRUM-STEER] step {step} gini={gini:.3f} "
                              f"spike_layers={spike_layers} strength={adaptive_strength:.2f}")

            # -- Step 4: decode --
            next_id = logits[0, -1].argmax().item()
            next_str = inner.tokenizer.decode([next_id])
            if next_id == inner.tokenizer.eos_token_id:
                break
            generated_tokens.append(next_str)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

        elapsed = time.time() - t_start
        text = "".join(generated_tokens)
        overhead_pct = (monitor_time / max(elapsed, 0.001)) * 100

        return SpectrumResult(
            prompt=prompt,
            text=text,
            tokens=generated_tokens,
            c_delta_trajectory=c_delta_trajectory,
            c_per_layer_trajectory=c_layer_trajectory,
            interventions=spectrum_interventions,
            intervention_count=len(spectrum_interventions),
            total_tokens=len(generated_tokens),
            generation_time_s=elapsed,
            monitor_overhead_s=monitor_time,
            overhead_pct=overhead_pct,
            gini_trajectory=gini_trajectory,
            spread_trajectory=spread_trajectory,
            early_mass_trajectory=early_mass_trajectory,
            spectrum_interventions=spectrum_interventions,
            steering_mode=self.mode,
        )

    def compare_modes(
        self,
        prompt: str,
        max_tokens: int = 50,
    ) -> dict:
        """
        Run the same prompt through unsteered, C_delta-steered, and
        spectrum-steered generation. Returns comparison dict.
        """
        print(f"\n{'='*70}")
        print(f"SPECTRUM STEERER -- MODE COMPARISON")
        print(f"{'='*70}")
        safe_prompt = prompt.encode('ascii', 'replace').decode()
        print(f"Prompt: {safe_prompt}")

        # Unsteered
        print("\n[1/3] Unsteered (monitor only)...")
        r_base = self.generate_spectrum(prompt, max_tokens, steer=False)

        # C_delta steered (legacy)
        print("[2/3] C_delta-steered (legacy)...")
        old_mode = self.mode
        self.mode = "c_delta"
        r_cdelta = self.generate_spectrum(prompt, max_tokens, steer=True, verbose=True)

        # Spectrum steered (new)
        print("[3/3] Spectrum-steered (new)...")
        self.mode = "spectrum"
        r_spectrum = self.generate_spectrum(prompt, max_tokens, steer=True, verbose=True)
        self.mode = old_mode

        # Report
        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"  Unsteered:  {r_base.text.encode('ascii','replace').decode()[:100]}")
        print(f"  C_delta:    {r_cdelta.text.encode('ascii','replace').decode()[:100]}")
        print(f"  Spectrum:   {r_spectrum.text.encode('ascii','replace').decode()[:100]}")
        print(f"\n  Interventions: base={r_base.intervention_count}, "
              f"c_delta={r_cdelta.intervention_count}, "
              f"spectrum={r_spectrum.intervention_count}")

        # Gini statistics
        if r_base.gini_trajectory:
            print(f"\n  Gini (unsteered):  mean={np.mean(r_base.gini_trajectory):.3f}, "
                  f"max={max(r_base.gini_trajectory):.3f}")
        if r_spectrum.gini_trajectory:
            print(f"  Gini (spectrum):   mean={np.mean(r_spectrum.gini_trajectory):.3f}, "
                  f"max={max(r_spectrum.gini_trajectory):.3f}")

        print(f"\n  Overhead: base={r_base.overhead_pct:.1f}%, "
              f"c_delta={r_cdelta.overhead_pct:.1f}%, "
              f"spectrum={r_spectrum.overhead_pct:.1f}%")

        return {
            "prompt": prompt,
            "unsteered": r_base,
            "c_delta_steered": r_cdelta,
            "spectrum_steered": r_spectrum,
        }

    def demo(self, prompt: str = "The president of Australia is", max_tokens: int = 30):
        """Quick spectrum demo with live Gini readout."""
        print(f"\n{'='*70}")
        print(f"FATHOM SPECTRUM MONITOR -- LIVE")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"Mode: {self.mode}, Gini threshold: {self.gini_threshold}")
        print(f"{'='*70}\n")

        result = self.generate_spectrum(prompt, max_tokens, steer=True, verbose=True)

        safe_text = result.text.encode('ascii', 'replace').decode()
        print(f"\nGenerated: {safe_text}")
        print(f"\nPer-token spectrum:")
        for i, tok in enumerate(result.tokens):
            safe_tok = tok.encode('ascii', 'replace').decode()
            g = result.gini_trajectory[i] if i < len(result.gini_trajectory) else 0
            s = result.spread_trajectory[i] if i < len(result.spread_trajectory) else 0
            em = result.early_mass_trajectory[i] if i < len(result.early_mass_trajectory) else 0
            flag = " << STEER" if i in result.spectrum_interventions else ""
            warn = " !! HIGH GINI" if g > self.gini_threshold else ""
            print(f"  [{i:2d}] '{safe_tok:15s}' gini={g:.3f} spread={s:.2f} "
                  f"early_mass={em:.3f}{warn}{flag}")

        print(f"\nStats:")
        print(f"  tokens: {result.total_tokens}")
        print(f"  interventions: {result.intervention_count}")
        print(f"  generation time: {result.generation_time_s:.2f}s")
        print(f"  monitor overhead: {result.overhead_pct:.1f}%")

        return result


# -- evaluation runner --

def run_truthfulqa_eval(steerer, n_items=20, seed=42):
    """
    Run spectrum-steered vs unsteered on TruthfulQA prompts.
    Measures whether steering reduces hallucination signatures.
    """
    import json

    data_path = Path(__file__).parent / "truthfulqa_results" / "truthfulqa_n200_seed42.json"
    if not data_path.exists():
        print(f"TruthfulQA data not found at {data_path}")
        return None

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(data["items"]), size=min(n_items, len(data["items"])), replace=False)

    results = []
    for idx in indices:
        item = data["items"][idx]
        q = item["question"]
        prompt = f"Q: {q}\nA:"

        # Unsteered
        r_base = steerer.generate_spectrum(prompt, max_tokens=30, steer=False)
        # Spectrum-steered
        r_steer = steerer.generate_spectrum(prompt, max_tokens=30, steer=True)

        results.append({
            "question": q,
            "base_text": r_base.text,
            "steer_text": r_steer.text,
            "base_gini_mean": float(np.mean(r_base.gini_trajectory)) if r_base.gini_trajectory else 0,
            "steer_gini_mean": float(np.mean(r_steer.gini_trajectory)) if r_steer.gini_trajectory else 0,
            "base_gini_max": float(max(r_base.gini_trajectory)) if r_base.gini_trajectory else 0,
            "steer_gini_max": float(max(r_steer.gini_trajectory)) if r_steer.gini_trajectory else 0,
            "interventions": r_steer.intervention_count,
        })

        safe_q = q[:50].encode('ascii', 'replace').decode()
        print(f"  [{len(results):2d}] {safe_q}...")
        print(f"       base gini={results[-1]['base_gini_mean']:.3f}, "
              f"steer gini={results[-1]['steer_gini_mean']:.3f}, "
              f"interventions={results[-1]['interventions']}")

    # Summary
    base_ginis = [r["base_gini_mean"] for r in results]
    steer_ginis = [r["steer_gini_mean"] for r in results]
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY (n={len(results)})")
    print(f"  Base mean Gini:   {np.mean(base_ginis):.4f}")
    print(f"  Steered mean Gini: {np.mean(steer_ginis):.4f}")
    print(f"  Gini reduction:   {np.mean(base_ginis) - np.mean(steer_ginis):+.4f}")
    total_interventions = sum(r["interventions"] for r in results)
    print(f"  Total interventions: {total_interventions}")
    print(f"{'='*60}")

    return results


from pathlib import Path

if __name__ == "__main__":
    import sys

    steerer = SpectrumSteerer(
        model_name="google/gemma-2-2b",
        gini_threshold=0.51,
        steering_strength=0.8,
        mode="spectrum",
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        results = run_truthfulqa_eval(steerer, n_items=n)
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "The president of Australia is"
        steerer.compare_modes(prompt)
    else:
        # Default: demo with a hallucination-prone prompt
        steerer.demo("The president of Australia is", max_tokens=25)
        print()
        steerer.demo("Eating watermelon seeds will", max_tokens=25)
        print()
        steerer.compare_modes("The Great Wall of China is visible from space because")
