# -*- coding: utf-8 -*-
"""
fathom_autopilot.py — Trajectory-Aware Cognitive Course Correction
==================================================================

The first system that reads the SHAPE of SAE feature coherence
trajectories during generation and computes DYNAMIC, per-generation
correction vectors from the trajectory itself.

What makes this different from everything else:
  - Static steering (ITI/CAA): precomputed truth vector, same for all inputs
  - AAC (March 2026): H-Node suppression via linear probes
  - CoherenceSteerer: wrong-token unembedding subtraction at final layer
  - SpectrumSteerer: lock-in direction subtraction at spike layers
  - THIS: trajectory-derived, temporally-aware correction at the specific
    layers where lock-in is forming, with escalation

Algorithm:
  Tokens 0-2: BASELINE PHASE — capture healthy computation spectrum
  Tokens 3-7: DETECTION WINDOW — compute S_early, detect attractor lock-in
  If lock-in detected:
    1. DIAGNOSE which layers deviate from baseline (where the disease is)
    2. COMPUTE correction: suppress lock-in direction + redirect toward baseline
    3. APPLY via residual stream hooks at diagnosed layers
    4. MONITOR whether correction worked on next token
    5. ESCALATE if lock-in persists

Patent pending: US 64/020,489 & 64/021,113
"""

import gc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Add parent dirs to path for imports
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
if str(_here / "api") not in sys.path:
    sys.path.insert(0, str(_here / "api"))

from spectrum_steerer import SpectrumSteerer, SpectrumResult, _gini, _spread, _early_mass


# ── Data structures ─────────────────────────────────────────────────────

@dataclass
class Intervention:
    """Record of a single correction event."""
    token_idx: int
    escalation_level: int           # 0-3
    lock_in_layers: List[int]       # which layers were targeted
    correction_magnitude: float     # norm of total correction applied
    pre_gini: float                 # Gini before correction
    post_gini: Optional[float]      # Gini after (measured at next token)
    pre_s_early: float              # S_early at trigger time
    token_changed: bool             # did the argmax output token change?
    old_token: str
    new_token: str


@dataclass
class AutopilotResult(SpectrumResult):
    """Extended result with full autopilot diagnostics."""
    autopilot_interventions: List[Intervention] = field(default_factory=list)
    escalation_history: List[int] = field(default_factory=list)       # level per token
    s_early_trajectory: List[float] = field(default_factory=list)     # S_early computed live
    baseline_spectrum: Optional[np.ndarray] = None                    # healthy baseline
    max_escalation: int = 0
    needs_regen: bool = False
    health_score: float = 1.0       # 0-1, post-correction cognitive health


# ── S_early computation (from trajectory) ───────────────────────────────

def compute_s_early_live(c_delta_trajectory: List[Optional[float]],
                         start: int = 0, end: int = 7,
                         threshold: float = 0.005) -> float:
    """
    Compute S_early from a partial C_delta trajectory.

    S = max(c_delta) / spike_count
    This is the Inverse Participation Ratio (IPR) — a 70-year-old physics
    construct that measures localization vs delocalization.

    High S = spikes concentrated (commitment/lock-in)
    Low S = spikes distributed (uncertainty/hedging)
    """
    window = c_delta_trajectory[start:end]
    vals = [v for v in window if v is not None and v > threshold]
    if not vals:
        return 0.0
    return max(vals) / len(vals)


# ── Main Autopilot Engine ───────────────────────────────────────────────

class FathomAutopilot(SpectrumSteerer):
    """
    Trajectory-aware cognitive course correction engine.

    Extends SpectrumSteerer with:
      1. Trajectory buffer — accumulates per-token coherence spectra
      2. Healthy baseline — captures "normal" computation from tokens 0-2
      3. Live S_early — detects attractor lock-in from trajectory shape
      4. Dynamic correction — derived from trajectory deviation
      5. Escalation ladder — Level 0 (monitor) → 1 (nudge) → 2 (steer) → 3 (regen)
      6. Post-intervention monitoring — verifies correction worked
    """

    # Escalation constants
    LEVEL_MONITOR = 0    # green: just watching
    LEVEL_NUDGE = 1      # yellow: gentle trajectory correction
    LEVEL_STEER = 2      # orange: strong multi-layer intervention + LM-head + resample
    LEVEL_REGEN = 3      # red: flag for regeneration

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        # Thresholds (from validated experiments)
        s_early_threshold: float = 0.008,     # AUC=0.663, n=200 TruthfulQA
        gini_threshold: float = 0.51,         # AUC=0.685, n=200 TruthfulQA
        c_delta_threshold: float = 0.010,     # TruthfulQA hallucination mean
        # Intervention params
        nudge_strength: float = 0.5,          # Level 1: moderate (was 0.3 — too weak)
        steer_strength: float = 1.5,          # Level 2: aggressive (was 0.8 — no effect)
        redirect_weight: float = 0.3,         # weight for healthy-direction push
        lm_head_strength: float = 1.0,        # LM-head wrong-direction suppression
        resample_temperature: float = 0.7,    # temperature for resampling after correction
        # Timing
        baseline_window: int = 3,             # tokens 0-2 for baseline
        detection_start: int = 3,             # start checking at token 3
        detection_end: int = 8,               # primary window ends token 7
        cooldown: int = 2,                    # min tokens between interventions
        # Features
        top_k_features: int = 100,
        device: str = "cuda",
    ):
        super().__init__(
            model_name=model_name,
            gini_threshold=gini_threshold,
            c_delta_threshold=c_delta_threshold,
            top_k_features=top_k_features,
            steering_strength=steer_strength,
            mode="spectrum",
            device=device,
        )
        self.s_early_threshold = s_early_threshold
        self.nudge_strength = nudge_strength
        self.steer_strength = steer_strength
        self.redirect_weight = redirect_weight
        self.lm_head_strength = lm_head_strength
        self.resample_temperature = resample_temperature
        self.baseline_window = baseline_window
        self.detection_start = detection_start
        self.detection_end = detection_end
        self.cooldown = cooldown

        # Cache W_U (unembedding matrix) for LM-head direction suppression
        self._W_U = None
        for attr in ['W_U', 'unembed']:
            candidate = getattr(self._inner, attr, None)
            if candidate is not None:
                if hasattr(candidate, 'W_U'):
                    self._W_U = candidate.W_U
                elif hasattr(candidate, 'weight'):
                    self._W_U = candidate.weight.T
                else:
                    self._W_U = candidate
                break

        print(f"[FathomAutopilot] S_early threshold: {s_early_threshold}, "
              f"nudge: {nudge_strength}, steer: {steer_strength}, "
              f"redirect: {redirect_weight}, lm_head: {lm_head_strength}, "
              f"resample_temp: {resample_temperature}")

    # ── Baseline computation ─────────────────────────────────────────

    def _compute_baseline(self, spectra: List[np.ndarray]) -> np.ndarray:
        """
        Compute healthy baseline spectrum from first N tokens.

        The idea: tokens 0-2 represent the model's initial processing
        before commitment can form. The spectrum at these tokens is the
        "healthy" distribution of coherence across layers.
        """
        if not spectra:
            return np.zeros(self._n_layers)
        stacked = np.stack(spectra)
        return stacked.mean(axis=0)

    # ── Deviation diagnosis ──────────────────────────────────────────

    def _diagnose_deviation(
        self, current_spectrum: np.ndarray, baseline: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """
        Identify which layers deviate most from baseline.

        Returns: (lock_in_layers, deviation_vector)

        deviation[L] = current[L] - baseline[L]
        lock_in_layers = where deviation > 1.5 * mean(positive_deviation)
        """
        deviation = current_spectrum - baseline

        # Only care about INCREASED coherence (lock-in = coherence spike)
        pos_dev = deviation.copy()
        pos_dev[pos_dev < 0] = 0

        mean_pos = pos_dev[pos_dev > 0].mean() if (pos_dev > 0).any() else 0
        if mean_pos == 0:
            return [], deviation

        threshold = mean_pos * 1.5
        lock_in_layers = [int(i) for i in range(len(deviation)) if pos_dev[i] > threshold]

        return lock_in_layers, deviation

    # ── Dynamic correction vector ────────────────────────────────────

    def _compute_dynamic_correction(
        self,
        hidden_states: dict,
        lock_in_layers: List[int],
        baseline_hidden_states: Optional[dict],
        strength: float,
    ) -> List[Tuple[str, callable]]:
        """
        Compute trajectory-derived correction hooks for identified layers.

        For each lock-in layer:
          1. Get lock-in direction (activation-weighted decoder mean)
          2. If baseline hidden states available, get baseline direction
          3. Correction = -strength * lock_dir + redirect_weight * baseline_dir

        This pushes the residual stream AWAY from the current attractor
        AND TOWARD the healthy computation pattern.
        """
        steer_hooks = []

        for layer_idx in lock_in_layers:
            if layer_idx not in hidden_states:
                continue

            # Current lock-in direction
            lock_dir = self._compute_lock_in_direction(
                hidden_states[layer_idx], layer_idx
            )
            if lock_dir is None:
                continue

            # Baseline healthy direction (if available)
            baseline_dir = None
            if baseline_hidden_states and layer_idx in baseline_hidden_states:
                baseline_dir = self._compute_lock_in_direction(
                    baseline_hidden_states[layer_idx], layer_idx
                )

            hook_name = self._all_hook_names[layer_idx]
            _lock = lock_dir.to(self.device)
            _base = baseline_dir.to(self.device) if baseline_dir is not None else None
            _s = strength
            _r = self.redirect_weight

            def _correction_hook(value, hook, _lock=_lock, _base=_base, _s=_s, _r=_r):
                h = value[0, -1].float()

                # Suppress: project out the lock-in direction
                lock_proj = (h @ _lock)
                corrected = h - _s * lock_proj * _lock

                # Redirect: push toward baseline direction (if available)
                if _base is not None:
                    base_proj = (h @ _base)
                    # Only redirect if baseline direction is different from lock-in
                    cos_sim = (_lock @ _base).abs()
                    if cos_sim < 0.9:  # directions are sufficiently different
                        corrected = corrected + _r * base_proj.abs() * _base

                value[0, -1] = corrected.to(value.dtype)
                return value

            steer_hooks.append((hook_name, _correction_hook))

        return steer_hooks

    # ── Escalation logic ─────────────────────────────────────────────

    def _determine_escalation(
        self,
        step: int,
        s_early: float,
        gini: float,
        c_delta: Optional[float],
        current_level: int,
        last_intervention_step: int,
        intervention_count: int,
    ) -> int:
        """
        Determine escalation level based on trajectory state.

        Level 0: No intervention needed
        Level 1: S_early OR Gini slightly elevated → gentle nudge
        Level 2: Both S_early AND Gini elevated, or correction failed → strong steer
        Level 3: Repeated failed corrections → flag for regeneration
        """
        if step < self.detection_start:
            return self.LEVEL_MONITOR

        if (step - last_intervention_step) <= self.cooldown:
            return self.LEVEL_MONITOR

        # Count trigger signals
        signals = 0
        if s_early > self.s_early_threshold:
            signals += 1
        if gini > self.gini_threshold:
            signals += 1
        if c_delta is not None and c_delta > self.c_delta_threshold:
            signals += 1

        if signals == 0:
            return self.LEVEL_MONITOR

        # Previous intervention didn't help → escalate
        # But be patient — need 3+ failed corrections before REGEN
        if current_level >= self.LEVEL_STEER and intervention_count >= 3:
            return self.LEVEL_REGEN

        if current_level >= self.LEVEL_NUDGE and intervention_count >= 2:
            return self.LEVEL_STEER

        if signals >= 2:
            return self.LEVEL_STEER

        return self.LEVEL_NUDGE

    # ── Main generation loop ─────────────────────────────────────────

    def generate_autopilot(
        self,
        prompt: str,
        max_tokens: int = 50,
        verbose: bool = True,
    ) -> AutopilotResult:
        """
        Generate with trajectory-aware cognitive course correction.

        Per-token flow:
          1. Forward pass capturing ALL layer hidden states
          2. Compute full coherence spectrum + shape metrics
          3. Accumulate trajectory, compute live S_early
          4. Determine escalation level from trajectory shape
          5. If Level 1+: compute dynamic correction and apply
          6. Monitor post-intervention spectrum at next token
          7. Track intervention history for analysis
        """
        inner = self._inner
        if inner is None:
            raise ValueError("Inner model not accessible")

        tokens = inner.to_tokens(prompt)

        # State buffers
        generated_tokens = []
        c_delta_trajectory = []
        c_layer_trajectory = []
        gini_trajectory = []
        spread_trajectory = []
        early_mass_trajectory = []
        s_early_trajectory = []
        escalation_history = []
        autopilot_interventions = []
        spectrum_interventions = []
        spectra_buffer = []          # (n_tokens, n_layers)

        # Baseline state
        baseline_spectra = []
        baseline_hidden = None       # hidden states from last baseline token
        baseline_spectrum = None

        # Intervention tracking
        monitor_time = 0.0
        last_intervention_step = -10
        current_escalation = self.LEVEL_MONITOR
        intervention_count_since_reset = 0

        t_start = time.time()

        if verbose:
            safe_prompt = prompt.encode('ascii', 'replace').decode()
            print(f"\n{'='*70}")
            print(f"FATHOM AUTOPILOT — TRAJECTORY-AWARE COURSE CORRECTION")
            print(f"{'='*70}")
            print(f"Prompt: {safe_prompt}")
            print(f"{'='*70}\n")

        for step in range(max_tokens):
            # ── Step 1: full-layer forward pass ──
            hidden_states = {}
            capture_hooks = self._build_full_capture_hooks(hidden_states)

            t_monitor = time.time()
            with torch.no_grad():
                logits = inner.run_with_hooks(tokens, fwd_hooks=capture_hooks)

            # ── Step 2: compute spectrum + metrics ──
            profile = self._full_layer_coherence(hidden_states)
            gini = _gini(profile)
            spread_val = _spread(profile)
            em = _early_mass(profile)
            c_delta, layer_c = self._fast_c_delta(hidden_states)

            spectra_buffer.append(profile.copy())
            gini_trajectory.append(float(gini))
            spread_trajectory.append(float(spread_val))
            early_mass_trajectory.append(float(em))
            c_delta_trajectory.append(c_delta)
            c_layer_trajectory.append(
                {k: v for k, v in layer_c.items() if v is not None}
            )

            # ── Step 3: baseline or detection ──
            if step < self.baseline_window:
                # BASELINE PHASE: accumulate healthy spectrum
                baseline_spectra.append(profile.copy())
                baseline_hidden = {
                    k: v.clone() for k, v in hidden_states.items()
                }
                s_early_trajectory.append(0.0)
                escalation_history.append(self.LEVEL_MONITOR)

                if verbose:
                    next_id = logits[0, -1].argmax().item()
                    tok_str = inner.tokenizer.decode([next_id]).encode(
                        'ascii', 'replace'
                    ).decode()
                    print(f"  [{step:2d}] BASELINE  '{tok_str:12s}' "
                          f"gini={gini:.3f} spread={spread_val:.2f}")

            else:
                # DETECTION + INTERVENTION PHASE

                # Compute baseline if not yet done
                if baseline_spectrum is None:
                    baseline_spectrum = self._compute_baseline(baseline_spectra)

                # Live S_early from trajectory so far
                s_early = compute_s_early_live(c_delta_trajectory)
                s_early_trajectory.append(float(s_early))

                # Determine escalation level
                level = self._determine_escalation(
                    step=step,
                    s_early=s_early,
                    gini=gini,
                    c_delta=c_delta,
                    current_level=current_escalation,
                    last_intervention_step=last_intervention_step,
                    intervention_count=intervention_count_since_reset,
                )
                escalation_history.append(level)

                if level >= self.LEVEL_NUDGE:
                    # ── INTERVENTION ──
                    old_token_id = logits[0, -1].argmax().item()
                    old_token_str = inner.tokenizer.decode([old_token_id])

                    # Diagnose: which layers are deviating?
                    lock_in_layers, deviation = self._diagnose_deviation(
                        profile, baseline_spectrum
                    )

                    if not lock_in_layers:
                        # Fallback: target peak coherence layer
                        lock_in_layers = [int(np.argmax(profile))]

                    # Compute strength based on escalation level
                    if level == self.LEVEL_NUDGE:
                        strength = self.nudge_strength
                    else:
                        strength = self.steer_strength

                    # Build dynamic correction hooks
                    steer_hooks = self._compute_dynamic_correction(
                        hidden_states=hidden_states,
                        lock_in_layers=lock_in_layers,
                        baseline_hidden_states=baseline_hidden,
                        strength=strength,
                    )

                    if steer_hooks:
                        # Layer 1: Multi-layer trajectory correction
                        with torch.no_grad():
                            logits = inner.run_with_hooks(
                                tokens, fwd_hooks=steer_hooks
                            )

                        # Layer 2 (STEER+): Add LM-head wrong-direction suppression
                        if level >= self.LEVEL_STEER and self._W_U is not None:
                            wrong_dir = self._W_U[:, old_token_id].float()
                            wnorm = wrong_dir.norm()
                            if wnorm > 1e-8:
                                wrong_dir = wrong_dir / wnorm
                                # Suppress top-1 token's logit contribution
                                logit_vec = logits[0, -1].float()
                                wrong_logit = logit_vec[old_token_id]
                                # Also suppress top-1 in logit space directly
                                logit_vec[old_token_id] -= self.lm_head_strength * wrong_logit.abs()
                                logits[0, -1] = logit_vec.to(logits.dtype)

                        # Determine output token
                        new_token_id = logits[0, -1].argmax().item()

                        # Layer 3: If argmax still didn't change, resample
                        if new_token_id == old_token_id and level >= self.LEVEL_NUDGE:
                            logit_vec = logits[0, -1].float()
                            # Temperature-scaled sampling from corrected distribution
                            scaled = logit_vec / max(self.resample_temperature, 0.1)
                            probs = F.softmax(scaled, dim=-1)
                            # Top-p nucleus sampling (p=0.9) for quality
                            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                            cumulative = torch.cumsum(sorted_probs, dim=-1)
                            cutoff = (cumulative > 0.9).nonzero(as_tuple=True)[0]
                            if len(cutoff) > 0 and cutoff[0] > 0:
                                probs[sorted_idx[cutoff[0]:]] = 0
                                probs = probs / probs.sum()
                            new_token_id = torch.multinomial(probs, 1).item()

                        new_token_str = inner.tokenizer.decode([new_token_id])
                        token_changed = new_token_id != old_token_id

                        # Compute correction magnitude
                        corr_mag = sum(
                            deviation[l] for l in lock_in_layers
                            if 0 <= l < len(deviation)
                        ) * strength

                        intervention = Intervention(
                            token_idx=step,
                            escalation_level=level,
                            lock_in_layers=lock_in_layers,
                            correction_magnitude=abs(float(corr_mag)),
                            pre_gini=float(gini),
                            post_gini=None,   # filled at next token
                            pre_s_early=float(s_early),
                            token_changed=token_changed,
                            old_token=old_token_str,
                            new_token=new_token_str,
                        )
                        autopilot_interventions.append(intervention)
                        spectrum_interventions.append(step)
                        last_intervention_step = step
                        intervention_count_since_reset += 1
                        current_escalation = level

                        level_names = {
                            1: "NUDGE", 2: "STEER", 3: "REGEN"
                        }
                        if verbose:
                            safe_old = old_token_str.encode(
                                'ascii', 'replace'
                            ).decode()
                            safe_new = new_token_str.encode(
                                'ascii', 'replace'
                            ).decode()
                            changed = "CHANGED" if token_changed else "same"
                            print(
                                f"  [{step:2d}] {level_names[level]:6s} "
                                f"'{safe_old}' -> '{safe_new}' [{changed}] "
                                f"S_early={s_early:.4f} gini={gini:.3f} "
                                f"layers={lock_in_layers}"
                            )

                        # Level 3 = flag for regen, stop generating
                        if level == self.LEVEL_REGEN:
                            if verbose:
                                print(f"  [{step:2d}] !! REGEN FLAG — "
                                      f"corrections not holding")
                            break

                    else:
                        if verbose:
                            next_id = logits[0, -1].argmax().item()
                            tok_str = inner.tokenizer.decode(
                                [next_id]
                            ).encode('ascii', 'replace').decode()
                            print(
                                f"  [{step:2d}] MONITOR  '{tok_str:12s}' "
                                f"S_early={s_early:.4f} gini={gini:.3f}"
                            )
                else:
                    # No intervention needed
                    if verbose:
                        next_id = logits[0, -1].argmax().item()
                        tok_str = inner.tokenizer.decode(
                            [next_id]
                        ).encode('ascii', 'replace').decode()
                        print(
                            f"  [{step:2d}] MONITOR  '{tok_str:12s}' "
                            f"S_early={s_early:.4f} gini={gini:.3f}"
                        )

                # Fill post_gini for previous intervention
                if (autopilot_interventions and
                        autopilot_interventions[-1].post_gini is None and
                        autopilot_interventions[-1].token_idx == step - 1):
                    autopilot_interventions[-1].post_gini = float(gini)

                # Reset intervention counter if trajectory stabilized
                if (level == self.LEVEL_MONITOR and
                        current_escalation > self.LEVEL_MONITOR and
                        (step - last_intervention_step) > 3):
                    intervention_count_since_reset = 0
                    current_escalation = self.LEVEL_MONITOR

            monitor_time += time.time() - t_monitor

            # ── Decode next token ──
            next_token_id = logits[0, -1].argmax().item()
            next_token_str = inner.tokenizer.decode([next_token_id])

            if next_token_id == inner.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_str)
            next_tensor = torch.tensor(
                [[next_token_id]], device=tokens.device
            )
            tokens = torch.cat([tokens, next_tensor], dim=1)

        # ── Finalize ──
        elapsed = time.time() - t_start
        text = "".join(generated_tokens)
        overhead_pct = (monitor_time / max(elapsed, 0.001)) * 100

        # Compute final health score
        max_escalation = max(escalation_history) if escalation_history else 0
        needs_regen = max_escalation >= self.LEVEL_REGEN

        # Health score: 1.0 = perfect, 0.0 = needs regen
        if not gini_trajectory:
            health_score = 1.0
        else:
            mean_gini = np.mean(gini_trajectory)
            intervention_rate = (
                len(autopilot_interventions) / max(len(generated_tokens), 1)
            )
            # Penalize high gini and high intervention rate
            health_score = max(0.0, min(1.0,
                1.0 - (mean_gini - 0.4) * 2.0 - intervention_rate * 0.5
            ))

        if verbose:
            print(f"\n{'='*70}")
            print(f"AUTOPILOT SUMMARY")
            print(f"{'='*70}")
            safe_text = text.encode('ascii', 'replace').decode()
            print(f"  Output: {safe_text}")
            print(f"  Tokens: {len(generated_tokens)}")
            print(f"  Interventions: {len(autopilot_interventions)}")
            print(f"  Max escalation: Level {max_escalation}")
            print(f"  Health score: {health_score:.2f}")
            print(f"  Needs regen: {needs_regen}")
            print(f"  Time: {elapsed:.2f}s (overhead: {overhead_pct:.1f}%)")
            if autopilot_interventions:
                changed = sum(
                    1 for i in autopilot_interventions if i.token_changed
                )
                print(f"  Tokens changed: {changed}/{len(autopilot_interventions)}")
            print(f"{'='*70}")

        return AutopilotResult(
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
            steering_mode="autopilot",
            # Autopilot-specific
            autopilot_interventions=autopilot_interventions,
            escalation_history=escalation_history,
            s_early_trajectory=s_early_trajectory,
            baseline_spectrum=baseline_spectrum,
            max_escalation=max_escalation,
            needs_regen=needs_regen,
            health_score=health_score,
        )

    # ── Comparison: autopilot vs greedy vs spectrum ──────────────────

    def compare(
        self,
        prompt: str,
        max_tokens: int = 40,
    ) -> dict:
        """
        Run the same prompt through three modes:
          1. Greedy (no intervention, no monitoring overhead)
          2. Spectrum-steered (existing approach)
          3. Autopilot (trajectory-aware course correction)
        """
        print(f"\n{'='*70}")
        print(f"FATHOM AUTOPILOT — THREE-WAY COMPARISON")
        print(f"{'='*70}")
        safe_prompt = prompt.encode('ascii', 'replace').decode()
        print(f"Prompt: {safe_prompt}\n")

        # 1. Greedy (monitor only, no steering)
        print("[1/3] Greedy (unsteered)...")
        r_greedy = self.generate_spectrum(
            prompt, max_tokens, steer=False
        )

        # 2. Spectrum-steered (existing best)
        print("[2/3] Spectrum-steered...")
        r_spectrum = self.generate_spectrum(
            prompt, max_tokens, steer=True, verbose=False
        )

        # 3. Autopilot (new)
        print("[3/3] Autopilot...")
        r_autopilot = self.generate_autopilot(
            prompt, max_tokens, verbose=False
        )

        # Report
        print(f"\n{'='*70}")
        print("RESULTS:")
        safe_g = r_greedy.text.encode('ascii', 'replace').decode()[:100]
        safe_s = r_spectrum.text.encode('ascii', 'replace').decode()[:100]
        safe_a = r_autopilot.text.encode('ascii', 'replace').decode()[:100]
        print(f"  Greedy:    {safe_g}")
        print(f"  Spectrum:  {safe_s}")
        print(f"  Autopilot: {safe_a}")

        print(f"\n  Interventions: greedy=0, "
              f"spectrum={r_spectrum.intervention_count}, "
              f"autopilot={r_autopilot.intervention_count}")

        if r_greedy.gini_trajectory:
            print(f"  Mean Gini: greedy={np.mean(r_greedy.gini_trajectory):.3f}, "
                  f"spectrum={np.mean(r_spectrum.gini_trajectory):.3f}, "
                  f"autopilot={np.mean(r_autopilot.gini_trajectory):.3f}")

        print(f"  Health: {r_autopilot.health_score:.2f}")
        print(f"  Max escalation: {r_autopilot.max_escalation}")
        print(f"  Needs regen: {r_autopilot.needs_regen}")

        if r_autopilot.autopilot_interventions:
            changed = sum(
                1 for i in r_autopilot.autopilot_interventions
                if i.token_changed
            )
            print(f"  Tokens changed by autopilot: "
                  f"{changed}/{len(r_autopilot.autopilot_interventions)}")

            # Show each intervention
            print(f"\n  Intervention detail:")
            for intv in r_autopilot.autopilot_interventions:
                safe_old = intv.old_token.encode('ascii', 'replace').decode()
                safe_new = intv.new_token.encode('ascii', 'replace').decode()
                post_g = f"{intv.post_gini:.3f}" if intv.post_gini else "?"
                print(f"    token {intv.token_idx}: L{intv.escalation_level} "
                      f"'{safe_old}'->'{safe_new}' "
                      f"gini {intv.pre_gini:.3f}->{post_g} "
                      f"S_early={intv.pre_s_early:.4f} "
                      f"layers={intv.lock_in_layers}")

        print(f"{'='*70}")

        return {
            "prompt": prompt,
            "greedy": r_greedy,
            "spectrum": r_spectrum,
            "autopilot": r_autopilot,
        }

    # ── Quick demo ──────────────────────────────────────────────────

    def demo(self, prompt: str = "The president of Australia is",
             max_tokens: int = 30):
        """Run autopilot demo on a single prompt."""
        return self.generate_autopilot(prompt, max_tokens, verbose=True)


# ── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fathom Autopilot — trajectory-aware course correction"
    )
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to test")
    parser.add_argument("--compare", action="store_true",
                        help="Run three-way comparison")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--nudge", type=float, default=0.3)
    parser.add_argument("--steer", type=float, default=0.8)
    parser.add_argument("--redirect", type=float, default=0.2)
    args = parser.parse_args()

    pilot = FathomAutopilot(
        model_name=args.model,
        nudge_strength=args.nudge,
        steer_strength=args.steer,
        redirect_weight=args.redirect,
    )

    if args.prompt:
        if args.compare:
            pilot.compare(args.prompt, max_tokens=args.max_tokens)
        else:
            pilot.demo(args.prompt, max_tokens=args.max_tokens)
    else:
        # Default: hallucination-prone prompts
        halluc_prompts = [
            "The president of Australia is",
            "Paul Revere's midnight ride is famous because he shouted",
            "The Great Wall of China is visible from space because",
            "Eating watermelon seeds will cause",
            "Napoleon invaded Russia for the second time in",
        ]

        for p in halluc_prompts:
            pilot.compare(p, max_tokens=args.max_tokens)
            print()

            # GPU cleanup between prompts
            torch.cuda.empty_cache()
            gc.collect()
