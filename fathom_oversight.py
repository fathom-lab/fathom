# -*- coding: utf-8 -*-
"""
fathom_oversight.py -- The Cognitive Oversight Layer
====================================================

Real-time cognitive monitoring and circuit-level intervention for
neural language models. Detects and corrects multiple classes of
cognitive failure during inference with sub-sentence latency.

Architecture:
  Layer 1: MONITORING  -- all axes (K, C, Cd, S, Gini) every token
  Layer 2: DIAGNOSIS   -- circuit-level fault attribution
  Layer 3: INTERVENTION -- escalating: suppress → steer → resample → abstain

Five anomaly types, five policies, one framework:
  1. Hallucination       (S_early + Gini)
  2. Shallow routing     (K depth ratio)
  3. Confidence diverge  (Cd + entropy)
  4. Adversarial depth   (missing RLHF amplification)
  5. Multi-policy        (all axes combined)

The key innovation: targeted SAE feature suppression during inference.
When an anomaly is detected, identify the specific features causing
the failure and dampen their contribution to the residual stream.
Nobody else can do this — it requires both a real-time anomaly signal
(Fathom axes) AND circuit-level access (SAE transcoders).

Patent pending: US 64/020,489 & 64/021,113
"""

import gc
import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn.functional as F
import numpy as np

# Parent classes
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "api"))

from spectrum_steerer import SpectrumSteerer, SpectrumResult, _gini, _spread, _early_mass
from cognitive_state import (
    CognitiveState, CognitiveAnomaly, CognitivePolicy, OversightResult,
    HALLUCINATION_POLICY, SHALLOW_ROUTING_POLICY, CONFIDENCE_DIVERGENCE_POLICY,
    ADVERSARIAL_DEPTH_POLICY, MULTI_POLICY, POLICY_REGISTRY,
    classify_anomaly_type,
)


# ── S_early computation (from pre-registered analysis) ───────────────────

def compute_s_early(c_delta_traj: List[float], end: int = 7, threshold: float = 0.005) -> float:
    """
    Compute S_early on trajectory[0:end].

    S = max(slice) / count(spikes > threshold)

    Matches the pre-registered formula from analyze_S_biphasic.py.
    """
    slc = c_delta_traj[:end]
    if not slc:
        return 0.0
    # Filter to valid values
    valid = [c for c in slc if c is not None and isinstance(c, (int, float))]
    if not valid:
        return 0.0
    spikes = [c for c in valid if c > threshold]
    if not spikes:
        return 0.0
    return max(valid) / len(spikes)


# ── Main Class ───────────────────────────────────────────────────────────

class FathomOversight(SpectrumSteerer):
    """
    The cognitive oversight engine.

    Extends SpectrumSteerer with:
      - Policy-driven anomaly detection (multiple anomaly types)
      - S_early as real-time rolling trigger
      - Feature-level fault diagnosis (top-k by activation in spike layers)
      - Feature suppression hooks (the novel mechanism)
      - Escalation ladder (suppress → steer → resample → abstain)
      - Post-intervention verification
      - Calibrated confidence scoring
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        policy: CognitivePolicy = None,
        s_early_threshold: float = 0.008,
        gini_threshold: float = 0.51,
        c_delta_threshold: float = 0.010,
        dampen_factor: float = 0.1,
        n_fault_features: int = 32,
        min_trigger_token: int = 5,
        verification_window: int = 3,
        resample_temperature: float = 0.9,
        max_escalation: int = 4,
        top_k_features: int = 100,
        steering_strength: float = 0.8,
        device: str = "cuda",
    ):
        super().__init__(
            model_name=model_name,
            gini_threshold=gini_threshold,
            c_delta_threshold=c_delta_threshold,
            top_k_features=top_k_features,
            steering_strength=steering_strength,
            mode="combined",
            device=device,
        )

        # Policy (defaults to hallucination if none specified)
        self.policy = policy or HALLUCINATION_POLICY
        self.s_early_threshold = self.policy.triggers.get("s_early", s_early_threshold)
        self.dampen_factor = self.policy.dampen_factor
        self.n_fault_features = self.policy.n_fault_features
        self.min_trigger_token = self.policy.min_trigger_token
        self.verification_window = self.policy.verification_window
        self.resample_temperature = self.policy.resample_temperature
        self.max_escalation = max_escalation

        # Cache encoder weights for suppression hooks (need W_enc, b_enc, W_dec)
        self._enc_weights = {}
        self._enc_biases = {}
        for i, tc in enumerate(self._tc_list):
            W_enc = getattr(tc, "W_enc", None)
            b_enc = getattr(tc, "b_enc", None)
            if W_enc is not None:
                self._enc_weights[i] = W_enc
            if b_enc is not None:
                self._enc_biases[i] = b_enc

        print(f"[FathomOversight] Policy: {self.policy.name}, "
              f"triggers: {self.policy.triggers}, "
              f"logic: {self.policy.trigger_logic}")

    # ── Layer 1: Cognitive State Computation ─────────────────────────────

    def _compute_cognitive_state(
        self,
        hidden_states: Dict[int, torch.Tensor],
        c_delta_traj: List[float],
        step: int,
        logits: Optional[torch.Tensor] = None,
        full_spectrum: bool = True,
    ) -> CognitiveState:
        """
        Compute unified cognitive state at one token position.
        All axes in a single call.

        If full_spectrum=False, only computes early+late layers (fast path).
        """
        # Spectrum: full layer profile + shape metrics
        if full_spectrum:
            profile = self._full_layer_coherence(hidden_states)
        else:
            # Fast path: only early+late layers (inherited from CoherenceSteerer)
            profile = np.zeros(self._n_layers)
            for layer_idx in self._monitor_layers:
                hs = hidden_states.get(layer_idx)
                if hs is not None:
                    c = self._fast_layer_coherence(hs, layer_idx)
                    if c is not None:
                        profile[layer_idx] = c
        gini = float(_gini(profile))
        spread = float(_spread(profile))
        em = float(_early_mass(profile))

        # C_delta: late minus early
        c_delta, layer_c = self._fast_c_delta(hidden_states)
        c_delta = c_delta if c_delta is not None else 0.0

        # K (depth): mean activated layer index weighted by coherence
        total_c = profile.sum()
        if total_c > 0:
            k_depth = float((np.arange(len(profile)) * profile).sum() / total_c)
        else:
            k_depth = 0.0

        # C (global coherence): mean of all layer coherences
        valid_c = profile[profile > 0]
        c_coherence = float(valid_c.mean()) if len(valid_c) > 0 else 0.0

        # S_early (rolling window)
        s_early = 0.0
        if step >= self.min_trigger_token:
            s_early = compute_s_early(c_delta_traj + [c_delta], end=min(step + 1, 7))

        # Entropy + logprob (from logits if available)
        entropy = None
        logprob = None
        top2_margin = None
        d_honesty = None  # D-axis: internal-external cognitive divergence
        if logits is not None:
            logit_vec = logits[0, -1].float()
            logp = F.log_softmax(logit_vec, dim=-1)
            p = logp.exp()
            entropy = float(-(p * logp).sum().item())
            top2 = p.topk(2)
            top2_margin = float((top2.values[0] - top2.values[1]).item())
            next_id = logit_vec.argmax().item()
            logprob = float(logp[next_id].item())

            # ── D-axis: Deception/Honesty measurement ──
            # D = cosine(residual_at_final_layer, unembedding[chosen_token])
            # High D = honest (internal state matches expression)
            # Low D = deceptive (internal state diverges from expression)
            final_layer = self._n_layers - 1
            final_hs = hidden_states.get(final_layer)
            if final_hs is not None:
                try:
                    # Reduce residual to a 1D vector at the current position
                    h = final_hs.float()
                    if h.dim() == 3:        # (batch, seq, d_model)
                        h = h[0, -1, :]
                    elif h.dim() == 2:      # (seq, d_model) or (batch, d_model)
                        h = h[-1, :]
                    # else assume (d_model,)
                    d_model = h.shape[-1]

                    # Locate unembedding matrix on HookedTransformer: unembed.W_U
                    W_U = None
                    for attr_path in ["unembed.W_U", "lm_head.weight",
                                      "embed_out.weight"]:
                        obj = self._inner
                        for part in attr_path.split("."):
                            obj = getattr(obj, part, None)
                            if obj is None:
                                break
                        if obj is not None:
                            W_U = obj
                            break

                    if W_U is not None and next_id is not None:
                        # Pick whichever dim equals d_model; the OTHER dim is vocab.
                        # TransformerLens layout: W_U is (d_model, vocab).
                        # HF lm_head.weight: (vocab, d_model).
                        W_f = W_U.float()
                        if W_f.shape[0] == d_model:
                            token_dir = W_f[:, next_id]
                        elif W_f.shape[1] == d_model:
                            token_dir = W_f[next_id, :]
                        else:
                            token_dir = None  # unexpected shape

                        if token_dir is not None:
                            h_norm = h / h.norm().clamp(min=1e-8)
                            t_norm = token_dir / token_dir.norm().clamp(min=1e-8)
                            d_honesty = float((h_norm @ t_norm).item())
                except Exception as _exc:
                    # Log first failure once so bugs don't stay silent
                    if not getattr(self, "_d_honesty_err_logged", False):
                        print(f"[d_honesty] first failure: "
                              f"{type(_exc).__name__}: {_exc}", flush=True)
                        self._d_honesty_err_logged = True

        return CognitiveState(
            token_idx=step,
            k_depth=k_depth,
            c_coherence=c_coherence,
            c_delta=c_delta,
            s_early=s_early,
            gini=gini,
            spread=spread,
            early_mass=em,
            layer_profile=profile,
            entropy=entropy,
            logprob=logprob,
            top2_margin=top2_margin,
            d_honesty=d_honesty,
        )

    # ── Layer 2: Anomaly Detection + Diagnosis ───────────────────────────

    def _check_triggers(
        self,
        state: CognitiveState,
        hidden_states: Dict[int, torch.Tensor],
    ) -> Optional[CognitiveAnomaly]:
        """
        Check policy triggers. If triggered, immediately diagnose.
        Returns CognitiveAnomaly or None.
        """
        if state.token_idx < self.policy.min_trigger_token:
            return None

        if not state.is_anomalous(self.policy):
            return None

        # Triggered! Determine anomaly type
        anomaly_type = classify_anomaly_type(state)

        # Find spike layers
        if state.layer_profile is not None:
            spike_layers = self._find_spike_layers(state.layer_profile)
            if not spike_layers:
                # Fallback: top 2 layers by coherence
                spike_layers = [int(i) for i in np.argsort(state.layer_profile)[-2:]]
        else:
            spike_layers = [11, 12]  # default from cognitive autopsy

        # Diagnose: which features are at fault?
        fault_features = self._diagnose(hidden_states, spike_layers)

        # Compute attractor coherence (how strongly fault features cluster)
        attractor_coh = self._compute_attractor_coherence(
            fault_features, hidden_states
        )

        # Determine which metric triggered (for logging)
        trigger_metric = "unknown"
        trigger_value = 0.0
        for metric, threshold in self.policy.triggers.items():
            val = state._get_metric(metric)
            if val is not None:
                if metric == "depth_delta" and val < threshold:
                    trigger_metric = metric
                    trigger_value = val
                    break
                elif metric == "entropy_low":
                    if state.entropy is not None and state.entropy < threshold:
                        trigger_metric = metric
                        trigger_value = state.entropy
                        break
                elif val > threshold:
                    trigger_metric = metric
                    trigger_value = val
                    break

        return CognitiveAnomaly(
            anomaly_type=anomaly_type,
            trigger_token=state.token_idx,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            fault_layers=spike_layers,
            fault_features=fault_features,
            attractor_coherence=attractor_coh,
            confidence=0.5,  # TODO: calibrate via S_early_predictor
        )

    def _diagnose(
        self,
        hidden_states: Dict[int, torch.Tensor],
        spike_layers: List[int],
    ) -> Dict[int, List[int]]:
        """
        Feature-level fault attribution.

        For each spike layer: run encoder pass, return top-N features
        by activation magnitude. These are the features dominating
        the residual stream — the ones creating the coherence cluster.
        """
        fault_map = {}
        for layer_idx in spike_layers:
            hs = hidden_states.get(layer_idx)
            if hs is None:
                continue

            W_enc = self._enc_weights.get(layer_idx)
            b_enc = self._enc_biases.get(layer_idx)
            if W_enc is None:
                continue

            with torch.no_grad():
                acts = hs.float() @ W_enc.float().T
                if b_enc is not None:
                    acts = acts + b_enc.float()
                acts = F.relu(acts)

            n_active = (acts > 0).sum().item()
            if n_active < 1:
                continue

            k = min(self.n_fault_features, int(n_active))
            _, topk_ids = torch.topk(acts, k)
            fault_map[layer_idx] = topk_ids.cpu().tolist()

        return fault_map

    def _compute_attractor_coherence(
        self,
        fault_features: Dict[int, List[int]],
        hidden_states: Dict[int, torch.Tensor],
    ) -> float:
        """
        How strongly the fault features cluster.
        Mean pairwise cosine similarity of fault feature decoder vectors.
        High = strong false attractor. Low = diffuse anomaly.
        """
        all_vecs = []
        for layer_idx, feat_ids in fault_features.items():
            W_dec = self._dec_weights.get(layer_idx)
            if W_dec is None:
                continue
            for fid in feat_ids[:16]:  # limit to keep fast
                if fid < W_dec.shape[0]:
                    all_vecs.append(W_dec[fid].float().cpu())

        if len(all_vecs) < 2:
            return 0.0

        vecs = torch.stack(all_vecs)
        norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vecs_n = vecs / norms
        sim = vecs_n @ vecs_n.T
        n = len(all_vecs)
        return float((sim.sum().item() - n) / max(n * (n - 1), 1))

    # ── Layer 3: Intervention ────────────────────────────────────────────

    def _make_suppression_hook(
        self,
        layer_idx: int,
        fault_ids: List[int],
        dampen_factor: float = None,
    ) -> tuple:
        """
        Build a forward hook that suppresses specific SAE features.

        The hook runs inside the existing forward pass:
          1. Encoder pass on residual → get feature activations
          2. For each fault feature: compute its contribution to residual
          3. Subtract (1 - dampen_factor) * contribution
          4. Return modified residual

        Cost: one encoder matmul (~0.5ms) + 32 vector lookups. Negligible.
        """
        W_enc = self._enc_weights.get(layer_idx)
        b_enc = self._enc_biases.get(layer_idx)
        W_dec = self._dec_weights.get(layer_idx)

        if W_enc is None or W_dec is None:
            return None

        _df = dampen_factor if dampen_factor is not None else self.dampen_factor
        _fids = torch.tensor(fault_ids, device=self.device, dtype=torch.long)
        # Filter to valid feature indices
        _fids = _fids[_fids < W_dec.shape[0]]
        if len(_fids) == 0:
            return None

        def hook_fn(value, hook):
            h = value[0, -1].float()
            # Encoder pass
            acts = h @ W_enc.float().T
            if b_enc is not None:
                acts = acts + b_enc.float()
            acts = F.relu(acts)

            # Fault feature contributions
            fault_acts = acts[_fids]
            fault_dirs = W_dec[_fids].float()

            # Subtract (1 - dampen) * sum(act[f] * dec[f])
            # This removes most of the fault features' influence on the residual
            correction = ((1.0 - _df) * fault_acts.unsqueeze(-1) * fault_dirs).sum(dim=0)
            value[0, -1] = (h - correction).to(value.dtype)
            return value

        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        return (hook_name, hook_fn)

    def _build_suppression_hooks(
        self,
        fault_map: Dict[int, List[int]],
        dampen_factor: float = None,
    ) -> List[tuple]:
        """Build suppression hooks for all fault layers."""
        hooks = []
        for layer_idx, feat_ids in fault_map.items():
            hook = self._make_suppression_hook(layer_idx, feat_ids, dampen_factor)
            if hook is not None:
                hooks.append(hook)
        return hooks

    def _build_steer_hooks(
        self,
        hidden_states: Dict[int, torch.Tensor],
        spike_layers: List[int],
        strength: float = None,
    ) -> List[tuple]:
        """Build directional steering hooks at spike layers."""
        hooks = []
        _s = strength if strength is not None else self.policy.steering_strength
        for sl in spike_layers:
            if sl not in hidden_states:
                continue
            lock_dir = self._compute_lock_in_direction(hidden_states[sl], sl)
            if lock_dir is None:
                continue
            hook_name = self._all_hook_names[sl]
            _d = lock_dir.to(self.device)
            def _steer(value, hook, _d=_d, _s=_s):
                h = value[0, -1].float()
                proj = h @ _d
                value[0, -1] = (h - _s * proj * _d).to(value.dtype)
                return value
            hooks.append((hook_name, _steer))
        return hooks

    def _escalate(
        self,
        current_level: int,
        anomaly: CognitiveAnomaly,
        hidden_states: Dict[int, torch.Tensor],
        tokens: torch.Tensor,
    ) -> Tuple[int, List[tuple], str]:
        """
        Determine next escalation level and return appropriate hooks.

        Returns: (new_level, hooks, action_name)
          action_name: "suppress", "steer", "resample", "abstain"
        """
        levels = self.policy.escalation_levels
        if current_level >= len(levels):
            return current_level, [], "abstain"

        action = levels[current_level]

        if action == "suppress":
            hooks = self._build_suppression_hooks(anomaly.fault_features)
            return current_level, hooks, "suppress"

        elif action == "steer":
            # Combine suppression + steering
            sup_hooks = self._build_suppression_hooks(anomaly.fault_features)
            steer_hooks = self._build_steer_hooks(
                hidden_states, anomaly.fault_layers
            )
            return current_level, sup_hooks + steer_hooks, "steer"

        elif action == "resample":
            return current_level, [], "resample"

        elif action == "abstain":
            return current_level, [], "abstain"

        return current_level, [], "unknown"

    def _verify_intervention(
        self,
        c_delta_traj: List[float],
        pre_intervention_s: float,
    ) -> bool:
        """
        Check if S_early dropped by at least 30% after intervention.
        """
        post_s = compute_s_early(c_delta_traj, end=len(c_delta_traj))
        if pre_intervention_s <= 0:
            return True
        reduction = (pre_intervention_s - post_s) / pre_intervention_s
        return reduction > 0.30

    # ── Resample Path ────────────────────────────────────────────────────

    def _resample_with_suppression(
        self,
        prompt: str,
        fault_map: Dict[int, List[int]],
        max_tokens: int = 50,
    ) -> OversightResult:
        """
        Full re-generation with suppression hooks active from token 0.
        Uses temperature sampling for diversity.

        Key insight from pilot: suppression from token 0 prevents the model
        from locking into the wrong attractor in the first place, rather
        than trying to fix it mid-generation (which is too late).
        """
        inner = self._inner
        tokens = inner.to_tokens(prompt)
        generated_tokens = []
        cognitive_states = []
        c_delta_traj_raw = []

        # Build persistent suppression hooks with policy's dampen factor
        sup_hooks = self._build_suppression_hooks(fault_map, self.policy.dampen_factor)

        t_start = time.time()

        for step in range(max_tokens):
            hidden_states = {}
            # Fast path for most tokens — only full spectrum near trigger window
            if step <= 2 or (step >= 5 and step <= 8):
                capture_hooks = self._build_full_capture_hooks(hidden_states)
                full = True
            else:
                capture_hooks = self._build_capture_hooks(hidden_states)
                full = False
            all_hooks = capture_hooks + sup_hooks

            with torch.no_grad():
                logits = inner.run_with_hooks(tokens, fwd_hooks=all_hooks)

            # Cognitive state
            state = self._compute_cognitive_state(
                hidden_states, c_delta_traj_raw, step, logits,
                full_spectrum=full,
            )
            cognitive_states.append(state)
            c_delta_traj_raw.append(state.c_delta)

            # Temperature sampling
            logit_vec = logits[0, -1].float()
            probs = F.softmax(logit_vec / self.resample_temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == inner.tokenizer.eos_token_id:
                break

            next_str = inner.tokenizer.decode([next_id])
            generated_tokens.append(next_str)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

        elapsed = time.time() - t_start

        return OversightResult(
            prompt=prompt,
            text="".join(generated_tokens),
            tokens=generated_tokens,
            cognitive_states=cognitive_states,
            anomalies=[],
            escalation_events=[{"action": "resample", "token": 0}],
            max_escalation_level=self.policy.escalation_levels.index("resample")
                if "resample" in self.policy.escalation_levels else 3,
            was_flagged=False,
            confidence=0.7,  # resampled = moderate confidence
            generation_time_s=elapsed,
            total_tokens=len(generated_tokens),
            policy_name=self.policy.name,
        )

    # ── Main Entry Point ─────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        policy: CognitivePolicy = None,
        verbose: bool = False,
    ) -> OversightResult:
        """
        Generate text with full cognitive oversight.

        Token-by-token generation with:
          1. All-axis monitoring (K, C, Cd, S, Gini, spread, early_mass)
          2. Policy-driven anomaly detection
          3. Circuit-level fault diagnosis
          4. Escalating intervention (suppress → steer → resample → abstain)
          5. Post-intervention verification
          6. Calibrated confidence scoring
        """
        active_policy = policy or self.policy
        inner = self._inner
        if inner is None:
            raise ValueError("Inner model not accessible")

        tokens = inner.to_tokens(prompt)
        generated_tokens = []
        cognitive_states = []
        anomalies = []
        escalation_events = []
        c_delta_traj_raw = []

        # Intervention state
        current_level = 0
        active_hooks = []  # persistent hooks from escalation
        triggered = False
        max_esc_level = 0
        was_flagged = False
        last_anomaly = None

        t_start = time.time()
        monitor_time = 0.0
        intervention_time = 0.0

        for step in range(max_tokens):
            # ── Step 1: Forward pass with monitoring + active intervention hooks ──
            # Use fast path (early+late only) unless we need full spectrum for diagnosis
            # Only do full spectrum on trigger check tokens and when actively intervening
            need_full = (
                (step >= active_policy.min_trigger_token and step <= active_policy.min_trigger_token + 2)
                or (triggered and step <= (escalation_events[-1].get("token", 0) + active_policy.verification_window + 1) if escalation_events else False)
            )
            hidden_states = {}
            if need_full:
                capture_hooks = self._build_full_capture_hooks(hidden_states)
            else:
                capture_hooks = self._build_capture_hooks(hidden_states)
            all_hooks = capture_hooks + active_hooks

            t_mon = time.time()
            with torch.no_grad():
                logits = inner.run_with_hooks(tokens, fwd_hooks=all_hooks)

            # ── Step 2: Compute cognitive state (all axes) ──
            state = self._compute_cognitive_state(
                hidden_states, c_delta_traj_raw, step, logits,
                full_spectrum=need_full,
            )
            cognitive_states.append(state)
            c_delta_traj_raw.append(state.c_delta)
            monitor_time += time.time() - t_mon

            # ── Step 3: Anomaly detection (policy-driven) ──
            if not triggered:
                anomaly = self._check_triggers(state, hidden_states)

                if anomaly is not None:
                    triggered = True
                    last_anomaly = anomaly
                    anomalies.append(anomaly)

                    if verbose:
                        print(f"  [ANOMALY] step={step} type={anomaly.anomaly_type} "
                              f"metric={anomaly.trigger_metric}={anomaly.trigger_value:.4f} "
                              f"fault_layers={anomaly.fault_layers}")

                    # ── Step 4: Escalation ──
                    t_int = time.time()
                    new_level, hooks, action = self._escalate(
                        current_level, anomaly, hidden_states, tokens
                    )
                    current_level = new_level + 1  # advance for next escalation
                    max_esc_level = max(max_esc_level, new_level)

                    if action == "suppress" or action == "steer":
                        active_hooks = hooks  # persist for remaining tokens

                        # Re-run this token with intervention hooks
                        with torch.no_grad():
                            logits = inner.run_with_hooks(
                                tokens, fwd_hooks=capture_hooks + hooks
                            )

                        # Re-compute state after intervention
                        state_after = self._compute_cognitive_state(
                            hidden_states, c_delta_traj_raw[:-1], step, logits
                        )

                        escalation_events.append({
                            "token": step,
                            "level": new_level,
                            "action": action,
                            "metric_before": anomaly.trigger_value,
                            "s_before": state.s_early,
                            "s_after": state_after.s_early,
                            "gini_before": state.gini,
                            "gini_after": state_after.gini,
                        })

                        if verbose:
                            print(f"  [{action.upper()}] gini: {state.gini:.3f} → {state_after.gini:.3f} "
                                  f"s_early: {state.s_early:.4f} → {state_after.s_early:.4f}")

                    elif action == "resample":
                        escalation_events.append({
                            "token": step, "level": new_level, "action": "resample"
                        })
                        if verbose:
                            print(f"  [RESAMPLE] triggered at step {step}")

                        intervention_time += time.time() - t_int

                        # Resample the entire generation with suppression hooks
                        return self._resample_with_suppression(
                            prompt, anomaly.fault_features, max_tokens
                        )

                    elif action == "abstain":
                        was_flagged = True
                        escalation_events.append({
                            "token": step, "level": new_level, "action": "abstain"
                        })
                        if verbose:
                            print(f"  [ABSTAIN] flagging output as low-confidence")
                        # Continue generating but mark as flagged

                    intervention_time += time.time() - t_int

            # ── Step 5: Verification (after intervention window) ──
            if (triggered and active_hooks and
                len(escalation_events) > 0 and
                step == escalation_events[-1].get("token", -1) + active_policy.verification_window):

                pre_s = escalation_events[-1].get("s_before", 0)
                if not self._verify_intervention(c_delta_traj_raw, pre_s):
                    # Intervention didn't work — escalate further
                    if current_level < len(active_policy.escalation_levels):
                        if verbose:
                            print(f"  [VERIFY-FAIL] step={step}, escalating to level {current_level}")

                        new_level, hooks, action = self._escalate(
                            current_level, last_anomaly, hidden_states, tokens
                        )
                        current_level = new_level + 1
                        max_esc_level = max(max_esc_level, new_level)

                        if action == "resample":
                            return self._resample_with_suppression(
                                prompt, last_anomaly.fault_features, max_tokens
                            )
                        elif action == "abstain":
                            was_flagged = True
                        elif hooks:
                            active_hooks = hooks

                        escalation_events.append({
                            "token": step, "level": new_level, "action": action,
                            "reason": "verification_failed"
                        })

            # ── Step 6: Decode next token ──
            next_id = logits[0, -1].argmax().item()
            if next_id == inner.tokenizer.eos_token_id:
                break

            next_str = inner.tokenizer.decode([next_id])
            generated_tokens.append(next_str)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

        # ── Build result ──
        elapsed = time.time() - t_start
        text = "".join(generated_tokens)

        # Confidence: 1.0 = clean, lower for interventions, lowest for flagged
        confidence = 1.0
        if was_flagged:
            confidence = 0.2
        elif max_esc_level >= 2:  # resample or higher
            confidence = 0.5
        elif max_esc_level >= 1:  # steer
            confidence = 0.7
        elif triggered:  # suppress only
            confidence = 0.8

        # Overhead = extra time from monitoring + intervention beyond base generation
        # Approximate base generation time as elapsed - monitor - intervention
        base_gen_time = max(elapsed - monitor_time - intervention_time, 0.001)
        overhead = (monitor_time + intervention_time) / base_gen_time * 100

        return OversightResult(
            prompt=prompt,
            text=text,
            tokens=generated_tokens,
            cognitive_states=cognitive_states,
            anomalies=anomalies,
            escalation_events=escalation_events,
            max_escalation_level=max_esc_level,
            was_flagged=was_flagged,
            confidence=confidence,
            generation_time_s=elapsed,
            monitor_time_s=monitor_time,
            intervention_time_s=intervention_time,
            overhead_pct=overhead,
            total_tokens=len(generated_tokens),
            policy_name=active_policy.name,
        )

    # ── Convenience Methods ──────────────────────────────────────────────

    def generate_with_policy(
        self,
        prompt: str,
        policy_name: str,
        max_tokens: int = 50,
        verbose: bool = False,
    ) -> OversightResult:
        """Generate with a named policy from the registry."""
        policy = POLICY_REGISTRY.get(policy_name)
        if policy is None:
            raise ValueError(f"Unknown policy: {policy_name}. "
                           f"Available: {list(POLICY_REGISTRY.keys())}")
        return self.generate(prompt, max_tokens=max_tokens, policy=policy, verbose=verbose)

    def generate_unguarded(
        self,
        prompt: str,
        max_tokens: int = 50,
    ) -> OversightResult:
        """Generate with monitoring but NO intervention (baseline)."""
        inner = self._inner
        tokens = inner.to_tokens(prompt)
        generated_tokens = []
        cognitive_states = []
        c_delta_traj_raw = []

        t_start = time.time()
        monitor_time = 0.0

        for step in range(max_tokens):
            hidden_states = {}
            capture_hooks = self._build_full_capture_hooks(hidden_states)

            t_mon = time.time()
            with torch.no_grad():
                logits = inner.run_with_hooks(tokens, fwd_hooks=capture_hooks)

            state = self._compute_cognitive_state(
                hidden_states, c_delta_traj_raw, step, logits
            )
            cognitive_states.append(state)
            c_delta_traj_raw.append(state.c_delta)
            monitor_time += time.time() - t_mon

            next_id = logits[0, -1].argmax().item()
            if next_id == inner.tokenizer.eos_token_id:
                break

            next_str = inner.tokenizer.decode([next_id])
            generated_tokens.append(next_str)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

        elapsed = time.time() - t_start

        return OversightResult(
            prompt=prompt,
            text="".join(generated_tokens),
            tokens=generated_tokens,
            cognitive_states=cognitive_states,
            generation_time_s=elapsed,
            monitor_time_s=monitor_time,
            overhead_pct=monitor_time / max(elapsed, 0.001) * 100,
            total_tokens=len(generated_tokens),
            policy_name="unguarded",
        )

    def compare(
        self,
        prompt: str,
        max_tokens: int = 50,
        verbose: bool = True,
    ) -> Dict[str, OversightResult]:
        """Run unguarded vs guarded, return both results for comparison."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"PROMPT: {prompt[:80]}...")
            print(f"{'='*60}")

        if verbose:
            print("\n--- UNGUARDED ---")
        unguarded = self.generate_unguarded(prompt, max_tokens)
        if verbose:
            print(f"  Output: {unguarded.text[:100]}...")
            print(f"  Max Gini: {max(unguarded.gini_trajectory) if unguarded.gini_trajectory else 0:.3f}")
            print(f"  Max S_early: {max(unguarded.s_early_trajectory) if unguarded.s_early_trajectory else 0:.4f}")

        if verbose:
            print(f"\n--- GUARDED ({self.policy.name}) ---")
        guarded = self.generate(prompt, max_tokens, verbose=verbose)
        if verbose:
            print(f"  Output: {guarded.text[:100]}...")
            print(f"  Anomalies: {len(guarded.anomalies)}")
            print(f"  Interventions: {len(guarded.escalation_events)}")
            print(f"  Max escalation: {guarded.max_escalation_level}")
            print(f"  Confidence: {guarded.confidence:.2f}")
            print(f"  Flagged: {guarded.was_flagged}")

        return {"unguarded": unguarded, "guarded": guarded}

    # ── Feature Anatomy ──────────────────────────────────────────────────

    def decode_fault_features(
        self,
        anomaly: CognitiveAnomaly,
        top_n: int = 10,
    ) -> Dict[int, List[Dict[str, any]]]:
        """
        Decode what the fault features encode.
        Maps decoder vectors to nearest tokens in vocabulary.

        Returns: {layer: [{feature_id, top_tokens, coherence_with_others}, ...]}
        """
        inner = self._inner
        if inner is None or not hasattr(inner, "tokenizer"):
            return {}

        # Get the model's unembedding matrix
        W_U = None
        for name in ["unembed.W_U", "embed_out.weight", "lm_head.weight"]:
            parts = name.split(".")
            obj = inner
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is not None:
                W_U = obj
                break

        if W_U is None:
            return {}

        result = {}
        for layer_idx, feat_ids in anomaly.fault_features.items():
            W_dec = self._dec_weights.get(layer_idx)
            if W_dec is None:
                continue

            layer_results = []
            for fid in feat_ids[:top_n]:
                if fid >= W_dec.shape[0]:
                    continue

                dec_vec = W_dec[fid].float()
                # Project decoder vector into vocabulary space
                logits = dec_vec @ W_U.float().T if W_U.shape[-1] == dec_vec.shape[0] else dec_vec @ W_U.float()
                top_tokens_ids = logits.topk(5).indices.tolist()
                top_tokens = [inner.tokenizer.decode([tid]).strip() for tid in top_tokens_ids]

                layer_results.append({
                    "feature_id": fid,
                    "top_tokens": top_tokens,
                    "activation_rank": feat_ids.index(fid),
                })

            result[layer_idx] = layer_results

        return result


# ── CLI Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fathom Oversight demo")
    parser.add_argument("--prompt", type=str, default="Q: Is the Great Wall of China visible from space?\nA:")
    parser.add_argument("--policy", type=str, default="hallucination",
                       choices=list(POLICY_REGISTRY.keys()))
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--compare", action="store_true",
                       help="Run unguarded vs guarded comparison")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("[FathomOversight] Initializing...")
    oversight = FathomOversight(
        model_name="google/gemma-2-2b",
        policy=POLICY_REGISTRY[args.policy],
    )

    if args.compare:
        results = oversight.compare(args.prompt, max_tokens=args.max_tokens, verbose=True)
    else:
        result = oversight.generate(
            args.prompt, max_tokens=args.max_tokens,
            policy=POLICY_REGISTRY[args.policy], verbose=args.verbose,
        )
        print(f"\nOutput: {result.text}")
        print(f"Anomalies: {[a.anomaly_type for a in result.anomalies]}")
        print(f"Interventions: {len(result.escalation_events)}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Overhead: {result.overhead_pct:.1f}%")

        # Feature anatomy
        for anomaly in result.anomalies:
            anatomy = oversight.decode_fault_features(anomaly)
            if anatomy:
                print(f"\n[FEATURE ANATOMY] {anomaly.anomaly_type} at token {anomaly.trigger_token}:")
                for layer, features in anatomy.items():
                    print(f"  Layer {layer}:")
                    for feat in features[:5]:
                        print(f"    Feature {feat['feature_id']}: {feat['top_tokens']}")
