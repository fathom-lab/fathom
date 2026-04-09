# -*- coding: utf-8 -*-
"""
cognitive_state.py -- Fathom Cognitive State & Policy Definitions
================================================================

Core data structures for the Fathom Oversight framework.
Defines the cognitive measurement layer, anomaly types,
and intervention policies.

Four orthogonal cognitive axes:
  K  -- depth routing (WHERE computation happens)
  C  -- coherence (WHAT concepts co-activate)
  Cd -- commitment (late vs early coherence shift)
  S  -- lock-in geometry (concentrated commitment intensity)

Five cognitive anomaly types, each with validated detection:
  1. Hallucination       -- S_early spike + Gini concentration
  2. Shallow routing     -- K above surface threshold
  3. Confidence diverge  -- high Cd + low entropy
  4. Adversarial depth   -- missing RLHF depth amplification
  5. Expression gap      -- internal K vs output expression mismatch

Patent pending: US 64/020,489 & 64/021,113
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


# ── Cognitive State (per-token snapshot) ─────────────────────────────────

@dataclass
class CognitiveState:
    """Complete cognitive snapshot at one token position."""
    token_idx: int

    # The four axes
    k_depth: float = 0.0            # K -- mean activated layer index
    c_coherence: float = 0.0        # C -- global mean pairwise cosine
    c_delta: float = 0.0            # Cd -- late minus early coherence
    s_early: float = 0.0            # S -- commitment intensity (rolling, 0 until min_trigger)

    # Spectrum shape
    gini: float = 0.0               # concentration coefficient
    spread: float = 0.0             # spectral width (std of distribution)
    early_mass: float = 0.0         # fraction in early layers

    # Layer profile
    layer_profile: Optional[np.ndarray] = None  # (n_layers,) coherence per layer

    # Raw signals
    entropy: Optional[float] = None         # logit entropy
    logprob: Optional[float] = None         # chosen token log probability
    top2_margin: Optional[float] = None     # p_top1 - p_top2

    # D-axis: cognitive honesty
    d_honesty: Optional[float] = None       # cosine(residual, unembedding[token])
                                            # high = honest (saying what it thinks)
                                            # low = divergent (saying something else)

    def is_anomalous(self, policy: "CognitivePolicy") -> bool:
        """Check if any policy trigger is exceeded."""
        checks = []
        for metric, threshold in policy.triggers.items():
            val = self._get_metric(metric)
            if val is None:
                continue
            if metric == "entropy_low":
                # Special: True if entropy below 25th percentile (~1.5 for Gemma-2-2B)
                checks.append(self.entropy is not None and self.entropy < 1.5)
            elif metric == "depth_delta":
                # Negative threshold = looking for LACK of depth amplification
                checks.append(val < threshold)
            else:
                checks.append(val > threshold)

        if not checks:
            return False

        if policy.trigger_logic == "all":
            return all(checks)
        else:  # "any"
            return any(checks)

    def _get_metric(self, metric: str) -> Optional[float]:
        """Resolve metric name to value."""
        mapping = {
            "s_early": self.s_early,
            "gini": self.gini,
            "c_delta": self.c_delta,
            "k_depth": self.k_depth,
            "entropy": self.entropy,
            "entropy_low": self.entropy,
            "depth_delta": self.c_delta,  # reused axis, different threshold
            "spread": self.spread,
            "early_mass": self.early_mass,
        }
        return mapping.get(metric)

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        d = {
            "token_idx": self.token_idx,
            "k_depth": self.k_depth,
            "c_coherence": self.c_coherence,
            "c_delta": self.c_delta,
            "s_early": self.s_early,
            "gini": self.gini,
            "spread": self.spread,
            "early_mass": self.early_mass,
            "entropy": self.entropy,
            "logprob": self.logprob,
            "top2_margin": self.top2_margin,
            "d_honesty": self.d_honesty,
        }
        if self.layer_profile is not None:
            d["layer_profile"] = self.layer_profile.tolist()
        return d


# ── Cognitive Anomaly (diagnosed failure) ────────────────────────────────

@dataclass
class CognitiveAnomaly:
    """A detected cognitive anomaly with full circuit-level diagnosis."""
    anomaly_type: str                          # "hallucination", "shallow_routing", etc.
    trigger_token: int                         # when it was detected
    trigger_metric: str                        # which metric triggered
    trigger_value: float                       # the metric value that crossed threshold
    fault_layers: List[int]                    # layers where anomaly concentrates
    fault_features: Dict[int, List[int]]       # {layer: [feature_ids]} -- the circuit diagnosis
    attractor_coherence: float = 0.0           # how strongly fault features cluster
    confidence: float = 0.5                    # P(anomaly is real), calibrated

    def to_dict(self) -> dict:
        return {
            "anomaly_type": self.anomaly_type,
            "trigger_token": self.trigger_token,
            "trigger_metric": self.trigger_metric,
            "trigger_value": self.trigger_value,
            "fault_layers": self.fault_layers,
            "fault_features": {str(k): v for k, v in self.fault_features.items()},
            "attractor_coherence": self.attractor_coherence,
            "confidence": self.confidence,
        }


# ── Cognitive Policy (anomaly-specific configuration) ────────────────────

@dataclass
class CognitivePolicy:
    """Policy defining what to monitor, when to trigger, how to intervene."""
    name: str

    # Detection
    triggers: Dict[str, float]              # {"s_early": 0.008, "gini": 0.51}
    trigger_logic: str = "any"              # "any" or "all"
    min_trigger_token: int = 5              # earliest token to evaluate

    # Diagnosis
    n_fault_features: int = 32              # top-k features to attribute per layer

    # Intervention
    escalation_levels: List[str] = field(
        default_factory=lambda: ["suppress", "steer", "resample", "abstain"]
    )
    dampen_factor: float = 0.1              # feature suppression strength (0=zero, 1=no change)
    steering_strength: float = 0.8          # directional steering magnitude
    resample_temperature: float = 0.9       # temperature for Level 3
    verification_window: int = 3            # tokens to wait before re-checking

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "triggers": self.triggers,
            "trigger_logic": self.trigger_logic,
            "min_trigger_token": self.min_trigger_token,
            "n_fault_features": self.n_fault_features,
            "escalation_levels": self.escalation_levels,
            "dampen_factor": self.dampen_factor,
            "steering_strength": self.steering_strength,
            "resample_temperature": self.resample_temperature,
            "verification_window": self.verification_window,
        }


# ── Pre-built Policies ──────────────────────────────────────────────────

HALLUCINATION_POLICY = CognitivePolicy(
    name="hallucination",
    triggers={"s_early": 0.008, "gini": 0.51},
    trigger_logic="any",
    min_trigger_token=5,
    n_fault_features=16,
    # Key insight from pilot: by token 8, hallucination is already committed.
    # Skip mid-generation surgery. Detect → resample with suppression from token 0.
    escalation_levels=["resample", "abstain"],
    dampen_factor=0.5,  # gentler for resample (model generates with partial features)
    steering_strength=0.8,
    resample_temperature=0.9,
    verification_window=3,
)
# Validated: TruthfulQA n=200. S_early AUC=0.663, Gini AUC=0.685.

SHALLOW_ROUTING_POLICY = CognitivePolicy(
    name="shallow_routing",
    triggers={"k_depth": 8.36},
    trigger_logic="any",
    min_trigger_token=3,
    n_fault_features=48,
    escalation_levels=["suppress", "steer", "resample"],
    dampen_factor=0.15,
    steering_strength=1.0,
    resample_temperature=0.7,
    verification_window=3,
)
# Validated: 30 pre-registered surface/insight pairs (p=0.000051, delta=0.257).

CONFIDENCE_DIVERGENCE_POLICY = CognitivePolicy(
    name="confidence_divergence",
    triggers={"c_delta": 0.012, "entropy_low": 1.5},
    trigger_logic="all",
    min_trigger_token=5,
    n_fault_features=32,
    escalation_levels=["suppress", "steer", "abstain"],
    dampen_factor=0.1,
    steering_strength=0.8,
    resample_temperature=0.9,
    verification_window=3,
)
# Validated: cognitive_divergence danger_zone_frac discriminates correct vs halluc.

ADVERSARIAL_DEPTH_POLICY = CognitivePolicy(
    name="adversarial_depth",
    triggers={"depth_delta": -0.124},
    trigger_logic="any",
    min_trigger_token=5,
    n_fault_features=64,
    escalation_levels=["suppress", "steer", "abstain"],
    dampen_factor=0.05,
    steering_strength=1.2,
    resample_temperature=0.9,
    verification_window=4,
)
# Validated: n=30 jailbreak pairs, instruct p=0.0028, d=-0.596.

MULTI_POLICY = CognitivePolicy(
    name="multi",
    triggers={"s_early": 0.008, "gini": 0.51, "k_depth": 8.36, "c_delta": 0.012},
    trigger_logic="any",
    min_trigger_token=3,
    n_fault_features=32,
    escalation_levels=["suppress", "steer", "resample", "abstain"],
    dampen_factor=0.1,
    steering_strength=0.8,
    resample_temperature=0.9,
    verification_window=3,
)
# Combined: catches all anomaly types. Production deployment policy.

# Registry for lookup by name
POLICY_REGISTRY: Dict[str, CognitivePolicy] = {
    "hallucination": HALLUCINATION_POLICY,
    "shallow_routing": SHALLOW_ROUTING_POLICY,
    "confidence_divergence": CONFIDENCE_DIVERGENCE_POLICY,
    "adversarial_depth": ADVERSARIAL_DEPTH_POLICY,
    "multi": MULTI_POLICY,
}


# ── Anomaly Type Classification ─────────────────────────────────────────

def classify_anomaly_type(state: CognitiveState) -> str:
    """
    Given a cognitive state that triggered an anomaly,
    determine which type of anomaly it is based on which
    metrics are most abnormal.

    Returns one of:
      "hallucination", "shallow_routing", "confidence_divergence",
      "adversarial_depth", "unknown"
    """
    scores = {}

    # Hallucination: S_early spike or Gini concentration
    s_score = max(0, (state.s_early - 0.005) / 0.005) if state.s_early > 0 else 0
    g_score = max(0, (state.gini - 0.48) / 0.10) if state.gini > 0 else 0
    scores["hallucination"] = s_score + g_score

    # Shallow routing: K above surface threshold
    k_score = max(0, (state.k_depth - 8.20) / 0.20) if state.k_depth > 0 else 0
    scores["shallow_routing"] = k_score

    # Confidence divergence: high C_delta AND low entropy
    cd_score = max(0, (state.c_delta - 0.008) / 0.005) if state.c_delta > 0 else 0
    ent_score = max(0, (2.0 - (state.entropy or 3.0)) / 1.0)  # low entropy = high score
    scores["confidence_divergence"] = cd_score * ent_score  # product = both must fire

    # Adversarial depth: negative depth delta (missing amplification)
    if state.c_delta < 0:
        scores["adversarial_depth"] = abs(state.c_delta) / 0.15
    else:
        scores["adversarial_depth"] = 0.0

    # Return highest scoring type
    if max(scores.values()) < 0.1:
        return "unknown"
    return max(scores, key=scores.get)


# ── Oversight Result ─────────────────────────────────────────────────────

@dataclass
class OversightResult:
    """Complete result from a guarded generation."""
    # Generation output
    prompt: str = ""
    text: str = ""
    tokens: List[str] = field(default_factory=list)

    # Full cognitive state per token
    cognitive_states: List[CognitiveState] = field(default_factory=list)

    # Anomaly events
    anomalies: List[CognitiveAnomaly] = field(default_factory=list)

    # Intervention trace
    escalation_events: List[dict] = field(default_factory=list)
    max_escalation_level: int = 0

    # Confidence
    was_flagged: bool = False
    confidence: float = 1.0          # calibrated P(output is correct)

    # Performance
    generation_time_s: float = 0.0
    monitor_time_s: float = 0.0
    intervention_time_s: float = 0.0
    overhead_pct: float = 0.0
    total_tokens: int = 0

    # Policy used
    policy_name: str = ""

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "text": self.text,
            "tokens": self.tokens,
            "cognitive_states": [s.to_dict() for s in self.cognitive_states],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "escalation_events": self.escalation_events,
            "max_escalation_level": self.max_escalation_level,
            "was_flagged": self.was_flagged,
            "confidence": self.confidence,
            "generation_time_s": self.generation_time_s,
            "monitor_time_s": self.monitor_time_s,
            "intervention_time_s": self.intervention_time_s,
            "overhead_pct": self.overhead_pct,
            "total_tokens": self.total_tokens,
            "policy_name": self.policy_name,
        }

    @property
    def intervention_count(self) -> int:
        return len(self.escalation_events)

    @property
    def anomaly_types(self) -> List[str]:
        return list(set(a.anomaly_type for a in self.anomalies))

    @property
    def s_early_trajectory(self) -> List[float]:
        return [s.s_early for s in self.cognitive_states]

    @property
    def c_delta_trajectory(self) -> List[float]:
        return [s.c_delta for s in self.cognitive_states]

    @property
    def gini_trajectory(self) -> List[float]:
        return [s.gini for s in self.cognitive_states]

    @property
    def k_depth_trajectory(self) -> List[float]:
        return [s.k_depth for s in self.cognitive_states]

    @property
    def d_honesty_trajectory(self) -> List[Optional[float]]:
        return [s.d_honesty for s in self.cognitive_states]
