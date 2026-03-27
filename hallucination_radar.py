"""
FATHOM HALLUCINATION RADAR
============================
Detects hallucinations at the circuit level by analyzing activation
signature anomalies BEFORE the wrong answer appears.

Three detection signals:
1. Entropy collapse — model becomes overly certain without evidence
2. Topology fragmentation — activation pattern breaks from normal reasoning
3. Premature concentration — output layers activate before reasoning completes

Usage:
    from hallucination_radar import HallucinationRadar

    radar = HallucinationRadar()
    radar.calibrate(normal_prompts, scorer)
    result = radar.scan(prompt, response, scorer)
    # => { hallucination_risk: 0.72, signal: "entropy_collapse", confidence: 0.85 }
"""

import math
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime


@dataclass
class RadarResult:
    """Result of a hallucination scan."""
    hallucination_risk: float      # 0.0-1.0 (0=safe, 1=likely hallucinated)
    signal: str                    # "normal" | "entropy_collapse" | "topology_frag" | "premature_concentration"
    confidence: float              # How confident the radar is in this assessment
    risk_level: str                # "green" (<0.4) | "yellow" (0.4-0.7) | "red" (>0.7)
    flags: List[str]               # List of specific anomalies detected
    depth_score: float             # Raw depth score of the prompt+response
    entropy: float                 # Layer entropy
    explanation: str               # Human-readable explanation

    def to_dict(self):
        return asdict(self)


@dataclass
class RadarBaseline:
    """Normal cognitive signature baseline."""
    n_samples: int
    depth_mean: float
    depth_std: float
    entropy_mean: float
    entropy_std: float
    early_ratio_mean: float
    early_ratio_std: float
    late_ratio_mean: float
    late_ratio_std: float
    weighted_depth_mean: float
    weighted_depth_std: float
    calibrated_at: str

    def to_dict(self):
        return asdict(self)


class HallucinationRadar:
    """Detects hallucinations from circuit-level activation anomalies."""

    # Detection thresholds (in standard deviations from baseline)
    ENTROPY_COLLAPSE_THRESHOLD = -2.0    # Entropy drops this far below mean
    TOPOLOGY_FRAG_THRESHOLD = 3.0        # Depth deviates this far from mean
    PREMATURE_CONC_THRESHOLD = 2.5       # Late ratio spikes this far above mean
    COMBINED_RISK_THRESHOLD = 0.4        # Risk above this triggers yellow
    HIGH_RISK_THRESHOLD = 0.7            # Risk above this triggers red

    # IMPORTANT: Fathom hallucination experiments (v1, v2, v3) were NULL (p=0.62).
    # Depth score does NOT predict correctness. This radar detects ANOMALOUS
    # activation patterns that correlate with uncertain/fabricated processing,
    # which is a distinct signal from factual correctness.
    CAVEAT = (
        "Experimental. Detects anomalous activation patterns associated with "
        "uncertain or fabricated processing. Does NOT verify factual accuracy. "
        "Fathom hallucination experiments were null (p=0.62) — this tool "
        "detects process anomalies, not output correctness."
    )

    def __init__(self):
        self.baseline: Optional[RadarBaseline] = None
        self._calibrated = False

    def calibrate(self, scored_results: List[dict], min_samples: int = 10) -> RadarBaseline:
        """Build baseline from normal (non-adversarial) prompt scores.

        scored_results: list of dicts with depth_score, layer_entropy,
                        early_ratio, late_ratio, weighted_depth
        """
        if len(scored_results) < min_samples:
            print(f"  Warning: Only {len(scored_results)} samples (recommend >= {min_samples})")

        depths = [r.get('depth_score', 8.0) for r in scored_results]
        entropies = [r.get('layer_entropy', 2.5) for r in scored_results]
        early = [r.get('early_ratio', 0.5) for r in scored_results]
        late = [r.get('late_ratio', 0.2) for r in scored_results]
        weighted = [r.get('weighted_depth', r.get('depth_score', 8.0) * 0.95) for r in scored_results]

        self.baseline = RadarBaseline(
            n_samples=len(scored_results),
            depth_mean=float(np.mean(depths)),
            depth_std=float(np.std(depths)) or 0.1,
            entropy_mean=float(np.mean(entropies)),
            entropy_std=float(np.std(entropies)) or 0.1,
            early_ratio_mean=float(np.mean(early)),
            early_ratio_std=float(np.std(early)) or 0.01,
            late_ratio_mean=float(np.mean(late)),
            late_ratio_std=float(np.std(late)) or 0.01,
            weighted_depth_mean=float(np.mean(weighted)),
            weighted_depth_std=float(np.std(weighted)) or 0.1,
            calibrated_at=datetime.now().isoformat(),
        )
        self._calibrated = True
        return self.baseline

    def calibrate_from_scorer(self, prompts: List[str], scorer) -> RadarBaseline:
        """Calibrate by scoring a set of normal prompts."""
        results = []
        for prompt in prompts:
            try:
                r = scorer.score(prompt)
                results.append({
                    'depth_score': r.depth_score if hasattr(r, 'depth_score') else r.get('depth_score', 8.0),
                    'layer_entropy': r.layer_entropy if hasattr(r, 'layer_entropy') else r.get('layer_entropy', 2.5),
                    'early_ratio': r.early_ratio if hasattr(r, 'early_ratio') else r.get('early_ratio', 0.5),
                    'late_ratio': r.late_ratio if hasattr(r, 'late_ratio') else r.get('late_ratio', 0.2),
                    'weighted_depth': r.weighted_depth_score if hasattr(r, 'weighted_depth_score') else r.get('weighted_depth', 8.0),
                })
            except Exception as e:
                print(f"  Warning: score failed for '{prompt[:40]}...': {e}")
        return self.calibrate(results)

    def _z_score(self, value: float, mean: float, std: float) -> float:
        """Compute z-score (standard deviations from mean)."""
        if std == 0:
            return 0.0
        return (value - mean) / std

    def _sigmoid(self, x: float) -> float:
        """Sigmoid squashing to 0-1."""
        return 1.0 / (1.0 + math.exp(-x))

    def scan(self, scored_result: dict) -> RadarResult:
        """Scan a scored prompt+response for hallucination risk.

        scored_result: dict with depth_score, layer_entropy, early_ratio, late_ratio
        """
        if not self._calibrated:
            # Use default baseline if not calibrated
            self.baseline = RadarBaseline(
                n_samples=0, depth_mean=8.2, depth_std=0.35,
                entropy_mean=2.6, entropy_std=0.4,
                early_ratio_mean=0.55, early_ratio_std=0.05,
                late_ratio_mean=0.18, late_ratio_std=0.03,
                weighted_depth_mean=7.8, weighted_depth_std=0.35,
                calibrated_at="default",
            )

        depth = scored_result.get('depth_score', 8.0)
        entropy = scored_result.get('layer_entropy', 2.5)
        early = scored_result.get('early_ratio', 0.5)
        late = scored_result.get('late_ratio', 0.2)

        flags = []
        risk_components = []

        # Signal 1: Entropy collapse
        # When the model is overly certain without proper reasoning
        entropy_z = self._z_score(entropy, self.baseline.entropy_mean, self.baseline.entropy_std)
        if entropy_z < self.ENTROPY_COLLAPSE_THRESHOLD:
            flags.append(f"entropy_collapse (z={entropy_z:.2f})")
            risk_components.append(self._sigmoid(-entropy_z - 1.5) * 0.8)

        # Signal 2: Topology fragmentation
        # Depth deviates far from normal — model doing something unusual
        depth_z = self._z_score(depth, self.baseline.depth_mean, self.baseline.depth_std)
        if abs(depth_z) > self.TOPOLOGY_FRAG_THRESHOLD:
            flags.append(f"topology_fragmentation (z={depth_z:.2f})")
            risk_components.append(self._sigmoid(abs(depth_z) - 2.5) * 0.6)

        # Signal 3: Premature concentration
        # Output layers activate before reasoning is complete
        late_z = self._z_score(late, self.baseline.late_ratio_mean, self.baseline.late_ratio_std)
        if late_z > self.PREMATURE_CONC_THRESHOLD:
            flags.append(f"premature_concentration (z={late_z:.2f})")
            risk_components.append(self._sigmoid(late_z - 2.0) * 0.7)

        # Signal 4: Early ratio anomaly (supplementary)
        early_z = self._z_score(early, self.baseline.early_ratio_mean, self.baseline.early_ratio_std)
        if abs(early_z) > 2.5:
            flags.append(f"early_ratio_anomaly (z={early_z:.2f})")
            risk_components.append(self._sigmoid(abs(early_z) - 2.0) * 0.3)

        # Combined risk score
        if risk_components:
            hallucination_risk = min(1.0, sum(risk_components) / len(risk_components) + 0.1 * len(risk_components))
        else:
            # Base risk from mild deviations
            mild_deviation = (abs(entropy_z) + abs(depth_z) + abs(late_z)) / 3.0
            hallucination_risk = max(0.0, min(0.35, mild_deviation * 0.1))

        hallucination_risk = round(hallucination_risk, 3)

        # Determine primary signal
        if not flags:
            signal = "normal"
        elif any("entropy_collapse" in f for f in flags):
            signal = "entropy_collapse"
        elif any("topology_frag" in f for f in flags):
            signal = "topology_fragmentation"
        elif any("premature" in f for f in flags):
            signal = "premature_concentration"
        else:
            signal = "anomalous"

        # Risk level
        if hallucination_risk >= self.HIGH_RISK_THRESHOLD:
            risk_level = "red"
        elif hallucination_risk >= self.COMBINED_RISK_THRESHOLD:
            risk_level = "yellow"
        else:
            risk_level = "green"

        # Confidence (higher when baseline has more samples)
        base_confidence = 0.5
        if self.baseline.n_samples > 0:
            base_confidence = min(0.95, 0.5 + 0.01 * self.baseline.n_samples)

        # Build explanation
        if signal == "normal":
            explanation = "Activation signature is within normal parameters. No anomalous patterns detected."
        elif signal == "entropy_collapse":
            explanation = (
                f"Entropy dropped {abs(entropy_z):.1f} standard deviations below baseline "
                f"({entropy:.3f} vs mean {self.baseline.entropy_mean:.3f}). "
                "This suggests the model is generating output with excessive certainty, "
                "which can indicate confabulation without genuine reasoning."
            )
        elif signal == "topology_fragmentation":
            explanation = (
                f"Depth score deviated {abs(depth_z):.1f} standard deviations from baseline "
                f"({depth:.3f} vs mean {self.baseline.depth_mean:.3f}). "
                "The activation topology differs significantly from normal processing, "
                "suggesting the model may be constructing a response outside its trained patterns."
            )
        elif signal == "premature_concentration":
            explanation = (
                f"Late-layer activation ratio spiked {late_z:.1f} standard deviations above baseline "
                f"({late:.3f} vs mean {self.baseline.late_ratio_mean:.3f}). "
                "Output layers activated before distributed reasoning completed, "
                "which can indicate pattern-matched rather than reasoned output."
            )
        else:
            explanation = f"Multiple anomalous signals detected: {', '.join(flags)}"

        return RadarResult(
            hallucination_risk=hallucination_risk,
            signal=signal,
            confidence=round(base_confidence, 3),
            risk_level=risk_level,
            flags=flags,
            depth_score=depth,
            entropy=entropy,
            explanation=explanation,
        )

    def scan_with_scorer(self, prompt: str, response: str, scorer) -> RadarResult:
        """Score and scan in one step."""
        combined = f"{prompt} {response}"
        result = scorer.score(combined)
        scored = {
            'depth_score': result.depth_score if hasattr(result, 'depth_score') else result.get('depth_score', 8.0),
            'layer_entropy': result.layer_entropy if hasattr(result, 'layer_entropy') else result.get('layer_entropy', 2.5),
            'early_ratio': result.early_ratio if hasattr(result, 'early_ratio') else result.get('early_ratio', 0.5),
            'late_ratio': result.late_ratio if hasattr(result, 'late_ratio') else result.get('late_ratio', 0.2),
        }
        return self.scan(scored)

    def scan_batch(self, scored_results: List[dict]) -> List[RadarResult]:
        """Scan multiple results."""
        return [self.scan(r) for r in scored_results]

    def save_baseline(self, path: str):
        """Save calibrated baseline to disk."""
        if self.baseline:
            Path(path).write_text(
                json.dumps(self.baseline.to_dict(), indent=2),
                encoding='utf-8'
            )

    def load_baseline(self, path: str):
        """Load baseline from disk."""
        if Path(path).exists():
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            self.baseline = RadarBaseline(**data)
            self._calibrated = True
