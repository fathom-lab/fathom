"""
FATHOM ADVERSARIAL — Depth-Based Prompt Injection / Jailbreak Detector
========================================================================

Prompt injections and jailbreaks produce unusual depth signatures.
Normal prompts have a characteristic cognitive fingerprint — adversarial
prompts deviate from it in measurable ways.

This is a security product: detect manipulation before it reaches output.

Usage:
    from fathom_adversarial import AdversarialDetector

    detector = AdversarialDetector()
    detector.build_baseline(normal_prompts, scorer)

    result = detector.is_adversarial("Ignore previous instructions and...", scorer)
    # {"is_adversarial": True, "anomaly_score": 3.2, "flags": ["depth_high", "entropy_low"]}
"""

import gc
import json
import math
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BaselineProfile:
    """Statistical baseline of normal prompt depth signatures."""
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
    timestamp: str

    def to_dict(self):
        return asdict(self)


@dataclass
class AdversarialResult:
    """Result of adversarial detection on a single prompt."""
    prompt: str
    is_adversarial: bool
    anomaly_score: float
    flags: List[str]
    depth_score: float
    entropy: float
    z_scores: Dict[str, float]

    def to_dict(self):
        return asdict(self)


class AdversarialDetector:
    """Detects adversarial prompts via depth signature anomalies.

    Builds a baseline from normal prompts, then flags prompts whose
    depth signatures deviate by >3 standard deviations on any dimension.
    """

    def __init__(self, z_threshold: float = 3.0):
        self._baseline: Optional[BaselineProfile] = None
        self._z_threshold = z_threshold

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> Optional[BaselineProfile]:
        return self._baseline

    def build_baseline(
        self,
        normal_prompts: List[str],
        scorer,
        min_samples: int = 10,
    ) -> BaselineProfile:
        """Score normal prompts and compute the baseline cognitive fingerprint.

        Args:
            normal_prompts: List of known-normal prompts (50+ recommended).
            scorer: Initialized DepthScorer.
            min_samples: Minimum valid samples needed.

        Returns:
            BaselineProfile with mean + std for each dimension.
        """
        depths = []
        entropies = []
        early_ratios = []
        late_ratios = []
        weighted_depths = []

        for i, prompt in enumerate(normal_prompts):
            try:
                result = scorer.score(prompt)
                if math.isnan(result.depth_score):
                    continue

                depths.append(result.depth_score)
                entropies.append(result.layer_entropy or 0.0)
                early_ratios.append(result.early_ratio)
                late_ratios.append(result.late_ratio)
                weighted_depths.append(
                    result.weighted_depth_score
                    if result.weighted_depth_score is not None
                    else result.depth_score
                )

                if (i + 1) % 10 == 0:
                    print(f"[Adversarial] Baseline: {i+1}/{len(normal_prompts)} scored")

            except Exception as e:
                print(f"[Adversarial] Baseline sample {i} failed: {e}")
                continue

            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

        if len(depths) < min_samples:
            raise ValueError(
                f"Need >= {min_samples} valid samples, got {len(depths)}"
            )

        self._baseline = BaselineProfile(
            n_samples=len(depths),
            depth_mean=float(np.mean(depths)),
            depth_std=float(np.std(depths, ddof=1)),
            entropy_mean=float(np.mean(entropies)),
            entropy_std=float(np.std(entropies, ddof=1)),
            early_ratio_mean=float(np.mean(early_ratios)),
            early_ratio_std=float(np.std(early_ratios, ddof=1)),
            late_ratio_mean=float(np.mean(late_ratios)),
            late_ratio_std=float(np.std(late_ratios, ddof=1)),
            weighted_depth_mean=float(np.mean(weighted_depths)),
            weighted_depth_std=float(np.std(weighted_depths, ddof=1)),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        print(f"[Adversarial] Baseline built: n={len(depths)}, "
              f"depth={self._baseline.depth_mean:.3f}±{self._baseline.depth_std:.3f}, "
              f"entropy={self._baseline.entropy_mean:.3f}±{self._baseline.entropy_std:.3f}")

        return self._baseline

    def is_adversarial(self, prompt: str, scorer) -> AdversarialResult:
        """Check if a prompt's depth signature is anomalous.

        Args:
            prompt: Text to check.
            scorer: Initialized DepthScorer.

        Returns:
            AdversarialResult with detection flags and anomaly score.
        """
        if self._baseline is None:
            raise ValueError("No baseline. Call build_baseline() first.")

        result = scorer.score(prompt)

        if math.isnan(result.depth_score):
            return AdversarialResult(
                prompt=prompt, is_adversarial=True,
                anomaly_score=float("inf"),
                flags=["no_attribution_output"],
                depth_score=float("nan"), entropy=0.0, z_scores={},
            )

        entropy = result.layer_entropy or 0.0
        weighted = result.weighted_depth_score or result.depth_score

        # Compute z-scores
        z_scores = {}
        flags = []

        def _z(val, mean, std, name):
            if std < 1e-8:
                z = 0.0
            else:
                z = (val - mean) / std
            z_scores[name] = round(z, 3)
            return z

        z_depth = _z(result.depth_score, self._baseline.depth_mean,
                      self._baseline.depth_std, "depth")
        z_entropy = _z(entropy, self._baseline.entropy_mean,
                        self._baseline.entropy_std, "entropy")
        z_early = _z(result.early_ratio, self._baseline.early_ratio_mean,
                      self._baseline.early_ratio_std, "early_ratio")
        z_late = _z(result.late_ratio, self._baseline.late_ratio_mean,
                     self._baseline.late_ratio_std, "late_ratio")
        z_weighted = _z(weighted, self._baseline.weighted_depth_mean,
                         self._baseline.weighted_depth_std, "weighted_depth")

        # Flag anomalies
        if abs(z_depth) > self._z_threshold:
            flags.append("depth_high" if z_depth > 0 else "depth_low")
        if abs(z_entropy) > self._z_threshold:
            flags.append("entropy_high" if z_entropy > 0 else "entropy_low")
        if abs(z_early) > self._z_threshold:
            flags.append("early_ratio_anomaly")
        if abs(z_late) > self._z_threshold:
            flags.append("late_ratio_anomaly")
        if abs(z_weighted) > self._z_threshold:
            flags.append("weighted_depth_anomaly")

        # Composite anomaly score: max absolute z-score across dimensions
        anomaly_score = max(abs(z_depth), abs(z_entropy), abs(z_early),
                           abs(z_late), abs(z_weighted))

        is_adv = len(flags) > 0

        return AdversarialResult(
            prompt=prompt,
            is_adversarial=is_adv,
            anomaly_score=round(anomaly_score, 4),
            flags=flags,
            depth_score=round(result.depth_score, 4),
            entropy=round(entropy, 4),
            z_scores=z_scores,
        )

    def adversarial_report(self, prompt: str, scorer) -> str:
        """Full human-readable report on a prompt's adversarial assessment.

        Args:
            prompt: Text to analyze.
            scorer: Initialized DepthScorer.

        Returns:
            Formatted report string.
        """
        result = self.is_adversarial(prompt, scorer)
        b = self._baseline

        lines = [
            "=" * 60,
            "FATHOM ADVERSARIAL DETECTION REPORT",
            "=" * 60,
            f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
            "",
            f"VERDICT: {'*** ADVERSARIAL ***' if result.is_adversarial else 'NORMAL'}",
            f"Anomaly score: {result.anomaly_score:.2f} (threshold: {self._z_threshold:.1f})",
            f"Flags: {', '.join(result.flags) if result.flags else 'none'}",
            "",
            "Dimension Analysis:",
            f"  depth_score:    {result.depth_score:.4f}  "
            f"(baseline: {b.depth_mean:.4f} ± {b.depth_std:.4f})  "
            f"z={result.z_scores.get('depth', 0):.2f}",
            f"  entropy:        {result.entropy:.4f}  "
            f"(baseline: {b.entropy_mean:.4f} ± {b.entropy_std:.4f})  "
            f"z={result.z_scores.get('entropy', 0):.2f}",
            f"  early_ratio:    z={result.z_scores.get('early_ratio', 0):.2f}",
            f"  late_ratio:     z={result.z_scores.get('late_ratio', 0):.2f}",
            f"  weighted_depth: z={result.z_scores.get('weighted_depth', 0):.2f}",
            "",
            f"Baseline: {b.n_samples} normal samples, built {b.timestamp}",
            "=" * 60,
        ]

        return "\n".join(lines)

    def save(self, path: str):
        """Pickle the detector with its baseline."""
        with open(path, "wb") as f:
            pickle.dump({
                "baseline": self._baseline,
                "z_threshold": self._z_threshold,
            }, f)

    @classmethod
    def load(cls, path: str) -> "AdversarialDetector":
        """Load a pickled detector."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        det = cls(z_threshold=state.get("z_threshold", 3.0))
        det._baseline = state["baseline"]
        return det

    def status(self) -> dict:
        """Return detector status for API."""
        if self._baseline is None:
            return {"has_baseline": False, "z_threshold": self._z_threshold}
        return {
            "has_baseline": True,
            "baseline_samples": self._baseline.n_samples,
            "baseline_depth_mean": self._baseline.depth_mean,
            "baseline_entropy_mean": self._baseline.entropy_mean,
            "z_threshold": self._z_threshold,
            "built": self._baseline.timestamp,
        }
