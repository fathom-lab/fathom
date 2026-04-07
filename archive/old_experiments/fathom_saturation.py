"""
FATHOM SATURATION — Cognitive Complexity Ceiling Detector
==========================================================

Every model has a complexity threshold. Beyond it, depth stops
increasing — the model saturates. This is the first tool to detect
when you've exceeded a model's reasoning capacity.

A model at saturation is doing its best but can't go deeper.

Usage:
    from fathom_saturation import SaturationAnalyzer

    analyzer = SaturationAnalyzer()
    point = analyzer.find_saturation_point("base prompt", scorer)
    print(point)  # {"saturation_level": 4, "max_depth": 7.82, ...}

    profile = analyzer.saturation_profile(scorer)
    # Full model saturation curve across 20 prompts
"""

import gc
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# Standard complexity escalation templates
COMPLEXITY_TEMPLATES = [
    # Level 0: Simple factual recall
    "{base}",
    # Level 1: Add a condition
    "Given that conditions vary, {base}",
    # Level 2: Add causal chain
    "Considering the underlying causes and their effects, {base}",
    # Level 3: Multi-hop reasoning
    "Taking into account the primary factors, their secondary effects, and the "
    "resulting feedback loops, {base}",
    # Level 4: Nested conditionals
    "If the initial conditions hold, and assuming that secondary factors interact "
    "with tertiary constraints in the way typically observed, then considering "
    "the combined impact, {base}",
    # Level 5: Abstract meta-reasoning
    "Analyzing not just the direct answer but why the reasoning process itself "
    "leads to that conclusion, and whether the logical structure would hold under "
    "alternative premises where the key assumptions are inverted, {base}",
    # Level 6: Recursive depth
    "First establish the foundational principles, then examine how those principles "
    "generate second-order effects that themselves create third-order consequences "
    "which feed back into the original principles, requiring a recursive analysis of "
    "the entire system to determine, {base}",
    # Level 7: Maximum complexity
    "Consider a system where the answer depends on understanding not just the "
    "components but the emergent properties that arise from their interaction, "
    "where those emergent properties themselves generate new interactions that "
    "recursively modify the system's behavior in ways that can only be predicted "
    "by reasoning about the meta-level dynamics of the system as a whole, then {base}",
    # Level 8: Philosophical depth
    "Before answering, examine the epistemological framework that makes the question "
    "meaningful, consider whether the concepts involved have stable referents across "
    "different theoretical contexts, evaluate the implicit assumptions, and then "
    "construct an answer that accounts for the fundamental uncertainty inherent in "
    "reasoning about complex adaptive systems where {base}",
    # Level 9: Maximum overload
    "Engaging with the full depth of this question requires simultaneously holding "
    "multiple incompatible theoretical frameworks, reasoning about their points of "
    "convergence and divergence, evaluating which framework best captures the causal "
    "structure of the domain, considering how observer effects modify the system being "
    "analyzed, and then synthesizing a response that transcends any single framework "
    "to address {base}",
]


# Standard baseline prompts for saturation profiling
PROFILE_PROMPTS = [
    "What is the capital of France?",
    "Why do economies experience inflation?",
    "How does photosynthesis convert sunlight to energy?",
    "Why is the sky blue during the day?",
    "What causes ocean tides to change?",
    "How do antibiotics fight bacterial infections?",
    "Why do some metals rust faster than others?",
    "What drives plate tectonics on Earth?",
    "How does natural selection lead to adaptation?",
    "Why do languages change over time?",
    "What causes the seasons to change?",
    "How does memory consolidation work during sleep?",
    "Why do markets experience boom and bust cycles?",
    "What makes some materials better conductors?",
    "How do ecosystems maintain stability?",
    "Why do stars eventually die?",
    "What causes political polarization?",
    "How does the immune system distinguish self from non-self?",
    "Why do complex systems exhibit emergent behavior?",
    "What drives the evolution of cooperation?",
]


@dataclass
class SaturationPoint:
    """Result of saturation analysis for a single base prompt."""
    base_prompt: str
    saturation_level: int  # complexity level where depth plateaus
    max_depth_score: float  # deepest score achieved
    depth_at_saturation: float  # score at saturation point
    depth_trajectory: List[float]  # depth at each complexity level
    plateau_detected: bool  # True if a clear plateau was found
    compute_time_s: float

    def to_dict(self):
        return asdict(self)


@dataclass
class SaturationProfile:
    """Full saturation profile for a model."""
    model_name: str
    n_prompts: int
    mean_saturation_level: float
    std_saturation_level: float
    mean_max_depth: float
    pct_saturated: float  # fraction of prompts that hit plateau
    per_prompt: List[dict]
    mean_curve: List[float]  # average depth at each complexity level
    timestamp: str
    compute_time_s: float

    def to_dict(self):
        return asdict(self)


SATURATION_INSIGHT = (
    "A model at saturation is doing its best but can't go deeper. "
    "This is the first tool to detect when you've exceeded a model's "
    "reasoning capacity."
)


class SaturationAnalyzer:
    """Detects cognitive complexity ceilings in language models."""

    def __init__(self):
        pass

    def find_saturation_point(
        self,
        base_prompt: str,
        scorer,
        n_steps: int = 10,
        plateau_threshold: float = 0.02,
    ) -> SaturationPoint:
        """Progressively increase prompt complexity and find the saturation point.

        Args:
            base_prompt: Core question to escalate.
            scorer: Initialized DepthScorer.
            n_steps: Number of complexity levels to test (max 10).
            plateau_threshold: Min depth change to count as still increasing.

        Returns:
            SaturationPoint with the detected ceiling.
        """
        t0 = time.time()
        n_steps = min(n_steps, len(COMPLEXITY_TEMPLATES))
        depths = []

        for level in range(n_steps):
            prompt = COMPLEXITY_TEMPLATES[level].format(base=base_prompt)

            try:
                result = scorer.score(prompt)
                ds = result.depth_score if not math.isnan(result.depth_score) else 0.0
                depths.append(ds)
                print(f"  [Saturation] Level {level}: depth={ds:.4f} "
                      f"prompt_len={len(prompt)}")
            except Exception as e:
                print(f"  [Saturation] Level {level} failed: {e}")
                depths.append(depths[-1] if depths else 0.0)

            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

        # Detect plateau: find where depth stops decreasing significantly
        # (lower depth = deeper reasoning, so we look for where it stops dropping)
        saturation_level = len(depths) - 1
        plateau_detected = False

        if len(depths) >= 3:
            for i in range(2, len(depths)):
                # Check if depth has plateaued (stopped decreasing)
                recent_change = abs(depths[i] - depths[i - 1])
                prev_change = abs(depths[i - 1] - depths[i - 2]) if i >= 2 else 1.0

                if recent_change < plateau_threshold and prev_change < plateau_threshold:
                    saturation_level = i - 1
                    plateau_detected = True
                    break

        max_depth = min(depths) if depths else 0.0  # lower = deeper
        depth_at_sat = depths[saturation_level] if depths else 0.0

        return SaturationPoint(
            base_prompt=base_prompt,
            saturation_level=saturation_level,
            max_depth_score=round(max_depth, 4),
            depth_at_saturation=round(depth_at_sat, 4),
            depth_trajectory=[round(d, 4) for d in depths],
            plateau_detected=plateau_detected,
            compute_time_s=round(time.time() - t0, 2),
        )

    def saturation_profile(
        self,
        scorer,
        prompts: Optional[List[str]] = None,
        n_steps: int = 8,
    ) -> SaturationProfile:
        """Build a full saturation profile for the model.

        Args:
            scorer: Initialized DepthScorer.
            prompts: Custom prompts, or uses PROFILE_PROMPTS[:20].
            n_steps: Complexity levels per prompt.

        Returns:
            SaturationProfile with aggregate and per-prompt results.
        """
        t0 = time.time()
        prompts = prompts or PROFILE_PROMPTS[:20]

        per_prompt = []
        all_trajectories = []
        sat_levels = []

        for i, prompt in enumerate(prompts):
            print(f"[Saturation] Profile {i+1}/{len(prompts)}: {prompt[:50]}...")
            sp = self.find_saturation_point(prompt, scorer, n_steps=n_steps)
            per_prompt.append(sp.to_dict())
            all_trajectories.append(sp.depth_trajectory)
            sat_levels.append(sp.saturation_level)

        # Compute mean curve (pad shorter trajectories)
        max_len = max(len(t) for t in all_trajectories) if all_trajectories else 0
        padded = []
        for t in all_trajectories:
            padded.append(t + [t[-1]] * (max_len - len(t)) if t else [0.0] * max_len)
        mean_curve = [round(float(np.mean([p[i] for p in padded])), 4) for i in range(max_len)]

        n_saturated = sum(1 for sp in per_prompt if sp["plateau_detected"])

        return SaturationProfile(
            model_name=scorer.model_name,
            n_prompts=len(prompts),
            mean_saturation_level=round(float(np.mean(sat_levels)), 2),
            std_saturation_level=round(float(np.std(sat_levels)), 2),
            mean_max_depth=round(float(np.mean([sp["max_depth_score"] for sp in per_prompt])), 4),
            pct_saturated=round(n_saturated / len(prompts), 3) if prompts else 0.0,
            per_prompt=per_prompt,
            mean_curve=mean_curve,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            compute_time_s=round(time.time() - t0, 2),
        )

    def is_saturated(
        self,
        prompt: str,
        depth_result,
        profile: Optional[SaturationProfile] = None,
    ) -> dict:
        """Predict whether the model hit its cognitive ceiling on this prompt.

        Args:
            prompt: The prompt text.
            depth_result: DepthResult from scorer.score().
            profile: Optional SaturationProfile for calibrated threshold.

        Returns:
            {"is_saturated": bool, "headroom": float, "reasoning": str}
        """
        ds = depth_result.depth_score
        if math.isnan(ds):
            return {
                "is_saturated": False,
                "headroom": 0.0,
                "reasoning": "No depth data — cannot assess saturation.",
            }

        # Use profile's mean max depth if available, else use calibration
        if profile is not None:
            floor = profile.mean_max_depth
        else:
            # Default: gemma-2-2b typical floor ~7.8
            floor = 7.8

        headroom = ds - floor  # positive = still has room, negative = past floor

        # Saturated if depth is within 0.05 of the floor
        is_sat = headroom <= 0.05

        if is_sat:
            reasoning = (
                f"Model appears saturated: depth_score={ds:.4f} is at or below "
                f"the estimated floor ({floor:.4f}). The model may not be able to "
                f"reason more deeply about this prompt."
            )
        else:
            reasoning = (
                f"Model has headroom: depth_score={ds:.4f} is {headroom:.4f} above "
                f"the estimated floor ({floor:.4f}). Deeper processing is possible."
            )

        return {
            "is_saturated": is_sat,
            "headroom": round(headroom, 4),
            "depth_score": round(ds, 4),
            "floor": round(floor, 4),
            "reasoning": reasoning,
        }
