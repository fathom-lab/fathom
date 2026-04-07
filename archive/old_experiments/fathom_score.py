"""
FATHOM SCORE — Single-Number Reasoning Quality Metric
======================================================

Like a credit score for reasoning depth. Combines depth_score, layer_entropy,
weighted_depth_score, and anomaly status into one 0-100 number with a letter
grade. Designed for instant human readability: "This prompt is an A (78/100)."
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class FathomScore:
    """Single-number reasoning quality assessment."""
    score: float            # 0-100 composite score
    grade: str              # F/D/C/B/A/S
    components: Dict        # breakdown of what contributed
    interpretation: str     # one-sentence human summary

    def to_dict(self):
        return asdict(self)


# ── Grade boundaries ─────────────────────────────────────────────────
# S >= 85, A >= 70, B >= 55, C >= 40, D >= 25, F < 25

def _grade(score: float) -> str:
    if score >= 85:
        return "S"
    elif score >= 70:
        return "A"
    elif score >= 55:
        return "B"
    elif score >= 40:
        return "C"
    elif score >= 25:
        return "D"
    else:
        return "F"


def compute_fathom_score(depth_result) -> FathomScore:
    """Compute the Fathom Score from a DepthResult.

    Formula (weighted composite of normalized components):
      - normalized_depth (40%): lower depth_score = deeper = better
      - entropy_component (30%): higher entropy = more distributed = better
      - weighted_depth_component (20%): weighted attribution depth
      - anomaly_penalty (10%): deducted if result is anomalous

    Args:
        depth_result: A DepthResult from DepthScorer.

    Returns:
        FathomScore with 0-100 score, grade, and interpretation.
    """
    n_layers = depth_result.n_layers if depth_result.n_layers > 0 else 26

    # ── Component 1: Depth score (40%) ───────────────────────────────
    # depth_score is mean layer index. Lower = deeper reasoning.
    # Normalize: 0 (deepest) -> 100, n_layers (shallowest) -> 0
    raw_depth = depth_result.depth_score
    if math.isnan(raw_depth):
        depth_norm = 0.0
    else:
        depth_norm = max(0.0, min(100.0, (1.0 - raw_depth / n_layers) * 100.0))

    # ── Component 2: Layer entropy (30%) ─────────────────────────────
    # Higher entropy = more distributed activation = deeper reasoning
    entropy = depth_result.layer_entropy if depth_result.layer_entropy is not None else 0.0
    max_entropy = math.log2(n_layers) if n_layers > 1 else 1.0
    entropy_norm = max(0.0, min(100.0, (entropy / max_entropy) * 100.0)) if max_entropy > 0 else 0.0

    # ── Component 3: Weighted depth (20%) ────────────────────────────
    # Same normalization as depth but using attribution-weighted version
    w_depth = depth_result.weighted_depth_score
    if w_depth is None or math.isnan(w_depth):
        w_depth_norm = depth_norm  # fall back to unweighted
    else:
        w_depth_norm = max(0.0, min(100.0, (1.0 - w_depth / n_layers) * 100.0))

    # ── Component 4: Anomaly penalty (10%) ───────────────────────────
    # Full 10 points if not anomalous, 0 if anomalous
    anomaly_penalty = 0.0 if depth_result.is_anomalous else 100.0

    # ── Composite ────────────────────────────────────────────────────
    score = (
        depth_norm * 0.40
        + entropy_norm * 0.30
        + w_depth_norm * 0.20
        + anomaly_penalty * 0.10
    )
    score = round(max(0.0, min(100.0, score)), 1)

    grade = _grade(score)

    components = {
        "depth_normalized": round(depth_norm, 2),
        "depth_weight": 0.40,
        "entropy_normalized": round(entropy_norm, 2),
        "entropy_weight": 0.30,
        "weighted_depth_normalized": round(w_depth_norm, 2),
        "weighted_depth_weight": 0.20,
        "anomaly_penalty": round(anomaly_penalty, 2),
        "anomaly_weight": 0.10,
        "raw_depth_score": round(raw_depth, 4) if not math.isnan(raw_depth) else None,
        "raw_entropy": round(entropy, 4),
        "raw_weighted_depth": round(w_depth, 4) if w_depth is not None and not math.isnan(w_depth) else None,
        "is_anomalous": depth_result.is_anomalous,
    }

    interpretation = interpret_fathom_score(score, grade)

    return FathomScore(
        score=score,
        grade=grade,
        components=components,
        interpretation=interpretation,
    )


def interpret_fathom_score(score: float, grade: Optional[str] = None) -> str:
    """One-sentence interpretation of a Fathom Score.

    Args:
        score: The 0-100 composite score.
        grade: Optional pre-computed grade.

    Returns:
        Human-readable interpretation string.
    """
    if grade is None:
        grade = _grade(score)

    score_int = int(round(score))

    if grade == "S":
        return (
            f"Exceptional deep reasoning engaged — distributed, multi-layer "
            f"processing with high integration (S-grade, {score_int}/100)"
        )
    elif grade == "A":
        return (
            f"Strong reasoning depth — significant cross-layer activation "
            f"with good distribution (A-grade, {score_int}/100)"
        )
    elif grade == "B":
        return (
            f"Moderate reasoning depth — meaningful processing beyond surface "
            f"recall with some distribution (B-grade, {score_int}/100)"
        )
    elif grade == "C":
        return (
            f"Mixed processing — some deeper engagement but significant "
            f"reliance on surface patterns (C-grade, {score_int}/100)"
        )
    elif grade == "D":
        return (
            f"Shallow processing detected — primarily late-layer recall "
            f"with limited distributed reasoning (D-grade, {score_int}/100)"
        )
    else:
        return (
            f"Surface-level pattern matching detected — minimal deep "
            f"reasoning engagement (F-grade, {score_int}/100)"
        )
