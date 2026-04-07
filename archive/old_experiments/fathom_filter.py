"""
FATHOM FILTER — Depth-Based Training Data Curation
====================================================

What if you could train a model only on deep-reasoning examples?
This tool scores training data by depth and filters to keep only
examples that teach genuine reasoning — not surface pattern matching.

This is a new method for high-quality dataset curation.

Usage:
    from fathom_filter import TrainingDataFilter

    f = TrainingDataFilter()
    scored = f.score_dataset(texts, scorer)
    deep = f.filter_deep(texts, scorer, min_grade="B")
    dist = f.depth_distribution(texts, scorer)
    curated = f.curate_reasoning_dataset(texts, scorer, target_n=1000)
"""

import gc
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# Grade boundaries (based on gemma-2-2b calibration)
# Lower depth_score = deeper reasoning
GRADE_BOUNDARIES = {
    "A+": (0.0, 7.5),    # Exceptionally deep
    "A":  (7.5, 7.9),    # Very deep reasoning
    "B+": (7.9, 8.1),    # Deep reasoning (near insight mean)
    "B":  (8.1, 8.25),   # Moderate reasoning
    "C":  (8.25, 8.4),   # Shallow reasoning (near surface mean)
    "D":  (8.4, 8.6),    # Very shallow
    "F":  (8.6, 100.0),  # Surface-only pattern
}


def grade_depth(depth_score: float) -> str:
    """Assign a letter grade to a depth score."""
    if math.isnan(depth_score):
        return "F"
    for grade, (low, high) in GRADE_BOUNDARIES.items():
        if low <= depth_score < high:
            return grade
    return "F"


@dataclass
class ScoredText:
    """A text scored by depth."""
    text: str
    depth_score: float
    weighted_depth: float
    entropy: float
    grade: str
    feature_count: int

    def to_dict(self):
        return asdict(self)


@dataclass
class DepthDistribution:
    """Distribution of depth scores across a dataset."""
    n_total: int
    n_scored: int
    n_failed: int
    mean_depth: float
    std_depth: float
    min_depth: float
    max_depth: float
    median_depth: float
    grade_counts: Dict[str, int]
    grade_pcts: Dict[str, float]
    pct_deep: float  # % scoring A+ through B+
    pct_shallow: float  # % scoring C through F
    histogram: List[Tuple[float, float, int]]  # (bin_low, bin_high, count)

    def to_dict(self):
        d = asdict(self)
        d["histogram"] = [
            {"low": h[0], "high": h[1], "count": h[2]}
            for h in self.histogram
        ]
        return d


class TrainingDataFilter:
    """Depth-based training data curation engine."""

    def __init__(self):
        pass

    def score_dataset(
        self,
        texts: List[str],
        scorer,
        batch_size: int = 10,
    ) -> List[ScoredText]:
        """Score a list of training examples by depth.

        Args:
            texts: List of text examples.
            scorer: Initialized DepthScorer.
            batch_size: Print progress every N examples.

        Returns:
            List of ScoredText objects, one per input.
        """
        scored = []

        for i, text in enumerate(texts):
            try:
                result = scorer.score(text)
                ds = result.depth_score
                wd = result.weighted_depth_score or ds
                ent = result.layer_entropy or 0.0

                scored.append(ScoredText(
                    text=text,
                    depth_score=round(ds, 4) if not math.isnan(ds) else float("nan"),
                    weighted_depth=round(wd, 4) if not math.isnan(wd) else float("nan"),
                    entropy=round(ent, 4),
                    grade=grade_depth(ds),
                    feature_count=result.feature_count,
                ))
            except Exception as e:
                scored.append(ScoredText(
                    text=text, depth_score=float("nan"),
                    weighted_depth=float("nan"), entropy=0.0,
                    grade="F", feature_count=0,
                ))

            if (i + 1) % batch_size == 0:
                print(f"[Filter] Scored {i+1}/{len(texts)}")

            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

        return scored

    def filter_deep(
        self,
        texts: List[str],
        scorer,
        min_depth_score: Optional[float] = None,
        min_grade: str = "B",
    ) -> List[ScoredText]:
        """Return only examples that score at or above the depth threshold.

        Args:
            texts: List of text examples.
            scorer: Initialized DepthScorer.
            min_depth_score: Max depth_score to include (lower = deeper).
                Overrides min_grade if set.
            min_grade: Minimum letter grade to include ("A+", "A", "B+", "B", etc).

        Returns:
            Filtered list of deep-reasoning examples.
        """
        scored = self.score_dataset(texts, scorer)

        if min_depth_score is not None:
            return [s for s in scored
                    if not math.isnan(s.depth_score) and s.depth_score <= min_depth_score]

        # Grade-based filtering
        grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
        try:
            cutoff_idx = grade_order.index(min_grade)
        except ValueError:
            cutoff_idx = 3  # default to B

        acceptable = set(grade_order[:cutoff_idx + 1])
        return [s for s in scored if s.grade in acceptable]

    def depth_distribution(
        self,
        texts: List[str],
        scorer,
    ) -> DepthDistribution:
        """Compute the depth score distribution across a dataset.

        Args:
            texts: List of text examples.
            scorer: Initialized DepthScorer.

        Returns:
            DepthDistribution with statistics and histogram.
        """
        scored = self.score_dataset(texts, scorer)

        valid_scores = [s.depth_score for s in scored if not math.isnan(s.depth_score)]
        n_failed = len(scored) - len(valid_scores)

        if not valid_scores:
            return DepthDistribution(
                n_total=len(texts), n_scored=0, n_failed=n_failed,
                mean_depth=float("nan"), std_depth=0.0,
                min_depth=float("nan"), max_depth=float("nan"),
                median_depth=float("nan"),
                grade_counts={}, grade_pcts={},
                pct_deep=0.0, pct_shallow=0.0, histogram=[],
            )

        arr = np.array(valid_scores)

        # Grade distribution
        grade_counts = {}
        for s in scored:
            grade_counts[s.grade] = grade_counts.get(s.grade, 0) + 1
        grade_pcts = {g: c / len(scored) for g, c in grade_counts.items()}

        deep_grades = {"A+", "A", "B+"}
        shallow_grades = {"C", "D", "F"}
        pct_deep = sum(grade_counts.get(g, 0) for g in deep_grades) / len(scored)
        pct_shallow = sum(grade_counts.get(g, 0) for g in shallow_grades) / len(scored)

        # Histogram
        bins = np.arange(7.0, 9.5, 0.1)
        counts, edges = np.histogram(arr, bins=bins)
        histogram = [
            (round(float(edges[i]), 2), round(float(edges[i+1]), 2), int(counts[i]))
            for i in range(len(counts))
        ]

        return DepthDistribution(
            n_total=len(texts),
            n_scored=len(valid_scores),
            n_failed=n_failed,
            mean_depth=round(float(arr.mean()), 4),
            std_depth=round(float(arr.std()), 4),
            min_depth=round(float(arr.min()), 4),
            max_depth=round(float(arr.max()), 4),
            median_depth=round(float(np.median(arr)), 4),
            grade_counts=grade_counts,
            grade_pcts={g: round(p, 4) for g, p in grade_pcts.items()},
            pct_deep=round(pct_deep, 4),
            pct_shallow=round(pct_shallow, 4),
            histogram=histogram,
        )

    def curate_reasoning_dataset(
        self,
        texts: List[str],
        scorer,
        target_n: int = 1000,
    ) -> List[ScoredText]:
        """Select the top-N deepest examples from a larger dataset.

        Args:
            texts: Full dataset of text examples.
            scorer: Initialized DepthScorer.
            target_n: Number of examples to select.

        Returns:
            The deepest target_n examples, sorted by depth (deepest first).
        """
        scored = self.score_dataset(texts, scorer)

        # Filter out NaN, sort by depth (lower = deeper)
        valid = [s for s in scored if not math.isnan(s.depth_score)]
        valid.sort(key=lambda s: s.depth_score)

        selected = valid[:target_n]

        print(f"[Filter] Curated {len(selected)} from {len(texts)} "
              f"(deepest: {selected[0].depth_score:.4f}, "
              f"cutoff: {selected[-1].depth_score:.4f})" if selected else
              f"[Filter] No valid examples to curate")

        return selected
