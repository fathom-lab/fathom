"""
FATHOM TEMPORAL — Depth Tracking Over Time
=============================================

What happens to reasoning depth as a model is fine-tuned?
Does RLHF make models shallower or deeper?

This is the tool for measuring cognitive drift in AI systems over time.
Nobody tracks this. We do.

Usage:
    from fathom_temporal import TemporalDepthTracker

    tracker = TemporalDepthTracker()
    snap = tracker.snapshot("gemma-2-2b", prompts, scorer)
    tracker.save_snapshot(snap, "snapshots/gemma_v1.json")

    # After fine-tuning...
    snap2 = tracker.snapshot("gemma-2-2b-ft", prompts, scorer)
    delta = tracker.compare_snapshots(snap, snap2)
    print(delta)  # "Model became 12% shallower after fine-tuning"
"""

import gc
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Standard probes for temporal tracking
TEMPORAL_PROBES = [
    "What is the capital of France?",
    "Why do economies experience inflation?",
    "How does photosynthesis work?",
    "Why is the sky blue?",
    "What causes ocean tides?",
    "How do antibiotics fight infection?",
    "Why do metals corrode over time?",
    "What drives tectonic plate movement?",
    "How does natural selection work?",
    "Why do languages evolve?",
    "What causes the seasons?",
    "How does sleep affect memory?",
    "Why do markets cycle between growth and recession?",
    "What makes some materials better conductors?",
    "How do ecosystems self-regulate?",
    "Why do stars have finite lifespans?",
    "What causes political polarization?",
    "How does the immune system identify threats?",
    "Why do complex systems show emergent properties?",
    "What drives the evolution of cooperative behavior?",
]


@dataclass
class DepthSnapshot:
    """Point-in-time depth profile of a model."""
    model_name: str
    timestamp: str
    n_probes: int
    mean_depth: float
    std_depth: float
    mean_entropy: float
    std_entropy: float
    mean_early_ratio: float
    mean_late_ratio: float
    grade_distribution: Dict[str, int]
    per_probe: List[dict]  # [{probe, depth, entropy, grade}]
    metadata: Dict[str, str]

    def to_dict(self):
        return asdict(self)


@dataclass
class SnapshotDelta:
    """Comparison between two snapshots."""
    model_a: str
    model_b: str
    timestamp_a: str
    timestamp_b: str
    depth_delta: float  # b - a (positive = got shallower)
    entropy_delta: float
    early_ratio_delta: float
    late_ratio_delta: float
    pct_depth_change: float  # percentage change
    direction: str  # "shallower", "deeper", or "stable"
    grade_shift: Dict[str, int]  # grade changes
    per_probe_deltas: List[dict]
    summary: str

    def to_dict(self):
        return asdict(self)


def _grade(depth_score: float) -> str:
    """Assign depth grade."""
    if math.isnan(depth_score):
        return "F"
    if depth_score < 7.5:
        return "A+"
    elif depth_score < 7.9:
        return "A"
    elif depth_score < 8.1:
        return "B+"
    elif depth_score < 8.25:
        return "B"
    elif depth_score < 8.4:
        return "C"
    elif depth_score < 8.6:
        return "D"
    else:
        return "F"


class TemporalDepthTracker:
    """Tracks reasoning depth across model versions over time."""

    def __init__(self):
        pass

    def snapshot(
        self,
        model_name: str,
        prompts: Optional[List[str]],
        scorer,
        metadata: Optional[Dict[str, str]] = None,
    ) -> DepthSnapshot:
        """Take a depth snapshot of a model.

        Args:
            model_name: Identifier for this model version.
            prompts: Probe prompts. Uses TEMPORAL_PROBES if None.
            scorer: Initialized DepthScorer.
            metadata: Optional metadata (version, commit hash, etc).

        Returns:
            DepthSnapshot with full depth profile.
        """
        prompts = prompts or TEMPORAL_PROBES
        metadata = metadata or {}
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        depths = []
        entropies = []
        early_ratios = []
        late_ratios = []
        per_probe = []

        for i, probe in enumerate(prompts):
            try:
                result = scorer.score(probe)
                ds = result.depth_score
                ent = result.layer_entropy or 0.0
                early = result.early_ratio
                late = result.late_ratio

                if not math.isnan(ds):
                    depths.append(ds)
                    entropies.append(ent)
                    early_ratios.append(early)
                    late_ratios.append(late)

                per_probe.append({
                    "probe": probe,
                    "depth": round(ds, 4) if not math.isnan(ds) else None,
                    "entropy": round(ent, 4),
                    "grade": _grade(ds),
                })

            except Exception as e:
                per_probe.append({
                    "probe": probe, "depth": None,
                    "entropy": 0.0, "grade": "F",
                })

            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

            if (i + 1) % 5 == 0:
                print(f"[Temporal] Snapshot {i+1}/{len(prompts)} "
                      f"(mean_depth={np.mean(depths):.4f})" if depths else
                      f"[Temporal] Snapshot {i+1}/{len(prompts)}")

        # Grade distribution
        grades = [p["grade"] for p in per_probe]
        grade_dist = {}
        for g in grades:
            grade_dist[g] = grade_dist.get(g, 0) + 1

        return DepthSnapshot(
            model_name=model_name,
            timestamp=timestamp,
            n_probes=len(prompts),
            mean_depth=round(float(np.mean(depths)), 4) if depths else float("nan"),
            std_depth=round(float(np.std(depths)), 4) if depths else 0.0,
            mean_entropy=round(float(np.mean(entropies)), 4) if entropies else 0.0,
            std_entropy=round(float(np.std(entropies)), 4) if entropies else 0.0,
            mean_early_ratio=round(float(np.mean(early_ratios)), 4) if early_ratios else 0.0,
            mean_late_ratio=round(float(np.mean(late_ratios)), 4) if late_ratios else 0.0,
            grade_distribution=grade_dist,
            per_probe=per_probe,
            metadata=metadata,
        )

    def compare_snapshots(
        self,
        snapshot_a: DepthSnapshot,
        snapshot_b: DepthSnapshot,
    ) -> SnapshotDelta:
        """Compare two snapshots to measure cognitive drift.

        Args:
            snapshot_a: Earlier snapshot (baseline).
            snapshot_b: Later snapshot (comparison).

        Returns:
            SnapshotDelta with change analysis.
        """
        depth_delta = snapshot_b.mean_depth - snapshot_a.mean_depth
        entropy_delta = snapshot_b.mean_entropy - snapshot_a.mean_entropy
        early_delta = snapshot_b.mean_early_ratio - snapshot_a.mean_early_ratio
        late_delta = snapshot_b.mean_late_ratio - snapshot_a.mean_late_ratio

        # Percentage change
        if snapshot_a.mean_depth != 0 and not math.isnan(snapshot_a.mean_depth):
            pct_change = (depth_delta / abs(snapshot_a.mean_depth)) * 100
        else:
            pct_change = 0.0

        # Direction (positive delta = got shallower = higher mean layer)
        if abs(depth_delta) < 0.02:
            direction = "stable"
        elif depth_delta > 0:
            direction = "shallower"
        else:
            direction = "deeper"

        # Grade shift
        grade_shift = {}
        for g in set(list(snapshot_a.grade_distribution.keys()) +
                     list(snapshot_b.grade_distribution.keys())):
            a_count = snapshot_a.grade_distribution.get(g, 0)
            b_count = snapshot_b.grade_distribution.get(g, 0)
            if a_count != b_count:
                grade_shift[g] = b_count - a_count

        # Per-probe deltas
        per_probe_deltas = []
        probes_a = {p["probe"]: p for p in snapshot_a.per_probe}
        for p_b in snapshot_b.per_probe:
            probe = p_b["probe"]
            if probe in probes_a:
                p_a = probes_a[probe]
                if p_a["depth"] is not None and p_b["depth"] is not None:
                    d = p_b["depth"] - p_a["depth"]
                    per_probe_deltas.append({
                        "probe": probe,
                        "depth_a": p_a["depth"],
                        "depth_b": p_b["depth"],
                        "delta": round(d, 4),
                        "grade_a": p_a["grade"],
                        "grade_b": p_b["grade"],
                    })

        # Summary
        abs_pct = abs(pct_change)
        summary = (
            f"Model {'became' if direction != 'stable' else 'remained'} "
            f"{direction}"
            f"{f' by {abs_pct:.1f}%' if direction != 'stable' else ''} "
            f"(depth: {snapshot_a.mean_depth:.4f} → {snapshot_b.mean_depth:.4f}, "
            f"delta: {depth_delta:+.4f})"
        )

        return SnapshotDelta(
            model_a=snapshot_a.model_name,
            model_b=snapshot_b.model_name,
            timestamp_a=snapshot_a.timestamp,
            timestamp_b=snapshot_b.timestamp,
            depth_delta=round(depth_delta, 4),
            entropy_delta=round(entropy_delta, 4),
            early_ratio_delta=round(early_delta, 4),
            late_ratio_delta=round(late_delta, 4),
            pct_depth_change=round(pct_change, 2),
            direction=direction,
            grade_shift=grade_shift,
            per_probe_deltas=per_probe_deltas,
            summary=summary,
        )

    def save_snapshot(self, snapshot: DepthSnapshot, path: str):
        """Save snapshot to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        print(f"[Temporal] Snapshot saved: {p}")

    def load_snapshot(self, path: str) -> DepthSnapshot:
        """Load snapshot from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DepthSnapshot(**data)
