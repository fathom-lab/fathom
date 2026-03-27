"""
FATHOM COGNITION ATLAS
========================
Maps ALL known reasoning systems onto a single topology space.
Humans (fMRI), AI models, and any system that processes information.

Coordinate space: (entropy, topology_ratio, phase_transition_magnitude)
Distance metric: Jensen-Shannon divergence between activation distributions.

The 0.830 universality coefficient is the seed. This builds the full atlas.

Usage:
    from cognition_atlas import CognitionAtlas

    atlas = CognitionAtlas()
    atlas.add_system("gemma-2-2b", ...)
    coords = atlas.get_coordinates_json()
    atlas.export_coordinates("atlas_data.json")
"""

import json
import math
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


@dataclass
class CognitiveSystem:
    """A reasoning system mapped in the atlas."""
    name: str
    system_type: str            # "transformer" | "biological" | "hybrid"
    provider: str
    surface_signature: List[float]   # [early, mid, late] phase distribution
    insight_signature: List[float]
    coordinates: Optional[List[float]] = None
    universality_score: float = 0.0
    fathom_score: float = 0.0
    n_measurements: int = 0
    measured_at: str = ""
    notes: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class AtlasCluster:
    name: str
    members: List[str]
    centroid: List[float]
    radius: float
    dominant_type: str


UNIVERSALITY_COEFFICIENT = 0.830

# ── Known systems (from DISCOVERY_UNIVERSAL.md + estimates) ──────────
KNOWN_SYSTEMS = {
    "gemma-2-2b": {
        "type": "transformer", "provider": "Google",
        "surface": [0.588, 0.243, 0.169],
        "insight": [0.583, 0.242, 0.175],
        "fathom_score": 78, "n": 30,
        "notes": "Measured. p=0.000051, delta=+0.257, 30 calibrated pairs.",
    },
    "human-fmri": {
        "type": "biological", "provider": "Human fMRI (literature)",
        "surface": [0.150, 0.250, 0.600],
        "insight": [0.450, 0.350, 0.200],
        "fathom_score": 0, "n": 2,
        "notes": "Reference from neuroscience. Early=prefrontal, Mid=parietal, Late=temporal.",
    },
    "gpt-4": {
        "type": "transformer", "provider": "OpenAI",
        "surface": [0.52, 0.28, 0.20],
        "insight": [0.48, 0.32, 0.20],
        "fathom_score": 0, "n": 0,
        "notes": "Estimated. Awaiting direct measurement.",
    },
    "claude-3.5-sonnet": {
        "type": "transformer", "provider": "Anthropic",
        "surface": [0.50, 0.30, 0.20],
        "insight": [0.46, 0.34, 0.20],
        "fathom_score": 0, "n": 0,
        "notes": "Estimated. Awaiting direct measurement.",
    },
    "llama-3.2-1b": {
        "type": "transformer", "provider": "Meta",
        "surface": [0.57, 0.25, 0.18],
        "insight": [0.55, 0.26, 0.19],
        "fathom_score": 0, "n": 0,
        "notes": "Estimated. Awaiting direct measurement.",
    },
    "mistral-7b": {
        "type": "transformer", "provider": "Mistral AI",
        "surface": [0.55, 0.26, 0.19],
        "insight": [0.52, 0.28, 0.20],
        "fathom_score": 0, "n": 0,
        "notes": "Estimated. Awaiting direct measurement.",
    },
}


class CognitionAtlas:
    """Universal coordinate space for reasoning systems."""

    def __init__(self):
        self.systems: Dict[str, CognitiveSystem] = {}
        self._load_known()

    def _load_known(self):
        for name, d in KNOWN_SYSTEMS.items():
            self.add_system(name, d["type"], d["provider"],
                            d["surface"], d["insight"],
                            fathom_score=d.get("fathom_score", 0),
                            n_measurements=d.get("n", 0),
                            notes=d.get("notes", ""))

    # ── Coordinate mapping ───────────────────────────────────────────

    def _entropy(self, dist: List[float]) -> float:
        total = sum(dist)
        if total == 0:
            return 0.0
        p = [x / total for x in dist]
        return -sum(pi * math.log2(pi + 1e-10) for pi in p)

    def _compute_coordinates(self, sys: CognitiveSystem) -> List[float]:
        """Map to 3-D atlas space:
        X = entropy of average signature
        Y = early/late processing ratio
        Z = phase-transition magnitude (surface -> insight shift)
        """
        avg = [(s + i) / 2 for s, i in zip(sys.surface_signature, sys.insight_signature)]
        x = self._entropy(avg)
        y = avg[0] / (avg[2] + 1e-10)
        delta = [i - s for s, i in zip(sys.surface_signature, sys.insight_signature)]
        z = math.sqrt(sum(d ** 2 for d in delta))
        return [round(x, 4), round(y, 4), round(z, 4)]

    def _js_divergence(self, p: List[float], q: List[float]) -> float:
        pa = np.array(p, dtype=float) + 1e-10
        qa = np.array(q, dtype=float) + 1e-10
        pa /= pa.sum(); qa /= qa.sum()
        m = (pa + qa) / 2
        return float(0.5 * np.sum(pa * np.log2(pa / m)) +
                      0.5 * np.sum(qa * np.log2(qa / m)))

    def _universality(self, sys: CognitiveSystem) -> float:
        ref_s = KNOWN_SYSTEMS["human-fmri"]["surface"]
        ref_i = KNOWN_SYSTEMS["human-fmri"]["insight"]
        jsd = (self._js_divergence(sys.insight_signature, ref_i) +
               self._js_divergence(sys.surface_signature, ref_s)) / 2
        return round(max(0.0, 1.0 - jsd), 4)

    # ── Public API ───────────────────────────────────────────────────

    def add_system(self, name, system_type, provider,
                   surface_signature, insight_signature,
                   fathom_score=0, n_measurements=0, notes=""):
        sys = CognitiveSystem(
            name=name, system_type=system_type, provider=provider,
            surface_signature=surface_signature,
            insight_signature=insight_signature,
            fathom_score=fathom_score,
            n_measurements=n_measurements,
            measured_at=datetime.now().isoformat(),
            notes=notes,
        )
        sys.coordinates = self._compute_coordinates(sys)
        sys.universality_score = self._universality(sys)
        self.systems[name] = sys

    def distance_matrix(self) -> Dict[str, float]:
        names = sorted(self.systems.keys())
        matrix = {}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i < j:
                    sa, sb = self.systems[a], self.systems[b]
                    jsd = (self._js_divergence(sa.insight_signature, sb.insight_signature) +
                           self._js_divergence(sa.surface_signature, sb.surface_signature)) / 2
                    matrix[f"{a} <-> {b}"] = round(jsd, 4)
        return matrix

    def find_clusters(self, threshold=0.15) -> List[AtlasCluster]:
        names = list(self.systems.keys())
        if not names:
            return []
        coords = np.array([self.systems[n].coordinates for n in names])
        clusters = [[i] for i in range(len(names))]
        while len(clusters) > 1:
            best_d, mi, mj = float("inf"), 0, 1
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    ds = [np.linalg.norm(coords[ci] - coords[cj])
                          for ci in clusters[i] for cj in clusters[j]]
                    avg = np.mean(ds)
                    if avg < best_d:
                        best_d, mi, mj = avg, i, j
            if best_d > threshold:
                break
            clusters[mi] = clusters[mi] + clusters[mj]
            clusters.pop(mj)
        result = []
        for idx, cl in enumerate(clusters):
            members = [names[i] for i in cl]
            cc = np.mean(coords[cl], axis=0)
            radius = float(max(np.linalg.norm(coords[c] - cc) for c in cl)) if len(cl) > 1 else 0.0
            types = [self.systems[m].system_type for m in members]
            result.append(AtlasCluster(
                name=f"Cluster {idx+1}", members=members,
                centroid=[round(x, 4) for x in cc],
                radius=round(radius, 4),
                dominant_type=max(set(types), key=types.count),
            ))
        return result

    def get_coordinates_json(self) -> str:
        data = {
            "universality_coefficient": UNIVERSALITY_COEFFICIENT,
            "generated_at": datetime.now().isoformat(),
            "systems": [
                {
                    "name": s.name, "type": s.system_type,
                    "provider": s.provider,
                    "coordinates": s.coordinates,
                    "universality_score": s.universality_score,
                    "fathom_score": s.fathom_score,
                    "surface_signature": s.surface_signature,
                    "insight_signature": s.insight_signature,
                    "n_measurements": s.n_measurements,
                    "notes": s.notes,
                }
                for s in self.systems.values()
            ],
            "distance_matrix": self.distance_matrix(),
            "clusters": [asdict(c) for c in self.find_clusters()],
        }
        return json.dumps(data, indent=2)

    def export_coordinates(self, path="atlas_coordinates.json"):
        Path(path).write_text(self.get_coordinates_json(), encoding="utf-8")

    def summary(self) -> str:
        lines = [
            "FATHOM COGNITION ATLAS", "=" * 60,
            f"Systems mapped: {len(self.systems)}",
            f"Universality coefficient: {UNIVERSALITY_COEFFICIENT}", "",
            f"{'System':<25} {'Type':<14} {'Coordinates':<30} {'Univ.':<8}",
            "-" * 80,
        ]
        for n, s in sorted(self.systems.items()):
            c = s.coordinates
            lines.append(f"{n:<25} {s.system_type:<14} [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]{'':<10} {s.universality_score:.3f}")
        return "\n".join(lines)


def generate_atlas_visualization_data() -> dict:
    return json.loads(CognitionAtlas().get_coordinates_json())
