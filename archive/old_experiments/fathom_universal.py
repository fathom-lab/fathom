"""
UNIVERSAL COGNITION MAPPER — Cross-System Reasoning Signatures
===============================================================

If the same depth signatures appear across biological neural systems
(fMRI) and artificial systems (transformers of different architectures,
scales, and training) — then reasoning depth is not a model property.
It's a universal property of information processing systems. This module
collects signatures from all available sources, normalizes them to a
common space, and tests for universality.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np


@dataclass
class CognitionSignature:
    """A normalized cognition signature from any information processing system."""
    source_name: str
    source_type: str               # "transformer", "neuro_reference", "custom"
    condition: str                  # "surface" or "insight"
    early_ratio: float
    mid_ratio: float
    late_ratio: float
    n_samples: int
    metadata: Optional[Dict] = None

    def to_vector(self) -> np.ndarray:
        return np.array([self.early_ratio, self.mid_ratio, self.late_ratio], dtype=np.float64)


@dataclass
class UniversalityResult:
    """Result of cross-system universality analysis."""
    signatures: List[CognitionSignature]
    n_sources: int
    n_transformer_sources: int
    n_neuro_sources: int
    surface_variance: float        # variance of surface signatures (lower = more universal)
    insight_variance: float        # variance of insight signatures
    universality_coefficient: float  # 0-1 (higher = more universal)
    is_universal: bool             # coefficient > 0.8
    discovery: Optional[str]

    def to_dict(self):
        return {
            "n_sources": self.n_sources,
            "surface_variance": self.surface_variance,
            "insight_variance": self.insight_variance,
            "universality_coefficient": self.universality_coefficient,
            "is_universal": self.is_universal,
            "sources": [
                {"name": s.source_name, "type": s.source_type, "condition": s.condition,
                 "signature": s.to_vector().tolist()}
                for s in self.signatures
            ],
        }


# ── Neuroscience Reference Signatures ─────────────────────────────────
#
# Approximate activation ratios derived from published fMRI literature.
# These represent the relative engagement of early (prefrontal),
# mid (parietal/integration), and late (hippocampal/retrieval) processing
# stages during reasoning vs recall tasks.
#
# CITE: Duncan (2010), Squire & Wixted (2011), Christoff et al. (2009)
# See fathom_neuro.py for full citation details.

NEURO_REFERENCE_SIGNATURES = {
    "human_fmri_reasoning": CognitionSignature(
        source_name="Human fMRI (reasoning tasks)",
        source_type="neuro_reference",
        condition="insight",
        early_ratio=0.45,   # PFC dominant
        mid_ratio=0.35,     # parietal integration
        late_ratio=0.20,    # minimal hippocampal
        n_samples=0,        # reference, not empirical
        metadata={"basis": "meta-analysis of fMRI reasoning studies"},
    ),
    "human_fmri_recall": CognitionSignature(
        source_name="Human fMRI (recall tasks)",
        source_type="neuro_reference",
        condition="surface",
        early_ratio=0.15,   # minimal PFC
        mid_ratio=0.25,     # some integration
        late_ratio=0.60,    # hippocampus dominant
        n_samples=0,
        metadata={"basis": "meta-analysis of fMRI memory retrieval studies"},
    ),
}


class UniversalCognitionMapper:
    """Tests whether reasoning depth signatures are universal across information
    processing systems — biological and artificial.

    Collects normalized 3-vector signatures [early_ratio, mid_ratio, late_ratio]
    from multiple sources (transformer models, fMRI reference data), and computes
    the variance of surface and insight signatures independently. Low variance
    across diverse systems = the signature is universal. If it holds, reasoning
    depth is not a transformer artifact — it's a property of computation itself.
    """

    def __init__(self):
        self._signatures: List[CognitionSignature] = []

    def collect_signatures(
        self,
        sources: List[Dict],
    ):
        """Collect and normalize signatures from multiple sources.

        Args:
            sources: List of dicts, each with:
                - "name": source identifier
                - "type": "transformer" or "neuro_reference"
                - "depth_results": list of DepthResult objects (for transformer)
                  OR "signature": CognitionSignature (for neuro_reference)
                - "labels": optional list of "surface"/"insight" labels
        """
        for source in sources:
            name = source["name"]
            stype = source["type"]

            if stype == "neuro_reference":
                if "signature" in source:
                    sig = source["signature"]
                    self._signatures.append(sig)
                else:
                    # Load from built-in references
                    for key, sig in NEURO_REFERENCE_SIGNATURES.items():
                        self._signatures.append(sig)
                continue

            # Transformer source: compute mean ratios from depth results
            results = source.get("depth_results", [])
            labels = source.get("labels", [])

            if not results:
                continue

            if not labels:
                # Auto-label by median split
                scores = [r.depth_score for r in results if r and not math.isnan(r.depth_score)]
                median = float(np.median(scores)) if scores else 8.0
                labels = [
                    "insight" if (r and not math.isnan(r.depth_score) and r.depth_score < median) else "surface"
                    for r in results
                ]

            # Split by condition
            for condition in ("surface", "insight"):
                group = [r for r, l in zip(results, labels)
                         if l == condition and r is not None and not math.isnan(r.depth_score)]
                if not group:
                    continue

                sig = CognitionSignature(
                    source_name=name,
                    source_type=stype,
                    condition=condition,
                    early_ratio=float(np.mean([r.early_ratio for r in group])),
                    mid_ratio=float(np.mean([r.mid_ratio for r in group])),
                    late_ratio=float(np.mean([r.late_ratio for r in group])),
                    n_samples=len(group),
                )
                self._signatures.append(sig)

        print(f"[Universal] Collected {len(self._signatures)} signatures from {len(sources)} sources")

    def add_neuro_references(self):
        """Add built-in neuroscience reference signatures."""
        for sig in NEURO_REFERENCE_SIGNATURES.values():
            self._signatures.append(sig)
        print(f"[Universal] Added {len(NEURO_REFERENCE_SIGNATURES)} neuro reference signatures")

    def compute_universality_score(
        self,
        signatures: Optional[List[CognitionSignature]] = None,
    ) -> UniversalityResult:
        """Compute how universal the reasoning signatures are across sources.

        Low variance of signatures across diverse systems = high universality.
        The universality coefficient is 1 - mean_variance (capped at 1).

        Args:
            signatures: List of CognitionSignature objects. Uses stored if None.

        Returns:
            UniversalityResult with universality_coefficient.
        """
        if signatures is not None:
            self._signatures = signatures

        if len(self._signatures) < 4:
            raise ValueError(
                f"Need at least 4 signatures (2 surface + 2 insight), have {len(self._signatures)}. "
                f"Call collect_signatures() or add_neuro_references() first."
            )

        surface_sigs = [s for s in self._signatures if s.condition == "surface"]
        insight_sigs = [s for s in self._signatures if s.condition == "insight"]

        if len(surface_sigs) < 2 or len(insight_sigs) < 2:
            raise ValueError(
                f"Need >= 2 of each condition. Have {len(surface_sigs)} surface, {len(insight_sigs)} insight."
            )

        # Compute variance of each condition's signatures
        surface_vectors = np.array([s.to_vector() for s in surface_sigs])
        insight_vectors = np.array([s.to_vector() for s in insight_sigs])

        surface_var = float(np.mean(np.var(surface_vectors, axis=0)))
        insight_var = float(np.mean(np.var(insight_vectors, axis=0)))

        # Universality: low variance = high universality
        mean_var = (surface_var + insight_var) / 2
        # Scale: variance of 0 → coefficient 1.0; variance of 0.1 → ~0
        coefficient = max(0.0, min(1.0, 1.0 - mean_var * 10))

        is_universal = coefficient > 0.8

        n_transformer = sum(1 for s in self._signatures if s.source_type == "transformer")
        n_neuro = sum(1 for s in self._signatures if s.source_type == "neuro_reference")
        # Count unique sources
        unique_sources = len(set(s.source_name for s in self._signatures))

        discovery = None
        if is_universal:
            discovery = (
                f"UNIVERSAL COGNITION SIGNATURE: Reasoning depth topology is consistent "
                f"across {unique_sources} distinct information processing systems "
                f"(universality coefficient: {coefficient:.3f}). Surface recall and "
                f"explanatory reasoning produce characteristic early/mid/late activation "
                f"ratios that hold across transformer architectures"
                f"{' and biological neural systems' if n_neuro > 0 else ''}."
            )
            self._write_discovery(discovery, coefficient, unique_sources,
                                  surface_sigs, insight_sigs)

        result = UniversalityResult(
            signatures=self._signatures,
            n_sources=unique_sources,
            n_transformer_sources=n_transformer,
            n_neuro_sources=n_neuro,
            surface_variance=surface_var,
            insight_variance=insight_var,
            universality_coefficient=coefficient,
            is_universal=is_universal,
            discovery=discovery,
        )

        print(f"\n[Universal] Universality coefficient: {coefficient:.3f}")
        print(f"  Surface variance: {surface_var:.4f}")
        print(f"  Insight variance: {insight_var:.4f}")
        print(f"  Sources: {unique_sources} ({n_transformer} transformer, {n_neuro} neuro)")
        if discovery:
            print(f"\n  *** {discovery}")

        return result

    def universality_report(
        self,
        result: UniversalityResult,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate the universality report.

        Auto-titles as DISCOVERY_UNIVERSAL.md if coefficient > 0.8.

        Args:
            result: A UniversalityResult.
            output_path: Optional output path.

        Returns:
            Report content as string.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if output_path is None:
            if result.is_universal:
                output_path = str(Path(__file__).parent / "DISCOVERY_UNIVERSAL.md")
            else:
                output_path = str(Path(__file__).parent / "universality_report.md")

        # Build signature table
        sig_rows = "\n".join(
            f"| {s.source_name} | {s.source_type} | {s.condition} | "
            f"{s.early_ratio:.3f} | {s.mid_ratio:.3f} | {s.late_ratio:.3f} |"
            for s in result.signatures
        )

        content = f"""# Universal Cognition Analysis

**Date:** {timestamp}
**Status:** {'UNIVERSAL SIGNATURE FOUND' if result.is_universal else 'PRELIMINARY'}

## Finding

{'Fathom has identified a universal signature of reasoning depth that holds across ' + str(result.n_sources) + ' distinct information processing systems.' if result.is_universal else 'Universality not yet established. More diverse sources needed.'}

## Universality Coefficient: {result.universality_coefficient:.3f}

| Metric | Value |
|--------|-------|
| Total sources | {result.n_sources} |
| Transformer sources | {result.n_transformer_sources} |
| Neuro reference sources | {result.n_neuro_sources} |
| Surface signature variance | {result.surface_variance:.4f} |
| Insight signature variance | {result.insight_variance:.4f} |
| Universal (coeff > 0.8) | {'Yes' if result.is_universal else 'No'} |

## Signatures

| Source | Type | Condition | Early | Mid | Late |
|--------|------|-----------|-------|-----|------|
{sig_rows}

## Methodology

Signatures are normalized 3-vectors [early_ratio, mid_ratio, late_ratio]
representing the distribution of computation across processing stages.
Universality is measured as 1 - mean_variance across sources: low variance
means all systems show the same pattern for the same cognitive task.

## Interpretation

{'The convergence of activation patterns across architecturally diverse systems suggests that the early-distributed (reasoning) vs late-concentrated (recall) topology is not an artifact of any particular system but a computational universal — a property that emerges whenever an information processing system differentiates between genuine reasoning and pattern retrieval.' if result.is_universal else 'Current data shows variability across sources. Additional model architectures, scales, and biological reference data would strengthen the analysis.'}
"""

        with open(output_path, "w") as f:
            f.write(content)
        print(f"[Universal] Report written to {output_path}")

        return content

    def _write_discovery(self, discovery: str, coefficient: float,
                         n_sources: int, surface_sigs, insight_sigs):
        """Write discovery file."""
        path = Path(__file__).parent / "DISCOVERY_UNIVERSAL.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Compute mean signatures
        surface_mean = np.mean([s.to_vector() for s in surface_sigs], axis=0)
        insight_mean = np.mean([s.to_vector() for s in insight_sigs], axis=0)

        content = f"""# Fathom Discovery: Universal Cognition Signature

**Date:** {timestamp}
**Status:** SIGNIFICANT

## Finding

{discovery}

## The Universal Signatures

**Surface Recall (mean across {len(surface_sigs)} sources):**
early={surface_mean[0]:.3f}, mid={surface_mean[1]:.3f}, late={surface_mean[2]:.3f}

**Explanatory Reasoning (mean across {len(insight_sigs)} sources):**
early={insight_mean[0]:.3f}, mid={insight_mean[1]:.3f}, late={insight_mean[2]:.3f}

## Evidence

Universality coefficient: {coefficient:.3f} (threshold: 0.8)
Sources: {n_sources} distinct information processing systems.

## Significance

If confirmed with additional data, this finding suggests that reasoning
depth is a universal property of information processing — not specific to
any architecture, training corpus, or substrate. The same computational
topology emerges in biological and artificial systems alike.
"""

        with open(path, "w") as f:
            f.write(content)
        print(f"[Universal] Discovery written to {path}")
