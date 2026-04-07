"""
REASONING FORENSICS ENGINE — Model Identification from Cognitive Fingerprints
=============================================================================

Every model has a characteristic cognitive fingerprint — the shape of its
depth distribution, the entropy of its layer activation, the ratio of early
to late processing. Given an unknown AI-generated text, the forensics engine
scores it and compares the fingerprint against a library of known models to
identify which model produced it. Cognitive ballistics for AI.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np


LIBRARY_PATH = Path(__file__).parent / "forensics_library.json"


@dataclass
class ModelMatch:
    """A candidate model match from forensic analysis."""
    model_id: str
    similarity: float              # cosine similarity to library fingerprint
    confidence: float              # calibrated confidence (0-1)
    distinguishing_dimensions: List[str]  # which fingerprint dims were most decisive


@dataclass
class ForensicResult:
    """Result of forensic model identification."""
    text_sample: str
    fingerprint: list              # the sample's cognitive fingerprint
    top_matches: List[ModelMatch]
    best_match: str                # model_id of top match
    best_confidence: float
    compute_time_s: float

    def to_dict(self):
        return {
            "text_sample": self.text_sample[:100] + "..." if len(self.text_sample) > 100 else self.text_sample,
            "fingerprint": self.fingerprint,
            "best_match": self.best_match,
            "best_confidence": self.best_confidence,
            "top_matches": [
                {"model": m.model_id, "similarity": m.similarity, "confidence": m.confidence}
                for m in self.top_matches
            ],
            "compute_time_s": self.compute_time_s,
        }


# Fingerprint dimension labels (must match DepthScorer.fingerprint() output)
FINGERPRINT_DIMS = [
    "weighted_depth",
    "layer_entropy",
    "early_ratio",
    "mid_ratio",
    "late_ratio",
    "feature_count_norm",
    "compute_efficiency",
]


class ForensicsEngine:
    """Identifies which model produced a text from its cognitive fingerprint.

    Builds a library of mean fingerprint vectors per model by scoring standard
    prompts, then matches unknown samples against the library using cosine
    similarity. The distinguishing dimensions reveal which aspects of reasoning
    are most model-specific — some models are deeper, some more distributed,
    some more efficient. These are cognitive signatures, not stylistic ones.
    """

    def __init__(self, library_path: Optional[str] = None):
        self._library_path = Path(library_path) if library_path else LIBRARY_PATH
        self._library = {}  # model_id → {"mean_fp": [...], "std_fp": [...], "n_prompts": int}
        self._load_library()

    def _load_library(self):
        """Load forensics library from disk."""
        if self._library_path.exists():
            try:
                with open(self._library_path) as f:
                    self._library = json.load(f)
                print(f"[Forensics] Loaded library with {len(self._library)} models")
            except Exception:
                self._library = {}

    def _save_library(self):
        """Save forensics library to disk."""
        with open(self._library_path, "w") as f:
            json.dump(self._library, f, indent=2)
        print(f"[Forensics] Saved library to {self._library_path}")

    def build_model_library(
        self,
        model_names: List[str],
        prompts: List[str],
        scorer_factory=None,
    ):
        """Score standard prompts across each model, build fingerprint library.

        Args:
            model_names: List of model identifiers to profile.
            prompts: Standard prompts to score (recommend 50+).
            scorer_factory: Callable(model_name) → DepthScorer instance.
                If None, uses a single scorer and tags results with its model_name.
        """
        for model_name in model_names:
            print(f"\n[Forensics] Profiling {model_name}...")

            if scorer_factory is not None:
                scorer = scorer_factory(model_name)
            else:
                raise ValueError("scorer_factory required for multi-model profiling.")

            fingerprints = []
            for i, prompt in enumerate(prompts):
                try:
                    result = scorer.score(prompt)
                    fp = scorer.fingerprint(result)
                    fingerprints.append(fp.tolist())
                except Exception as e:
                    print(f"  Prompt {i} failed: {e}")
                    continue

            if not fingerprints:
                print(f"  No valid fingerprints for {model_name}")
                continue

            fp_array = np.array(fingerprints)
            self._library[model_name] = {
                "mean_fp": fp_array.mean(axis=0).tolist(),
                "std_fp": fp_array.std(axis=0).tolist(),
                "n_prompts": len(fingerprints),
            }
            print(f"  Stored {len(fingerprints)} fingerprints for {model_name}")

        self._save_library()

    def add_model_to_library(
        self,
        model_name: str,
        prompts: List[str],
        scorer,
    ):
        """Add a single model to the forensics library using an existing scorer.

        Args:
            model_name: Model identifier.
            prompts: Prompts to score.
            scorer: An initialized DepthScorer (already loaded with this model).
        """
        fingerprints = []
        print(f"[Forensics] Profiling {model_name} ({len(prompts)} prompts)...")

        for i, prompt in enumerate(prompts):
            try:
                result = scorer.score(prompt)
                fp = scorer.fingerprint(result)
                fingerprints.append(fp.tolist())
            except Exception as e:
                print(f"  Prompt {i} failed: {e}")
                continue

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(prompts)}] scored...")

        if not fingerprints:
            raise ValueError(f"No valid fingerprints collected for {model_name}")

        fp_array = np.array(fingerprints)
        self._library[model_name] = {
            "mean_fp": fp_array.mean(axis=0).tolist(),
            "std_fp": fp_array.std(axis=0).tolist(),
            "n_prompts": len(fingerprints),
        }

        self._save_library()
        print(f"[Forensics] Added {model_name}: {len(fingerprints)} fingerprints")

    def identify_model(
        self,
        text_sample: str,
        scorer,
    ) -> ForensicResult:
        """Identify which model produced a text sample.

        Scores the sample, computes its cognitive fingerprint, and compares
        against the library of known model fingerprints.

        Args:
            text_sample: AI-generated text to identify.
            scorer: An initialized DepthScorer.

        Returns:
            ForensicResult with ranked model matches.
        """
        if not self._library:
            raise RuntimeError(
                "Forensics library is empty. Call build_model_library() or "
                "add_model_to_library() first."
            )

        t0 = time.time()

        # Score the sample
        result = scorer.score(text_sample)
        sample_fp = scorer.fingerprint(result)

        # Compare against library
        matches = []
        for model_id, entry in self._library.items():
            mean_fp = np.array(entry["mean_fp"])
            std_fp = np.array(entry["std_fp"])

            # Cosine similarity
            similarity = self._cosine_similarity(sample_fp, mean_fp)

            # Confidence: based on similarity and how many std devs away
            z_scores = np.abs(sample_fp - mean_fp) / np.maximum(std_fp, 1e-8)
            mean_z = float(np.mean(z_scores))
            confidence = max(0.0, min(1.0, similarity * (1.0 - min(1.0, mean_z / 5.0))))

            # Which dimensions distinguish this match
            dim_diffs = np.abs(sample_fp - mean_fp)
            sorted_dims = np.argsort(-dim_diffs)
            distinguishing = [FINGERPRINT_DIMS[d] for d in sorted_dims[:3] if d < len(FINGERPRINT_DIMS)]

            matches.append(ModelMatch(
                model_id=model_id,
                similarity=float(similarity),
                confidence=float(confidence),
                distinguishing_dimensions=distinguishing,
            ))

        matches.sort(key=lambda m: m.similarity, reverse=True)

        compute_time = time.time() - t0

        return ForensicResult(
            text_sample=text_sample,
            fingerprint=sample_fp.tolist(),
            top_matches=matches[:3],
            best_match=matches[0].model_id if matches else "unknown",
            best_confidence=matches[0].confidence if matches else 0.0,
            compute_time_s=compute_time,
        )

    def forensics_confidence(self, result: ForensicResult) -> Dict:
        """Assess identification confidence with detailed breakdown.

        Args:
            result: A ForensicResult from identify_model().

        Returns:
            Dict with confidence score, margin, and distinguishing features.
        """
        if not result.top_matches:
            return {
                "confidence": 0.0,
                "margin": 0.0,
                "assessment": "No models in library to compare against.",
                "distinguishing_features": [],
            }

        top = result.top_matches[0]
        margin = 0.0
        if len(result.top_matches) >= 2:
            margin = top.similarity - result.top_matches[1].similarity

        if top.confidence >= 0.8 and margin >= 0.1:
            assessment = f"High confidence: {top.model_id} with clear separation from alternatives."
        elif top.confidence >= 0.5:
            assessment = f"Moderate confidence: {top.model_id} is the best match but alternatives are plausible."
        else:
            assessment = f"Low confidence: {top.model_id} is the closest match but the signal is weak."

        return {
            "confidence": top.confidence,
            "margin": margin,
            "assessment": assessment,
            "distinguishing_features": top.distinguishing_dimensions,
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
