"""
FATHOM CORRECTNESS PREDICTOR — Depth-Based Binary Classifier
==============================================================

Predicts whether an AI output is correct or hallucinated using only
the depth signature from DepthScorer.

KEY FINDING (inverted from original hypothesis):
    Correct answers score SHALLOWER (higher depth_score) than incorrect
    answers. Shallow = efficient factual retrieval = correct. Deep =
    confused/uncertain processing = incorrect.

    Direction is consistent across initial pairs, mean delta +0.188.

Usage:
    from fathom_correctness import CorrectnessPredictor

    predictor = CorrectnessPredictor()
    predictor.train(scored_pairs)
    result = predictor.predict(depth_result)
    # {"is_correct": True, "confidence": 0.82, "depth_signal": "shallow_recall"}
"""

import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Direction Note ────────────────────────────────────────────────────

DIRECTION_NOTE = (
    "Correct answers score shallower (higher depth_score). "
    "Incorrect/hallucinated answers score deeper (lower depth_score, "
    "more distributed processing). This is the INVERTED direction from "
    "the original hypothesis."
)


# ── Feature extraction ────────────────────────────────────────────────

FEATURE_NAMES = [
    "depth_score",
    "weighted_depth",
    "layer_entropy",
    "early_ratio",
    "late_ratio",
]


def _extract_features_from_tuple(tup) -> np.ndarray:
    """Extract feature vector from a training tuple.

    Expected tuple: (depth_score, weighted_depth, layer_entropy,
                     early_ratio, late_ratio, is_correct)
    """
    return np.array(tup[:5], dtype=np.float64)


def _extract_features_from_depth_result(depth_result) -> np.ndarray:
    """Extract feature vector from a DepthResult object."""
    wd = depth_result.weighted_depth_score
    if wd is None:
        wd = depth_result.depth_score
    ent = depth_result.layer_entropy
    if ent is None:
        ent = 0.0
    return np.array([
        depth_result.depth_score,
        wd,
        ent,
        depth_result.early_ratio,
        depth_result.late_ratio,
    ], dtype=np.float64)


# ── Threshold-based fallback classifier ───────────────────────────────

class ThresholdClassifier:
    """Simple threshold on depth_score. Fallback when sklearn unavailable."""

    def __init__(self):
        self.threshold = 8.23  # midpoint of calibration means
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Find optimal depth_score threshold maximizing F1."""
        depth_scores = X[:, 0]  # first feature is depth_score
        best_f1 = 0.0
        best_thresh = self.threshold

        # Scan thresholds
        sorted_scores = np.sort(np.unique(depth_scores))
        for i in range(len(sorted_scores) - 1):
            thresh = (sorted_scores[i] + sorted_scores[i + 1]) / 2.0
            # Correct = shallow = depth_score >= threshold
            preds = (depth_scores >= thresh).astype(int)
            tp = ((preds == 1) & (y == 1)).sum()
            fp = ((preds == 1) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        self.threshold = best_thresh
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        depth_scores = X[:, 0]
        return (depth_scores >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Rough probability estimate based on distance from threshold."""
        depth_scores = X[:, 0]
        # Sigmoid-like transform centered on threshold
        dist = depth_scores - self.threshold
        prob_correct = 1.0 / (1.0 + np.exp(-5.0 * dist))
        prob_incorrect = 1.0 - prob_correct
        return np.column_stack([prob_incorrect, prob_correct])


# ── Main class ────────────────────────────────────────────────────────

class CorrectnessPredictor:
    """Binary classifier: correct vs hallucinated, from depth signature.

    Direction: correct = shallower (higher depth_score).
    """

    def __init__(self):
        self._model = None
        self._scaler = None
        self._threshold = 8.23  # default midpoint
        self._is_trained = False
        self._n_training_samples = 0
        self._use_sklearn = HAS_SKLEARN

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def n_training_samples(self) -> int:
        return self._n_training_samples

    def train(self, scored_pairs: list):
        """Train on scored pairs.

        Args:
            scored_pairs: list of tuples:
                (depth_score, weighted_depth, layer_entropy,
                 early_ratio, late_ratio, is_correct)
                where is_correct is 1/True for correct, 0/False for incorrect.
        """
        if len(scored_pairs) < 4:
            raise ValueError(f"Need >= 4 samples, got {len(scored_pairs)}")

        X = np.array([_extract_features_from_tuple(t) for t in scored_pairs])
        y = np.array([int(t[5]) for t in scored_pairs])

        # Replace NaN with column means
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                col_mean = np.nanmean(X[:, col])
                X[mask, col] = col_mean

        self._n_training_samples = len(scored_pairs)

        if self._use_sklearn:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            self._model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            )
            self._model.fit(X_scaled, y)
        else:
            self._model = ThresholdClassifier()
            self._model.fit(X, y)
            self._scaler = None

        # Also calibrate the standalone threshold
        self._threshold = self._calibrate_threshold_internal(X, y)
        self._is_trained = True

    def _calibrate_threshold_internal(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find optimal depth_score threshold maximizing F1."""
        depth_scores = X[:, 0]
        best_f1 = 0.0
        best_thresh = 8.23

        sorted_scores = np.sort(np.unique(depth_scores))
        for i in range(len(sorted_scores) - 1):
            thresh = (sorted_scores[i] + sorted_scores[i + 1]) / 2.0
            preds = (depth_scores >= thresh).astype(int)
            tp = ((preds == 1) & (y == 1)).sum()
            fp = ((preds == 1) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        return best_thresh

    def calibrate_threshold(self, scored_pairs: list) -> float:
        """Public interface: find optimal depth_score threshold from data.

        Args:
            scored_pairs: same format as train().

        Returns:
            Optimal threshold value.
        """
        X = np.array([_extract_features_from_tuple(t) for t in scored_pairs])
        y = np.array([int(t[5]) for t in scored_pairs])

        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                X[mask, col] = np.nanmean(X[:, col])

        self._threshold = self._calibrate_threshold_internal(X, y)
        return self._threshold

    def predict(self, depth_result) -> dict:
        """Predict correctness from a DepthResult.

        Args:
            depth_result: DepthResult from DepthScorer.score()

        Returns:
            {
                "is_correct": bool,
                "confidence": float (0-1),
                "depth_signal": "shallow_recall" | "deep_uncertain"
            }
        """
        features = _extract_features_from_depth_result(depth_result)
        features_2d = features.reshape(1, -1)

        # Replace NaN
        for col in range(features_2d.shape[1]):
            if np.isnan(features_2d[0, col]):
                features_2d[0, col] = 0.0

        if self._is_trained and self._model is not None:
            if self._use_sklearn and self._scaler is not None:
                X_scaled = self._scaler.transform(features_2d)
                prob = self._model.predict_proba(X_scaled)[0]
            else:
                prob = self._model.predict_proba(features_2d)[0]

            is_correct = bool(prob[1] >= 0.5)
            confidence = float(max(prob))
        else:
            # Fallback: threshold on depth_score
            ds = depth_result.depth_score
            is_correct = ds >= self._threshold
            dist = abs(ds - self._threshold)
            confidence = min(1.0, 0.5 + dist * 2.0)

        signal = "shallow_recall" if is_correct else "deep_uncertain"

        return {
            "is_correct": is_correct,
            "confidence": round(confidence, 4),
            "depth_signal": signal,
        }

    def predict_proba(self, depth_result) -> float:
        """Probability that the response is correct (0-1).

        Args:
            depth_result: DepthResult from DepthScorer.score()

        Returns:
            float: P(correct) in [0, 1]
        """
        result = self.predict(depth_result)
        if result["is_correct"]:
            return result["confidence"]
        else:
            return 1.0 - result["confidence"]

    def save(self, path: str):
        """Pickle the trained predictor."""
        state = {
            "model": self._model,
            "scaler": self._scaler,
            "threshold": self._threshold,
            "is_trained": self._is_trained,
            "n_training_samples": self._n_training_samples,
            "use_sklearn": self._use_sklearn,
            "direction_note": DIRECTION_NOTE,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "CorrectnessPredictor":
        """Load a pickled predictor."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        pred = cls()
        pred._model = state["model"]
        pred._scaler = state.get("scaler")
        pred._threshold = state["threshold"]
        pred._is_trained = state["is_trained"]
        pred._n_training_samples = state.get("n_training_samples", 0)
        pred._use_sklearn = state.get("use_sklearn", HAS_SKLEARN)
        return pred

    def status(self) -> dict:
        """Return predictor status for API."""
        return {
            "is_trained": self._is_trained,
            "threshold": self._threshold,
            "n_training_samples": self._n_training_samples,
            "backend": "sklearn" if self._use_sklearn else "threshold",
            "direction": DIRECTION_NOTE,
        }
