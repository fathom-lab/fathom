#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPTH PROBE — Black-Box Depth Estimation via Text Features
=============================================================

Trains a lightweight regression model to predict Fathom depth scores
from text features alone. Calibrated against white-box measurements
on Gemma-2-2B and Llama-3.2-1B.

Usage:
    python depth_probe.py                    # Train and evaluate
    python depth_probe.py --calibration calibration_dataset.json

Requirements:
    pip install scikit-learn joblib numpy scipy
"""

import json
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
from scipy import stats

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import joblib
except ImportError:
    print("[ERROR] scikit-learn and joblib required: pip install scikit-learn joblib")
    sys.exit(1)

from text_features import extract_features, features_to_vector, feature_names, TextFeatures

BASE_DIR = Path(__file__).parent
DEFAULT_CALIBRATION = BASE_DIR / "calibration_dataset.json"
DEFAULT_MODEL_PATH = BASE_DIR / "depth_probe.pkl"


@dataclass
class ProbeResult:
    """Evaluation results from probe training."""
    pearson_r: float
    pearson_p: float
    mae: float
    rmse: float
    n_samples: int
    n_features: int
    per_architecture: Dict[str, Dict]
    feature_importances: Dict[str, float]
    model_type: str


class DepthProbe:
    """Lightweight depth estimation probe trained on text features."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._scaler = None
        self._feature_names = None
        self._training_mean = None
        self._training_cov_inv = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, calibration_path: Path = DEFAULT_CALIBRATION) -> ProbeResult:
        """Train probe with Leave-One-Topic-Out cross-validation."""
        # Load calibration data
        with open(calibration_path) as f:
            dataset = json.load(f)

        print(f"[PROBE] Loaded {len(dataset)} calibration rows")

        # Extract features and targets
        # Add architecture as one-hot features to handle scale differences
        base_X = np.array([row["feature_vector"] for row in dataset])
        arch_gemma = np.array([1.0 if row["architecture"] == "gemma" else 0.0 for row in dataset]).reshape(-1, 1)
        arch_llama = np.array([1.0 if row["architecture"] == "llama" else 0.0 for row in dataset]).reshape(-1, 1)
        X = np.hstack([base_X, arch_gemma, arch_llama])

        y = np.array([row["white_box_depth_normalized"] for row in dataset])
        topics = [row["topic"] for row in dataset]
        archs = [row["architecture"] for row in dataset]
        self._feature_names = feature_names() + ["arch_gemma", "arch_llama"]

        n_samples, n_features = X.shape
        print(f"[PROBE] Features: {n_features}, Samples: {n_samples}")

        # ── Leave-One-Topic-Out Cross-Validation ─────────────────────
        unique_topics = sorted(set(topics))
        all_preds = np.zeros(n_samples)
        all_actual = np.zeros(n_samples)

        for held_out_topic in unique_topics:
            # Split
            train_mask = np.array([t != held_out_topic for t in topics])
            test_mask = ~train_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(X_test) == 0:
                continue

            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Train Ridge
            model = RidgeCV(alphas=np.logspace(-3, 3, 50))
            model.fit(X_train_s, y_train)

            preds = model.predict(X_test_s)
            all_preds[test_mask] = preds
            all_actual[test_mask] = y_test

        # ── Compute Metrics ──────────────────────────────────────────
        # Filter to only rows that were actually predicted
        mask = all_actual != 0  # all normalized depths should be non-zero
        # Actually, some could be near zero. Use a different approach:
        # Since we predict all rows, just use all
        r, p = stats.pearsonr(all_actual, all_preds)
        mae = mean_absolute_error(all_actual, all_preds)
        rmse = float(np.sqrt(mean_squared_error(all_actual, all_preds)))

        print(f"\n[LOTO CV RESULTS]")
        print(f"  Pearson r = {r:.4f} (p = {p:.6f})")
        print(f"  MAE       = {mae:.4f}")
        print(f"  RMSE      = {rmse:.4f}")

        # ── Per-Architecture Breakdown ───────────────────────────────
        per_arch = {}
        for arch in set(archs):
            arch_mask = np.array([a == arch for a in archs])
            if arch_mask.sum() < 3:
                continue
            r_arch, _ = stats.pearsonr(all_actual[arch_mask], all_preds[arch_mask])
            mae_arch = mean_absolute_error(all_actual[arch_mask], all_preds[arch_mask])
            per_arch[arch] = {
                "pearson_r": round(r_arch, 4),
                "mae": round(mae_arch, 4),
                "n": int(arch_mask.sum()),
            }
            print(f"  {arch}: r={r_arch:.4f}, MAE={mae_arch:.4f}, n={arch_mask.sum()}")

        # ── Train Final Model on All Data ────────────────────────────
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = RidgeCV(alphas=np.logspace(-3, 3, 50))
        self._model.fit(X_scaled, y)

        # Store training distribution for confidence estimation
        self._training_mean = X_scaled.mean(axis=0)
        try:
            cov = np.cov(X_scaled.T)
            self._training_cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(n_features))
        except np.linalg.LinAlgError:
            self._training_cov_inv = np.eye(n_features)

        # Feature importances (Ridge coefficients)
        coefficients = self._model.coef_
        importances = dict(zip(self._feature_names, [round(float(c), 4) for c in coefficients]))

        print(f"\n[FEATURE IMPORTANCES] (Ridge coefficients)")
        sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, coef in sorted_imp:
            bar = "+" * int(abs(coef) * 50) if coef > 0 else "-" * int(abs(coef) * 50)
            print(f"  {name:30s} {coef:+.4f} {bar}")

        print(f"\n[MODEL] Alpha selected: {self._model.alpha_:.4f}")

        # Check if we should try GBR
        result_model_type = "ridge"
        if r < 0.25:
            print(f"\n[WARN] Ridge r={r:.4f} < 0.25 — trying GradientBoosting...")
            gbr = GradientBoostingRegressor(
                n_estimators=50, max_depth=2, min_samples_leaf=10,
                learning_rate=0.1, random_state=42,
            )
            # Re-run LOTO with GBR
            gbr_preds = np.zeros(n_samples)
            for held_out_topic in unique_topics:
                train_mask = np.array([t != held_out_topic for t in topics])
                test_mask = ~train_mask
                if test_mask.sum() == 0:
                    continue
                scaler = StandardScaler()
                X_ts = scaler.fit_transform(X[train_mask])
                X_vs = scaler.transform(X[test_mask])
                gbr_tmp = GradientBoostingRegressor(
                    n_estimators=50, max_depth=2, min_samples_leaf=10,
                    learning_rate=0.1, random_state=42,
                )
                gbr_tmp.fit(X_ts, y[train_mask])
                gbr_preds[test_mask] = gbr_tmp.predict(X_vs)

            r_gbr, p_gbr = stats.pearsonr(all_actual, gbr_preds)
            print(f"  GBR Pearson r = {r_gbr:.4f}")

            if r_gbr > r:
                print(f"  GBR wins. Retraining on full data...")
                self._model = gbr
                self._model.fit(X_scaled, y)
                r, p = r_gbr, p_gbr
                mae = mean_absolute_error(all_actual, gbr_preds)
                rmse = float(np.sqrt(mean_squared_error(all_actual, gbr_preds)))
                result_model_type = "gradient_boosting"

                # GBR feature importances
                importances = dict(zip(self._feature_names, [round(float(c), 4) for c in gbr.feature_importances_]))

        return ProbeResult(
            pearson_r=round(r, 4),
            pearson_p=round(p, 6),
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            n_samples=n_samples,
            n_features=n_features,
            per_architecture=per_arch,
            feature_importances=importances,
            model_type=result_model_type,
        )

    def _add_arch_features(self, vec: np.ndarray, architecture: str = "gemma") -> np.ndarray:
        """Append architecture one-hot features to feature vector."""
        arch_gemma = 1.0 if architecture == "gemma" else 0.0
        arch_llama = 1.0 if architecture == "llama" else 0.0
        return np.hstack([vec, [[arch_gemma, arch_llama]]])

    def predict(self, text: str, architecture: str = "gemma") -> float:
        """Predict normalized depth (0-1) from text."""
        if self._model is None:
            raise RuntimeError("Probe not trained. Call train() or load() first.")

        features = extract_features(text)
        vec = features_to_vector(features).reshape(1, -1)
        vec = self._add_arch_features(vec, architecture)
        vec_scaled = self._scaler.transform(vec)

        return float(self._model.predict(vec_scaled)[0])

    def predict_with_confidence(self, text: str, architecture: str = "gemma") -> tuple:
        """Predict depth and confidence score."""
        if self._model is None:
            raise RuntimeError("Probe not trained. Call train() or load() first.")

        features = extract_features(text)
        vec = features_to_vector(features).reshape(1, -1)
        vec = self._add_arch_features(vec, architecture)
        vec_scaled = self._scaler.transform(vec)

        predicted = float(self._model.predict(vec_scaled)[0])

        # Mahalanobis distance for confidence
        diff = vec_scaled[0] - self._training_mean
        if self._training_cov_inv is not None:
            mahal = float(np.sqrt(diff @ self._training_cov_inv @ diff))
        else:
            mahal = float(np.sqrt(np.sum(diff ** 2)))

        # Convert to 0-1 confidence (higher = more in-distribution)
        # Use sigmoid-like decay: close to training data = high confidence
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + mahal / 3.0)))

        return predicted, confidence, features

    def save(self, path: Path = DEFAULT_MODEL_PATH):
        """Save trained probe to disk."""
        payload = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "training_mean": self._training_mean,
            "training_cov_inv": self._training_cov_inv,
            "version": "1.0",
        }
        joblib.dump(payload, path)
        print(f"[SAVED] Probe -> {path}")

    def load(self, path: Path = DEFAULT_MODEL_PATH):
        """Load trained probe from disk."""
        payload = joblib.load(path)
        self._model = payload["model"]
        self._scaler = payload["scaler"]
        self._feature_names = payload["feature_names"]
        self._training_mean = payload.get("training_mean")
        self._training_cov_inv = payload.get("training_cov_inv")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Fathom Depth Probe")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    if not args.calibration.exists():
        print(f"[ERROR] Calibration file not found: {args.calibration}")
        print("Run build_calibration_data.py first.")
        sys.exit(1)

    print("=" * 60)
    print("  FATHOM DEPTH PROBE — Training")
    print("=" * 60)

    probe = DepthProbe()
    result = probe.train(args.calibration)

    # Save
    probe.save(args.output)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  PROBE TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model: {result.model_type}")
    print(f"  Pearson r = {result.pearson_r:.4f} (p = {result.pearson_p:.6f})")
    print(f"  MAE = {result.mae:.4f}")
    print(f"  RMSE = {result.rmse:.4f}")
    print(f"  Samples = {result.n_samples}")

    if result.pearson_r > 0.5:
        print(f"\n  *** STRONG CORRELATION — probe is viable ***")
    elif result.pearson_r > 0.3:
        print(f"\n  ** MODERATE CORRELATION — probe shows signal **")
    elif result.pearson_r > 0.0:
        print(f"\n  * WEAK CORRELATION — needs more data *")
    else:
        print(f"\n  NO CORRELATION — text features insufficient")

    # Quick demo predictions
    print(f"\n[DEMO] Quick predictions:")
    demos = [
        ("The patient has a fever and feels warm", "surface"),
        ("Fever triggers prostaglandin synthesis in the hypothalamus", "insight"),
        ("The stock market went down today", "surface"),
        ("Yield curve inversions predict recessions by signaling credit contraction", "insight"),
    ]
    for text, expected in demos:
        pred, conf, _ = probe.predict_with_confidence(text)
        print(f"  [{expected:7s}] depth={pred:.4f} conf={conf:.2f} | {text[:60]}")


if __name__ == "__main__":
    main()
