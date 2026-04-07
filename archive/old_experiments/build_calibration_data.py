#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BUILD CALIBRATION DATASET — Joins Prompts + White-Box Scores + Text Features
==============================================================================

Combines data from:
  - experiment_12_power_pairs.json (30 surface/insight pair texts)
  - results_power_replication.json (Gemma-2-2B depth scores, 30 pairs)
  - llama_experiment_results.json (Llama-3.2-1B depth scores, 19 pairs)

Extracts text features for each prompt and outputs a unified JSON
for training the black-box depth probe.

Usage:
    python build_calibration_data.py
"""

import json
import sys
from pathlib import Path

from text_features import extract_features, features_to_vector, feature_names

BASE_DIR = Path(__file__).parent
PAIRS_FILE = BASE_DIR / ".." / "research" / "experiment_12_power_pairs.json"
GEMMA_RESULTS = BASE_DIR / "results_power_replication.json"
LLAMA_RESULTS = BASE_DIR / "llama_experiment_results.json"
OUTPUT_FILE = BASE_DIR / "calibration_dataset.json"


def load_pairs():
    """Load the 30 surface/insight pair texts."""
    for candidate in [PAIRS_FILE, BASE_DIR / "experiment_12_power_pairs.json"]:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            if isinstance(data, dict) and "pairs" in data:
                return data["pairs"]
            return data
    raise FileNotFoundError(f"Pairs file not found")


def load_gemma_results():
    """Load Gemma-2-2B depth scores keyed by topic."""
    if not GEMMA_RESULTS.exists():
        print(f"[WARN] Gemma results not found: {GEMMA_RESULTS}")
        return {}
    with open(GEMMA_RESULTS) as f:
        data = json.load(f)
    results = {}
    for pair in data.get("pairs", []):
        results[pair["topic"]] = {
            "surface_depth": pair["surface_mean_layer"],
            "insight_depth": pair["insight_mean_layer"],
            "surface_features": pair.get("surface_n"),
            "insight_features": pair.get("insight_n"),
        }
    return results


def load_llama_results():
    """Load Llama-3.2-1B depth scores keyed by pair_id."""
    if not LLAMA_RESULTS.exists():
        print(f"[WARN] Llama results not found: {LLAMA_RESULTS}")
        return {}
    with open(LLAMA_RESULTS) as f:
        data = json.load(f)
    results = {}
    for pair in data:
        results[pair["pair_id"]] = {
            "surface_depth": pair["surface_depth"],
            "insight_depth": pair["insight_depth"],
            "surface_entropy": pair.get("surface_entropy"),
            "insight_entropy": pair.get("insight_entropy"),
            "surface_features": pair.get("surface_features"),
            "insight_features": pair.get("insight_features"),
        }
    return results


def normalize_depth(raw_depth, n_layers):
    """Normalize depth to 0-1 scale. Higher = deeper reasoning (lower layer index)."""
    return round(1.0 - raw_depth / n_layers, 6)


def main():
    print("=" * 60)
    print("  BUILD CALIBRATION DATASET")
    print("=" * 60)

    # Load data
    pairs = load_pairs()
    gemma = load_gemma_results()
    llama = load_llama_results()

    print(f"[DATA] Pairs: {len(pairs)}")
    print(f"[DATA] Gemma results: {len(gemma)} topics")
    print(f"[DATA] Llama results: {len(llama)} topics")

    dataset = []
    fn = feature_names()

    for pair in pairs:
        topic = pair.get("topic", pair.get("id", "unknown"))
        surface_text = pair.get("surface", "")
        insight_text = pair.get("insight", "")
        surface_target = pair.get("surface_target", "")
        insight_target = pair.get("insight_target", "")

        # Handle nested format
        if isinstance(surface_text, dict):
            surface_text = surface_text.get("prompt", "")
            surface_target = surface_text.get("target", surface_target)
        if isinstance(insight_text, dict):
            insight_text = insight_text.get("prompt", "")
            insight_target = insight_text.get("target", insight_target)

        # Extract features for both prompts
        surface_features = extract_features(surface_text)
        insight_features = extract_features(insight_text)
        surface_vec = features_to_vector(surface_features).tolist()
        insight_vec = features_to_vector(insight_features).tolist()

        # ── Gemma data (26 layers) ───────────────────────────────────
        if topic in gemma:
            g = gemma[topic]
            dataset.append({
                "text": surface_text,
                "target": surface_target,
                "prompt_type": "surface",
                "topic": topic,
                "features": surface_features.to_dict(),
                "feature_vector": surface_vec,
                "white_box_depth_raw": g["surface_depth"],
                "white_box_depth_normalized": normalize_depth(g["surface_depth"], 26),
                "feature_count": g.get("surface_features"),
                "model": "google/gemma-2-2b",
                "n_layers": 26,
                "architecture": "gemma",
            })
            dataset.append({
                "text": insight_text,
                "target": insight_target,
                "prompt_type": "insight",
                "topic": topic,
                "features": insight_features.to_dict(),
                "feature_vector": insight_vec,
                "white_box_depth_raw": g["insight_depth"],
                "white_box_depth_normalized": normalize_depth(g["insight_depth"], 26),
                "feature_count": g.get("insight_features"),
                "model": "google/gemma-2-2b",
                "n_layers": 26,
                "architecture": "gemma",
            })

        # ── Llama data (16 layers) ───────────────────────────────────
        if topic in llama:
            l = llama[topic]
            dataset.append({
                "text": surface_text,
                "target": surface_target,
                "prompt_type": "surface",
                "topic": topic,
                "features": surface_features.to_dict(),
                "feature_vector": surface_vec,
                "white_box_depth_raw": l["surface_depth"],
                "white_box_depth_normalized": normalize_depth(l["surface_depth"], 16),
                "white_box_entropy": l.get("surface_entropy"),
                "feature_count": l.get("surface_features"),
                "model": "meta-llama/Llama-3.2-1B",
                "n_layers": 16,
                "architecture": "llama",
            })
            dataset.append({
                "text": insight_text,
                "target": insight_target,
                "prompt_type": "insight",
                "topic": topic,
                "features": insight_features.to_dict(),
                "feature_vector": insight_vec,
                "white_box_depth_raw": l["insight_depth"],
                "white_box_depth_normalized": normalize_depth(l["insight_depth"], 16),
                "white_box_entropy": l.get("insight_entropy"),
                "feature_count": l.get("insight_features"),
                "model": "meta-llama/Llama-3.2-1B",
                "n_layers": 16,
                "architecture": "llama",
            })

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    # Summary
    n_gemma = sum(1 for d in dataset if d["architecture"] == "gemma")
    n_llama = sum(1 for d in dataset if d["architecture"] == "llama")
    n_surface = sum(1 for d in dataset if d["prompt_type"] == "surface")
    n_insight = sum(1 for d in dataset if d["prompt_type"] == "insight")

    print(f"\n[OUTPUT] {len(dataset)} calibration rows saved to {OUTPUT_FILE}")
    print(f"  Gemma: {n_gemma} | Llama: {n_llama}")
    print(f"  Surface: {n_surface} | Insight: {n_insight}")

    # Quick feature sanity check
    if dataset:
        print(f"\n[SANITY] First row features:")
        for k, v in dataset[0]["features"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
