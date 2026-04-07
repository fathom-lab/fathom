"""
FATHOM LEADERBOARD — Benchmark Runner
=======================================

Runs the Fathom benchmark on a model and updates leaderboard/scores.json.

Usage:
    python fathom_leaderboard.py                       # Score default model (gemma-2-2b)
    python fathom_leaderboard.py --model google/gemma-2-2b
    python fathom_leaderboard.py --from-results        # Import from results_power_replication.json
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
LEADERBOARD_DIR = PROJECT_ROOT / "leaderboard"
SCORES_PATH = LEADERBOARD_DIR / "scores.json"


def load_scores() -> dict:
    """Load existing leaderboard scores."""
    if SCORES_PATH.exists():
        return json.loads(SCORES_PATH.read_text(encoding="utf-8"))
    return {
        "benchmark_version": "1.0",
        "benchmark_pairs": 30,
        "benchmark_source": "Power replication set (p=0.000051)",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "models": [],
    }


def save_scores(data: dict):
    """Save leaderboard scores."""
    LEADERBOARD_DIR.mkdir(exist_ok=True)
    data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    SCORES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_benchmark_pairs() -> list:
    """Load the 30 paper pairs from reproduce_paper.py."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from reproduce_paper import PAIRS
    return PAIRS


def grade_from_depth(depth_score: float) -> tuple:
    """Convert depth score to grade and fathom_score."""
    # Based on gemma-2-2b calibration (26 layers)
    # Lower depth = deeper reasoning
    if depth_score <= 7.5:
        return "A+", 90
    elif depth_score <= 7.8:
        return "A", 82
    elif depth_score <= 8.0:
        return "B+", 74
    elif depth_score <= 8.2:
        return "B", 66
    elif depth_score <= 8.5:
        return "C", 55
    elif depth_score <= 9.0:
        return "D", 40
    else:
        return "F", 25


def import_from_results():
    """Import gemma-2-2b results from results_power_replication.json."""
    results_path = PROJECT_ROOT / "results_power_replication.json"
    if not results_path.exists():
        print("[ERROR] results_power_replication.json not found")
        return

    results = json.loads(results_path.read_text(encoding="utf-8"))

    surface_mean = results["surface_mean_layer"]
    insight_mean = results["insight_mean_layer"]
    delta = results["mean_delta"]
    pct = results["pct_surface_deeper"]

    grade, fathom_score = grade_from_depth(insight_mean)

    entry = {
        "model_name": "google/gemma-2-2b",
        "display_name": "Gemma 2 2B",
        "provider": "Google",
        "params": "2B",
        "mean_depth_score": round(surface_mean, 3),
        "mean_insight_depth": round(insight_mean, 3),
        "benchmark_delta": round(delta, 3),
        "pct_directional": round(pct, 1),
        "grade": grade,
        "fathom_score": fathom_score,
        "tested_date": datetime.now().strftime("%Y-%m-%d"),
        "status": "verified",
        "notes": "Primary model. Paper core finding.",
    }

    scores = load_scores()

    # Update or add
    existing_idx = next(
        (i for i, m in enumerate(scores["models"])
         if m["model_name"] == entry["model_name"]),
        None
    )
    if existing_idx is not None:
        scores["models"][existing_idx] = entry
    else:
        scores["models"].append(entry)

    save_scores(scores)
    print(f"[OK] Imported {entry['display_name']}: delta={delta:+.3f}, grade={grade}")


def run_benchmark(model_name: str, transcoder_set: str = "gemma", max_features: int = 150):
    """Run the full benchmark on a model."""
    import torch
    from depth_scorer import DepthScorer

    pairs = get_benchmark_pairs()
    print(f"[Benchmark] Scoring {len(pairs)} pairs on {model_name}...")

    scorer = DepthScorer(
        model_name=model_name,
        model_type=transcoder_set,
        max_feature_nodes=max_features,
    )

    surface_depths = []
    insight_depths = []

    for i, pair in enumerate(pairs):
        try:
            r_s = scorer.score(pair["surface"], target=pair.get("surface_target"))
            torch.cuda.empty_cache()
            gc.collect()

            r_i = scorer.score(pair["insight"], target=pair.get("insight_target"))
            torch.cuda.empty_cache()
            gc.collect()

            if not math.isnan(r_s.depth_score) and not math.isnan(r_i.depth_score):
                surface_depths.append(r_s.depth_score)
                insight_depths.append(r_i.depth_score)
                d = r_s.depth_score - r_i.depth_score
                print(f"  [{i+1}/{len(pairs)}] {pair['topic']:<22} "
                      f"s={r_s.depth_score:.3f} i={r_i.depth_score:.3f} d={d:+.3f}")
        except Exception as e:
            print(f"  [{i+1}/{len(pairs)}] {pair['topic']:<22} FAILED: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    del scorer
    torch.cuda.empty_cache()
    gc.collect()

    if not surface_depths:
        print("[ERROR] No valid pairs scored")
        return

    s_mean = float(np.mean(surface_depths))
    i_mean = float(np.mean(insight_depths))
    delta = s_mean - i_mean
    pct = sum(1 for s, i in zip(surface_depths, insight_depths) if s > i) / len(surface_depths) * 100

    grade, fathom_score = grade_from_depth(i_mean)

    # Determine provider / display name
    provider = model_name.split("/")[0] if "/" in model_name else "Unknown"
    display_name = model_name.split("/")[-1] if "/" in model_name else model_name

    entry = {
        "model_name": model_name,
        "display_name": display_name,
        "provider": provider,
        "params": "—",
        "mean_depth_score": round(s_mean, 3),
        "mean_insight_depth": round(i_mean, 3),
        "benchmark_delta": round(delta, 3),
        "pct_directional": round(pct, 1),
        "grade": grade,
        "fathom_score": fathom_score,
        "tested_date": datetime.now().strftime("%Y-%m-%d"),
        "status": "verified",
        "notes": f"Benchmark run. {len(surface_depths)}/{len(pairs)} pairs valid.",
    }

    scores = load_scores()
    existing_idx = next(
        (i for i, m in enumerate(scores["models"])
         if m["model_name"] == entry["model_name"]),
        None
    )
    if existing_idx is not None:
        scores["models"][existing_idx] = entry
    else:
        scores["models"].append(entry)

    save_scores(scores)

    print(f"\n[RESULT] {display_name}")
    print(f"  Surface mean: {s_mean:.3f}")
    print(f"  Insight mean: {i_mean:.3f}")
    print(f"  Delta: {delta:+.3f}")
    print(f"  Direction: {pct:.1f}%")
    print(f"  Grade: {grade} ({fathom_score}/100)")
    print(f"  Saved to {SCORES_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Fathom Benchmark Leaderboard Runner")
    parser.add_argument("--model", default="google/gemma-2-2b", help="Model to benchmark")
    parser.add_argument("--transcoder", default="gemma", help="Transcoder set name")
    parser.add_argument("--max-features", type=int, default=150, help="Max feature nodes")
    parser.add_argument("--from-results", action="store_true",
                        help="Import from results_power_replication.json")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    if args.from_results:
        import_from_results()
    else:
        run_benchmark(args.model, args.transcoder, args.max_features)


if __name__ == "__main__":
    main()
