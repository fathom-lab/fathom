#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
"""
SCORE REMAINING PAIRS — Bulletproof VRAM-safe scorer
=====================================================
Scores all pairs that are missing base and/or instruct results.
Saves checkpoint after EVERY SINGLE PAIR. OOM-safe.

Usage:
    python score_remaining_pairs.py --model base       # score missing base pairs
    python score_remaining_pairs.py --model instruct   # score missing instruct pairs
    python score_remaining_pairs.py --model both       # score everything needed
    python score_remaining_pairs.py --analyze           # just run analysis on what we have

VRAM-safe: max_features=100, batch_size=4, gc after every pair.
Run base and instruct in SEPARATE invocations to avoid VRAM fragmentation.
"""

import json
import gc
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
CHECKPOINT = BASE_DIR / "scoring_checkpoint.json"

# Import the pair loader and analysis from the main experiment
from experiment_selective_amplification import (
    load_all_pairs, load_existing_results,
    analyze_selective_amplification, generate_report,
    ANALYSIS_FILE, REPORT_FILE,
)

# ── JSON helpers ─────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"base": {}, "instruct": {}}


def save_checkpoint(data):
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_one_pair(model, prompt, target, max_features=100, batch_size=4):
    """Score a single prompt+target. Returns depth dict or None."""
    import torch
    from circuit_tracer import attribute

    full_text = prompt + target

    try:
        graph = attribute(
            full_text,
            model,
            attribution_targets=[target],
            max_feature_nodes=max_features,
            batch_size=batch_size,
        )

        layers = graph.active_features[:, 0].float().cpu().numpy()
        if len(layers) == 0:
            return None

        depth = float(np.mean(layers))
        n_features = len(layers)

        layer_counts = {}
        for l in layers:
            k = int(l)
            layer_counts[k] = layer_counts.get(k, 0) + 1

        probs = np.array(list(layer_counts.values())) / n_features
        entropy = -float(np.sum(probs * np.log2(probs + 1e-10)))

        return {
            "depth": depth,
            "n_features": n_features,
            "entropy": entropy,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"OOM! Clearing cache and retrying with max_features=50...")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                graph = attribute(
                    full_text, model,
                    attribution_targets=[target],
                    max_feature_nodes=50,
                    batch_size=2,
                )
                layers = graph.active_features[:, 0].float().cpu().numpy()
                if len(layers) == 0:
                    return None
                return {
                    "depth": float(np.mean(layers)),
                    "n_features": len(layers),
                    "entropy": 0.0,
                }
            except Exception:
                return None
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def score_missing(model_type, pairs, base_results, inst_results):
    """Score all pairs missing results for the given model type."""
    import torch
    from circuit_tracer import ReplacementModel

    if model_type == "base":
        model_name = "google/gemma-2-2b"
        existing = base_results
        needed = [p for p in pairs if p["id"] not in existing]
    else:
        model_name = "google/gemma-2-2b-it"
        existing = inst_results
        # Need instruct for pairs that have base scores
        needed = [p for p in pairs if p["id"] in base_results and p["id"] not in existing]

    if not needed:
        print(f"[{model_type.upper()}] Nothing to score!")
        return {}

    print(f"\n[{model_type.upper()}] {len(needed)} pairs to score on {model_name}")

    # Load checkpoint
    ckpt = load_checkpoint()
    ckpt_key = model_type
    if ckpt_key not in ckpt:
        ckpt[ckpt_key] = {}

    # Skip already checkpointed
    to_score = [p for p in needed if p["id"] not in ckpt[ckpt_key]]
    if len(to_score) < len(needed):
        print(f"  Resuming: {len(needed) - len(to_score)} already in checkpoint")

    if not to_score:
        print("  All pairs already checkpointed!")
        return ckpt[ckpt_key]

    # Clear VRAM
    torch.cuda.empty_cache()
    gc.collect()

    print(f"  Loading {model_name}...")
    model = ReplacementModel.from_pretrained(
        model_name, "gemma",
        dtype=torch.bfloat16,
        backend="transformerlens",
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded. VRAM: {vram:.2f} GB")

    results = dict(ckpt[ckpt_key])  # start from checkpoint
    n = len(to_score)

    for i, pair in enumerate(to_score):
        pid = pair["id"]
        sp = pair["surface"]["prompt"]
        st = pair["surface"]["target"]
        ip = pair["insight"]["prompt"]
        it = pair["insight"]["target"]

        print(f"  [{i+1}/{n}] {pid}", end=" ", flush=True)

        t0 = time.time()
        sr = score_one_pair(model, sp, st)
        ir = score_one_pair(model, ip, it)
        elapsed = time.time() - t0

        # Clear after every pair
        torch.cuda.empty_cache()
        gc.collect()

        if sr is None or ir is None:
            print(f"SKIP (attribution failed) [{elapsed:.1f}s]")
            continue

        delta = sr["depth"] - ir["depth"]
        result = {
            "surface_depth": sr["depth"],
            "insight_depth": ir["depth"],
            "delta": delta,
            "surface_n": sr["n_features"],
            "insight_n": ir["n_features"],
        }

        results[pid] = result
        direction = "S>I" if delta > 0 else "I>S"
        print(f"d={delta:+.3f} ({direction}) [{elapsed:.1f}s]")

        # Save checkpoint after EVERY pair
        ckpt[ckpt_key] = results
        save_checkpoint(ckpt)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def merge_and_analyze(pairs, base_results, inst_results):
    """Merge checkpoint data and run analysis."""
    ckpt = load_checkpoint()

    # Merge checkpoint into results
    for topic, data in ckpt.get("base", {}).items():
        if topic not in base_results:
            base_results[topic] = data

    for topic, data in ckpt.get("instruct", {}).items():
        if topic not in inst_results:
            inst_results[topic] = data

    # Save merged results
    merged_file = BASE_DIR / "selective_amplification_results.json"
    with open(merged_file, "w") as f:
        json.dump({"base": base_results, "instruct": inst_results}, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Merged results -> {merged_file}")

    labeled = [p for p in pairs if p["sensitivity"] != "unknown"]
    matched = [p for p in labeled if p["id"] in base_results and p["id"] in inst_results]
    print(f"\n[MATCHED] {len(matched)} pairs with both base and instruct scores")

    # Run analysis
    analysis = analyze_selective_amplification(labeled, base_results, inst_results)

    if analysis:
        with open(ANALYSIS_FILE, "w") as f:
            json.dump(analysis, f, indent=2, cls=NumpyEncoder)

        report = generate_report(analysis)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)

        pt = analysis["primary_test"]
        gs = analysis["group_stats"]

        print(f"\n{'=' * 60}")
        print(f"  SELECTIVE AMPLIFICATION RESULTS (n={analysis['n_matched']})")
        print(f"{'=' * 60}")
        for sens in ["high", "medium", "low"]:
            if sens in gs:
                g = gs[sens]
                d = "AMPLIFIED" if g["mean_change"] > 0 else "COMPRESSED"
                print(f"  {sens.upper():>8}: {g['mean_change']:+.4f} (n={g['n']}) {d}")
        print(f"\n  p-value:  {pt['p_value']:.6f}")
        print(f"  Cohen's d: {pt['cohens_d']:.4f}")
        print(f"  VERDICT: {analysis['verdict']}")
        print(f"{'=' * 60}")

        if pt["significant"]:
            print("\n  *** SELECTIVE AMPLIFICATION CONFIRMED ***")
            print("  *** RLHF REDISTRIBUTES DEPTH TOWARD SAFETY DOMAINS ***")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score remaining pairs")
    parser.add_argument("--model", choices=["base", "instruct", "both"], default="both")
    parser.add_argument("--analyze", action="store_true", help="Just analyze, no scoring")
    args = parser.parse_args()

    print("=" * 60)
    print("  SCORE REMAINING PAIRS")
    print("  VRAM-safe, checkpoint-per-pair")
    print("=" * 60)

    pairs = load_all_pairs()
    base_results, inst_results = load_existing_results()
    labeled = [p for p in pairs if p["sensitivity"] != "unknown"]

    print(f"\nLabeled pairs: {len(labeled)}")
    print(f"Base scores: {len(base_results)}")
    print(f"Instruct scores: {len(inst_results)}")
    matched = sum(1 for p in labeled if p["id"] in base_results and p["id"] in inst_results)
    print(f"Currently matched: {matched}")

    if args.analyze:
        merge_and_analyze(labeled, base_results, inst_results)
        return

    if args.model in ("base", "both"):
        new_base = score_missing("base", labeled, base_results, inst_results)
        base_results.update(new_base)

    if args.model in ("instruct", "both"):
        new_inst = score_missing("instruct", labeled, base_results, inst_results)
        inst_results.update(new_inst)

    # Final analysis
    merge_and_analyze(labeled, base_results, inst_results)


if __name__ == "__main__":
    main()
