#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score new E11 pairs on gemma-2-2b-it (instruct model) for selective amplification experiment.

Checkpoint after every pair. OOM retry at lower max_features.
Safe for 8GB VRAM with max_features=100, batch_size=4.

Usage:
    python score_new_instruct_pairs.py           # score all E11 pairs
    python score_new_instruct_pairs.py --smoke   # score first 2 pairs only
    python score_new_instruct_pairs.py --resume  # skip already-checkpointed pairs
"""

import gc
import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RESEARCH_DIR = BASE_DIR.parent / "research"

E11_PAIRS_FILE = RESEARCH_DIR / "experiment_11_depth_pairs.json"
CHECKPOINT_FILE = BASE_DIR / "new_instruct_checkpoint.json"
OUTPUT_FILE = BASE_DIR / "new_instruct_results.json"

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
MAX_FEATURES_DEFAULT = 100   # main VRAM lever (default was 500)
MAX_FEATURES_FALLBACK = 50   # fallback if OOM
BATCH_SIZE = 4               # lower than default 16


def clear_vram():
    """Aggressively free VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_gb():
    """Return current VRAM usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def load_e11_pairs():
    """Load experiment_11 pairs from JSON."""
    raw = json.loads(E11_PAIRS_FILE.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "pairs" in raw:
        return raw["pairs"]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Unexpected E11 format: {type(raw)}")


def load_checkpoint():
    """Load pairs already scored in a previous run."""
    if not CHECKPOINT_FILE.exists():
        return {}
    data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {r["pair_id"]: r for r in data}


def save_checkpoint(results_dict):
    """Save checkpoint (dict of pair_id -> result)."""
    records = list(results_dict.values())
    CHECKPOINT_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def score_pair(model, pair, max_features):
    """Score one surface/insight pair. Returns (surface_depth, insight_depth, delta)."""
    from circuit_tracer import attribute

    surface_prompt = pair["surface"]
    surface_target = pair["surface_target"]
    insight_prompt = pair["insight"]
    insight_target = pair["insight_target"]

    # Validate single-token targets
    def is_single_token(text):
        try:
            ids = model.tokenizer.encode(text, add_special_tokens=False)
            return len(ids) == 1
        except Exception:
            return False

    if not is_single_token(surface_target):
        raise ValueError(f"Multi-token target: '{surface_target}'")
    if not is_single_token(insight_target):
        raise ValueError(f"Multi-token target: '{insight_target}'")

    depths = []
    for prompt, target in [(surface_prompt, surface_target), (insight_prompt, insight_target)]:
        clear_vram()
        graph = attribute(
            prompt + " " + target,
            model,
            attribution_targets=[target],
            max_feature_nodes=max_features,
            batch_size=BATCH_SIZE,
        )
        layers = graph.active_features[:, 0].float().cpu().numpy()
        depth = float(np.mean(layers))
        depths.append(depth)
        del graph
        clear_vram()

    surface_depth, insight_depth = depths
    delta = surface_depth - insight_depth
    return surface_depth, insight_depth, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Score only first 2 pairs")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already-checkpointed pairs")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    print("=" * 60)
    print("SELECTIVE AMPLIFICATION — Score New Instruct Pairs")
    print(f"Model: {MODEL_NAME}")
    print(f"max_features: {MAX_FEATURES_DEFAULT} (fallback: {MAX_FEATURES_FALLBACK})")
    print(f"batch_size: {BATCH_SIZE}")
    print("=" * 60)

    # Load pairs
    all_pairs = load_e11_pairs()
    if args.smoke:
        all_pairs = all_pairs[:2]
        print(f"[SMOKE] Running 2 pairs only")

    print(f"\nLoaded {len(all_pairs)} E11 pairs")

    # Load checkpoint
    checkpoint = load_checkpoint() if args.resume else {}
    pending = [p for p in all_pairs if p["topic"] not in checkpoint]
    already_done = len(all_pairs) - len(pending)

    if already_done:
        print(f"[RESUME] {already_done} pairs already scored, skipping")
    print(f"[PENDING] {len(pending)} pairs to score\n")

    if not pending:
        print("All pairs already scored!")
        # Write final output
        OUTPUT_FILE.write_text(
            json.dumps(list(checkpoint.values()), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"Output: {OUTPUT_FILE}")
        return

    # Kill any other CUDA processes / clear VRAM
    clear_vram()
    if torch.cuda.is_available():
        used = vram_gb()
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM before load: {used:.2f}/{total:.1f} GB")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    sys.path.insert(0, str(BASE_DIR.parent / "research" / "circuit-tracer"))
    from circuit_tracer import ReplacementModel

    t0 = time.time()
    model = ReplacementModel.from_pretrained(
        MODEL_NAME, "gemma",
        dtype=torch.bfloat16,
        backend="transformerlens"
    )
    load_time = time.time() - t0

    if torch.cuda.is_available():
        used = vram_gb()
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Model loaded in {load_time:.1f}s  |  VRAM: {used:.2f}/{total:.1f} GB")

    clear_vram()

    # Score pairs
    results = dict(checkpoint)
    n_success = 0
    n_skip = 0
    n_fail = 0

    for i, pair in enumerate(pending):
        topic = pair["topic"]
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] {i+1}/{len(pending)} {topic}", end=" ", flush=True)

        scored = False
        for max_features in [MAX_FEATURES_DEFAULT, MAX_FEATURES_FALLBACK]:
            try:
                t_start = time.time()
                surface_depth, insight_depth, delta = score_pair(model, pair, max_features)
                elapsed = time.time() - t_start

                direction = "S>I" if delta > 0 else "I>S"
                print(f"d={delta:+.3f} ({direction}) [{elapsed:.1f}s]", end="", flush=True)
                if max_features == MAX_FEATURES_FALLBACK:
                    print(f" [fallback max_features={max_features}]", end="", flush=True)

                result = {
                    "pair_id": topic,
                    "surface_depth": surface_depth,
                    "insight_depth": insight_depth,
                    "delta": delta,
                    "direction": direction,
                    "max_features_used": max_features,
                    "time_s": round(elapsed, 1),
                    "timestamp": datetime.now().isoformat(),
                }
                results[topic] = result
                save_checkpoint(results)
                n_success += 1
                scored = True
                break

            except ValueError as e:
                # Token validation error - skip
                print(f" SKIP ({e})", flush=True)
                n_skip += 1
                scored = True
                break

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                err_str = str(e)
                if "out of memory" in err_str.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    if max_features == MAX_FEATURES_DEFAULT:
                        print(f" OOM at {max_features}, retrying at {MAX_FEATURES_FALLBACK}...", end=" ", flush=True)
                        clear_vram()
                        time.sleep(2)
                        continue
                    else:
                        print(f" OOM even at {max_features} - SKIP", flush=True)
                        n_skip += 1
                        scored = True
                        break
                else:
                    print(f" ERROR: {err_str[:80]}", flush=True)
                    n_fail += 1
                    scored = True
                    break

            except Exception as e:
                print(f" ERROR: {type(e).__name__}: {str(e)[:80]}", flush=True)
                n_fail += 1
                scored = True
                break

        if not scored:
            print(f" SKIP (exhausted retries)", flush=True)
            n_skip += 1

        clear_vram()

    # Write final output
    final = list(results.values())
    OUTPUT_FILE.write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print(f"DONE")
    print(f"  Scored:   {n_success}")
    print(f"  Skipped:  {n_skip}")
    print(f"  Failed:   {n_fail}")
    print(f"  Output:   {OUTPUT_FILE}")
    print("=" * 60)
    print()
    print("Next: merge results and run analysis:")
    print("  python experiment_selective_amplification.py --analyze-only")


if __name__ == "__main__":
    main()
