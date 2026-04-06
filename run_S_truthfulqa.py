"""
run_S_truthfulqa.py — Natural-hallucination S-axis experiment
===============================================================
Runs Gemma-2-2B-IT free generation on 200 TruthfulQA questions,
captures per-token C_delta trajectory for each response.

Pre-registered: osf_prereg_S_truthfulqa_v2.md (April 5, 2026, LOCKED)
Data source:    truthfulqa_results/truthfulqa_n200_seed42.json (seed=42, locked)
Analysis:       analyze_S_truthfulqa.py (deterministic labeling + tests)

Usage:
    # Micro-pilot (5 items) — verifies pipeline
    python run_S_truthfulqa.py --pilot

    # Full run (200 items, ~2.5h on RTX 4070)
    python run_S_truthfulqa.py

    # Resume after interruption
    python run_S_truthfulqa.py --resume

    # Override max_tokens (default 40, matches pre-reg)
    python run_S_truthfulqa.py --max-tokens 40
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "truthfulqa_results")
INPUT_PATH = os.path.join(RESULTS_DIR, "truthfulqa_n200_seed42.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "S_truthfulqa_n200.json")
PILOT_PATH = os.path.join(RESULTS_DIR, "S_truthfulqa_pilot.json")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "S_truthfulqa_checkpoint.json")


def format_prompt(question):
    """Locked prompt format from pre-registration."""
    return f"Q: {question}\nA:"


def save_checkpoint(state):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_one(steerer, question, max_tokens):
    prompt = format_prompt(question)
    r = steerer.generate_monitored(prompt, max_tokens=max_tokens)
    return {
        "prompt": prompt,
        "text": r.text,
        "c_delta_trajectory": [float(c) if c is not None else 0.0
                               for c in r.c_delta_trajectory],
        "tokens": list(r.tokens) if hasattr(r, "tokens") else [],
        "total_tokens": r.total_tokens,
        "generation_time_s": r.generation_time_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="Run only first 5 items")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--max-tokens", type=int, default=40,
                        help="Max generation tokens (default 40, matches pre-reg)")
    parser.add_argument("--model", default="google/gemma-2-2b-it",
                        help="Model name")
    args = parser.parse_args()

    print("=" * 70)
    print("S-axis × TruthfulQA — natural hallucination experiment")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Mode:        {'MICRO-PILOT (n=5)' if args.pilot else 'FULL (n=200)'}")
    print(f"Pre-reg:     osf_prereg_S_truthfulqa_v2.md")
    print(f"Input:       {INPUT_PATH}")
    print()

    # Load locked dataset
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data_in = json.load(f)
    items = data_in["items"]
    if args.pilot:
        items = items[:5]
    n_items = len(items)
    print(f"Loaded {n_items} items (seed={data_in['seed']})")

    # Checkpoint resume
    start_idx = 0
    all_results = []
    if args.resume:
        ckpt = load_checkpoint()
        if ckpt is not None:
            start_idx = ckpt["next_idx"]
            all_results = ckpt["results"]
            print(f"Resuming from item {start_idx + 1}/{n_items}")

    # Load model
    from coherence_steerer import CoherenceSteerer
    print(f"\n[load] Initializing CoherenceSteerer({args.model})...")
    steerer = CoherenceSteerer(model_name=args.model)
    print("[load] Model ready.\n")

    t_start = time.time()

    for i in range(start_idx, n_items):
        item = items[i]
        t_item = time.time()
        q_snip = item["question"][:60].replace("\n", " ")
        print(f"[{i+1}/{n_items}] {q_snip}")

        try:
            r = run_one(steerer, item["question"], args.max_tokens)
        except Exception as e:
            print(f"  [ERR] {e}")
            save_checkpoint({"next_idx": i, "results": all_results})
            raise

        preview = r["text"][:70].replace("\n", "\\n")
        print(f"  ({r['total_tokens']}t, {time.time()-t_item:.1f}s): {preview!r}")

        all_results.append({
            "hf_index": item["hf_index"],
            "category": item["category"],
            "question": item["question"],
            "best_answer": item["best_answer"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
            "generation": r,
        })

        save_checkpoint({"next_idx": i + 1, "results": all_results})

    elapsed = time.time() - t_start

    output = {
        "experiment": "S_truthfulqa_natural_hallucination",
        "prereg": "osf_prereg_S_truthfulqa_v2.md",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "n_items": len(all_results),
        "mode": "pilot" if args.pilot else "full",
        "data_seed": data_in["seed"],
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed,
        "results": all_results,
    }

    out_path = PILOT_PATH if args.pilot else OUTPUT_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Complete: {len(all_results)} items in {elapsed:.1f}s "
          f"({elapsed/max(1,len(all_results)):.1f}s/item)")
    print(f"Saved to: {out_path}")
    print(f"{'='*70}")

    # Quick sanity check
    if len(all_results) > 0:
        import numpy as np
        lens = [r["generation"]["total_tokens"] for r in all_results]
        print(f"\nGeneration lengths:")
        print(f"  mean={np.mean(lens):.1f}  "
              f"median={int(np.median(lens))}  "
              f"range=[{min(lens)},{max(lens)}]")


if __name__ == "__main__":
    main()
