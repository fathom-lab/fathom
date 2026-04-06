"""
run_S_triplet.py — S-axis triplet experiment driver
=====================================================
Runs the pre-registered 3-condition S-axis study.

For each of 30 topics, generates 3 responses (hedge / correct-commit /
wrong-commit), captures per-token C_delta trajectory, saves results.

Pre-registered: osf_prereg_S_triplet.md (April 5, 2026, LOCKED)
Prompts:        prompts_S_triplet.json
Analysis:       analyze_S_triplet.py (post-run, deterministic)

Usage:
    # Pilot mode (3 topics, verifies pipeline)
    python run_S_triplet.py --pilot

    # Full run (30 topics)
    python run_S_triplet.py

    # Resume from checkpoint
    python run_S_triplet.py --resume

    # Custom max_tokens (default 30, matches pre-registration)
    python run_S_triplet.py --max-tokens 30
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "truthfulqa_results")
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "prompts_S_triplet.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "S_triplet_n30.json")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "S_triplet_checkpoint.json")


def load_prompts():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_checkpoint(state):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_one(steerer, prompt, max_tokens):
    """Run a single prompt, return trajectory + text."""
    r = steerer.generate_monitored(prompt, max_tokens=max_tokens)
    return {
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
                        help="Run only first 3 topics (pilot mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--max-tokens", type=int, default=30,
                        help="Max generation tokens per prompt (default 30)")
    parser.add_argument("--model", default="google/gemma-2-2b-it",
                        help="Model name (default gemma-2-2b-it)")
    args = parser.parse_args()

    print("=" * 70)
    print("S-axis triplet experiment")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Mode:        {'PILOT (n=3)' if args.pilot else 'FULL (n=30)'}")
    print(f"Pre-reg:     osf_prereg_S_triplet.md")
    print(f"Output:      {OUTPUT_PATH}")
    print()

    data = load_prompts()
    topics = data["topics"]
    if args.pilot:
        topics = topics[:3]
    n_topics = len(topics)
    print(f"Loaded {n_topics} topics")

    # Checkpoint resume
    start_idx = 0
    all_results = []
    if args.resume:
        ckpt = load_checkpoint()
        if ckpt is not None:
            start_idx = ckpt["next_idx"]
            all_results = ckpt["results"]
            print(f"Resuming from topic {start_idx + 1}/{n_topics}")

    # Load model
    from coherence_steerer import CoherenceSteerer
    print(f"\n[load] Initializing CoherenceSteerer({args.model})...")
    steerer = CoherenceSteerer(model_name=args.model)
    print("[load] Model ready.\n")

    t_start = time.time()

    for i in range(start_idx, n_topics):
        topic = topics[i]
        t_topic_start = time.time()
        print(f"[{i+1}/{n_topics}] Topic: {topic['topic']}")

        try:
            r_H = run_one(steerer, topic["H"], args.max_tokens)
            print(f"  H  ({r_H['total_tokens']:>2}t): {r_H['text'][:60].strip()!r}")
            r_Cp = run_one(steerer, topic["C_plus"], args.max_tokens)
            print(f"  C+ ({r_Cp['total_tokens']:>2}t): {r_Cp['text'][:60].strip()!r}")
            r_Cm = run_one(steerer, topic["C_minus"], args.max_tokens)
            print(f"  C- ({r_Cm['total_tokens']:>2}t): {r_Cm['text'][:60].strip()!r}")
        except Exception as e:
            print(f"  [ERR] {e}")
            save_checkpoint({"next_idx": i, "results": all_results})
            raise

        all_results.append({
            "id": topic["id"],
            "topic": topic["topic"],
            "prompts": {
                "H": topic["H"],
                "C_plus": topic["C_plus"],
                "C_minus": topic["C_minus"],
            },
            "H": r_H,
            "C_plus": r_Cp,
            "C_minus": r_Cm,
        })

        t_topic = time.time() - t_topic_start
        print(f"  [{t_topic:.1f}s]")

        # Save checkpoint every topic
        save_checkpoint({"next_idx": i + 1, "results": all_results})

    elapsed = time.time() - t_start

    # Save final results
    output = {
        "experiment": "S_triplet_confident_wrongness",
        "prereg": "osf_prereg_S_triplet.md",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "n_topics": len(all_results),
        "mode": "pilot" if args.pilot else "full",
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed,
        "results": all_results,
    }

    out_path = OUTPUT_PATH
    if args.pilot:
        out_path = out_path.replace(".json", "_pilot.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Complete: {len(all_results)} topics in {elapsed:.1f}s "
          f"({elapsed/len(all_results):.1f}s/topic)")
    print(f"Saved to: {out_path}")
    print(f"{'='*70}")

    # Quick sanity check
    if len(all_results) > 0:
        lens_H = [r["H"]["total_tokens"] for r in all_results]
        lens_Cp = [r["C_plus"]["total_tokens"] for r in all_results]
        lens_Cm = [r["C_minus"]["total_tokens"] for r in all_results]
        import numpy as np
        print("\nLength summary (pre-registered target: 30 tokens):")
        print(f"  H:  mean={np.mean(lens_H):.1f}  range=[{min(lens_H)},{max(lens_H)}]")
        print(f"  C+: mean={np.mean(lens_Cp):.1f}  range=[{min(lens_Cp)},{max(lens_Cp)}]")
        print(f"  C-: mean={np.mean(lens_Cm):.1f}  range=[{min(lens_Cm)},{max(lens_Cm)}]")
        spread = max(np.mean(lens_H), np.mean(lens_Cp), np.mean(lens_Cm)) - \
                 min(np.mean(lens_H), np.mean(lens_Cp), np.mean(lens_Cm))
        print(f"  inter-condition spread: {spread:.1f} tokens "
              f"({'OK' if spread <= 5 else 'WARNING — exceeds 5-token threshold'})")


if __name__ == "__main__":
    main()
