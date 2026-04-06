"""
run_S_truthfulqa_baselines.py — baseline comparison run
==========================================================
Re-runs same 200 TruthfulQA items with SAME prompts and SAME model,
but captures per-token logit entropy + chosen-token logprob + top2
margin alongside C_delta.

Uses CoherenceSteererExt (already built) which captures all these
per-token quantities.

Purpose: head-to-head comparison of S_early vs:
  - logit entropy
  - chosen-token logprob
  - top2 margin (p_top1 - p_top2)

Usage:
    # Micro-pilot (5 items)
    python run_S_truthfulqa_baselines.py --pilot

    # Full run (200 items, ~3.5h)
    python run_S_truthfulqa_baselines.py

    # Resume
    python run_S_truthfulqa_baselines.py --resume
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
OUTPUT_PATH = os.path.join(RESULTS_DIR, "S_baselines_n200.json")
PILOT_PATH = os.path.join(RESULTS_DIR, "S_baselines_pilot.json")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "S_baselines_checkpoint.json")


def format_prompt(question):
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
    r = steerer.generate_with_entropy(prompt, max_tokens=max_tokens)
    return {
        "prompt": prompt,
        "text": r.text,
        "c_delta_trajectory": [float(c) if c is not None else 0.0
                               for c in r.c_delta_trajectory],
        "entropy_trajectory": [float(e) for e in r.entropy_trajectory],
        "logprob_trajectory": [float(lp) for lp in r.logprob_trajectory],
        "top2_margin_trajectory": [float(m) for m in r.top2_margin_trajectory],
        "tokens": list(r.tokens) if hasattr(r, "tokens") else [],
        "total_tokens": r.total_tokens,
        "generation_time_s": r.generation_time_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    args = parser.parse_args()

    print("=" * 70)
    print("S-axis BASELINES run (C_delta + entropy + logprob + top2 margin)")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Mode:        {'PILOT (n=5)' if args.pilot else 'FULL (n=200)'}")
    print(f"Output:      {OUTPUT_PATH}")
    print()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data_in = json.load(f)
    items = data_in["items"]
    if args.pilot:
        items = items[:5]
    n_items = len(items)
    print(f"Loaded {n_items} items")

    start_idx = 0
    all_results = []
    if args.resume:
        ckpt = load_checkpoint()
        if ckpt is not None:
            start_idx = ckpt["next_idx"]
            all_results = ckpt["results"]
            print(f"Resuming from item {start_idx + 1}/{n_items}")

    from coherence_steerer_ext import CoherenceSteererExt
    print(f"\n[load] Initializing CoherenceSteererExt({args.model})...")
    steerer = CoherenceSteererExt(model_name=args.model)
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
        if r["entropy_trajectory"]:
            print(f"  H[0]={r['entropy_trajectory'][0]:.3f} "
                  f"logp[0]={r['logprob_trajectory'][0]:.3f} "
                  f"margin[0]={r['top2_margin_trajectory'][0]:.3f}")

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
        "experiment": "S_baselines_entropy_logprob",
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
    print(f"Complete: {len(all_results)} in {elapsed:.1f}s ({elapsed/max(1,len(all_results)):.1f}s/item)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
