"""
run_sae_free_truthfulqa.py — SAE-free IPR probe on TruthfulQA
===================================================================
Runs the SAE-free IPR measurement on the same 200 TruthfulQA items
used in the main experiment, on any target model.

Purpose: Test whether the S_early commitment-intensity signature
replicates across architectures WITHOUT requiring pre-trained SAEs.

Usage:
    # Test on Gemma-2-2B-IT (validation: should match SAE-based result)
    python run_sae_free_truthfulqa.py --model google/gemma-2-2b-it --pilot

    # Cross-architecture test on Llama-3.1-8B
    python run_sae_free_truthfulqa.py --model meta-llama/Llama-3.1-8B-Instruct --pilot

    # Full run (n=200)
    python run_sae_free_truthfulqa.py --model meta-llama/Llama-3.1-8B-Instruct
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


def safe_model_slug(name):
    return name.replace("/", "_").replace("-", "_")


def format_prompt(question):
    return f"Q: {question}\nA:"


def run_one(probe, question, max_tokens):
    r = probe.generate_with_ipr(format_prompt(question), max_tokens=max_tokens)
    return {
        "prompt": r.prompt,
        "text": r.text,
        "c_delta_trajectory": list(r.c_delta_trajectory),
        "ipr_early_trajectory": list(r.ipr_early_trajectory),
        "ipr_late_trajectory": list(r.ipr_late_trajectory),
        "tokens": list(r.tokens),
        "total_tokens": r.total_tokens,
        "generation_time_s": r.generation_time_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pilot", action="store_true", help="Only first 5")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args()

    slug = safe_model_slug(args.model)
    output_path = os.path.join(RESULTS_DIR, f"sae_free_ipr_{slug}_n200.json")
    pilot_path = os.path.join(RESULTS_DIR, f"sae_free_ipr_{slug}_pilot.json")
    checkpoint_path = os.path.join(RESULTS_DIR, f"sae_free_ipr_{slug}_checkpoint.json")

    print("=" * 70)
    print("SAE-FREE IPR RUN ON TRUTHFULQA")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"top-k:      {args.top_k}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"Mode:       {'PILOT (n=5)' if args.pilot else 'FULL (n=200)'}")
    print()

    with open(INPUT_PATH) as f:
        data_in = json.load(f)
    items = data_in["items"]
    if args.pilot:
        items = items[:5]
    n_items = len(items)
    print(f"Loaded {n_items} items")

    start_idx = 0
    all_results = []
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_idx = ckpt["next_idx"]
        all_results = ckpt["results"]
        print(f"Resuming from item {start_idx+1}/{n_items}")

    from sae_free_ipr import SAEFreeIPR
    probe = SAEFreeIPR(model_name=args.model, top_k=args.top_k)

    t_start = time.time()
    for i in range(start_idx, n_items):
        item = items[i]
        t_item = time.time()
        q_snip = item["question"][:60].replace("\n", " ")
        print(f"[{i+1}/{n_items}] {q_snip}")
        try:
            r = run_one(probe, item["question"], args.max_tokens)
        except Exception as e:
            print(f"  [ERR] {e}")
            with open(checkpoint_path, "w") as f:
                json.dump({"next_idx": i, "results": all_results}, f)
            raise

        preview = r["text"][:70].replace("\n", "\\n")
        try:
            print(f"  ({r['total_tokens']}t, {time.time()-t_item:.1f}s): {preview!r}")
        except UnicodeEncodeError:
            print(f"  ({r['total_tokens']}t, {time.time()-t_item:.1f}s): [unicode in output]")
        if r["c_delta_trajectory"]:
            print(f"  C_delta[0]={r['c_delta_trajectory'][0]:+.5f}  "
                  f"IPR_e[0]={r['ipr_early_trajectory'][0]:.5f}  "
                  f"IPR_l[0]={r['ipr_late_trajectory'][0]:.5f}")

        all_results.append({
            "hf_index": item["hf_index"],
            "category": item["category"],
            "question": item["question"],
            "best_answer": item["best_answer"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
            "generation": r,
        })
        with open(checkpoint_path, "w") as f:
            json.dump({"next_idx": i+1, "results": all_results}, f)

    elapsed = time.time() - t_start
    output = {
        "experiment": "sae_free_ipr_truthfulqa",
        "model": args.model,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "n_items": len(all_results),
        "mode": "pilot" if args.pilot else "full",
        "data_seed": data_in["seed"],
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed,
        "results": all_results,
    }
    out_path = pilot_path if args.pilot else output_path
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Complete: {len(all_results)} in {elapsed:.1f}s "
          f"({elapsed/max(1,len(all_results)):.1f}s/item)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
