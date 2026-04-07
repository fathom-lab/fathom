"""
run_cosine_probe_sequence.py — run cosine probe on multiple models
=====================================================================
Usage:
    python run_cosine_probe_sequence.py --pilot    # n=5 per model
    python run_cosine_probe_sequence.py            # n=200 per model
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

MODELS = [
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]


def safe_slug(name):
    return name.replace("/", "_").replace("-", "_")


def format_prompt(question):
    return f"Q: {question}\nA:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--models", nargs="*", default=MODELS)
    args = parser.parse_args()

    with open(INPUT_PATH) as f:
        data_in = json.load(f)
    items = data_in["items"]
    if args.pilot:
        items = items[:5]

    print("=" * 72)
    print("COSINE PROBE CROSS-ARCHITECTURE")
    print("=" * 72)
    print(f"Mode: {'PILOT' if args.pilot else 'FULL'} (n={len(items)})")
    print(f"Models: {len(args.models)}")
    print()

    for mi, model_name in enumerate(args.models):
        slug = safe_slug(model_name)
        suffix = "pilot" if args.pilot else "n200"
        out_path = os.path.join(RESULTS_DIR, f"cosine_probe_{slug}_{suffix}.json")
        ckpt_path = os.path.join(RESULTS_DIR, f"cosine_probe_{slug}_checkpoint.json")

        # Skip if already done
        if os.path.exists(out_path):
            print(f"[{mi+1}/{len(args.models)}] {model_name}: ALREADY DONE, skipping")
            continue

        # Resume from checkpoint
        start_idx = 0
        all_results = []
        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            start_idx = ckpt["next_idx"]
            all_results = ckpt["results"]
            print(f"[{mi+1}/{len(args.models)}] {model_name}: resuming from {start_idx}")
        else:
            print(f"\n{'='*72}")
            print(f"[{mi+1}/{len(args.models)}] {model_name}")
            print(f"{'='*72}")

        from sae_free_cosine_probe import CosineProbe
        probe = CosineProbe(model_name=model_name)

        t_start = time.time()
        for i in range(start_idx, len(items)):
            item = items[i]
            t_item = time.time()
            try:
                r = probe.generate_with_alignment(
                    format_prompt(item["question"]), max_tokens=40
                )
                gen = {
                    "text": r.text,
                    "c_delta_trajectory": r.c_delta_trajectory,
                    "c_align_trajectory": r.c_align_trajectory,
                    "c_early_self_trajectory": r.c_early_self_trajectory,
                    "c_late_self_trajectory": r.c_late_self_trajectory,
                    "tokens": r.tokens,
                    "total_tokens": r.total_tokens,
                    "generation_time_s": r.generation_time_s,
                }
            except Exception as e:
                print(f"  [ERR] item {i}: {e}")
                with open(ckpt_path, "w") as f:
                    json.dump({"next_idx": i, "results": all_results}, f)
                raise

            try:
                preview = r.text[:60].replace("\n", "\\n")
                print(f"  [{i+1}/{len(items)}] ({r.total_tokens}t, {time.time()-t_item:.1f}s) "
                      f"C_d[0]={r.c_delta_trajectory[0]:+.4f} {preview!r}")
            except UnicodeEncodeError:
                print(f"  [{i+1}/{len(items)}] ({r.total_tokens}t, {time.time()-t_item:.1f}s) "
                      f"C_d[0]={r.c_delta_trajectory[0]:+.4f} [unicode]")

            all_results.append({
                "hf_index": item["hf_index"],
                "category": item["category"],
                "question": item["question"],
                "best_answer": item["best_answer"],
                "correct_answers": item["correct_answers"],
                "incorrect_answers": item["incorrect_answers"],
                "generation": gen,
            })
            with open(ckpt_path, "w") as f:
                json.dump({"next_idx": i + 1, "results": all_results}, f)

        elapsed = time.time() - t_start

        output = {
            "experiment": "cosine_probe_truthfulqa",
            "model": model_name,
            "max_tokens": 40,
            "n_items": len(all_results),
            "mode": "pilot" if args.pilot else "full",
            "data_seed": data_in["seed"],
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": elapsed,
            "results": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {out_path} ({elapsed:.1f}s)")

        # Free GPU memory
        del probe
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*72}\nSEQUENCE COMPLETE\n{'='*72}")


if __name__ == "__main__":
    main()
