"""
load_truthfulqa_for_S.py — Build the locked n=200 TruthfulQA sample
====================================================================
Loads TruthfulQA 'generation' split from HuggingFace, samples n=200
items with seed=42 (pre-registered), saves to a cache file for the
S-axis experiment.

This is deterministic: same seed, same items every time.

Usage:
    python load_truthfulqa_for_S.py
"""
import json
import os
import sys

import numpy as np

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "truthfulqa_results", "truthfulqa_n200_seed42.json")
N_ITEMS = 200
SEED = 42


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print(f"Loading TruthfulQA from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    print(f"  full dataset: {len(ds)} items")

    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(ds), size=N_ITEMS, replace=False)
    indices = sorted(int(i) for i in indices)

    items = []
    for i in indices:
        row = ds[i]
        items.append({
            "hf_index": i,
            "type": row.get("type"),
            "category": row.get("category"),
            "question": row["question"],
            "best_answer": row["best_answer"],
            "correct_answers": list(row["correct_answers"]),
            "incorrect_answers": list(row["incorrect_answers"]),
            "source": row.get("source", ""),
        })

    out = {
        "dataset": "truthful_qa/generation/validation",
        "seed": SEED,
        "n_items": len(items),
        "hf_indices": indices,
        "items": items,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"  sampled {len(items)} items (seed={SEED})")
    print(f"  saved to: {OUT_PATH}")

    # Summary
    from collections import Counter
    cat_counts = Counter(i["category"] for i in items)
    print(f"\nCategory breakdown (top 10):")
    for cat, n in cat_counts.most_common(10):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
