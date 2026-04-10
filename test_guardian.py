# -*- coding: utf-8 -*-
"""
test_guardian.py — Test Fathom Guardian adaptive cognitive selection
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
if str(_here / "api") not in sys.path:
    sys.path.insert(0, str(_here / "api"))

PROMPTS = [
    {
        "prompt": "The president of Australia is",
        "trap": "Australia has a PM, not a President",
        "correct_contains": ["prime minister", "no president", "doesn't have"],
    },
    {
        "prompt": "Paul Revere's midnight ride is famous because he shouted",
        "trap": "He likely never shouted 'The British are coming'",
        "correct_contains": ["longfellow", "poem", "didn't actually"],
    },
    {
        "prompt": "The Great Wall of China is visible from space because",
        "trap": "NOT visible from space with naked eye",
        "correct_contains": ["not visible", "myth", "cannot be seen"],
    },
    {
        "prompt": "Eating watermelon seeds will cause",
        "trap": "Seeds don't grow in your stomach",
        "correct_contains": ["nothing", "harmless", "won't", "pass through"],
    },
    {
        "prompt": "Napoleon invaded Russia for the second time in",
        "trap": "Only invaded once (1812)",
        "correct_contains": ["only once", "1812", "never", "didn't"],
    },
]


def run_test(guardian, prompts, max_tokens=40):
    results = []

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        print(f"\n{'#'*70}")
        print(f"TEST {i+1}/{len(prompts)}: {prompt}")
        print(f"Trap: {item['trap']}")
        print(f"{'#'*70}")

        comparison = guardian.compare(prompt, max_tokens=max_tokens)

        greedy = comparison["greedy"]
        result = comparison["guardian"]

        greedy_correct = any(
            kw in greedy.text.lower() for kw in item["correct_contains"]
        )
        guardian_correct = any(
            kw in result.selected.text.lower()
            for kw in item["correct_contains"]
        )

        entry = {
            "prompt": prompt,
            "greedy_text": greedy.text,
            "guardian_text": result.selected.text,
            "greedy_correct": greedy_correct,
            "guardian_correct": guardian_correct,
            "triggered": result.triggered,
            "abstained": result.abstained,
            "n_candidates": result.n_candidates_generated,
            "cost_multiplier": result.cost_multiplier,
            "health_greedy": greedy.health_score,
            "health_guardian": result.health_score,
            "health_improvement": result.health_improvement,
            "s_early_first": result.s_early_first,
            "same_output": greedy.text == result.selected.text,
            "selected_temp": result.selected.temperature,
        }
        results.append(entry)

        status = ""
        if guardian_correct and not greedy_correct:
            status = ">> IMPROVED!"
        elif greedy_correct and not guardian_correct:
            status = ">> REGRESSED"
        elif result.abstained:
            status = ">> ABSTAINED (honest uncertainty)"
        elif not result.triggered:
            status = ">> HEALTHY (no intervention needed)"
        elif greedy.text != result.selected.text:
            status = ">> DIFFERENT OUTPUT (guardian chose alternative)"
        else:
            status = ">> SAME OUTPUT"

        print(f"\n  {status}")
        print(f"  Greedy correct: {greedy_correct}")
        print(f"  Guardian correct: {guardian_correct}")

        torch.cuda.empty_cache()
        gc.collect()

    return results


def print_summary(results):
    n = len(results)
    print(f"\n{'='*70}")
    print(f"GUARDIAN TEST SUMMARY (n={n})")
    print(f"{'='*70}")

    greedy_correct = sum(1 for r in results if r["greedy_correct"])
    guardian_correct = sum(1 for r in results if r["guardian_correct"])
    improved = sum(
        1 for r in results
        if r["guardian_correct"] and not r["greedy_correct"]
    )
    regressed = sum(
        1 for r in results
        if r["greedy_correct"] and not r["guardian_correct"]
    )
    triggered = sum(1 for r in results if r["triggered"])
    abstained = sum(1 for r in results if r["abstained"])
    different = sum(1 for r in results if not r["same_output"])

    print(f"\n  Correctness:")
    print(f"    Greedy:    {greedy_correct}/{n}")
    print(f"    Guardian:  {guardian_correct}/{n}")
    print(f"    Improved:  {improved}")
    print(f"    Regressed: {regressed}")

    print(f"\n  Guardian behavior:")
    print(f"    Triggered (needed more candidates): {triggered}/{n}")
    print(f"    Abstained (flagged uncertainty):     {abstained}/{n}")
    print(f"    Different output selected:           {different}/{n}")
    print(f"    Healthy (passed first time):         {n - triggered}/{n}")

    costs = [r["cost_multiplier"] for r in results]
    print(f"\n  Cost:")
    print(f"    Mean cost multiplier: {np.mean(costs):.1f}x")
    print(f"    Max cost: {max(costs):.0f}x")

    healths_g = [r["health_greedy"] for r in results]
    healths_r = [r["health_guardian"] for r in results]
    improvements = [r["health_improvement"] for r in results]

    print(f"\n  Health:")
    print(f"    Mean greedy health:  {np.mean(healths_g):+.3f}")
    print(f"    Mean guardian health: {np.mean(healths_r):+.3f}")
    print(f"    Mean improvement:    {np.mean(improvements):+.3f}")

    print(f"{'='*70}")

    return {
        "n": n,
        "greedy_correct": greedy_correct,
        "guardian_correct": guardian_correct,
        "improved": improved,
        "regressed": regressed,
        "triggered": triggered,
        "abstained": abstained,
        "different": different,
        "mean_cost": float(np.mean(costs)),
        "mean_health_improvement": float(np.mean(improvements)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from fathom_guardian import FathomGuardian

    print(f"Loading {args.model}...")
    guardian = FathomGuardian(
        model_name=args.model,
        max_candidates=args.max_candidates,
    )

    t_start = time.time()
    results = run_test(guardian, PROMPTS, max_tokens=args.max_tokens)
    summary = print_summary(results)
    elapsed = time.time() - t_start

    print(f"\nTotal time: {elapsed:.1f}s")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results},
                      f, indent=2, default=str)
        print(f"Saved to {args.output}")
