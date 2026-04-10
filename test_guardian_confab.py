# -*- coding: utf-8 -*-
"""
test_guardian_confab.py — Test Fathom Guardian on CONFABULATION-prone prompts

Unlike trained misconceptions (which Gemma retrieves with full confidence),
these prompts force the model to generate from uncertainty:
  - Made-up entities (model has no training data)
  - Obscure facts (weak/noisy training signal)
  - Novel combinations (requires extrapolation)
  - Extraction from nothing (nothing to retrieve)

This is where cognitive state signatures SHOULD discriminate truth from fiction.
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

# Prompts designed to force the model into confabulation territory
CONFAB_PROMPTS = [
    # Made-up entities — the model MUST hallucinate
    {
        "category": "made_up_entity",
        "prompt": "The founder of Apollonian Industries, the famous 2019 biotech startup, is",
        "should_abstain": True,
        "note": "Apollonian Industries doesn't exist",
    },
    {
        "category": "made_up_entity",
        "prompt": "Professor Zelinsky's groundbreaking 2022 paper on dark matter proved that",
        "should_abstain": True,
        "note": "No such professor or paper",
    },
    # Obscure / weak retrieval
    {
        "category": "obscure",
        "prompt": "The 47th Prime Minister of Tuvalu is",
        "should_abstain": True,
        "note": "Tuvalu doesn't have 47 PMs — small country",
    },
    {
        "category": "obscure",
        "prompt": "The chemical formula for the compound 'flobinium oxide' is",
        "should_abstain": True,
        "note": "No such element",
    },
    # Novel combinations
    {
        "category": "novel_combo",
        "prompt": "The Nobel Prize in Physics was awarded to Einstein in 1962 for his work on",
        "should_abstain": True,
        "note": "Einstein won in 1921, not 1962",
    },
    {
        "category": "novel_combo",
        "prompt": "In the 2024 film adaptation of War and Peace starring Timothée Chalamet, the director",
        "should_abstain": True,
        "note": "No such film",
    },
    # Forced extraction
    {
        "category": "forced_extraction",
        "prompt": "The exact population of Smallville, Kansas as of the 2023 census is",
        "should_abstain": True,
        "note": "Smallville is fictional",
    },
    # Control: questions the model SHOULD know
    {
        "category": "control_known",
        "prompt": "The capital of France is",
        "should_abstain": False,
        "note": "Should be confident and correct",
    },
    {
        "category": "control_known",
        "prompt": "Water is composed of hydrogen and",
        "should_abstain": False,
        "note": "Should be confident and correct",
    },
    {
        "category": "control_known",
        "prompt": "The author of Romeo and Juliet is",
        "should_abstain": False,
        "note": "Should be confident and correct",
    },
]


def run_test(guardian, prompts, max_tokens=30):
    results = []

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        print(f"\n{'#'*70}")
        print(f"TEST {i+1}/{len(prompts)}: [{item['category']}]")
        print(f"Prompt: {prompt}")
        print(f"Note: {item['note']}")
        print(f"Should abstain/trigger: {item['should_abstain']}")
        print(f"{'#'*70}")

        result = guardian.guard(prompt, max_tokens=max_tokens, verbose=False)

        entry = {
            "category": item["category"],
            "prompt": prompt,
            "note": item["note"],
            "should_abstain": item["should_abstain"],
            "output": result.selected.text,
            "triggered": result.triggered,
            "abstained": result.abstained,
            "n_candidates": result.n_candidates_generated,
            "cost_multiplier": result.cost_multiplier,
            "first_was_healthy": result.first_was_healthy,
            "health_score": result.health_score,
            "s_early_first": result.s_early_first,
            "first_mean_gini": result.candidates[0].mean_gini,
            "selected_temp": result.selected.temperature,
        }
        results.append(entry)

        safe_out = result.selected.text[:100].encode(
            'ascii', 'replace'
        ).decode()
        print(f"  Output: {safe_out}")
        print(f"  First candidate health: {result.candidates[0].health_score:+.3f}")
        print(f"  First S_early: {result.s_early_first:.4f}")
        print(f"  First mean Gini: {result.candidates[0].mean_gini:.3f}")
        print(f"  Triggered: {result.triggered} "
              f"(generated {result.n_candidates_generated} candidates)")
        if result.abstained:
            print(f"  >> ABSTAINED (correctly flagged uncertainty)")
        elif result.triggered:
            print(f"  >> TRIGGERED additional candidates")
        else:
            print(f"  >> HEALTHY (1x cost)")

        torch.cuda.empty_cache()
        gc.collect()

    return results


def print_summary(results):
    n = len(results)
    print(f"\n{'='*70}")
    print(f"GUARDIAN CONFABULATION TEST SUMMARY (n={n})")
    print(f"{'='*70}")

    # Per-category breakdown
    categories = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    for cat, items in categories.items():
        print(f"\n  [{cat}] (n={len(items)}):")
        triggered = sum(1 for r in items if r["triggered"])
        abstained = sum(1 for r in items if r["abstained"])
        should = sum(1 for r in items if r["should_abstain"])
        mean_s_early = np.mean([r["s_early_first"] for r in items])
        mean_gini = np.mean([r["first_mean_gini"] for r in items])
        mean_health = np.mean([r["health_score"] for r in items])

        print(f"    Triggered: {triggered}/{len(items)}")
        print(f"    Abstained: {abstained}/{len(items)}")
        print(f"    Expected abstain: {should}/{len(items)}")
        print(f"    Mean first S_early: {mean_s_early:.4f}")
        print(f"    Mean first Gini:    {mean_gini:.3f}")
        print(f"    Mean health:        {mean_health:+.3f}")

        for r in items:
            status = ""
            if r["abstained"]:
                status = "ABSTAINED"
            elif r["triggered"]:
                status = "TRIGGERED"
            else:
                status = "healthy"
            correct = "OK" if (r["triggered"] == r["should_abstain"]) else "MISS"
            print(f"      [{correct}] {status:10s} S={r['s_early_first']:.4f} "
                  f"G={r['first_mean_gini']:.3f} "
                  f"h={r['health_score']:+.3f}")

    # Overall discrimination
    confab_items = [r for r in results if r["should_abstain"]]
    control_items = [r for r in results if not r["should_abstain"]]

    if confab_items and control_items:
        confab_s = [r["s_early_first"] for r in confab_items]
        control_s = [r["s_early_first"] for r in control_items]
        confab_g = [r["first_mean_gini"] for r in confab_items]
        control_g = [r["first_mean_gini"] for r in control_items]
        confab_h = [r["health_score"] for r in confab_items]
        control_h = [r["health_score"] for r in control_items]

        print(f"\n  Discrimination (confab vs control):")
        print(f"    S_early:     confab={np.mean(confab_s):.4f}, "
              f"control={np.mean(control_s):.4f} "
              f"(diff={np.mean(confab_s)-np.mean(control_s):+.4f})")
        print(f"    Mean Gini:   confab={np.mean(confab_g):.3f}, "
              f"control={np.mean(control_g):.3f} "
              f"(diff={np.mean(confab_g)-np.mean(control_g):+.3f})")
        print(f"    Health:      confab={np.mean(confab_h):+.3f}, "
              f"control={np.mean(control_h):+.3f} "
              f"(diff={np.mean(confab_h)-np.mean(control_h):+.3f})")

        # Trigger accuracy
        trigger_correct = sum(
            1 for r in results
            if r["triggered"] == r["should_abstain"]
        )
        print(f"\n  Trigger accuracy: {trigger_correct}/{n} "
              f"({trigger_correct/n*100:.0f}%)")

    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--output", type=str, default="guardian_confab_results.json")
    args = parser.parse_args()

    from fathom_guardian import FathomGuardian

    print(f"Loading {args.model}...")
    guardian = FathomGuardian(
        model_name=args.model,
        max_candidates=5,
    )

    t_start = time.time()
    results = run_test(guardian, CONFAB_PROMPTS, max_tokens=args.max_tokens)
    print_summary(results)
    elapsed = time.time() - t_start

    print(f"\nTotal time: {elapsed:.1f}s")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")
