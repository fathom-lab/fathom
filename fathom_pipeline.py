# -*- coding: utf-8 -*-
"""
fathom_pipeline.py -- The Complete Fathom Pipeline
====================================================

One command. Measure -> Diagnose -> Fix -> Verify.

    python fathom_pipeline.py google/gemma-2-2b

This is the first automated circuit-level hallucination treatment
pipeline. It:

  1. MEASURE: Computes per-layer coherence spectrum on TruthfulQA
  2. DIAGNOSE: Runs cognitive autopsy to identify fault layers
  3. FIX: Performs targeted transcoder surgery on fault layers
  4. VERIFY: Tests fixed model on held-out data

No human-labeled preferences. No RLHF. No constitutional AI.
Just: find where computation concentrates, and redistribute it.

Usage:
    python fathom_pipeline.py                          # full pipeline
    python fathom_pipeline.py --diagnose-only          # just measure + diagnose
    python fathom_pipeline.py --apply surgical_checkpoints/  # apply existing fix

Patent pending: US 64/020,489 & 64/021,113
"""

import sys
import time
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent


def run_pipeline(
    model_name: str = "google/gemma-2-2b",
    n_train: int = 20,
    n_test: int = 30,
    n_epochs: int = 3,
    gini_weight: float = 1.0,
    kl_weight: float = 0.5,
    lr: float = 1e-4,
):
    """Run the complete Fathom pipeline."""
    t_start = time.time()

    print()
    print("  =============================================")
    print("  F A T H O M   P I P E L I N E")
    print("  Measure -> Diagnose -> Fix -> Verify")
    print("  =============================================")
    print(f"  Model: {model_name}")
    print(f"  Training: {n_train} prompts x {n_epochs} epochs")
    print(f"  Evaluation: {n_test} held-out prompts")
    print()

    # ── STEP 1: DIAGNOSE ──
    print("  [1/4] DIAGNOSIS -- Identifying fault layers...")
    print("  " + "-" * 50)

    from analysis.cognitive_divergence import load_coherence_profiles, cognitive_autopsy, gini
    import numpy as np

    profiles = load_coherence_profiles()
    autopsies = [cognitive_autopsy(p["correct_profile"], p["halluc_profile"]) for p in profiles]

    # Find top fault layers
    fault_contrib = np.mean([a["layer_contribution"] for a in autopsies], axis=0)
    fault_ranking = np.argsort(fault_contrib)[::-1]

    print(f"  Analyzed {len(profiles)} paired profiles")
    print(f"  Top fault layers:")
    target_layers = []
    for rank, layer in enumerate(fault_ranking[:5]):
        contrib = fault_contrib[layer]
        if contrib > 0.05:  # >5% contribution
            target_layers.append(int(layer))
        print(f"    #{rank+1}: Layer {layer} ({100*contrib:.1f}% contribution)"
              + (" << TARGET" if contrib > 0.05 else ""))

    if not target_layers:
        target_layers = [int(fault_ranking[0]), int(fault_ranking[1])]
    print(f"  Target layers for surgery: {target_layers}")

    # ── STEP 2: SURGERY ──
    print(f"\n  [2/4] SURGERY -- Fine-tuning transcoders {target_layers}...")
    print("  " + "-" * 50)

    from surgical_fix import TranscoderSurgeon

    surgeon = TranscoderSurgeon(
        model_name=model_name,
        target_layers=target_layers,
        lr=lr,
        gini_weight=gini_weight,
        kl_weight=kl_weight,
    )

    results = surgeon.train(
        n_epochs=n_epochs,
        batch_prompts=n_train,
        eval_every=10,
    )

    # ── STEP 3: VERIFY ──
    print(f"\n  [3/4] VERIFICATION -- Testing on held-out data...")
    print("  " + "-" * 50)

    from surgical_verify import (
        load_held_out_questions, generate_answer, score_answer,
        compute_spectrum_metrics
    )

    # The surgeon still has the model loaded with surgical weights
    model = surgeon._inner
    test_items = load_held_out_questions(n_test=n_test, n_train=n_train)

    fixed_results = []
    for i, item in enumerate(test_items):
        text = generate_answer(model, item["question"], max_tokens=40)
        tokens = model.to_tokens(f"Q: {item['question']}\nA:")
        metrics = compute_spectrum_metrics(model, tokens, target_layers)
        score = score_answer(text, item["correct_answers"], item["incorrect_answers"])

        fixed_results.append({
            "question": item["question"],
            "generated": text,
            "score": score,
            "metrics": {str(k): v for k, v in metrics.items()},
        })

    # Quick summary
    n = len(fixed_results)
    correct = sum(1 for r in fixed_results if r["score"]["is_correct"])
    incorrect = sum(1 for r in fixed_results if r["score"]["is_incorrect"])
    gini_vals = [r["metrics"].get(str(target_layers[0]), {}).get("gini", 0)
                 for r in fixed_results]

    print(f"  Results on {n} held-out questions:")
    print(f"    Correct: {correct} ({100*correct/n:.0f}%)")
    print(f"    Incorrect: {incorrect} ({100*incorrect/n:.0f}%)")
    print(f"    Mean Gini (L{target_layers[0]}): {np.mean(gini_vals):.4f}")

    # ── STEP 4: REPORT ──
    print(f"\n  [4/4] REPORT")
    print("  " + "-" * 50)

    elapsed = time.time() - t_start

    report = {
        "model": model_name,
        "pipeline_version": "1.0",
        "elapsed_s": elapsed,
        "diagnosis": {
            "n_profiles": len(profiles),
            "target_layers": target_layers,
            "fault_contribution": {str(l): float(fault_contrib[l]) for l in target_layers},
        },
        "surgery": {
            "n_epochs": n_epochs,
            "n_train": n_train,
            "lr": lr,
            "pre_gini": results.get("pre_eval", {}).get("incorrect", {}).get("gini", {}),
            "post_gini": results.get("post_eval", {}).get("incorrect", {}).get("gini", {}),
        },
        "verification": {
            "n_test": n,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": correct / n if n > 0 else 0,
            "mean_gini": float(np.mean(gini_vals)),
        },
    }

    out_path = REPO / "truthfulqa_results" / "fathom_pipeline_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  =============================================")
    print(f"  PIPELINE COMPLETE in {elapsed:.0f}s")
    print(f"  =============================================")
    print(f"  Target layers: {target_layers}")
    for l in target_layers:
        pre = results.get("pre_eval", {}).get("incorrect", {}).get("gini", {}).get(l, "?")
        post = results.get("post_eval", {}).get("incorrect", {}).get("gini", {}).get(l, "?")
        if isinstance(pre, float) and isinstance(post, float):
            print(f"  Layer {l} Gini: {pre:.4f} -> {post:.4f} ({post-pre:+.4f})")
    print(f"  Held-out accuracy: {correct}/{n} ({100*correct/n:.0f}%)")
    print(f"  Report: {out_path.name}")
    print(f"  Checkpoints: surgical_checkpoints/")
    print()

    return report


if __name__ == "__main__":
    if "--diagnose-only" in sys.argv:
        from analysis.cognitive_divergence import load_coherence_profiles, cognitive_autopsy
        import numpy as np
        profiles = load_coherence_profiles()
        autopsies = [cognitive_autopsy(p["correct_profile"], p["halluc_profile"]) for p in profiles]
        fc = np.mean([a["layer_contribution"] for a in autopsies], axis=0)
        print("Fault contribution by layer:")
        for i in np.argsort(fc)[::-1][:10]:
            print(f"  Layer {i}: {100*fc[i]:.1f}%")
    else:
        model = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "google/gemma-2-2b"
        run_pipeline(model_name=model)
