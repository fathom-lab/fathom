"""
TruthfulQA × Fathom Coherence Experiment
=========================================

Tests the hallucination-as-cognitive-lock-in hypothesis:
  H1: Hallucinated responses show different C (coherence) than correct responses
  H2: K (depth) does NOT significantly differ between hallucinated and correct
  H3: Hallucinations cluster in (normal K, high C) quadrant — wrong attractor

Pre-registered: OSF_PREREGISTRATION_V2.md (April 1, 2026)
Patent pending: US 64/020,489 & 64/021,113

Usage:
    python truthfulqa_coherence.py [--n 50] [--model google/gemma-2-2b]
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_truthfulqa(n_questions=50, seed=42):
    """Load TruthfulQA multiple choice questions."""
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    except Exception as e:
        print(f"[!] Failed to load TruthfulQA from HuggingFace: {e}")
        print("[!] Trying local fallback...")
        # Fallback: generate synthetic test prompts for structure validation
        return _generate_fallback_prompts(n_questions)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_questions, len(ds)), replace=False)

    questions = []
    for idx in indices:
        item = ds[int(idx)]
        question = item["question"]
        # mc1_targets: first answer is correct, rest are incorrect
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]

        correct_answers = [c for c, l in zip(choices, labels) if l == 1]
        incorrect_answers = [c for c, l in zip(choices, labels) if l == 0]

        if correct_answers and incorrect_answers:
            questions.append({
                "question": question,
                "correct_answer": correct_answers[0],
                "incorrect_answer": incorrect_answers[0],  # most plausible wrong answer
            })

    print(f"[+] Loaded {len(questions)} TruthfulQA questions")
    return questions


def _generate_fallback_prompts(n):
    """Fallback prompts if TruthfulQA dataset isn't available."""
    # These are NOT substitutes — just for testing the pipeline
    pairs = [
        ("What is the capital of France?", "Paris", "London"),
        ("What causes the seasons on Earth?", "The tilt of Earth's axis", "Distance from the Sun"),
        ("What is the speed of light?", "About 300,000 km/s", "About 300,000 miles per hour"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare", "Charles Dickens"),
        ("What is the largest planet in our solar system?", "Jupiter", "Saturn"),
    ]
    questions = []
    for i in range(min(n, len(pairs))):
        q, correct, incorrect = pairs[i]
        questions.append({
            "question": q,
            "correct_answer": correct,
            "incorrect_answer": incorrect,
        })
    return questions


def run_experiment(n_questions=50, model_name="google/gemma-2-2b"):
    """Run the full TruthfulQA coherence experiment."""
    from depth_scorer import DepthScorer

    print(f"\n{'='*60}")
    print(f"FATHOM COHERENCE EXPERIMENT — TruthfulQA")
    print(f"Model: {model_name}")
    print(f"Target n: {n_questions}")
    print(f"{'='*60}\n")

    # Load questions
    questions = load_truthfulqa(n_questions)
    if not questions:
        print("[!] No questions loaded. Exiting.")
        return

    # Initialize scorer
    print("[+] Loading model and scorer...")
    scorer = DepthScorer(model_name=model_name)

    results = []
    t_start = time.time()

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['question'][:60]}...")

        # Score the correct answer framing
        correct_prompt = f"Q: {q['question']}\nA: {q['correct_answer']}"
        # Score the hallucinated/incorrect answer framing
        hallucinated_prompt = f"Q: {q['question']}\nA: {q['incorrect_answer']}"

        try:
            correct_result = scorer.score(correct_prompt)
            gc.collect()
            torch.cuda.empty_cache()

            hallucinated_result = scorer.score(hallucinated_prompt)
            gc.collect()
            torch.cuda.empty_cache()

            entry = {
                "question": q["question"],
                "correct_answer": q["correct_answer"],
                "incorrect_answer": q["incorrect_answer"],
                "correct": {
                    "K": correct_result.depth_score,
                    "C": correct_result.coherence_score,
                    "C_delta": correct_result.c_delta,
                    "layer_coherence": correct_result.layer_coherence,
                    "feature_count": correct_result.feature_count,
                    "entropy": correct_result.layer_entropy,
                    "early_ratio": correct_result.early_ratio,
                    "mid_ratio": correct_result.mid_ratio,
                    "late_ratio": correct_result.late_ratio,
                },
                "hallucinated": {
                    "K": hallucinated_result.depth_score,
                    "C": hallucinated_result.coherence_score,
                    "C_delta": hallucinated_result.c_delta,
                    "layer_coherence": hallucinated_result.layer_coherence,
                    "feature_count": hallucinated_result.feature_count,
                    "entropy": hallucinated_result.layer_entropy,
                    "early_ratio": hallucinated_result.early_ratio,
                    "mid_ratio": hallucinated_result.mid_ratio,
                    "late_ratio": hallucinated_result.late_ratio,
                },
            }
            results.append(entry)

            cd_correct = correct_result.c_delta
            cd_halluc = hallucinated_result.c_delta
            cd_str_c = f"Cd={cd_correct:+.4f}" if cd_correct is not None else "Cd=None"
            cd_str_h = f"Cd={cd_halluc:+.4f}" if cd_halluc is not None else "Cd=None"
            c_correct = correct_result.coherence_score or 0
            c_halluc = hallucinated_result.coherence_score or 0
            print(f"    correct:      K={correct_result.depth_score:.3f}  C={c_correct:.4f}  {cd_str_c}  features={correct_result.feature_count}")
            print(f"    hallucinated: K={hallucinated_result.depth_score:.3f}  C={c_halluc:.4f}  {cd_str_h}  features={hallucinated_result.feature_count}")

        except Exception as e:
            print(f"    [!] Error: {e}")
            continue

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Completed {len(results)}/{len(questions)} pairs in {elapsed:.1f}s")

    # Save results
    outdir = Path("truthfulqa_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"coherence_{model_name.replace('/', '_')}_n{len(results)}.json"

    output = {
        "experiment": "truthfulqa_coherence",
        "model": model_name,
        "n_pairs": len(results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": elapsed,
        "results": results,
    }

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[+] Results saved: {outfile}")

    # Quick analysis
    analyze_results(results)

    return output


def analyze_results(results):
    """Print statistical summary of K and C across correct vs hallucinated."""
    from scipy import stats

    correct_K = [r["correct"]["K"] for r in results if r["correct"]["K"] is not None]
    halluc_K = [r["hallucinated"]["K"] for r in results if r["hallucinated"]["K"] is not None]
    correct_C = [r["correct"]["C"] for r in results if r["correct"]["C"] is not None]
    halluc_C = [r["hallucinated"]["C"] for r in results if r["hallucinated"]["C"] is not None]

    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    if correct_K and halluc_K:
        k_stat, k_p = stats.mannwhitneyu(correct_K, halluc_K, alternative="two-sided")
        k_d = (np.mean(halluc_K) - np.mean(correct_K)) / np.sqrt(
            (np.std(correct_K)**2 + np.std(halluc_K)**2) / 2
        ) if np.std(correct_K) > 0 or np.std(halluc_K) > 0 else 0
        print(f"\nK (Depth) — H2: depth should NOT differ")
        print(f"  correct:      mean={np.mean(correct_K):.4f}  std={np.std(correct_K):.4f}")
        print(f"  hallucinated: mean={np.mean(halluc_K):.4f}  std={np.std(halluc_K):.4f}")
        print(f"  Mann-Whitney U={k_stat:.1f}  p={k_p:.6f}  Cohen's d={k_d:.3f}")
        print(f"  {'** SIGNIFICANT **' if k_p < 0.05 else 'not significant'}")

    if correct_C and halluc_C:
        c_stat, c_p = stats.mannwhitneyu(correct_C, halluc_C, alternative="two-sided")
        c_d = (np.mean(halluc_C) - np.mean(correct_C)) / np.sqrt(
            (np.std(correct_C)**2 + np.std(halluc_C)**2) / 2
        ) if np.std(correct_C) > 0 or np.std(halluc_C) > 0 else 0
        print(f"\nC (Coherence) — H1: coherence SHOULD differ")
        print(f"  correct:      mean={np.mean(correct_C):.4f}  std={np.std(correct_C):.4f}")
        print(f"  hallucinated: mean={np.mean(halluc_C):.4f}  std={np.std(halluc_C):.4f}")
        print(f"  Mann-Whitney U={c_stat:.1f}  p={c_p:.6f}  Cohen's d={c_d:.3f}")
        print(f"  {'** SIGNIFICANT **' if c_p < 0.05 else 'not significant'}")

        if np.mean(halluc_C) > np.mean(correct_C):
            print(f"\n  -> Hallucinated responses show HIGHER coherence (wrong attractor hypothesis)")
        else:
            print(f"\n  -> Hallucinated responses show LOWER coherence (coherence collapse hypothesis)")
    else:
        print("\n[!] Coherence scores unavailable — SAE decoder vectors not accessible.")
        print("[!] This means coherence computation needs a different access path.")
        print("[!] K scores are still valid. C requires SAE decoder weight access.")

    # C_delta analysis (THE key metric — late-layer coherence collapse)
    correct_Cd = [r["correct"]["C_delta"] for r in results if r["correct"].get("C_delta") is not None]
    halluc_Cd = [r["hallucinated"]["C_delta"] for r in results if r["hallucinated"].get("C_delta") is not None]

    if correct_Cd and halluc_Cd:
        cd_stat, cd_p = stats.mannwhitneyu(correct_Cd, halluc_Cd, alternative="two-sided")
        cd_d = (np.mean(halluc_Cd) - np.mean(correct_Cd)) / np.sqrt(
            (np.std(correct_Cd)**2 + np.std(halluc_Cd)**2) / 2
        ) if np.std(correct_Cd) > 0 or np.std(halluc_Cd) > 0 else 0
        print(f"\nC_delta (Late-Early coherence) — HALLUCINATION SIGNATURE")
        print(f"  correct:      mean={np.mean(correct_Cd):+.4f}  std={np.std(correct_Cd):.4f}")
        print(f"  hallucinated: mean={np.mean(halluc_Cd):+.4f}  std={np.std(halluc_Cd):.4f}")
        print(f"  Mann-Whitney U={cd_stat:.1f}  p={cd_p:.6f}  Cohen's d={cd_d:.3f}")
        print(f"  {'** SIGNIFICANT **' if cd_p < 0.05 else 'not significant'}")
        if np.mean(halluc_Cd) < np.mean(correct_Cd):
            print(f"\n  -> Hallucinated responses show MORE NEGATIVE C_delta (late-layer coherence collapse)")
        else:
            print(f"\n  -> Hallucinated responses show LESS NEGATIVE C_delta")

    # Quadrant analysis
    if correct_K and correct_C and halluc_K and halluc_C:
        median_K = np.median(correct_K + halluc_K)
        median_C = np.median(correct_C + halluc_C)

        def quadrant(k, c):
            if k >= median_K and c >= median_C: return "high-K, high-C"
            if k >= median_K and c < median_C: return "high-K, low-C"
            if k < median_K and c >= median_C: return "low-K, high-C"
            return "low-K, low-C"

        print(f"\nQUADRANT DISTRIBUTION (median split: K={median_K:.3f}, C={median_C:.4f})")
        for label, k_vals, c_vals in [
            ("correct", correct_K, correct_C),
            ("hallucinated", halluc_K, halluc_C),
        ]:
            quads = [quadrant(k, c) for k, c in zip(k_vals, c_vals)]
            print(f"  {label}:")
            for q in ["high-K, high-C", "high-K, low-C", "low-K, high-C", "low-K, low-C"]:
                count = quads.count(q)
                pct = count / len(quads) * 100 if quads else 0
                print(f"    {q}: {count} ({pct:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthfulQA coherence experiment")
    parser.add_argument("--n", type=int, default=50, help="Number of questions")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b", help="Model name")
    args = parser.parse_args()

    run_experiment(n_questions=args.n, model_name=args.model)
