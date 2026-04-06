"""
run_truthfulqa_n200.py — TruthfulQA C_delta experiment, n=200, with checkpointing.

Resumes from checkpoint if interrupted. Saves every 10 pairs.
Target: n=200, model=google/gemma-2-2b

Usage:
    python run_truthfulqa_n200.py [--resume]
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout — prevents charmap errors on Windows for unicode in questions
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

MODEL      = "google/gemma-2-2b"
N_TARGET   = 200
SEED       = 99          # different seed from n=50 run (seed=42) — fully fresh sample
CHECKPOINT = Path("truthfulqa_results/n200_checkpoint.json")
OUTFILE    = Path("truthfulqa_results/coherence_n200_final.json")
SAVE_EVERY = 10          # checkpoint every N pairs


def load_questions(n, seed):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    questions = []
    for idx in indices:
        item = ds[int(idx)]
        choices = item["mc1_targets"]["choices"]
        labels  = item["mc1_targets"]["labels"]
        correct   = [c for c, l in zip(choices, labels) if l == 1]
        incorrect = [c for c, l in zip(choices, labels) if l == 0]
        if correct and incorrect:
            questions.append({
                "question":         item["question"],
                "correct_answer":   correct[0],
                "incorrect_answer": incorrect[0],
            })
    print(f"[+] Loaded {len(questions)} TruthfulQA questions (seed={seed})")
    return questions


def score_pair(scorer, q):
    correct_prompt     = f"Q: {q['question']}\nA: {q['correct_answer']}"
    hallucinated_prompt = f"Q: {q['question']}\nA: {q['incorrect_answer']}"

    cr = scorer.score(correct_prompt)
    gc.collect(); torch.cuda.empty_cache()

    hr = scorer.score(hallucinated_prompt)
    gc.collect(); torch.cuda.empty_cache()

    def extract(r):
        return {
            "K":             r.depth_score,
            "C":             r.coherence_score,
            "C_delta":       r.c_delta,
            "layer_coherence": r.layer_coherence,
            "feature_count": r.feature_count,
            "entropy":       r.layer_entropy,
            "early_ratio":   r.early_ratio,
            "mid_ratio":     r.mid_ratio,
            "late_ratio":    r.late_ratio,
        }

    return {
        "question":         q["question"],
        "correct_answer":   q["correct_answer"],
        "incorrect_answer": q["incorrect_answer"],
        "correct":          extract(cr),
        "hallucinated":     extract(hr),
    }


def save_checkpoint(results, elapsed):
    CHECKPOINT.parent.mkdir(exist_ok=True)
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({
            "n_complete": len(results),
            "elapsed_s":  elapsed,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results":    results,
        }, f, indent=2, default=str)
    print(f"    [checkpoint] {len(results)} pairs saved.")


def analyze(results):
    from scipy import stats

    def vals(key, sub): return [r[sub][key] for r in results if r[sub].get(key) is not None]

    print(f"\n{'='*60}")
    print(f"RESULTS — n={len(results)}")
    print(f"{'='*60}")

    for metric, label, h in [
        ("K",       "Depth (K) — expect NO difference",       "H2"),
        ("C",       "Coherence (C) — expect difference",      "H1"),
        ("C_delta", "C_delta — HALLUCINATION SIGNATURE",      "H1b"),
    ]:
        cvals = vals(metric, "correct")
        hvals = vals(metric, "hallucinated")
        if not cvals or not hvals:
            print(f"\n{h} {label}: no data")
            continue
        stat, p = stats.mannwhitneyu(cvals, hvals, alternative="two-sided")
        pooled_std = np.sqrt((np.std(cvals)**2 + np.std(hvals)**2) / 2)
        d = (np.mean(hvals) - np.mean(cvals)) / pooled_std if pooled_std > 0 else 0
        sig = "** SIGNIFICANT **" if p < 0.05 else "not significant"
        print(f"\n{h} — {label}")
        print(f"  correct:      mean={np.mean(cvals):+.4f}  std={np.std(cvals):.4f}")
        print(f"  hallucinated: mean={np.mean(hvals):+.4f}  std={np.std(hvals):.4f}")
        print(f"  p={p:.6f}  Cohen's d={d:.3f}  {sig}")


def main(resume=False):
    results   = []
    start_idx = 0

    if resume and CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            ckpt = json.load(f)
        results   = ckpt["results"]
        start_idx = len(results)
        print(f"[+] Resuming from checkpoint: {start_idx} pairs done")

    questions = load_questions(N_TARGET, SEED)

    if start_idx >= len(questions):
        print("[+] Already complete. Running final analysis.")
        analyze(results)
        return

    print(f"[+] Loading model: {MODEL}")
    from depth_scorer import DepthScorer
    scorer = DepthScorer(model_name=MODEL)

    t_start = time.time()

    for i, q in enumerate(questions[start_idx:], start=start_idx):
        idx_display = i + 1
        print(f"\n[{idx_display}/{len(questions)}] {q['question'][:70]}...")

        try:
            entry = score_pair(scorer, q)
            results.append(entry)

            cd_c = entry["correct"]["C_delta"]
            cd_h = entry["hallucinated"]["C_delta"]
            print(f"    correct:      K={entry['correct']['K']:.3f}  C_delta={cd_c:+.4f}")
            print(f"    hallucinated: K={entry['hallucinated']['K']:.3f}  C_delta={cd_h:+.4f}")
            if cd_h is not None and cd_c is not None:
                print(f"    delta: {cd_h - cd_c:+.4f}  {'HALLUC HIGHER [!]' if cd_h > cd_c else 'correct higher'}")

        except Exception as e:
            print(f"    [!] Error on pair {idx_display}: {e}")
            continue

        if len(results) % SAVE_EVERY == 0:
            save_checkpoint(results, time.time() - t_start)

    elapsed = time.time() - t_start
    print(f"\n[+] Done: {len(results)}/{len(questions)} pairs in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Final save
    OUTFILE.parent.mkdir(exist_ok=True)
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump({
            "experiment":  "truthfulqa_coherence_n200",
            "model":       MODEL,
            "n_pairs":     len(results),
            "seed":        SEED,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s":   elapsed,
            "results":     results,
        }, f, indent=2, default=str)
    print(f"[+] Final results: {OUTFILE}")

    analyze(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    main(resume=args.resume)
