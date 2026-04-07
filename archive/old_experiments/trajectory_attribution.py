"""
trajectory_attribution.py — Attribution-Based Cognitive Trajectory
==================================================================

The RIGHT way: uses circuit_tracer.attribute() per token instead of
the fast encoder-only K_fast proxy. Slower but the signal is proven
(p=0.000051 for depth, p=0.040 for C_delta).

Per token captures: K (attribution depth), C_delta, coherence_score,
layer_entropy, full layer_profile, layer_coherence.

Expected runtime: ~10-30s per token, ~50 pairs × ~15 tokens = ~750
attribution passes = 2-6 hours on RTX 4070.

Patent pending: US 64/020,489 & 64/021,113
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_truthfulqa(n: int = 50, seed: int = 42):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    questions = []
    for idx in indices:
        item = ds[int(idx)]
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]
        correct = [c for c, l in zip(choices, labels) if l == 1]
        incorrect = [c for c, l in zip(choices, labels) if l == 0]
        if correct and incorrect:
            questions.append({
                "question": item["question"],
                "correct_answer": correct[0],
                "incorrect_answer": incorrect[0],
            })
    print(f"[+] Loaded {len(questions)} TruthfulQA pairs (seed={seed})")
    return questions


def extract_result(r):
    """Extract serializable fields from DepthResult."""
    return {
        "depth_score": r.depth_score,
        "coherence_score": r.coherence_score,
        "c_delta": r.c_delta,
        "layer_entropy": r.layer_entropy,
        "feature_count": r.feature_count,
        "early_ratio": r.early_ratio,
        "mid_ratio": r.mid_ratio,
        "late_ratio": r.late_ratio,
        "layer_profile": {str(k): v for k, v in (r.layer_profile or {}).items()},
        "layer_coherence": {str(k): v for k, v in (r.layer_coherence or {}).items()} if r.layer_coherence else None,
    }


def run_trajectory(scorer, prompt, max_tokens=15):
    """
    Generate tokens and run full attribution at each position.

    Returns list of per-token results with ALL metrics.
    """
    inner = scorer._get_inner_model()
    # Fallback: the ReplacementModel itself may have to_tokens (TransformerLens)
    if inner is None and hasattr(scorer._model, "to_tokens"):
        inner = scorer._model
    if inner is None:
        raise ValueError("Inner model not accessible")

    trajectory = []
    current_prompt = prompt

    for step in range(max_tokens):
        # Predict next token
        tokens = inner.to_tokens(current_prompt)
        with torch.no_grad():
            logits = inner(tokens)
        pred_id = logits[0, -1].argmax().item()
        pred_token = inner.tokenizer.decode([pred_id])

        # Verify single-token
        re_encoded = inner.tokenizer.encode(pred_token, add_special_tokens=False)
        if len(re_encoded) != 1:
            print(f"    [step {step}] multi-token prediction, stopping")
            break

        # EOS check
        eos_id = getattr(inner.tokenizer, "eos_token_id", None)
        if pred_id == eos_id:
            print(f"    [step {step}] EOS")
            break

        # Full attribution
        try:
            result = scorer.score(current_prompt, target=pred_token)
            entry = extract_result(result)
            entry["token_idx"] = step
            entry["token_str"] = pred_token
            trajectory.append(entry)
        except Exception as e:
            print(f"    [step {step}] attribution failed: {e}")
            trajectory.append({
                "token_idx": step,
                "token_str": pred_token,
                "depth_score": None,
                "c_delta": None,
            })

        current_prompt += pred_token
        torch.cuda.empty_cache()
        gc.collect()

    return trajectory


def run_experiment(
    n_questions: int = 50,
    max_tokens: int = 15,
    seed: int = 42,
    output_dir: str = "trajectory_attribution_results",
):
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    checkpoint_path = out_path / "checkpoint.json"

    # Resume
    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            completed = json.load(f)
        print(f"[+] Resuming: {len(completed)} pairs done")

    questions = load_truthfulqa(n_questions, seed)

    # Load scorer
    print("[+] Loading DepthScorer...")
    from depth_scorer import DepthScorer
    scorer = DepthScorer(model_name="google/gemma-2-2b")

    results = []
    for i, q in enumerate(questions):
        qid = f"q{i}"
        if qid in completed:
            results.append(completed[qid])
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(questions)}] {q['question'][:80]}")
        print(f"{'='*60}")

        # Correct
        correct_prompt = f"Q: {q['question']}\nA: {q['correct_answer']}"
        print(f"  [correct] attributing...")
        t0 = time.time()
        try:
            c_traj = run_trajectory(scorer, correct_prompt, max_tokens)
            c_time = time.time() - t0
            print(f"  [correct] {len(c_traj)} tokens in {c_time:.1f}s")
            correct_data = {
                "prompt": correct_prompt,
                "label": "correct",
                "answer": q["correct_answer"],
                "trajectory": c_traj,
                "time_s": c_time,
            }
        except Exception as e:
            print(f"  [correct] FAILED: {e}")
            correct_data = None

        gc.collect(); torch.cuda.empty_cache()

        # Hallucinated
        halluc_prompt = f"Q: {q['question']}\nA: {q['incorrect_answer']}"
        print(f"  [hallucinated] attributing...")
        t0 = time.time()
        try:
            h_traj = run_trajectory(scorer, halluc_prompt, max_tokens)
            h_time = time.time() - t0
            print(f"  [hallucinated] {len(h_traj)} tokens in {h_time:.1f}s")
            halluc_data = {
                "prompt": halluc_prompt,
                "label": "hallucinated",
                "answer": q["incorrect_answer"],
                "trajectory": h_traj,
                "time_s": h_time,
            }
        except Exception as e:
            print(f"  [hallucinated] FAILED: {e}")
            halluc_data = None

        gc.collect(); torch.cuda.empty_cache()

        pair = {
            "question": q["question"],
            "correct": correct_data,
            "hallucinated": halluc_data,
        }
        results.append(pair)
        completed[qid] = pair

        # Print summary
        if correct_data and halluc_data and c_traj and h_traj:
            c_ks = [t["depth_score"] for t in c_traj if t.get("depth_score") is not None]
            h_ks = [t["depth_score"] for t in h_traj if t.get("depth_score") is not None]
            c_cds = [t["c_delta"] for t in c_traj if t.get("c_delta") is not None]
            h_cds = [t["c_delta"] for t in h_traj if t.get("c_delta") is not None]
            if c_ks and h_ks:
                print(f"  K_attrib — correct: {np.mean(c_ks):.3f}  halluc: {np.mean(h_ks):.3f}")
            if c_cds and h_cds:
                print(f"  C_delta  — correct: {np.mean(c_cds):.4f}  halluc: {np.mean(h_cds):.4f}")

        # Checkpoint every 5
        if (i + 1) % 5 == 0:
            print(f"\n  [checkpoint] saving {len(completed)} pairs...")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(completed, f)

    # Final save
    final_path = out_path / "trajectories_final.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved {len(results)} pairs to {final_path}")

    # Quick stats
    print(f"\n{'='*60}")
    print("ATTRIBUTION TRAJECTORY SUMMARY")
    print(f"{'='*60}")

    all_c_k = []; all_h_k = []
    all_c_cd = []; all_h_cd = []
    for pair in results:
        c = pair.get("correct"); h = pair.get("hallucinated")
        if not c or not h: continue
        ct = c.get("trajectory", []); ht = h.get("trajectory", [])
        all_c_k.extend([t["depth_score"] for t in ct if t.get("depth_score") is not None])
        all_h_k.extend([t["depth_score"] for t in ht if t.get("depth_score") is not None])
        all_c_cd.extend([t["c_delta"] for t in ct if t.get("c_delta") is not None])
        all_h_cd.extend([t["c_delta"] for t in ht if t.get("c_delta") is not None])

    if all_c_k:
        print(f"K (attribution): correct={np.mean(all_c_k):.3f} halluc={np.mean(all_h_k):.3f} delta={np.mean(all_h_k)-np.mean(all_c_k):+.3f}")
    if all_c_cd:
        print(f"C_delta: correct={np.mean(all_c_cd):.4f} halluc={np.mean(all_h_cd):.4f} delta={np.mean(all_h_cd)-np.mean(all_c_cd):+.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of pairs")
    parser.add_argument("--max-tokens", type=int, default=15, help="Max tokens per generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="trajectory_attribution_results")
    args = parser.parse_args()

    run_experiment(args.n, args.max_tokens, args.seed, args.output)
