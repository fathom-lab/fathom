"""
fathom_diagnostics.py — Full Three-Axis Cognitive Diagnostic
============================================================
Axis K: Depth (from depth_scorer.py)
Axis C: Coherence (per-layer, from depth_scorer.py)
Axis D: Drift (NEW — divergence from correct answer feature space)

Lock-in signature:  high C + high D + normal K
Coherence collapse: low C + normal K + high D
Depth anomaly:      high/low K regardless of C, D

Lock-in score = C * D — single number combining coherence with drift.

Based on darkflobi's three-axis framework (April 1, 2026).
Adapted to work with circuit_tracer ReplacementModel + TranscoderSet.

Usage:
    python fathom_diagnostics.py --model google/gemma-2-2b --n 50

Patent pending: US 64/020,489 & 64/021,113
"""

import gc
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# DRIFT (D axis) — distance from correct answer feature space
# ═══════════════════════════════════════════════════════════════════════════════

def compute_drift_from_results(correct_result, hallucinated_result) -> dict:
    """
    D = 1 - cosine_sim(hallucinated layer_profile, correct layer_profile)

    Uses the layer concentration profiles from DepthResult as a proxy
    for feature-space position. D=0 means identical processing pattern.
    D→1 means orthogonal processing.

    Also computes per-layer drift using layer_coherence profiles.
    """
    # Layer profile drift (feature count distribution across layers)
    cp = correct_result.layer_profile
    hp = hallucinated_result.layer_profile

    all_layers = sorted(set(list(cp.keys()) + list(hp.keys())))
    if not all_layers:
        return {"D_mean": None, "D_profile": None}

    correct_vec = np.array([cp.get(l, 0) for l in all_layers], dtype=np.float64)
    halluc_vec = np.array([hp.get(l, 0) for l in all_layers], dtype=np.float64)

    # Normalize to distributions
    c_norm = np.linalg.norm(correct_vec)
    h_norm = np.linalg.norm(halluc_vec)

    if c_norm > 0 and h_norm > 0:
        cos_sim = np.dot(correct_vec, halluc_vec) / (c_norm * h_norm)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        D_profile = 1.0 - cos_sim
    else:
        D_profile = None

    # Per-layer coherence drift (if available)
    D_coherence = None
    if correct_result.layer_coherence and hallucinated_result.layer_coherence:
        c_coh = correct_result.layer_coherence
        h_coh = hallucinated_result.layer_coherence
        shared_layers = [l for l in c_coh if c_coh[l] is not None and l in h_coh and h_coh[l] is not None]
        if len(shared_layers) >= 2:
            c_vec = np.array([c_coh[l] for l in shared_layers])
            h_vec = np.array([h_coh[l] for l in shared_layers])
            cn = np.linalg.norm(c_vec)
            hn = np.linalg.norm(h_vec)
            if cn > 0 and hn > 0:
                D_coherence = 1.0 - np.dot(c_vec, h_vec) / (cn * hn)

    D_mean = D_profile if D_profile is not None else D_coherence

    return {
        "D_mean": float(D_mean) if D_mean is not None else None,
        "D_profile": float(D_profile) if D_profile is not None else None,
        "D_coherence": float(D_coherence) if D_coherence is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOCK-IN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def classify_cognitive_state(
    K: float,
    C: Optional[float],
    D: Optional[float],
    C_delta: Optional[float],
    K_baseline: float = 8.65,  # from TruthfulQA baseline
    C_threshold: float = 0.003,
    D_threshold: float = 0.1,
    Cd_threshold: float = 0.0074,  # midpoint of correct/hallucinated C_delta
) -> str:
    """
    Three-axis cognitive state classifier.

    LOCK_IN:            high C_delta + high D + normal K — wrong attractor
    COHERENCE_COLLAPSE: low C + high D + normal K — scattered features
    DEEP_LOCK_IN:       high C_delta + high D + anomalous K
    NOMINAL:            normal everything
    ANOMALOUS:          doesn't fit clean category
    """
    if C_delta is None or D is None:
        return "UNKNOWN"

    k_high = K > K_baseline * 1.03
    k_low = K < K_baseline * 0.97
    cd_high = C_delta > Cd_threshold
    d_high = D > D_threshold

    if cd_high and d_high and not k_high:
        return "LOCK_IN"
    elif not cd_high and d_high and not k_high:
        return "COHERENCE_COLLAPSE"
    elif cd_high and d_high and k_high:
        return "DEEP_LOCK_IN"
    elif not d_high and not k_high and not k_low:
        return "NOMINAL"
    else:
        return "ANOMALOUS"


# ═══════════════════════════════════════════════════════════════════════════════
# FULL THREE-AXIS EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(n_samples=50, model_name="google/gemma-2-2b"):
    """Run full three-axis diagnostic on TruthfulQA."""
    from depth_scorer import DepthScorer
    from datasets import load_dataset
    from scipy import stats

    print(f"\n{'='*60}")
    print(f"FATHOM THREE-AXIS DIAGNOSTIC — TruthfulQA")
    print(f"Model: {model_name}")
    print(f"Axes: K (depth) + C/C_delta (coherence) + D (drift)")
    print(f"Target n: {n_samples}")
    print(f"{'='*60}\n")

    # Load TruthfulQA
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = np.random.RandomState(42)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    # Init scorer
    scorer = DepthScorer(model_name=model_name)

    results = []
    t_start = time.time()

    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        question = item["question"]
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]

        correct = [c for c, l in zip(choices, labels) if l == 1]
        incorrect = [c for c, l in zip(choices, labels) if l == 0]

        if not correct or not incorrect:
            continue

        correct_prompt = f"Q: {question}\nA: {correct[0]}"
        halluc_prompt = f"Q: {question}\nA: {incorrect[0]}"

        print(f"[{i+1}/{n_samples}] {question[:60]}...")

        try:
            correct_result = scorer.score(correct_prompt)
            gc.collect(); torch.cuda.empty_cache()

            halluc_result = scorer.score(halluc_prompt)
            gc.collect(); torch.cuda.empty_cache()

            # Compute drift
            drift = compute_drift_from_results(correct_result, halluc_result)

            # Classify
            state = classify_cognitive_state(
                K=halluc_result.depth_score,
                C=halluc_result.coherence_score,
                D=drift["D_mean"],
                C_delta=halluc_result.c_delta,
            )

            # Lock-in score = C_delta * D
            lock_in = None
            if halluc_result.c_delta is not None and drift["D_mean"] is not None:
                lock_in = halluc_result.c_delta * drift["D_mean"]

            entry = {
                "question": question,
                # Correct
                "K_correct": correct_result.depth_score,
                "C_correct": correct_result.coherence_score,
                "Cd_correct": correct_result.c_delta,
                "features_correct": correct_result.feature_count,
                # Hallucinated
                "K_hallucinated": halluc_result.depth_score,
                "C_hallucinated": halluc_result.coherence_score,
                "Cd_hallucinated": halluc_result.c_delta,
                "features_hallucinated": halluc_result.feature_count,
                # Drift
                "D_mean": drift["D_mean"],
                "D_profile": drift["D_profile"],
                "D_coherence": drift["D_coherence"],
                # Classification
                "state": state,
                "lock_in_score": lock_in,
            }
            results.append(entry)

            d_str = f"D={drift['D_mean']:.4f}" if drift['D_mean'] is not None else "D=None"
            cd_str = f"Cd={halluc_result.c_delta:+.4f}" if halluc_result.c_delta is not None else "Cd=None"
            li_str = f"LI={lock_in:.6f}" if lock_in is not None else "LI=None"
            print(f"    {state:20s} K={halluc_result.depth_score:.3f} {cd_str} {d_str} {li_str}")

        except Exception as e:
            print(f"    [!] Error: {e}")
            continue

    elapsed = time.time() - t_start
    print(f"\nCompleted {len(results)}/{n_samples} in {elapsed:.1f}s")

    # Save
    outdir = Path("truthfulqa_results")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"three_axis_{model_name.replace('/', '_')}_n{len(results)}.json"

    output = {
        "experiment": "three_axis_diagnostic",
        "model": model_name,
        "n_pairs": len(results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": elapsed,
        "results": results,
    }

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[+] Results saved: {outfile}")

    # ── Statistics ──
    analyze_three_axis(results)

    return output


def analyze_three_axis(results):
    """Full three-axis statistical analysis."""
    from scipy import stats

    print(f"\n{'='*60}")
    print("THREE-AXIS STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    def test_axis(label, correct_key, halluc_key):
        c_vals = [r[correct_key] for r in results if r[correct_key] is not None]
        h_vals = [r[halluc_key] for r in results if r[halluc_key] is not None]
        if len(c_vals) < 3 or len(h_vals) < 3:
            print(f"\n{label}: insufficient data")
            return
        stat, p = stats.mannwhitneyu(c_vals, h_vals, alternative="two-sided")
        pooled_std = np.sqrt((np.std(c_vals)**2 + np.std(h_vals)**2) / 2)
        d = (np.mean(h_vals) - np.mean(c_vals)) / pooled_std if pooled_std > 0 else 0
        sig = "** SIGNIFICANT **" if p < 0.05 else "not significant"
        print(f"\n{label}")
        print(f"  correct:      mean={np.mean(c_vals):.4f}  std={np.std(c_vals):.4f}")
        print(f"  hallucinated: mean={np.mean(h_vals):.4f}  std={np.std(h_vals):.4f}")
        print(f"  U={stat:.1f}  p={p:.6f}  d={d:.3f}  {sig}")

    test_axis("K (Depth)", "K_correct", "K_hallucinated")
    test_axis("C (Global Coherence)", "C_correct", "C_hallucinated")
    test_axis("C_delta (Late-Early Coherence)", "Cd_correct", "Cd_hallucinated")

    # Drift (only hallucinated has meaningful D)
    d_vals = [r["D_mean"] for r in results if r["D_mean"] is not None]
    if d_vals:
        print(f"\nD (Drift from correct)")
        print(f"  hallucinated: mean={np.mean(d_vals):.4f}  std={np.std(d_vals):.4f}")

    # Lock-in scores
    li_vals = [r["lock_in_score"] for r in results if r["lock_in_score"] is not None]
    if li_vals:
        print(f"\nLock-In Score (C_delta x D)")
        print(f"  mean={np.mean(li_vals):.6f}  std={np.std(li_vals):.6f}")

    # State distribution
    states = [r["state"] for r in results]
    print(f"\nCognitive State Distribution:")
    for state in sorted(set(states)):
        count = states.count(state)
        print(f"  {state}: {count} ({count/len(states)*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fathom three-axis diagnostic")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    args = parser.parse_args()

    run_experiment(n_samples=args.n, model_name=args.model)
