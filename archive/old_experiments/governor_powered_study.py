"""
governor_powered_study.py — Powered Cognitive Governor Experiment
=================================================================

Three-arm study on n>=30 TruthfulQA hallucinations:
  Arm A: Monitored baseline (no intervention)
  Arm B: Directional steering (alpha=2.0)
  Arm C: Noise injection (control)

Measures:
  - C_delta trajectory per arm
  - Mean C_delta pre/post intervention
  - Intervention count per question
  - Cohen's d effect size, 95% CI, paired t-test p-value
  - Output text comparison

Selects top-N hallucination candidates from n200 baseline by C_delta.

Usage:
    python governor_powered_study.py [--n 30] [--alpha 2.0] [--max-tokens 30]

Patent pending: US 64/020,489 & 64/021,113
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "truthfulqa_results")
BASELINE_PATH = os.path.join(RESULTS_DIR, "n200_checkpoint.json")


def load_candidates(n: int, min_cdelta: float = 0.008) -> list:
    """Load top-N hallucination candidates from n200 baseline by C_delta."""
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    scored = []
    for i, r in enumerate(results):
        h_cd = r["hallucinated"]["C_delta"]
        c_cd = r["correct"]["C_delta"]
        if h_cd >= min_cdelta:
            scored.append({
                "idx": i,
                "question": r["question"],
                "correct_answer": r["correct_answer"],
                "incorrect_answer": r["incorrect_answer"],
                "baseline_halluc_cdelta": h_cd,
                "baseline_correct_cdelta": c_cd,
                "baseline_gap": h_cd - c_cd,
            })

    # Sort by halluc C_delta descending, take top N
    scored.sort(key=lambda x: x["baseline_halluc_cdelta"], reverse=True)
    selected = scored[:n]
    print(f"[Candidates] {len(scored)} above threshold {min_cdelta}, selected top {len(selected)}")
    return selected


def run_arm_monitored(steerer, question: str, max_tokens: int):
    """Arm A: baseline monitoring only."""
    return steerer.generate_monitored(question, max_tokens=max_tokens)


def run_arm_steered(steerer, question: str, max_tokens: int, alpha: float):
    """Arm B: directional steering."""
    old_strength = steerer.steering_strength
    steerer.steering_strength = alpha
    result = steerer.generate_steered(question, max_tokens=max_tokens)
    steerer.steering_strength = old_strength
    return result


def run_arm_noise(steerer, question: str, max_tokens: int, noise_scale: float = 0.5):
    """
    Arm C: noise injection control.
    Replaces the steering direction with random noise of same magnitude.
    Uses the steered pipeline but with randomized intervention.
    """
    # Save original method
    import types
    original_compute = steerer._compute_lock_in_direction

    def _random_direction(self_ref, hidden_state, layer_idx):
        """Return random direction instead of lock-in direction."""
        d_model = hidden_state.shape[-1]
        rand_dir = torch.randn(d_model, device=hidden_state.device, dtype=torch.float32)
        rand_dir = rand_dir / rand_dir.norm()
        return rand_dir * noise_scale

    # Monkey-patch for noise arm
    import torch
    steerer._compute_lock_in_direction = types.MethodType(
        lambda self, h, l: _random_direction(self, h, l), steerer
    )
    old_strength = steerer.steering_strength
    steerer.steering_strength = 2.0  # Same as steered arm

    result = steerer.generate_steered(question, max_tokens=max_tokens)

    # Restore
    steerer._compute_lock_in_direction = original_compute
    steerer.steering_strength = old_strength
    return result


def compute_stats(monitored_cdeltas, steered_cdeltas, noise_cdeltas):
    """Compute paired statistics across arms."""
    results = {}

    # Filter to pairs where both have valid C_delta
    def mean_cdelta(trajectory):
        valid = [x for x in trajectory if x is not None]
        return np.mean(valid) if valid else None

    # Arm A vs Arm B (monitored vs steered)
    paired_ab = [(a, b) for a, b in zip(monitored_cdeltas, steered_cdeltas)
                 if a is not None and b is not None]
    if len(paired_ab) >= 2:
        a_vals = np.array([x[0] for x in paired_ab])
        b_vals = np.array([x[1] for x in paired_ab])
        diffs = a_vals - b_vals  # positive = steering reduced C_delta

        t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
        pooled_std = np.sqrt((np.std(a_vals)**2 + np.std(b_vals)**2) / 2)
        cohens_d = np.mean(diffs) / pooled_std if pooled_std > 0 else 0

        # 95% CI on mean difference
        se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        ci_low = np.mean(diffs) - 1.96 * se
        ci_high = np.mean(diffs) + 1.96 * se

        results["monitored_vs_steered"] = {
            "n_pairs": len(paired_ab),
            "mean_monitored": float(np.mean(a_vals)),
            "mean_steered": float(np.mean(b_vals)),
            "mean_reduction": float(np.mean(diffs)),
            "std_reduction": float(np.std(diffs, ddof=1)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
        }

    # Arm A vs Arm C (monitored vs noise)
    paired_ac = [(a, c) for a, c in zip(monitored_cdeltas, noise_cdeltas)
                 if a is not None and c is not None]
    if len(paired_ac) >= 2:
        a_vals = np.array([x[0] for x in paired_ac])
        c_vals = np.array([x[1] for x in paired_ac])
        diffs = a_vals - c_vals

        t_stat, p_val = stats.ttest_rel(a_vals, c_vals)
        pooled_std = np.sqrt((np.std(a_vals)**2 + np.std(c_vals)**2) / 2)
        cohens_d = np.mean(diffs) / pooled_std if pooled_std > 0 else 0

        se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        ci_low = np.mean(diffs) - 1.96 * se
        ci_high = np.mean(diffs) + 1.96 * se

        results["monitored_vs_noise"] = {
            "n_pairs": len(paired_ac),
            "mean_monitored": float(np.mean(a_vals)),
            "mean_noise": float(np.mean(c_vals)),
            "mean_reduction": float(np.mean(diffs)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
        }

    # Arm B vs Arm C (steered vs noise — the key comparison)
    paired_bc = [(b, c) for b, c in zip(steered_cdeltas, noise_cdeltas)
                 if b is not None and c is not None]
    if len(paired_bc) >= 2:
        b_vals = np.array([x[0] for x in paired_bc])
        c_vals = np.array([x[1] for x in paired_bc])
        diffs = c_vals - b_vals  # positive = steering better than noise

        t_stat, p_val = stats.ttest_rel(c_vals, b_vals)
        pooled_std = np.sqrt((np.std(b_vals)**2 + np.std(c_vals)**2) / 2)
        cohens_d = np.mean(diffs) / pooled_std if pooled_std > 0 else 0

        se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        ci_low = np.mean(diffs) - 1.96 * se
        ci_high = np.mean(diffs) + 1.96 * se

        results["steered_vs_noise"] = {
            "n_pairs": len(paired_bc),
            "mean_steered": float(np.mean(b_vals)),
            "mean_noise": float(np.mean(c_vals)),
            "mean_advantage": float(np.mean(diffs)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Powered Governor Study")
    parser.add_argument("--n", type=int, default=30, help="Number of hallucination pairs")
    parser.add_argument("--alpha", type=float, default=2.0, help="Steering strength")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max tokens per generation")
    parser.add_argument("--min-cdelta", type=float, default=0.008, help="Min baseline C_delta for candidate selection")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"COGNITIVE GOVERNOR — POWERED STUDY")
    print(f"n={args.n} | alpha={args.alpha} | max_tokens={args.max_tokens}")
    print(f"{'='*70}\n")

    # Load candidates
    candidates = load_candidates(args.n, args.min_cdelta)
    if len(candidates) < args.n:
        print(f"WARNING: Only {len(candidates)} candidates found (requested {args.n})")

    # Initialize steerer (loads model once)
    from coherence_steerer import CoherenceSteerer
    steerer = CoherenceSteerer(
        model_name="google/gemma-2-2b",
        steering_strength=args.alpha,
        c_delta_threshold=args.min_cdelta,  # use selection threshold as detection threshold
    )

    # Run three arms
    all_results = []
    monitored_mean_cdeltas = []
    steered_mean_cdeltas = []
    noise_mean_cdeltas = []

    t_start = time.time()

    for i, cand in enumerate(candidates):
        q = cand["question"]
        print(f"\n[{i+1}/{len(candidates)}] {q[:70]}")

        # Arm A: monitored baseline
        r_mon = run_arm_monitored(steerer, q, args.max_tokens)
        mon_valid = [x for x in r_mon.c_delta_trajectory if x is not None]
        mon_mean = float(np.mean(mon_valid)) if mon_valid else None

        # Arm B: directional steering
        r_steer = run_arm_steered(steerer, q, args.max_tokens, args.alpha)
        steer_valid = [x for x in r_steer.c_delta_trajectory if x is not None]
        steer_mean = float(np.mean(steer_valid)) if steer_valid else None

        # Arm C: noise injection
        r_noise = run_arm_noise(steerer, q, args.max_tokens)
        noise_valid = [x for x in r_noise.c_delta_trajectory if x is not None]
        noise_mean = float(np.mean(noise_valid)) if noise_valid else None

        monitored_mean_cdeltas.append(mon_mean)
        steered_mean_cdeltas.append(steer_mean)
        noise_mean_cdeltas.append(noise_mean)

        # Summary
        mon_cd_str = f"{mon_mean:.6f}" if mon_mean else "N/A"
        steer_cd_str = f"{steer_mean:.6f}" if steer_mean else "N/A"
        noise_cd_str = f"{noise_mean:.6f}" if noise_mean else "N/A"
        reduction = ""
        if mon_mean and steer_mean:
            pct = ((mon_mean - steer_mean) / mon_mean) * 100 if mon_mean != 0 else 0
            reduction = f" ({pct:+.1f}%)"

        safe_text_mon = r_mon.text.encode('ascii', 'replace').decode()[:60]
        safe_text_steer = r_steer.text.encode('ascii', 'replace').decode()[:60]

        print(f"  Monitored:  C_delta={mon_cd_str}  \"{safe_text_mon}\"")
        print(f"  Steered:    C_delta={steer_cd_str}{reduction}  interventions={r_steer.intervention_count}")
        print(f"  Noise:      C_delta={noise_cd_str}  interventions={r_noise.intervention_count}")

        all_results.append({
            "question": q,
            "correct_answer": cand["correct_answer"],
            "incorrect_answer": cand["incorrect_answer"],
            "baseline_halluc_cdelta": cand["baseline_halluc_cdelta"],
            "monitored": {
                "text": r_mon.text,
                "mean_cdelta": mon_mean,
                "c_delta_trajectory": r_mon.c_delta_trajectory,
                "total_tokens": r_mon.total_tokens,
                "overhead_pct": r_mon.overhead_pct,
            },
            "steered": {
                "text": r_steer.text,
                "mean_cdelta": steer_mean,
                "c_delta_trajectory": r_steer.c_delta_trajectory,
                "interventions": r_steer.interventions,
                "intervention_count": r_steer.intervention_count,
                "total_tokens": r_steer.total_tokens,
                "overhead_pct": r_steer.overhead_pct,
            },
            "noise": {
                "text": r_noise.text,
                "mean_cdelta": noise_mean,
                "c_delta_trajectory": r_noise.c_delta_trajectory,
                "interventions": r_noise.interventions,
                "intervention_count": r_noise.intervention_count,
                "total_tokens": r_noise.total_tokens,
            },
        })

        # Checkpoint every 5 pairs
        if (i + 1) % 5 == 0:
            _save_checkpoint(args, all_results, monitored_mean_cdeltas,
                           steered_mean_cdeltas, noise_mean_cdeltas, t_start)

    elapsed = time.time() - t_start

    # Final statistics
    stat_results = compute_stats(
        monitored_mean_cdeltas,
        steered_mean_cdeltas,
        noise_mean_cdeltas,
    )

    # Intervention summary
    total_interventions = sum(r["steered"]["intervention_count"] for r in all_results)
    questions_with_intervention = sum(1 for r in all_results if r["steered"]["intervention_count"] > 0)

    output = {
        "experiment": "cognitive_governor_powered_study",
        "model": "google/gemma-2-2b",
        "n_pairs": len(candidates),
        "steering_strength": args.alpha,
        "max_tokens": args.max_tokens,
        "c_delta_threshold": steerer.c_delta_threshold,
        "min_cdelta_selection": args.min_cdelta,
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed,
        "statistics": stat_results,
        "intervention_summary": {
            "total_interventions": total_interventions,
            "questions_with_intervention": questions_with_intervention,
            "intervention_rate": questions_with_intervention / len(candidates) if candidates else 0,
        },
        "results": all_results,
    }

    out_path = os.path.join(RESULTS_DIR, "governor_powered_study.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"GOVERNOR POWERED STUDY — COMPLETE")
    print(f"{'='*70}")
    print(f"N pairs: {len(candidates)}")
    print(f"Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Interventions: {total_interventions} total, {questions_with_intervention}/{len(candidates)} questions")

    for comp_name, comp_stats in stat_results.items():
        print(f"\n  {comp_name}:")
        for k, v in comp_stats.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

    print(f"\nSaved to: {out_path}")
    print(f"{'='*70}\n")


def _save_checkpoint(args, results, mon, steer, noise, t_start):
    """Save intermediate checkpoint."""
    partial_stats = compute_stats(mon, steer, noise)
    checkpoint = {
        "experiment": "cognitive_governor_powered_study",
        "status": "in_progress",
        "n_complete": len(results),
        "n_target": args.n,
        "elapsed_s": time.time() - t_start,
        "timestamp": datetime.now().isoformat(),
        "partial_statistics": partial_stats,
        "results": results,
    }
    cp_path = os.path.join(RESULTS_DIR, "governor_checkpoint.json")
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"  [checkpoint saved: {len(results)} pairs]")


if __name__ == "__main__":
    main()
