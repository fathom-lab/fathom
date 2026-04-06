"""
governor_preselection.py — Preselected Hallucination Governor Experiment
=========================================================================

The clean governor experiment. Instead of hoping the steerer triggers on
random questions, we PRESELECT the top-N questions where the model
demonstrably hallucinates (verified from n=200 baseline C_delta gap).

Two modes:
  --mode prefix: Feed the incorrect answer prefix, let model continue.
                 Mechanistic claim: "governor corrects known lock-in."
                 Goes in Section 5.3.

  --mode free:   Let the model generate freely from the question.
                 Ecological validity: "governor detects in naturalistic generation."
                 Goes in Section 5.4 or supplementary.

Three arms per question:
  A: Monitored baseline (no intervention)
  B: Directional steering (alpha=2.0)
  C: Noise injection (control)

Selection: Top-30 from n=200 baseline by hallucinated-correct C_delta gap.
100% trigger rate by construction in prefix mode.

Usage:
    python governor_preselection.py --mode prefix --n 30 --alpha 2.0
    python governor_preselection.py --mode free --n 30 --alpha 2.0

Patent pending: US 64/020,489 & 64/021,113
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Optional

import numpy as np
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "truthfulqa_results")
BASELINE_PATH = os.path.join(RESULTS_DIR, "n200_checkpoint.json")


def select_hallucination_set(n: int) -> list:
    """
    Select top-N questions where model most clearly hallucinates.
    Ranked by C_delta gap (hallucinated - correct).
    """
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    candidates = []
    for i, r in enumerate(results):
        h_cd = r["hallucinated"]["C_delta"]
        c_cd = r["correct"]["C_delta"]
        gap = h_cd - c_cd
        if gap > 0:  # only take pairs where hallucination shows higher C_delta
            candidates.append({
                "idx": i,
                "question": r["question"],
                "correct_answer": r["correct_answer"],
                "incorrect_answer": r["incorrect_answer"],
                "baseline_halluc_cdelta": h_cd,
                "baseline_correct_cdelta": c_cd,
                "cdelta_gap": gap,
            })

    candidates.sort(key=lambda x: x["cdelta_gap"], reverse=True)
    selected = candidates[:n]
    print(f"[Selection] {len(candidates)} pairs with positive gap, selected top {len(selected)}")
    print(f"[Selection] Gap range: {selected[0]['cdelta_gap']:.6f} to {selected[-1]['cdelta_gap']:.6f}")
    return selected


def build_prompt(question: str, incorrect_answer: str, mode: str) -> str:
    """Build prompt based on mode."""
    if mode == "prefix":
        # Feed the question + start of incorrect answer to force hallucination path
        # Take first ~10 words of incorrect answer as prefix
        words = incorrect_answer.split()
        prefix = " ".join(words[:min(5, len(words))])
        return f"Q: {question}\nA: {prefix}"
    else:  # free
        return f"Q: {question}\nA:"


def run_three_arms(steerer, prompt: str, max_tokens: int, alpha: float):
    """Run all three arms for one question."""
    import types
    import torch

    # Arm A: monitored baseline
    r_mon = steerer.generate_monitored(prompt, max_tokens=max_tokens)

    # Arm B: directional steering
    old_strength = steerer.steering_strength
    steerer.steering_strength = alpha
    r_steer = steerer.generate_steered(prompt, max_tokens=max_tokens)
    steerer.steering_strength = old_strength

    # Arm C: noise injection control
    original_compute = steerer._compute_lock_in_direction

    def _random_direction(self_ref, hidden_state, layer_idx):
        d_model = hidden_state.shape[-1]
        rand_dir = torch.randn(d_model, device=hidden_state.device, dtype=torch.float32)
        rand_dir = rand_dir / rand_dir.norm()
        return rand_dir * 0.5

    steerer._compute_lock_in_direction = types.MethodType(
        lambda self, h, l: _random_direction(self, h, l), steerer
    )
    old_s = steerer.steering_strength
    steerer.steering_strength = alpha
    r_noise = steerer.generate_steered(prompt, max_tokens=max_tokens)
    steerer._compute_lock_in_direction = original_compute
    steerer.steering_strength = old_s

    return r_mon, r_steer, r_noise


def mean_cdelta(trajectory):
    """Mean C_delta from trajectory, ignoring None."""
    valid = [x for x in trajectory if x is not None]
    return float(np.mean(valid)) if valid else None


def compute_paired_stats(a_vals, b_vals, label=""):
    """Compute paired t-test stats."""
    paired = [(a, b) for a, b in zip(a_vals, b_vals)
              if a is not None and b is not None]
    if len(paired) < 2:
        return {"n": len(paired), "note": "insufficient pairs"}

    a = np.array([x[0] for x in paired])
    b = np.array([x[1] for x in paired])
    diffs = a - b

    t_stat, p_val = stats.ttest_rel(a, b)
    pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    cohens_d = np.mean(diffs) / pooled_std if pooled_std > 0 else 0.0

    # Wilcoxon signed-rank (non-parametric, robust)
    try:
        w_stat, w_p = stats.wilcoxon(diffs)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")

    se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    ci_low = np.mean(diffs) - 1.96 * se
    ci_high = np.mean(diffs) + 1.96 * se

    return {
        "n_pairs": len(paired),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_reduction": float(np.mean(diffs)),
        "std_reduction": float(np.std(diffs, ddof=1)),
        "t_statistic": float(t_stat),
        "p_value_ttest": float(p_val),
        "wilcoxon_statistic": float(w_stat),
        "p_value_wilcoxon": float(w_p),
        "cohens_d": float(cohens_d),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
    }


def main():
    parser = argparse.ArgumentParser(description="Preselection Governor Experiment")
    parser.add_argument("--mode", choices=["prefix", "free"], required=True,
                        help="prefix=force hallucination path, free=naturalistic generation")
    parser.add_argument("--n", type=int, default=30, help="Number of preselected pairs")
    parser.add_argument("--alpha", type=float, default=2.0, help="Steering strength")
    parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens per generation")
    parser.add_argument("--threshold", type=float, default=0.005, help="C_delta detection threshold")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"GOVERNOR PRESELECTION EXPERIMENT — {args.mode.upper()} MODE")
    print(f"n={args.n} | alpha={args.alpha} | max_tokens={args.max_tokens} | threshold={args.threshold}")
    print(f"{'='*70}\n")

    # Select hallucination set
    candidates = select_hallucination_set(args.n)

    # Initialize steerer
    from coherence_steerer import CoherenceSteerer
    steerer = CoherenceSteerer(
        model_name="google/gemma-2-2b",
        steering_strength=args.alpha,
        c_delta_threshold=args.threshold,
    )

    # Run experiment
    all_results = []
    mon_means = []
    steer_means = []
    noise_means = []

    t_start = time.time()

    for i, cand in enumerate(candidates):
        prompt = build_prompt(cand["question"], cand["incorrect_answer"], args.mode)
        print(f"\n[{i+1}/{len(candidates)}] {cand['question'][:60]}")
        print(f"  prompt: {prompt[:80]}")

        r_mon, r_steer, r_noise = run_three_arms(steerer, prompt, args.max_tokens, args.alpha)

        m_cd = mean_cdelta(r_mon.c_delta_trajectory)
        s_cd = mean_cdelta(r_steer.c_delta_trajectory)
        n_cd = mean_cdelta(r_noise.c_delta_trajectory)

        mon_means.append(m_cd)
        steer_means.append(s_cd)
        noise_means.append(n_cd)

        # Summary
        m_str = f"{m_cd:.6f}" if m_cd else "N/A"
        s_str = f"{s_cd:.6f}" if s_cd else "N/A"
        n_str = f"{n_cd:.6f}" if n_cd else "N/A"
        red = ""
        if m_cd and s_cd and m_cd != 0:
            pct = ((m_cd - s_cd) / m_cd) * 100
            red = f" ({pct:+.1f}%)"

        safe_mon = r_mon.text.encode('ascii', 'replace').decode()[:50]
        safe_steer = r_steer.text.encode('ascii', 'replace').decode()[:50]

        print(f"  Mon:   C_delta={m_str}  \"{safe_mon}\"")
        print(f"  Steer: C_delta={s_str}{red}  int={r_steer.intervention_count}  \"{safe_steer}\"")
        print(f"  Noise: C_delta={n_str}  int={r_noise.intervention_count}")

        all_results.append({
            "question": cand["question"],
            "correct_answer": cand["correct_answer"],
            "incorrect_answer": cand["incorrect_answer"],
            "prompt": prompt,
            "baseline_cdelta_gap": cand["cdelta_gap"],
            "monitored": {
                "text": r_mon.text,
                "mean_cdelta": m_cd,
                "c_delta_trajectory": r_mon.c_delta_trajectory,
                "total_tokens": r_mon.total_tokens,
                "overhead_pct": r_mon.overhead_pct,
            },
            "steered": {
                "text": r_steer.text,
                "mean_cdelta": s_cd,
                "c_delta_trajectory": r_steer.c_delta_trajectory,
                "interventions": r_steer.interventions,
                "intervention_count": r_steer.intervention_count,
                "total_tokens": r_steer.total_tokens,
            },
            "noise": {
                "text": r_noise.text,
                "mean_cdelta": n_cd,
                "c_delta_trajectory": r_noise.c_delta_trajectory,
                "interventions": r_noise.interventions,
                "intervention_count": r_noise.intervention_count,
                "total_tokens": r_noise.total_tokens,
            },
        })

        # Checkpoint every 5
        if (i + 1) % 5 == 0:
            _save_checkpoint(args, all_results, mon_means, steer_means, noise_means, t_start)

    elapsed = time.time() - t_start

    # Final statistics
    stats_mon_steer = compute_paired_stats(mon_means, steer_means, "monitored_vs_steered")
    stats_mon_noise = compute_paired_stats(mon_means, noise_means, "monitored_vs_noise")
    stats_steer_noise = compute_paired_stats(steer_means, noise_means, "steered_vs_noise")

    # Intervention summary
    total_int = sum(r["steered"]["intervention_count"] for r in all_results)
    with_int = sum(1 for r in all_results if r["steered"]["intervention_count"] > 0)

    output = {
        "experiment": f"governor_preselection_{args.mode}",
        "mode": args.mode,
        "model": "google/gemma-2-2b",
        "n_pairs": len(candidates),
        "steering_strength": args.alpha,
        "max_tokens": args.max_tokens,
        "c_delta_threshold": args.threshold,
        "selection_method": "top-N by baseline hallucinated-correct C_delta gap",
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed,
        "statistics": {
            "monitored_vs_steered": stats_mon_steer,
            "monitored_vs_noise": stats_mon_noise,
            "steered_vs_noise": stats_steer_noise,
        },
        "intervention_summary": {
            "total_interventions": total_int,
            "questions_with_intervention": with_int,
            "intervention_rate": with_int / len(candidates) if candidates else 0,
        },
        "results": all_results,
    }

    out_path = os.path.join(RESULTS_DIR, f"governor_preselection_{args.mode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"GOVERNOR PRESELECTION — {args.mode.upper()} MODE — COMPLETE")
    print(f"{'='*70}")
    print(f"N: {len(candidates)} | Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Interventions: {total_int} total, {with_int}/{len(candidates)} questions ({with_int/len(candidates)*100:.0f}%)")

    for name, s in [("Mon vs Steer", stats_mon_steer), ("Mon vs Noise", stats_mon_noise), ("Steer vs Noise", stats_steer_noise)]:
        if "n_pairs" in s:
            print(f"\n  {name} (n={s['n_pairs']}):")
            print(f"    reduction: {s['mean_reduction']:.6f}")
            print(f"    t-test p:  {s['p_value_ttest']:.4f}")
            print(f"    wilcoxon:  {s['p_value_wilcoxon']:.4f}")
            print(f"    Cohen d:   {s['cohens_d']:.4f}")
            print(f"    95% CI:    [{s['ci_95_low']:.6f}, {s['ci_95_high']:.6f}]")

    print(f"\nSaved to: {out_path}")
    print(f"{'='*70}\n")


def _save_checkpoint(args, results, mon, steer, noise, t_start):
    partial = compute_paired_stats(mon, steer)
    checkpoint = {
        "experiment": f"governor_preselection_{args.mode}",
        "mode": args.mode,
        "status": "in_progress",
        "n_complete": len(results),
        "n_target": args.n,
        "elapsed_s": time.time() - t_start,
        "timestamp": datetime.now().isoformat(),
        "partial_statistics": partial,
        "results": results,
    }
    cp_path = os.path.join(RESULTS_DIR, f"governor_preselection_{args.mode}_checkpoint.json")
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"  [checkpoint saved: {len(results)} pairs]")


if __name__ == "__main__":
    main()
