# -*- coding: utf-8 -*-
"""
run_all_experiments.py -- Run the complete Fathom Oversight evaluation suite
============================================================================

Runs ALL experiments sequentially on a single GPU session (avoids
reloading the model between experiments):

  1. Hallucination intervention (TruthfulQA, n items)
  2. Depth routing correction (surface/insight pairs)
  3. Multi-policy comparison (all policies on same items)

Usage:
  python run_all_experiments.py --pilot          # Quick: n=10 halluc, n=10 multi
  python run_all_experiments.py --medium         # Medium: n=30 halluc, n=30 multi
  python run_all_experiments.py --full           # Full: n=200 halluc, n=50 multi
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from eval_oversight import OversightEvaluator


def main():
    parser = argparse.ArgumentParser(description="Fathom Oversight — Full Experiment Suite")
    parser.add_argument("--pilot", action="store_true", help="Quick run (n=10)")
    parser.add_argument("--medium", action="store_true", help="Medium run (n=30)")
    parser.add_argument("--full", action="store_true", help="Full run (n=200)")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.pilot:
        n_halluc, n_multi = 10, 10
    elif args.medium:
        n_halluc, n_multi = 30, 30
    elif args.full:
        n_halluc, n_multi = 200, 50
    else:
        print("Specify --pilot, --medium, or --full")
        return

    print("=" * 60)
    print("FATHOM OVERSIGHT — COMPLETE EVALUATION SUITE")
    print(f"Hallucination: n={n_halluc}, Multi-policy: n={n_multi}")
    print(f"Model: {args.model}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    evaluator = OversightEvaluator(
        model_name=args.model,
        max_tokens=40,
        device=args.device,
    )

    t_start = time.time()
    reports = {}

    # Experiment 1: Hallucination Intervention
    print("\n\n" + "=" * 60)
    print("EXPERIMENT 1: HALLUCINATION INTERVENTION")
    print("=" * 60)
    reports["hallucination"] = evaluator.eval_hallucination(n=n_halluc)

    # Experiment 2: Depth Routing Correction
    print("\n\n" + "=" * 60)
    print("EXPERIMENT 2: DEPTH ROUTING CORRECTION")
    print("Can we make the model think HARDER by suppressing shortcuts?")
    print("=" * 60)
    reports["depth_routing"] = evaluator.eval_depth_routing()

    # Experiment 3: Multi-Policy Comparison
    print("\n\n" + "=" * 60)
    print("EXPERIMENT 3: MULTI-POLICY — DO DIFFERENT POLICIES CATCH DIFFERENT FAILURES?")
    print("=" * 60)
    reports["multi_policy"] = evaluator.eval_multi_policy(n=n_multi)

    elapsed = time.time() - t_start

    # Final summary
    print("\n\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Finished: {datetime.now().isoformat()}")

    if "hallucination" in reports:
        s = reports["hallucination"].get("statistics", {})
        print(f"\nHallucination: {s.get('unguarded_halluc_rate', 0):.1%} → "
              f"{s.get('guarded_halluc_rate', 0):.1%} "
              f"(p={s.get('mcnemar_p', 1):.4f})")

    if "depth_routing" in reports:
        print(f"Depth routing: K shift = {reports['depth_routing'].get('mean_k_shift', 0):+.4f} "
              f"(p={reports['depth_routing'].get('p_value', 1):.4f})")

    if "multi_policy" in reports:
        tc = reports["multi_policy"].get("trigger_counts", {})
        print(f"Multi-policy triggers: {tc}")

    print(f"\nResults saved to: oversight_results/")


if __name__ == "__main__":
    main()
