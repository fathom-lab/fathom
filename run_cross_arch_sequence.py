"""
run_cross_arch_sequence.py — queue SAE-free IPR runs across architectures
==============================================================================
Runs the SAE-free probe on multiple model families sequentially.
Waits for baseline GPU run to finish, then cycles through:
  1. Gemma-2-2B-IT  (validation against SAE S_early)
  2. Llama-3.2-3B-Instruct
  3. Qwen2.5-3B-Instruct
  4. Phi-3.5-mini-instruct

Each model: 200 TruthfulQA items, ~2-3h on RTX 4070 laptop (8GB).

Pilot mode: n=5 for each, ~10-15 min per model, quick sanity check.

Usage:
    # Quick pilots on all 4 models (~1h total)
    python run_cross_arch_sequence.py --pilot

    # Full runs (8-12h total)
    python run_cross_arch_sequence.py

    # Specific models
    python run_cross_arch_sequence.py --models "meta-llama/Llama-3.2-3B-Instruct"
"""

import argparse
import subprocess
import sys
import os
import time

DEFAULT_MODELS = [
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]


def wait_for_gpu(max_wait_s=14400, poll_interval=60):
    """Wait until GPU memory usage < 2GB."""
    import subprocess
    t_start = time.time()
    while time.time() - t_start < max_wait_s:
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                text=True).strip()
            used_mb = int(out.split('\n')[0])
            if used_mb < 2000:
                print(f"  GPU free ({used_mb} MB used)")
                return True
            print(f"  GPU busy ({used_mb} MB used), waiting...")
        except Exception as e:
            print(f"  GPU check failed: {e}")
            return False
        time.sleep(poll_interval)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("CROSS-ARCHITECTURE SEQUENCE")
    print("=" * 72)
    print(f"Mode: {'PILOT' if args.pilot else 'FULL'}")
    print(f"Models ({len(args.models)}):")
    for m in args.models:
        print(f"  - {m}")
    print()

    if not args.no_wait:
        print("Waiting for GPU to free (checking every 60s)...")
        ok = wait_for_gpu()
        if not ok:
            print("GPU never freed. Exiting.")
            sys.exit(1)

    for i, model in enumerate(args.models):
        print()
        print("=" * 72)
        print(f"[{i+1}/{len(args.models)}] Running SAE-free IPR on {model}")
        print("=" * 72)
        cmd = [sys.executable, "run_sae_free_truthfulqa.py", "--model", model]
        if args.pilot: cmd.append("--pilot")
        if args.resume: cmd.append("--resume")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [FAIL] {model} returned {result.returncode}")
            print(f"  Continuing with next model...")
        else:
            print(f"  [OK] {model} complete")

    print()
    print("=" * 72)
    print("SEQUENCE COMPLETE")
    print("=" * 72)
    print("Next: python analyze_cross_arch_meta.py")


if __name__ == "__main__":
    main()
