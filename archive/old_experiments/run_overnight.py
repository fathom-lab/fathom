"""
run_overnight.py — Overnight Pipeline: E-Index + Attribution Trajectories
=========================================================================

Chains two GPU-intensive runs:
1. Feature frequency index (E axis) — ~30-60 min
2. Attribution-based trajectories (K/C/E) — ~3-5 hours

Run: python run_overnight.py
"""

import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"STARTING: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n{'='*60}")
    print(f"FINISHED: {desc} — {status}")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    return result.returncode == 0


if __name__ == "__main__":
    print(f"FATHOM OVERNIGHT PIPELINE")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Build feature frequency index
    ok1 = run(
        [sys.executable, "build_feature_freq_index.py",
         "--corpus", "wikipedia",
         "--n-tokens", "100000",
         "--output", "feature_freq_index.npz",
         "--batch-size", "64"],
        "Feature Frequency Index (E axis foundation)"
    )

    if not ok1:
        print("WARNING: Feature freq index failed. Continuing with attribution anyway.")

    # Step 2: Attribution-based trajectories
    ok2 = run(
        [sys.executable, "trajectory_attribution.py",
         "--n", "30",
         "--max-tokens", "15",
         "--seed", "42",
         "--output", "trajectory_attribution_results"],
        "Attribution Trajectories (K/C/E per token)"
    )

    print(f"\n{'='*60}")
    print(f"OVERNIGHT PIPELINE COMPLETE")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"E-Index: {'OK' if ok1 else 'FAILED'}")
    print(f"Attribution: {'OK' if ok2 else 'FAILED'}")
    print(f"{'='*60}")
