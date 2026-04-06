"""
run_all_S_analyses.py — one command to run every S-axis analysis
==================================================================
Usage (after main GPU run completes):
    python run_all_S_analyses.py

Runs, in order:
  1. Primary pre-registered test (analyze_S_truthfulqa.py)
  2. Predictive horizon curve (analyze_S_horizon.py)
  3. Trajectory shape features + classifier (analyze_trajectory_features.py)
  4. Category × geometry signature (analyze_category_geometry.py)
  5. Cross-validated calibrated predictor (S_calibrated_predictor.py)
  6. Visualization bundle (visualize_S_trajectories.py)

Outputs: console summary + JSON audit files + PDF/PNG figures.
All outputs named after the input file.
"""

import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(ROOT, "truthfulqa_results", "S_truthfulqa_n200.json")


def run_step(name, cmd, out_path):
    """Run one analysis step, return success bool."""
    print("\n" + "=" * 72)
    print(f"STEP: {name}")
    print("=" * 72)
    t_start = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    elapsed = time.time() - t_start
    if result.returncode == 0:
        print(f"[OK] {name} complete ({elapsed:.1f}s)")
        return True
    print(f"[FAIL] {name} returned {result.returncode} after {elapsed:.1f}s")
    return False


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(data_path):
        print(f"ERROR: data file not found: {data_path}")
        print(f"Run 'python run_S_truthfulqa.py' first.")
        sys.exit(1)

    py = sys.executable
    steps = [
        ("Primary pre-registered test",
         [py, "analyze_S_truthfulqa.py", data_path]),
        ("Predictive horizon curve",
         [py, "analyze_S_horizon.py", data_path]),
        ("Trajectory shape features + multi-feature classifier",
         [py, "analyze_trajectory_features.py", data_path]),
        ("Category × geometry signature",
         [py, "analyze_category_geometry.py", data_path]),
        ("Cross-validated calibrated predictor",
         [py, "S_calibrated_predictor.py", data_path]),
        ("Visualization bundle",
         [py, "visualize_S_trajectories.py", data_path]),
        ("Sliding-window S (dynamic)",
         [py, "analyze_S_sliding.py", data_path]),
        ("Horizon permutation test (null distribution)",
         [py, "analyze_S_horizon_permutation.py", data_path]),
        ("Trajectory manifold (PCA + kNN + clustering)",
         [py, "analyze_trajectory_manifold.py", data_path]),
        ("Power analysis",
         [py, "analyze_S_power.py", data_path]),
        ("Trajectory specimen cards (HTML)",
         [py, "generate_trajectory_cards.py", data_path]),
        ("Executive summary",
         [py, "make_executive_summary.py", data_path]),
    ]

    print("\n" + "=" * 72)
    print("S-AXIS × TRUTHFULQA — FULL ANALYSIS SUITE")
    print("=" * 72)
    print(f"Input: {data_path}")
    print(f"Steps: {len(steps)}")

    t0 = time.time()
    results = []
    for name, cmd in steps:
        success = run_step(name, cmd, data_path)
        results.append((name, success))
    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("ANALYSIS SUITE COMPLETE")
    print("=" * 72)
    for name, success in results:
        marker = "[OK]" if success else "[FAIL]"
        print(f"  {marker} {name}")
    print(f"\nTotal time: {elapsed:.1f}s")
    if all(s for _, s in results):
        print("ALL ANALYSES COMPLETE")
    else:
        print("SOME ANALYSES FAILED — review individual step outputs above")
        sys.exit(1)


if __name__ == "__main__":
    main()
