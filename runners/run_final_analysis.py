"""
run_final_analysis.py — one command to run ALL analyses after all GPU runs complete
=======================================================================================
Runs in order:
  1. SAE-free agreement test (Gemma SAE vs SAE-free)
  2. SAE-free adaptive threshold analysis (per model)
  3. Cross-architecture meta-analysis
  4. Summary

Usage:
    python run_final_analysis.py
"""
import subprocess
import sys
import os
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
py = sys.executable


def run(name, cmd):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    result = subprocess.run(cmd, cwd=ROOT)
    status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
    print(f"[{status}] {name}")
    return result.returncode == 0


def main():
    print("=" * 72)
    print("FINAL ANALYSIS SUITE")
    print("=" * 72)

    # Check what SAE-free files exist
    sae_free_files = sorted(glob.glob(
        os.path.join(ROOT, "truthfulqa_results", "sae_free_ipr_*_pilot.json")))
    sae_free_full = sorted(glob.glob(
        os.path.join(ROOT, "truthfulqa_results", "sae_free_ipr_*_n200.json")))
    print(f"SAE-free pilot files: {len(sae_free_files)}")
    for f in sae_free_files:
        print(f"  {os.path.basename(f)}")
    print(f"SAE-free full files:  {len(sae_free_full)}")
    for f in sae_free_full:
        print(f"  {os.path.basename(f)}")
    print()

    all_files = sae_free_full if sae_free_full else sae_free_files

    results = []

    # 1. Agreement test (Gemma only — if we have both SAE and SAE-free)
    gemma_files = [f for f in all_files if "gemma" in f.lower()]
    if gemma_files:
        results.append(run("SAE vs SAE-free agreement (Gemma)",
                          [py, "analyze_sae_free_agreement.py", gemma_files[0]]))

    # 2. Adaptive threshold analysis per model
    for f in all_files:
        name = os.path.basename(f).replace("sae_free_ipr_", "").replace("_n200.json", "").replace("_pilot.json", "")
        results.append(run(f"Adaptive threshold ({name})",
                          [py, "analyze_sae_free_adaptive.py", f]))

    # 3. Cross-architecture meta-analysis (only if we have full runs from multiple models)
    if len(sae_free_full) >= 2:
        results.append(run("Cross-architecture meta-analysis",
                          [py, "analyze_cross_arch_meta.py"]))
    else:
        print("\n[SKIP] Cross-arch meta requires >= 2 full runs")

    # Summary
    print()
    print("=" * 72)
    print("FINAL ANALYSIS COMPLETE")
    print("=" * 72)
    n_ok = sum(1 for r in results if r)
    n_fail = sum(1 for r in results if not r)
    print(f"  {n_ok} OK, {n_fail} FAIL out of {len(results)} analyses")


if __name__ == "__main__":
    main()
