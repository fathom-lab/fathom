# -*- coding: utf-8 -*-
"""
d_axis_temporal.py -- H-D4 temporal localization analysis
==========================================================

Pre-registered secondary hypothesis:
    H-D4: The D divergence on adversarial prompts concentrates in
          early tokens (0-6) rather than late tokens (7+).
          Predicted: |d_z| for early window exceeds |d_z| for late
          window on the instruct model, mirroring the S-axis
          biphasic finding.

This is a CPU-only analysis — all data is in the saved trajectories
from `oversight_results/d_axis_pilot_gemma-2-2b{,-it}_fast.json`.

Usage:
    python analysis/d_axis_temporal.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paired_test(a, b, label):
    """Paired t-test on (a - b). Returns dict."""
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    t, p = stats.ttest_rel(a, b)
    dz = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
    se = np.std(diff, ddof=1) / np.sqrt(n)
    return {
        "label": label,
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "t": float(t),
        "p": float(p),
        "dz": dz,
        "ci95_low": float(diff.mean() - 1.96 * se),
        "ci95_high": float(diff.mean() + 1.96 * se),
    }


def extract_windows(dataset):
    """
    From a saved pilot dataset, extract per-pair early/late mean D
    for direct and adversarial conditions.
    Returns dict: {
        "direct_early": [...], "direct_late": [...],
        "adv_early": [...],    "adv_late":    [...]
    }
    """
    de, dl, ae, al = [], [], [], []
    for p in dataset["pairs"]:
        d = p.get("direct_summary")
        a = p.get("adversarial_summary")
        if d and a:
            de.append(d["d_early_mean"])
            dl.append(d["d_late_mean"])
            ae.append(a["d_early_mean"])
            al.append(a["d_late_mean"])
    return {
        "direct_early": de,
        "direct_late": dl,
        "adv_early": ae,
        "adv_late": al,
    }


def main():
    base = load(ROOT / "oversight_results" / "d_axis_pilot_gemma-2-2b_fast.json")
    it = load(ROOT / "oversight_results" / "d_axis_pilot_gemma-2-2b-it_fast.json")

    base_w = extract_windows(base)
    it_w = extract_windows(it)

    def run(w, model_label):
        return {
            "full":  paired_test(
                [a for a, d in zip(w["adv_early"], w["direct_early"])],  # placeholder
                [d for a, d in zip(w["adv_early"], w["direct_early"])],
                "full",
            ) if False else None,  # not used, kept for clarity
            "early": paired_test(w["adv_early"], w["direct_early"],
                                  f"{model_label} early (tokens 0-6)"),
            "late":  paired_test(w["adv_late"],  w["direct_late"],
                                  f"{model_label} late (tokens 7+)"),
        }

    # Per-model early vs late
    base_res = {
        "early": paired_test(base_w["adv_early"], base_w["direct_early"],
                             "Base early"),
        "late": paired_test(base_w["adv_late"], base_w["direct_late"],
                            "Base late"),
    }
    it_res = {
        "early": paired_test(it_w["adv_early"], it_w["direct_early"],
                             "Instruct early"),
        "late": paired_test(it_w["adv_late"], it_w["direct_late"],
                            "Instruct late"),
    }

    # Per-pair diff-in-diff on early window and late window separately
    base_diff_early = np.array(base_w["adv_early"]) - np.array(base_w["direct_early"])
    it_diff_early = np.array(it_w["adv_early"]) - np.array(it_w["direct_early"])
    base_diff_late = np.array(base_w["adv_late"]) - np.array(base_w["direct_late"])
    it_diff_late = np.array(it_w["adv_late"]) - np.array(it_w["direct_late"])

    dod_early = paired_test(it_diff_early, base_diff_early, "DoD early")
    dod_late = paired_test(it_diff_late, base_diff_late, "DoD late")

    report = {
        "experiment": "d_axis_temporal_H_D4",
        "base": base_res,
        "instruct": it_res,
        "diff_in_diff": {
            "early": dod_early,
            "late": dod_late,
        },
        "pre_reg_H_D4": (
            "|dz| on the early window (tokens 0-6) should exceed "
            "|dz| on the late window (tokens 7+) on the instruct model, "
            "mirroring the S-axis biphasic finding."
        ),
    }

    out_path = ROOT / "oversight_results" / "d_axis_temporal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 62)
    print("H-D4 TEMPORAL LOCALIZATION (early vs late D windows)")
    print("=" * 62)

    def row(label, r):
        print(f"  {label:<22s} t={r['t']:+6.2f} p={r['p']:.4f} "
              f"dz={r['dz']:+5.2f}  mean(adv-direct)={r['mean_diff']:+.4f}")

    print("\nPer-model paired tests on (adv - direct):")
    row("Base early (0-6)",      base_res["early"])
    row("Base late (7+)",        base_res["late"])
    row("Instruct early (0-6)",  it_res["early"])
    row("Instruct late (7+)",    it_res["late"])

    print("\nDiff-in-diff (instruct-base) on each window:")
    row("DoD early (0-6)", dod_early)
    row("DoD late (7+)",   dod_late)

    print(f"\nH-D4 verdict:")
    it_early_mag = abs(it_res["early"]["dz"])
    it_late_mag = abs(it_res["late"]["dz"])
    print(f"  |dz| instruct early = {it_early_mag:.2f}")
    print(f"  |dz| instruct late  = {it_late_mag:.2f}")
    if it_early_mag > it_late_mag:
        print("  [SUPPORTED] Early magnitude exceeds late magnitude on instruct")
    else:
        print("  [NOT SUPPORTED] Late magnitude >= early on instruct")

    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
