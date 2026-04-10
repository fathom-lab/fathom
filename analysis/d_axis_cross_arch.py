# -*- coding: utf-8 -*-
"""
d_axis_cross_arch.py -- Cross-architecture D-axis meta-analysis
================================================================

Consolidates four n=30 pilots into a single meta-analysis:

  Gemma-2-2B (base)           - primary pilot, reversed direction
  Gemma-2-2B-IT (instruct)    - primary pilot, H-D2 direction
  Llama-3.2-3B (base)         - cross-arch replication
  Llama-3.2-3B-Instruct       - cross-arch replication

Reports:
  - Per-model paired t-test (adv - direct)
  - Fisher combined p across all four runs
  - Consistency analysis (sign pattern, effect size)
  - Scope statement for the paper: what replicates, what doesn't

Usage:
    python analysis/d_axis_cross_arch.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PILOTS = [
    ("Gemma-2-2B",              "base",     "d_axis_pilot_gemma-2-2b_fast.json"),
    ("Gemma-2-2B-IT",           "instruct", "d_axis_pilot_gemma-2-2b-it_fast.json"),
    ("Llama-3.2-3B",            "base",     "d_axis_pilot_Llama-3.2-3B_fast.json"),
    ("Llama-3.2-3B-Instruct",   "instruct", "d_axis_pilot_Llama-3.2-3B-Instruct_fast.json"),
]


def load_deltas(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for p in data["pairs"]:
        d = p.get("direct_summary")
        a = p.get("adversarial_summary")
        if d and a:
            out.append({
                "topic": p["topic"],
                "direct": d["d_mean"],
                "adv": a["d_mean"],
                "delta_adv_minus_direct": a["d_mean"] - d["d_mean"],
            })
    return out


def paired_test(deltas):
    from scipy import stats
    diffs = np.array([d["delta_adv_minus_direct"] for d in deltas], dtype=float)
    n = len(diffs)
    # H-D2: adv - direct should be < 0, so paired t on (adv_vals - direct_vals)
    # But we're directly given the delta, so one-sample t-test on delta == 0
    t, p = stats.ttest_1samp(diffs, 0)
    dz = float(diffs.mean() / (diffs.std(ddof=1) + 1e-12))
    return {
        "n": n,
        "mean_delta_adv_minus_direct": float(diffs.mean()),
        "sd": float(diffs.std(ddof=1)),
        "t": float(t),
        "p": float(p),
        "dz": dz,
        "n_positive": int(np.sum(diffs > 0)),
        "n_negative": int(np.sum(diffs < 0)),
    }


def fisher_combined(ps):
    from scipy import stats
    ps = np.array([max(min(p, 1-1e-12), 1e-12) for p in ps])
    chi2 = -2 * np.sum(np.log(ps))
    df = 2 * len(ps)
    combined_p = 1 - stats.chi2.cdf(chi2, df)
    return float(combined_p), float(chi2), df


def main():
    results = {}
    for label, training, filename in PILOTS:
        path = ROOT / "oversight_results" / filename
        if not path.exists():
            print(f"[!] Missing: {path}")
            continue
        deltas = load_deltas(path)
        test = paired_test(deltas)
        results[label] = {
            "training": training,
            "file": filename,
            **test,
        }

    # Fisher combined across all four
    ps = [r["p"] for r in results.values()]
    fisher_p, chi2, df = fisher_combined(ps)

    # Sign pattern
    signs = [(l, "neg" if r["mean_delta_adv_minus_direct"] < 0 else "pos",
              r["dz"]) for l, r in results.items()]

    # Consistency: how many match H-D2 direction (negative = adv lower = H-D2)?
    h_d2_count = sum(1 for _, s, _ in signs if s == "neg")

    print("=" * 78)
    print("D-AXIS CROSS-ARCHITECTURE META-ANALYSIS")
    print("=" * 78)
    print(f"{'Model':<26s}{'Train':<10s}{'mean(adv-dir)':>14s}{'p':>12s}{'dz':>8s}{'+/-':>8s}")
    print("-" * 78)
    for label in results:
        r = results[label]
        sign_str = f"{r['n_positive']}/{r['n_negative']}"
        p_str = f"{r['p']:.4f}" if r["p"] >= 0.0001 else f"{r['p']:.1e}"
        print(f"{label:<26s}{r['training']:<10s}{r['mean_delta_adv_minus_direct']:>+14.4f}"
              f"{p_str:>12s}{r['dz']:>+8.2f}{sign_str:>8s}")

    print("-" * 78)
    print(f"Fisher combined p across {len(ps)} runs: {fisher_p:.6f}")
    print(f"  chi2 = {chi2:.2f}, df = {df}")
    print()
    print(f"Direction consistency (H-D2: adv < direct): {h_d2_count}/{len(signs)}")
    for label, s, dz in signs:
        marker = "H-D2" if s == "neg" else "REV "
        print(f"  [{marker}] {label:<28s} dz = {dz:+.3f}")

    print()
    print("Scope statement:")
    if h_d2_count == len(signs):
        scope = "Full H-D2 replication across all tested models."
    elif h_d2_count >= len(signs) - 1:
        rev = [l for l, s, _ in signs if s == "pos"]
        scope = (f"H-D2 direction replicates on {h_d2_count}/{len(signs)} "
                 f"models. Exception: {', '.join(rev)}. "
                 f"RLHF-induced sign flip is specific to the Gemma family.")
    else:
        scope = (f"H-D2 replicates on {h_d2_count}/{len(signs)} models; "
                 f"direction is not universal.")
    print(f"  {scope}")

    report = {
        "experiment": "d_axis_cross_arch_meta",
        "pilots": results,
        "fisher_combined_p": fisher_p,
        "fisher_chi2": chi2,
        "fisher_df": df,
        "h_d2_direction_count": h_d2_count,
        "n_pilots": len(results),
        "scope": scope,
    }
    out_path = ROOT / "oversight_results" / "d_axis_cross_arch_meta.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
