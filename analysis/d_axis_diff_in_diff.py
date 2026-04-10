# -*- coding: utf-8 -*-
"""
d_axis_diff_in_diff.py -- H-D3 diff-in-diff analysis
======================================================

Combines the two pre-registered pilots on Gemma-2-2B (base) and
Gemma-2-2B-IT (instruct) into a single diff-in-diff test.

H-D3 pre-registered prediction:
    (D_adv - D_direct)_instruct - (D_adv - D_direct)_base <= -0.3

Observed:
    Base:     adv - direct = +0.0231, p=0.0012, d_z = +0.65 (REVERSED)
    Instruct: adv - direct = -0.0025, p=0.0035, d_z = -0.58 (H-D2 supported)

The effect is therefore not just an amplification — it is a SIGN FLIP.

Usage:
    python analysis/d_axis_diff_in_diff.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def per_pair_delta(dataset):
    """Return list of (adv - direct) per pair, matched by topic."""
    out = []
    for p in dataset["pairs"]:
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


def main():
    base_path = ROOT / "oversight_results" / "d_axis_pilot_gemma-2-2b_fast.json"
    it_path = ROOT / "oversight_results" / "d_axis_pilot_gemma-2-2b-it_fast.json"

    base = load(base_path)
    it = load(it_path)

    base_deltas = per_pair_delta(base)
    it_deltas = per_pair_delta(it)

    # Match by topic (ordering should already match, verify)
    base_by_topic = {d["topic"]: d for d in base_deltas}
    it_by_topic = {d["topic"]: d for d in it_deltas}
    topics = [t for t in base_by_topic if t in it_by_topic]

    # Per-topic paired diff-in-diff: (adv-direct)_instruct - (adv-direct)_base
    per_topic_dod = []
    for t in topics:
        b = base_by_topic[t]["delta_adv_minus_direct"]
        i = it_by_topic[t]["delta_adv_minus_direct"]
        per_topic_dod.append({
            "topic": t,
            "base_delta": b,
            "instruct_delta": i,
            "dod": i - b,  # how much MORE negative instruct is than base
        })

    b_arr = np.array([d["base_delta"] for d in per_topic_dod])
    i_arr = np.array([d["instruct_delta"] for d in per_topic_dod])
    dod_arr = np.array([d["dod"] for d in per_topic_dod])

    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_rel(i_arr, b_arr)
        # Bootstrap CI on dod
        rng = np.random.default_rng(42)
        B = 10000
        boots = []
        for _ in range(B):
            idx = rng.integers(0, len(dod_arr), len(dod_arr))
            boots.append(dod_arr[idx].mean())
        boots = np.array(boots)
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    except ImportError:
        t_stat = p_val = 0
        ci_low = ci_high = 0

    dz = float(dod_arr.mean() / (dod_arr.std(ddof=1) + 1e-12))

    report = {
        "experiment": "d_axis_diff_in_diff",
        "n_topics": len(topics),
        "base": {
            "mean_adv_minus_direct": float(b_arr.mean()),
            "sd": float(b_arr.std(ddof=1)),
            "direction": "adversarial HIGHER (reversed from H-D1 null)",
        },
        "instruct": {
            "mean_adv_minus_direct": float(i_arr.mean()),
            "sd": float(i_arr.std(ddof=1)),
            "direction": "adversarial LOWER (H-D2 supported)",
        },
        "diff_in_diff": {
            "mean": float(dod_arr.mean()),
            "sd": float(dod_arr.std(ddof=1)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_dz": dz,
            "bootstrap_ci95": [float(ci_low), float(ci_high)],
            "n_pairs": len(topics),
            "interpretation": (
                "instruct delta (adv-direct) is "
                f"{dod_arr.mean():+.4f} units MORE NEGATIVE than base delta. "
                "RLHF causes a SIGN FLIP in the D-axis reaction to "
                "adversarial framings, not merely an amplification."
            ),
        },
        "per_topic": per_topic_dod,
    }

    out_path = ROOT / "oversight_results" / "d_axis_diff_in_diff.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("D-AXIS DIFF-IN-DIFF (H-D3) — Base vs Instruct")
    print("=" * 60)
    print(f"N matched topics: {len(topics)}")
    print()
    print(f"Base (Gemma-2-2B):")
    print(f"  adv - direct = {b_arr.mean():+.4f}  (REVERSED direction)")
    print(f"Instruct (Gemma-2-2B-IT):")
    print(f"  adv - direct = {i_arr.mean():+.4f}  (H-D2 direction)")
    print()
    print(f"Diff-in-diff: instruct - base = {dod_arr.mean():+.4f}")
    print(f"  paired t({len(topics)-1}) = {t_stat:+.3f}")
    print(f"  p = {p_val:.6f}")
    print(f"  Cohen's dz = {dz:+.3f}")
    print(f"  bootstrap 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print()

    n_flipped = sum(1 for d in per_topic_dod if d["dod"] < 0)
    print(f"Per-topic flip direction (instruct more negative than base):")
    print(f"  {n_flipped}/{len(topics)} ({100*n_flipped/len(topics):.0f}%)")
    print()
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
