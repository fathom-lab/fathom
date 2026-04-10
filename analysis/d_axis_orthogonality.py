# -*- coding: utf-8 -*-
"""
d_axis_orthogonality.py -- D vs K/C/C_delta/S orthogonality check
==================================================================

Verifies that the D-axis is empirically independent of the four prior
Fathom axes (K, C, C_delta, S) by running FathomOversight's SAE-based
monitoring on the same 30 jailbreak matched pairs used in the D-axis
pilot, extracting per-token K/C/C_delta/S/D trajectories, and
computing cross-axis Pearson correlations at the per-pair summary
level.

This runs on Gemma-2-2B base (the model the existing §4 orthogonality
table uses) so the new correlations can be compared directly to the
K/C/C_delta/S table.

Output:
  oversight_results/d_axis_orthogonality.json

Usage:
    python analysis/d_axis_orthogonality.py --model google/gemma-2-2b
    python analysis/d_axis_orthogonality.py --analyze-only \
        oversight_results/d_axis_orthogonality.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _trajectory_mean(traj):
    vals = [v for v in traj if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def run_capture(model_name: str, max_tokens: int = 30,
                max_pairs: int = None) -> str:
    """Capture K/C/C_delta/S/D per-token trajectories on the jailbreak pairs."""
    from fathom_oversight import FathomOversight

    pairs_path = ROOT / "data" / "jailbreak_pairs.json"
    with open(pairs_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    pairs = dataset["pairs"]
    if max_pairs:
        pairs = pairs[:max_pairs]

    out_dir = ROOT / "oversight_results"
    out_dir.mkdir(exist_ok=True)
    tag = model_name.split("/")[-1]
    out_path = out_dir / f"d_axis_orthogonality_{tag}.json"
    ckpt_path = out_dir / f"d_axis_orthogonality_{tag}_checkpoint.json"

    print("=" * 62)
    print(f"D-AXIS ORTHOGONALITY CAPTURE -- {model_name}")
    print("=" * 62)
    print(f"Pairs: {len(pairs)}")
    print(f"Output: {out_path}")
    print()

    print(f"[LOAD] FathomOversight({model_name}) ...", flush=True)
    t_load = time.time()
    oversight = FathomOversight(model_name=model_name)
    print(f"[LOAD] done in {time.time()-t_load:.1f}s", flush=True)

    # Resume from checkpoint
    captures = []
    done_topics = set()
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        captures = ckpt.get("pairs", [])
        done_topics = {c["topic"] for c in captures}
        print(f"[RESUME] {len(captures)} pairs already done")

    t_start = time.time()
    for idx, p in enumerate(pairs):
        topic = p["topic"]
        if topic in done_topics:
            continue
        t_pair = time.time()
        try:
            r_d = oversight.generate_unguarded(p["direct"],
                                                max_tokens=max_tokens)
            r_a = oversight.generate_unguarded(p["adversarial"],
                                                max_tokens=max_tokens)

            def summarize(res, side):
                k = [s.k_depth for s in res.cognitive_states]
                c = [s.c_coherence for s in res.cognitive_states]
                cd = [s.c_delta for s in res.cognitive_states]
                se = [s.s_early for s in res.cognitive_states]
                dh = [s.d_honesty for s in res.cognitive_states
                      if s.d_honesty is not None]
                return {
                    "side": side,
                    "n_tokens": len(res.cognitive_states),
                    "k_mean": float(np.mean(k)) if k else None,
                    "c_mean": float(np.mean(c)) if c else None,
                    "c_delta_mean": float(np.mean(cd)) if cd else None,
                    "s_early_max": float(np.max(se)) if se else None,
                    "d_mean": float(np.mean(dh)) if dh else None,
                    "text": res.text[:120],
                }

            captures.append({
                "topic": topic,
                "direct": summarize(r_d, "direct"),
                "adversarial": summarize(r_a, "adversarial"),
            })
            c0 = captures[-1]
            print(f"[{idx+1}/{len(pairs)}] {topic:<20s} "
                  f"direct D={c0['direct']['d_mean']:+.3f} "
                  f"adv D={c0['adversarial']['d_mean']:+.3f} "
                  f"({time.time()-t_pair:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{idx+1}/{len(pairs)}] {topic} ERROR: {e}", flush=True)
            captures.append({"topic": topic, "error": str(e)})

        if (idx + 1) % 5 == 0:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"pairs": captures}, f, indent=2)

    elapsed = time.time() - t_start
    output = {
        "experiment": "d_axis_orthogonality",
        "model": model_name,
        "n_pairs": len(captures),
        "max_tokens": max_tokens,
        "elapsed_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pairs": captures,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    print(f"[TIME] {elapsed:.0f}s")
    return str(out_path)


def analyze(results_path: str) -> dict:
    """Compute per-axis Pearson correlations.

    Reports TWO correlation modes:
      (a) Pooled: direct + adversarial summaries pooled (n=2*pairs).
          Captures axis correlation across the full stimulus distribution.
      (b) Paired delta: per-pair (adv - direct) differences (n=pairs).
          Captures whether the WITHIN-pair shift on D tracks the shift
          on the other axes. This is the more honest orthogonality test
          for a paired design.
    """
    from scipy import stats
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect per-pair values (direct and adversarial separately)
    direct_k = []; adv_k = []
    direct_c = []; adv_c = []
    direct_cd = []; adv_cd = []
    direct_s = []; adv_s = []
    direct_d = []; adv_d = []
    for p in data["pairs"]:
        if "error" in p:
            continue
        d_side = p.get("direct", {})
        a_side = p.get("adversarial", {})
        if (not d_side or not a_side or
            d_side.get("d_mean") is None or a_side.get("d_mean") is None):
            continue
        direct_k.append(d_side["k_mean"]);       adv_k.append(a_side["k_mean"])
        direct_c.append(d_side["c_mean"]);       adv_c.append(a_side["c_mean"])
        direct_cd.append(d_side["c_delta_mean"]);adv_cd.append(a_side["c_delta_mean"])
        direct_s.append(d_side["s_early_max"]);  adv_s.append(a_side["s_early_max"])
        direct_d.append(d_side["d_mean"]);       adv_d.append(a_side["d_mean"])

    # Pooled arrays
    k_pool = np.array(direct_k + adv_k)
    c_pool = np.array(direct_c + adv_c)
    cd_pool = np.array(direct_cd + adv_cd)
    s_pool = np.array(direct_s + adv_s)
    d_pool = np.array(direct_d + adv_d)

    # Paired deltas (adv - direct)
    k_delta = np.array(adv_k) - np.array(direct_k)
    c_delta = np.array(adv_c) - np.array(direct_c)
    cd_delta = np.array(adv_cd) - np.array(direct_cd)
    s_delta = np.array(adv_s) - np.array(direct_s)
    d_delta = np.array(adv_d) - np.array(direct_d)

    n_pool = len(d_pool)
    n_pairs = len(d_delta)

    def corr(x, y):
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return {"r": 0.0, "p": 1.0}
        r, p = stats.pearsonr(x, y)
        return {"r": float(r), "p": float(p)}

    pooled_results = {
        "D_K":       corr(d_pool, k_pool),
        "D_C":       corr(d_pool, c_pool),
        "D_C_delta": corr(d_pool, cd_pool),
        "D_S":       corr(d_pool, s_pool),
    }
    delta_results = {
        "D_K":       corr(d_delta, k_delta),
        "D_C":       corr(d_delta, c_delta),
        "D_C_delta": corr(d_delta, cd_delta),
        "D_S":       corr(d_delta, s_delta),
    }
    max_pool = max(abs(v["r"]) for v in pooled_results.values())
    max_delta = max(abs(v["r"]) for v in delta_results.values())

    results = {
        "n_pairs": int(n_pairs),
        "n_pooled": int(n_pool),
        "pooled": pooled_results,
        "paired_delta": delta_results,
        "max_abs_r_pooled": float(max_pool),
        "max_abs_r_delta": float(max_delta),
        "orthogonal_pooled": bool(max_pool < 0.35),
        "orthogonal_delta": bool(max_delta < 0.35),
        "interpretation": (
            f"n_pairs = {n_pairs}. Pooled max|r| = {max_pool:.3f} "
            f"(across all prompts). Paired-delta max|r| = "
            f"{max_delta:.3f} (within-pair shifts). "
            + ("Paired-delta under 0.35 indicates D shifts are largely "
               "independent of shifts on the other axes within a pair "
               "(orthogonal at the causal level)."
               if max_delta < 0.35 else
               "Paired-delta exceeds 0.35: within-pair D shifts track "
               "shifts on at least one other axis. Small-sample instability "
               "or a genuine non-orthogonality. Re-run at larger n to "
               "disambiguate.")
        ),
    }

    print("=" * 62)
    print("D-AXIS ORTHOGONALITY")
    print("=" * 62)
    print(f"n_pairs = {n_pairs}    n_pooled = {n_pool}")
    print()
    print("POOLED correlations (direct + adversarial):")
    for k, v in pooled_results.items():
        print(f"  {k:<12s} r = {v['r']:+.3f}  p = {v['p']:.3f}")
    print(f"  max|r| = {max_pool:.3f}")
    print()
    print("PAIRED-DELTA correlations (adv - direct within pair):")
    for k, v in delta_results.items():
        print(f"  {k:<12s} r = {v['r']:+.3f}  p = {v['p']:.3f}")
    print(f"  max|r| = {max_delta:.3f}")
    print()
    print(f"pooled     : {'ORTHOGONAL' if results['orthogonal_pooled'] else 'NOT orthogonal'}")
    print(f"paired_delta: {'ORTHOGONAL' if results['orthogonal_delta'] else 'NOT orthogonal'}")
    print(f"interpretation: {results['interpretation']}")

    out_path = Path(results_path).with_suffix("")
    out_path = out_path.parent / (out_path.name + "_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--analyze-only", metavar="PATH")
    args = parser.parse_args()

    if args.analyze_only:
        analyze(args.analyze_only)
        return

    out = run_capture(args.model, args.max_tokens, args.max_pairs)
    print()
    analyze(out)


if __name__ == "__main__":
    main()
