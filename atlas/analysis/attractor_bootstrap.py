# -*- coding: utf-8 -*-
"""
attractor_bootstrap.py -- Bootstrap CIs + permutation test for the RLHF Attractor

Adds statistical rigor to the v0.2.1 attractor finding:

  1. Bootstrap the LOO direction test across prompt subsamples.
     For each family, resample the 15 prompts per category with
     replacement, recompute per-category means, re-derive the LOO
     attractor direction, compute cos(d_LOO, d_true). Repeat B times.
     Report per-family median + 95% percentile CI.

  2. Permutation test on the attractor structure.
     Shuffle the (model, training) labels, recompute the LOO mean
     cosine under each shuffle. Report how often shuffled cosines
     exceed the observed cosine. P = fraction of shuffles >= observed.

  3. Leave-one-prompt-out robustness.
     For each of the 15 prompts per category, remove it and recompute
     the attractor + LOO test. Report the full spread of cosines and
     the worst-case drop from the full-data value.

Output: atlas/analysis/attractor_bootstrap_stats.json

Usage:
    python atlas/analysis/attractor_bootstrap.py
    python atlas/analysis/attractor_bootstrap.py --n-boot 5000
    python atlas/analysis/attractor_bootstrap.py --n-perm 5000
"""

import argparse
import json
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ATLAS = ROOT / "atlas"

CATEGORY_ORDER = ["retrieval", "reasoning", "refusal", "creative",
                   "adversarial", "hallucination"]

MODEL_GROUPS = {
    "Gemma-2-2B": {
        "base": "google/gemma-2-2b",
        "instruct": "google/gemma-2-2b-it",
    },
    "Llama-3.2-3B": {
        "base": "meta-llama/Llama-3.2-3B",
        "instruct": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "Qwen2.5-3B": {
        "base": "Qwen/Qwen2.5-3B",
        "instruct": "Qwen/Qwen2.5-3B-Instruct",
    },
}


def load_capture(model_name: str) -> dict:
    cap_dir = ATLAS / "captures"
    slug = model_name.replace("/", "__").replace(".", "_")
    matches = sorted(cap_dir.glob(f"{slug}__v*.json"))
    if not matches:
        raise FileNotFoundError(model_name)
    with open(matches[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def per_prompt_d(capture: dict) -> dict:
    """{category: [per-prompt D_mean, ...]}."""
    out = {}
    for cat in CATEGORY_ORDER:
        out[cat] = []
        for pid, rec in capture["probes"].get(cat, {}).items():
            if "error" in rec:
                continue
            dm = rec.get("summaries", {}).get("D", {}).get("mean")
            if dm is not None:
                out[cat].append(dm)
    return out


def signature_from_per_prompt(per_prompt: dict) -> np.ndarray:
    """Mean over prompts per category -> 6-vector."""
    return np.array([np.mean(per_prompt[c]) if per_prompt[c] else 0.0
                     for c in CATEGORY_ORDER], dtype=float)


def shape_normalize(s: np.ndarray) -> np.ndarray:
    return s - s.mean()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a @ b) / (na * nb)) if na and nb else 0.0


def loo_cosine(sigs: dict, family_order: list) -> float:
    """Mean LOO cosine across families for the given signature dict."""
    mean = 0.0
    for f_held in family_order:
        other_its = [sigs[(f, "instruct")]
                     for f in family_order if f != f_held]
        attractor_loo = np.mean(other_its, axis=0)
        s_base = sigs[(f_held, "base")]
        s_it = sigs[(f_held, "instruct")]
        d_loo = attractor_loo - s_base
        d_true = s_it - s_base
        mean += cosine(d_loo, d_true)
    return mean / len(family_order)


def per_family_loo_cosines(sigs: dict, family_order: list) -> dict:
    out = {}
    for f_held in family_order:
        other_its = [sigs[(f, "instruct")]
                     for f in family_order if f != f_held]
        attractor_loo = np.mean(other_its, axis=0)
        s_base = sigs[(f_held, "base")]
        s_it = sigs[(f_held, "instruct")]
        d_loo = attractor_loo - s_base
        d_true = s_it - s_base
        out[f_held] = cosine(d_loo, d_true)
    return out


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load per-prompt D values per (family, training)
    per_prompt = {}
    captures = {}
    for family, vs in MODEL_GROUPS.items():
        for training in ("base", "instruct"):
            cap = load_capture(vs[training])
            captures[(family, training)] = cap
            per_prompt[(family, training)] = per_prompt_d(cap)

    family_order = list(MODEL_GROUPS.keys())

    # ── Observed LOO (shape-normalized) ──
    obs_shape_sigs = {
        k: shape_normalize(signature_from_per_prompt(v))
        for k, v in per_prompt.items()
    }
    observed_mean_cos = loo_cosine(obs_shape_sigs, family_order)
    observed_per_family = per_family_loo_cosines(obs_shape_sigs, family_order)

    print("=" * 72)
    print("ATTRACTOR BOOTSTRAP + PERMUTATION ANALYSIS")
    print("=" * 72)
    print()
    print("Observed LOO cosines:")
    for f, c in observed_per_family.items():
        print(f"  {f:<18s} cos = {c:+.3f}")
    print(f"  mean             = {observed_mean_cos:+.3f}")
    print()

    # ── Bootstrap: resample prompts within each category ──
    print(f"Bootstrap: {args.n_boot} resamples ...")
    boot_per_family = {f: [] for f in family_order}
    boot_mean = []

    for b in range(args.n_boot):
        # Build resampled signatures
        boot_sigs = {}
        for key, cat_dict in per_prompt.items():
            sig_vals = []
            for cat in CATEGORY_ORDER:
                vals = cat_dict[cat]
                if not vals:
                    sig_vals.append(0.0)
                    continue
                idx = rng.integers(0, len(vals), size=len(vals))
                resampled = np.array(vals)[idx]
                sig_vals.append(float(resampled.mean()))
            raw = np.array(sig_vals)
            boot_sigs[key] = shape_normalize(raw)
        per_fam = per_family_loo_cosines(boot_sigs, family_order)
        for f, c in per_fam.items():
            boot_per_family[f].append(c)
        boot_mean.append(loo_cosine(boot_sigs, family_order))
        if (b + 1) % 500 == 0:
            print(f"  {b+1}/{args.n_boot} done")

    ci_per_family = {}
    for f in family_order:
        arr = np.array(boot_per_family[f])
        ci_per_family[f] = {
            "median": float(np.median(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
            "frac_positive": float(np.mean(arr > 0)),
        }

    mean_arr = np.array(boot_mean)
    mean_ci = {
        "median": float(np.median(mean_arr)),
        "ci95_low": float(np.percentile(mean_arr, 2.5)),
        "ci95_high": float(np.percentile(mean_arr, 97.5)),
        "frac_positive": float(np.mean(mean_arr > 0)),
    }

    print()
    print("Bootstrap 95% CIs on per-family LOO cosines:")
    for f in family_order:
        c = ci_per_family[f]
        print(f"  {f:<18s} median = {c['median']:+.3f}  "
              f"95% CI = [{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]  "
              f"P(>0) = {c['frac_positive']:.3f}")
    print(f"  mean (all 3)       median = {mean_ci['median']:+.3f}  "
          f"95% CI = [{mean_ci['ci95_low']:+.3f}, "
          f"{mean_ci['ci95_high']:+.3f}]  "
          f"P(>0) = {mean_ci['frac_positive']:.3f}")
    print()

    # ── Permutation test: shuffle (family, training) labels ──
    # The null hypothesis: training has no systematic effect, so swapping
    # base/instruct labels within a family should give similar LOO
    # cosines. Under H0, the observed mean cosine should not stand out
    # against the permutation distribution.
    print(f"Permutation test: {args.n_perm} shuffles ...")
    perm_means = []
    keys = list(per_prompt.keys())
    base_sigs = {k: shape_normalize(signature_from_per_prompt(v))
                 for k, v in per_prompt.items()}

    for p in range(args.n_perm):
        # For each family, flip base/instruct with 50% prob
        shuffled = dict(base_sigs)
        for f in family_order:
            if rng.random() < 0.5:
                shuffled[(f, "base")], shuffled[(f, "instruct")] = (
                    shuffled[(f, "instruct")], shuffled[(f, "base")]
                )
        perm_means.append(loo_cosine(shuffled, family_order))
        if (p + 1) % 500 == 0:
            print(f"  {p+1}/{args.n_perm} done")

    perm_arr = np.array(perm_means)
    p_value = float(np.mean(perm_arr >= observed_mean_cos))
    p_value_abs = float(np.mean(np.abs(perm_arr) >= abs(observed_mean_cos)))

    print()
    print("Permutation test:")
    print(f"  observed mean LOO cos       = {observed_mean_cos:+.3f}")
    print(f"  permutation distribution    median = {float(np.median(perm_arr)):+.3f}")
    print(f"  permutation 95th percentile = {float(np.percentile(perm_arr, 95)):+.3f}")
    print(f"  permutation 5th percentile  = {float(np.percentile(perm_arr, 5)):+.3f}")
    print(f"  p (one-sided, perm >= obs)  = {p_value:.4f}")
    print(f"  p (two-sided, |perm| >= |obs|) = {p_value_abs:.4f}")
    print()

    # ── Save ──
    report = {
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "observed_mean_cos": observed_mean_cos,
        "observed_per_family": observed_per_family,
        "bootstrap": {
            "per_family": ci_per_family,
            "mean": mean_ci,
        },
        "permutation": {
            "observed": observed_mean_cos,
            "perm_median": float(np.median(perm_arr)),
            "perm_p5": float(np.percentile(perm_arr, 5)),
            "perm_p95": float(np.percentile(perm_arr, 95)),
            "p_one_sided": p_value,
            "p_two_sided": p_value_abs,
        },
    }

    out_path = ATLAS / "analysis" / "attractor_bootstrap_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
