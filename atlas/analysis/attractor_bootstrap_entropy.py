# -*- coding: utf-8 -*-
"""
attractor_bootstrap_entropy.py -- Same bootstrap + permutation test
as attractor_bootstrap.py, but on the entropy atlas signatures
instead of D. Gives us the comparable statistical rigor check for
the entropy bridge finding.

Output: atlas/analysis/attractor_bootstrap_stats_entropy.json
"""

import argparse
import json
import sys
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


def per_prompt_entropy(capture: dict) -> dict:
    out = {}
    for cat in CATEGORY_ORDER:
        out[cat] = []
        for pid, rec in capture["probes"].get(cat, {}).items():
            if "error" in rec:
                continue
            em = rec.get("summaries", {}).get("entropy", {}).get("mean")
            if em is not None:
                out[cat].append(em)
    return out


def per_prompt_entropy_early(capture: dict) -> dict:
    out = {}
    for cat in CATEGORY_ORDER:
        out[cat] = []
        for pid, rec in capture["probes"].get(cat, {}).items():
            if "error" in rec:
                continue
            em = rec.get("summaries", {}).get("entropy", {}).get("early_mean")
            if em is not None:
                out[cat].append(em)
    return out


def signature_from_per_prompt(per_prompt: dict) -> np.ndarray:
    return np.array([np.mean(per_prompt[c]) if per_prompt[c] else 0.0
                     for c in CATEGORY_ORDER], dtype=float)


def shape_normalize(s):
    return s - s.mean()


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a @ b) / (na * nb)) if na and nb else 0.0


def loo_cosine_mean(sigs, family_order):
    mean = 0.0
    for f_held in family_order:
        other_its = [sigs[(f, "instruct")]
                     for f in family_order if f != f_held]
        attractor_loo = np.mean(other_its, axis=0)
        s_base = sigs[(f_held, "base")]
        s_it = sigs[(f_held, "instruct")]
        mean += cosine(attractor_loo - s_base, s_it - s_base)
    return mean / len(family_order)


def per_family_loo(sigs, family_order):
    out = {}
    for f_held in family_order:
        other_its = [sigs[(f, "instruct")]
                     for f in family_order if f != f_held]
        attractor_loo = np.mean(other_its, axis=0)
        s_base = sigs[(f_held, "base")]
        s_it = sigs[(f_held, "instruct")]
        out[f_held] = cosine(attractor_loo - s_base, s_it - s_base)
    return out


def bootstrap_loo(per_prompt_full, per_prompt_early, family_order,
                   n_boot, rng):
    full_per_fam = {f: [] for f in family_order}
    early_per_fam = {f: [] for f in family_order}
    full_mean = []
    early_mean = []

    for _ in range(n_boot):
        full_sigs = {}
        early_sigs = {}
        for key, cat_dict in per_prompt_full.items():
            vals = []
            for cat in CATEGORY_ORDER:
                x = cat_dict[cat]
                if not x:
                    vals.append(0.0)
                    continue
                idx = rng.integers(0, len(x), size=len(x))
                vals.append(float(np.mean(np.array(x)[idx])))
            full_sigs[key] = shape_normalize(np.array(vals))
        for key, cat_dict in per_prompt_early.items():
            vals = []
            for cat in CATEGORY_ORDER:
                x = cat_dict[cat]
                if not x:
                    vals.append(0.0)
                    continue
                idx = rng.integers(0, len(x), size=len(x))
                vals.append(float(np.mean(np.array(x)[idx])))
            early_sigs[key] = shape_normalize(np.array(vals))
        f_full = per_family_loo(full_sigs, family_order)
        f_early = per_family_loo(early_sigs, family_order)
        for f in family_order:
            full_per_fam[f].append(f_full[f])
            early_per_fam[f].append(f_early[f])
        full_mean.append(loo_cosine_mean(full_sigs, family_order))
        early_mean.append(loo_cosine_mean(early_sigs, family_order))

    return full_per_fam, early_per_fam, full_mean, early_mean


def ci_summary(arr):
    arr = np.array(arr)
    return {
        "median": float(np.median(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "frac_positive": float(np.mean(arr > 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    family_order = list(MODEL_GROUPS.keys())

    per_prompt_full = {}
    per_prompt_early = {}
    for family, vs in MODEL_GROUPS.items():
        for training in ("base", "instruct"):
            cap = load_capture(vs[training])
            per_prompt_full[(family, training)] = per_prompt_entropy(cap)
            per_prompt_early[(family, training)] = per_prompt_entropy_early(cap)

    obs_full_sigs = {k: shape_normalize(signature_from_per_prompt(v))
                      for k, v in per_prompt_full.items()}
    obs_early_sigs = {k: shape_normalize(signature_from_per_prompt(v))
                       for k, v in per_prompt_early.items()}

    obs_full_mean = loo_cosine_mean(obs_full_sigs, family_order)
    obs_early_mean = loo_cosine_mean(obs_early_sigs, family_order)
    obs_full_per_fam = per_family_loo(obs_full_sigs, family_order)
    obs_early_per_fam = per_family_loo(obs_early_sigs, family_order)

    print("=" * 72)
    print("ENTROPY ATTRACTOR BOOTSTRAP + PERMUTATION")
    print("=" * 72)
    print()
    print("Observed LOO cosines:")
    print(f"  {'family':<18s} {'full':>8s} {'early':>8s}")
    print("-" * 40)
    for f in family_order:
        print(f"  {f:<18s} {obs_full_per_fam[f]:>+8.3f} "
              f"{obs_early_per_fam[f]:>+8.3f}")
    print(f"  {'mean':<18s} {obs_full_mean:>+8.3f} {obs_early_mean:>+8.3f}")
    print()

    # Bootstrap
    print(f"Bootstrap: {args.n_boot} resamples ...")
    full_pf, early_pf, full_mean_boot, early_mean_boot = bootstrap_loo(
        per_prompt_full, per_prompt_early, family_order, args.n_boot, rng)

    ci_full = {f: ci_summary(full_pf[f]) for f in family_order}
    ci_early = {f: ci_summary(early_pf[f]) for f in family_order}
    ci_full_mean = ci_summary(full_mean_boot)
    ci_early_mean = ci_summary(early_mean_boot)

    print()
    print("Bootstrap 95% CI -- full trajectory:")
    for f in family_order:
        c = ci_full[f]
        print(f"  {f:<18s} median {c['median']:+.3f}  "
              f"CI [{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]  "
              f"P(>0) {c['frac_positive']:.3f}")
    print(f"  mean               median {ci_full_mean['median']:+.3f}  "
          f"CI [{ci_full_mean['ci95_low']:+.3f}, "
          f"{ci_full_mean['ci95_high']:+.3f}]  "
          f"P(>0) {ci_full_mean['frac_positive']:.3f}")
    print()
    print("Bootstrap 95% CI -- early window:")
    for f in family_order:
        c = ci_early[f]
        print(f"  {f:<18s} median {c['median']:+.3f}  "
              f"CI [{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]  "
              f"P(>0) {c['frac_positive']:.3f}")
    print(f"  mean               median {ci_early_mean['median']:+.3f}  "
          f"CI [{ci_early_mean['ci95_low']:+.3f}, "
          f"{ci_early_mean['ci95_high']:+.3f}]  "
          f"P(>0) {ci_early_mean['frac_positive']:.3f}")
    print()

    # Permutation (label-swap within family)
    print(f"Permutation: {args.n_perm} shuffles ...")
    perm_full = []
    perm_early = []
    for _ in range(args.n_perm):
        shuffled_full = dict(obs_full_sigs)
        shuffled_early = dict(obs_early_sigs)
        for f in family_order:
            if rng.random() < 0.5:
                shuffled_full[(f, "base")], shuffled_full[(f, "instruct")] = (
                    shuffled_full[(f, "instruct")], shuffled_full[(f, "base")]
                )
                shuffled_early[(f, "base")], shuffled_early[(f, "instruct")] = (
                    shuffled_early[(f, "instruct")], shuffled_early[(f, "base")]
                )
        perm_full.append(loo_cosine_mean(shuffled_full, family_order))
        perm_early.append(loo_cosine_mean(shuffled_early, family_order))

    pf = np.array(perm_full)
    pe = np.array(perm_early)
    p_full = float(np.mean(pf >= obs_full_mean))
    p_early = float(np.mean(pe >= obs_early_mean))

    print()
    print("Permutation test:")
    print(f"  full trajectory: observed {obs_full_mean:+.3f}, "
          f"perm median {float(np.median(pf)):+.3f}, "
          f"perm 95th {float(np.percentile(pf, 95)):+.3f}, "
          f"p={p_full:.4f}")
    print(f"  early window:    observed {obs_early_mean:+.3f}, "
          f"perm median {float(np.median(pe)):+.3f}, "
          f"perm 95th {float(np.percentile(pe, 95)):+.3f}, "
          f"p={p_early:.4f}")

    report = {
        "observed": {
            "full_mean": obs_full_mean,
            "early_mean": obs_early_mean,
            "full_per_fam": obs_full_per_fam,
            "early_per_fam": obs_early_per_fam,
        },
        "bootstrap": {
            "full_per_family": ci_full,
            "early_per_family": ci_early,
            "full_mean": ci_full_mean,
            "early_mean": ci_early_mean,
        },
        "permutation": {
            "full_p_one_sided": p_full,
            "early_p_one_sided": p_early,
            "full_perm_median": float(np.median(pf)),
            "early_perm_median": float(np.median(pe)),
            "full_perm_p95": float(np.percentile(pf, 95)),
            "early_perm_p95": float(np.percentile(pe, 95)),
        },
    }

    out_path = ATLAS / "analysis" / "attractor_bootstrap_stats_entropy.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
