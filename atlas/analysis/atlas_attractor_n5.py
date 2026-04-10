# -*- coding: utf-8 -*-
"""
atlas_attractor_n5.py -- RLHF Attractor analysis at n=5 families

v0.3 expansion: runs the exact same pre-registered analysis from
atlas_attractor.py, atlas_entropy.py, attractor_bootstrap.py, and
attractor_bootstrap_entropy.py, but across n=5 (family x training)
pairs instead of n=3. The new families are:

  - Gemma-3-1b  (Gemma generation 3, different from Gemma-2)
  - Llama-3.2-1B (Llama-3.2 scale variant, different from the 3B)
  - Qwen2.5-1.5B (Qwen2.5 scale variant, different from the 3B)

plus the 3 original families from v0.2:

  - Gemma-2-2B
  - Llama-3.2-3B
  - Qwen2.5-3B

Produces the final decision-rule evaluation described in
atlas/PREREG_v0.3_attractor_replication.md section 5.

Usage:
    python atlas/analysis/atlas_attractor_n5.py
"""

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
    "Gemma-3-1B": {
        "base": "google/gemma-3-1b-pt",
        "instruct": "google/gemma-3-1b-it",
    },
    "Llama-3.2-1B": {
        "base": "meta-llama/Llama-3.2-1B",
        "instruct": "meta-llama/Llama-3.2-1B-Instruct",
    },
    "Qwen2.5-1.5B": {
        "base": "Qwen/Qwen2.5-1.5B",
        "instruct": "Qwen/Qwen2.5-1.5B-Instruct",
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


def per_prompt_D(capture: dict) -> dict:
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


def per_prompt_entropy(capture: dict, which: str = "mean") -> dict:
    """which: 'mean' or 'early_mean'"""
    out = {}
    for cat in CATEGORY_ORDER:
        out[cat] = []
        for pid, rec in capture["probes"].get(cat, {}).items():
            if "error" in rec:
                continue
            em = rec.get("summaries", {}).get("entropy", {}).get(which)
            if em is not None:
                out[cat].append(em)
    return out


def signature(per_prompt: dict) -> np.ndarray:
    return np.array([np.mean(per_prompt[c]) if per_prompt[c] else 0.0
                     for c in CATEGORY_ORDER], dtype=float)


def shape_normalize(s):
    return s - s.mean()


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a @ b) / (na * nb)) if na and nb else 0.0


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


def loo_mean(sigs, family_order):
    return float(np.mean(list(per_family_loo(sigs, family_order).values())))


def bootstrap_loo(per_prompt_sigs, family_order, n_boot, rng):
    """
    Bootstrap the per-family LOO cosines by resampling prompts
    within each category for each (family, training) combination.
    Returns: per-family lists of bootstrap cosines + mean cosine list.
    """
    per_fam = {f: [] for f in family_order}
    means = []
    for _ in range(n_boot):
        sigs = {}
        for key, cat_dict in per_prompt_sigs.items():
            vals = []
            for cat in CATEGORY_ORDER:
                x = cat_dict[cat]
                if not x:
                    vals.append(0.0)
                    continue
                idx = rng.integers(0, len(x), size=len(x))
                vals.append(float(np.mean(np.array(x)[idx])))
            sigs[key] = shape_normalize(np.array(vals))
        pf = per_family_loo(sigs, family_order)
        for f in family_order:
            per_fam[f].append(pf[f])
        means.append(loo_mean(sigs, family_order))
    return per_fam, means


def permutation_loo(sigs, family_order, n_perm, rng):
    """
    Label-swap permutation: for each family, with 50% probability
    swap base/instruct labels. Recompute mean LOO cosine.
    """
    out = []
    for _ in range(n_perm):
        shuffled = dict(sigs)
        for f in family_order:
            if rng.random() < 0.5:
                shuffled[(f, "base")], shuffled[(f, "instruct")] = (
                    shuffled[(f, "instruct")], shuffled[(f, "base")]
                )
        out.append(loo_mean(shuffled, family_order))
    return out


def ci_summary(arr):
    arr = np.array(arr)
    return {
        "median": float(np.median(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "frac_positive": float(np.mean(arr > 0)),
    }


def main():
    rng = np.random.default_rng(42)
    n_boot = 2000
    n_perm = 2000

    # Load every model
    caps = {}
    missing = []
    for family, vs in MODEL_GROUPS.items():
        for training in ("base", "instruct"):
            try:
                caps[(family, training)] = load_capture(vs[training])
            except FileNotFoundError as e:
                missing.append((family, training, str(e)))
    if missing:
        print("[!] missing captures:")
        for m in missing:
            print(f"    {m}")
        print()

    family_order = [f for f in MODEL_GROUPS
                     if (f, "base") in caps and (f, "instruct") in caps]
    print(f"Active families: {family_order}  (n={len(family_order)})")
    print()

    # ── D analysis ──
    d_per_prompt = {k: per_prompt_D(v) for k, v in caps.items()}
    d_sigs = {k: shape_normalize(signature(v))
              for k, v in d_per_prompt.items()}
    d_per_fam = per_family_loo(d_sigs, family_order)
    d_mean = loo_mean(d_sigs, family_order)

    d_boot_per_fam, d_boot_means = bootstrap_loo(
        d_per_prompt, family_order, n_boot, rng)
    d_boot_ci_per_fam = {f: ci_summary(d_boot_per_fam[f]) for f in family_order}
    d_boot_ci_mean = ci_summary(d_boot_means)
    d_perm = permutation_loo(d_sigs, family_order, n_perm, rng)
    d_perm_arr = np.array(d_perm)
    d_perm_p = float(np.mean(d_perm_arr >= d_mean))

    # ── Entropy (full) analysis ──
    e_per_prompt = {k: per_prompt_entropy(v, "mean") for k, v in caps.items()}
    e_sigs = {k: shape_normalize(signature(v))
              for k, v in e_per_prompt.items()}
    e_per_fam = per_family_loo(e_sigs, family_order)
    e_mean = loo_mean(e_sigs, family_order)
    e_boot_per_fam, e_boot_means = bootstrap_loo(
        e_per_prompt, family_order, n_boot, rng)
    e_boot_ci_per_fam = {f: ci_summary(e_boot_per_fam[f])
                          for f in family_order}
    e_boot_ci_mean = ci_summary(e_boot_means)
    e_perm = permutation_loo(e_sigs, family_order, n_perm, rng)
    e_perm_p = float(np.mean(np.array(e_perm) >= e_mean))

    # ── Entropy (early window) analysis ──
    ee_per_prompt = {k: per_prompt_entropy(v, "early_mean") for k, v in caps.items()}
    ee_sigs = {k: shape_normalize(signature(v))
               for k, v in ee_per_prompt.items()}
    ee_per_fam = per_family_loo(ee_sigs, family_order)
    ee_mean = loo_mean(ee_sigs, family_order)
    ee_boot_per_fam, ee_boot_means = bootstrap_loo(
        ee_per_prompt, family_order, n_boot, rng)
    ee_boot_ci_per_fam = {f: ci_summary(ee_boot_per_fam[f])
                           for f in family_order}
    ee_boot_ci_mean = ci_summary(ee_boot_means)
    ee_perm = permutation_loo(ee_sigs, family_order, n_perm, rng)
    ee_perm_p = float(np.mean(np.array(ee_perm) >= ee_mean))

    # ── Print report ──
    print("=" * 78)
    print(f"ATLAS v0.3 ATTRACTOR ANALYSIS (n={len(family_order)} families)")
    print("=" * 78)
    print()
    print(f"Sealed decision rule (from PREREG_v0.3_attractor_replication.md):")
    print("  H1 supported iff ALL of:")
    print("    (a) mean LOO cosine (entropy early-window)   >=  0.40")
    print("    (b) permutation p (one-sided)                <   0.05")
    print("    (c) bootstrap 95% CI on mean LOO cos         > 0 (lower bound)")
    print()

    def print_block(name, per_fam, mean, ci_per_fam, ci_mean, perm_p):
        print(f"## {name}")
        print(f"{'family':<18s}{'obs cos':>10s}  bootstrap CI   {'P(>0)':>8s}")
        print("-" * 65)
        for f in family_order:
            ci = ci_per_fam[f]
            print(f"{f:<18s}{per_fam[f]:>+10.3f}  "
                  f"[{ci['ci95_low']:+.3f}, {ci['ci95_high']:+.3f}]  "
                  f"{ci['frac_positive']:>8.1%}")
        print(f"{'mean':<18s}{mean:>+10.3f}  "
              f"[{ci_mean['ci95_low']:+.3f}, {ci_mean['ci95_high']:+.3f}]  "
              f"{ci_mean['frac_positive']:>8.1%}")
        print(f"permutation p (one-sided, n={n_perm}): {perm_p:.4f}")
        print()

    print_block("D (full trajectory)", d_per_fam, d_mean,
                 d_boot_ci_per_fam, d_boot_ci_mean, d_perm_p)
    print_block("Entropy (full trajectory)", e_per_fam, e_mean,
                 e_boot_ci_per_fam, e_boot_ci_mean, e_perm_p)
    print_block("Entropy (early window, 0-6)", ee_per_fam, ee_mean,
                 ee_boot_ci_per_fam, ee_boot_ci_mean, ee_perm_p)

    # ── H1 decision ──
    primary_mean = ee_mean
    primary_ci_low = ee_boot_ci_mean["ci95_low"]
    primary_perm_p = ee_perm_p

    cond_a = primary_mean >= 0.40
    cond_b = primary_perm_p < 0.05
    cond_c = primary_ci_low > 0.0

    print("=" * 78)
    print("H1 DECISION (per PREREG_v0.3_attractor_replication.md)")
    print("=" * 78)
    print(f"  (a) mean entropy-early cos >= 0.40:  "
          f"{primary_mean:+.3f}   {'PASS' if cond_a else 'FAIL'}")
    print(f"  (b) permutation p < 0.05:            "
          f"{primary_perm_p:.4f}   {'PASS' if cond_b else 'FAIL'}")
    print(f"  (c) bootstrap CI lower > 0:          "
          f"{primary_ci_low:+.3f}   {'PASS' if cond_c else 'FAIL'}")
    print()
    if cond_a and cond_b and cond_c:
        verdict = "H1 SUPPORTED"
    else:
        fails = []
        if not cond_a: fails.append("(a)")
        if not cond_b: fails.append("(b)")
        if not cond_c: fails.append("(c)")
        verdict = f"H1 NOT SUPPORTED (failed: {', '.join(fails)})"
    print(f"VERDICT: {verdict}")
    print()

    # Save
    report = {
        "n_families": len(family_order),
        "family_order": family_order,
        "prereg": "atlas/PREREG_v0.3_attractor_replication.md",
        "D": {
            "per_family_loo": d_per_fam,
            "mean_loo": d_mean,
            "bootstrap_ci_per_family": d_boot_ci_per_fam,
            "bootstrap_ci_mean": d_boot_ci_mean,
            "permutation_p_one_sided": d_perm_p,
        },
        "entropy_full": {
            "per_family_loo": e_per_fam,
            "mean_loo": e_mean,
            "bootstrap_ci_per_family": e_boot_ci_per_fam,
            "bootstrap_ci_mean": e_boot_ci_mean,
            "permutation_p_one_sided": e_perm_p,
        },
        "entropy_early": {
            "per_family_loo": ee_per_fam,
            "mean_loo": ee_mean,
            "bootstrap_ci_per_family": ee_boot_ci_per_fam,
            "bootstrap_ci_mean": ee_boot_ci_mean,
            "permutation_p_one_sided": ee_perm_p,
        },
        "H1_decision": {
            "primary_measurement": "entropy early window",
            "mean_loo": primary_mean,
            "condition_a_passed": cond_a,
            "condition_b_passed": cond_b,
            "condition_c_passed": cond_c,
            "verdict": verdict,
        },
    }
    out_path = ATLAS / "analysis" / "atlas_v0.3_attractor_n5.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
