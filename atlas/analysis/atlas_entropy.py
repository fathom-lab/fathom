# -*- coding: utf-8 -*-
"""
atlas_entropy.py -- Entropy-signature atlas analysis
=======================================================

Builds a parallel atlas using logit entropy instead of D as the
per-token measurement. The motivation: D requires white-box access
to the final-layer residual stream and the unembedding matrix, which
only open-weight models provide. Entropy only requires logprobs,
which every modern LLM API exposes via `logprobs=true`. If the
entropy signatures show the same category-level structure as the
D signatures on our open models, we have a validated bridge
measurement for extending the atlas to closed frontier models
(GPT-4, Claude, Gemini).

This script:
  1. Loads all 6 open-model atlas captures
  2. Extracts per-category mean entropy per model (shape-normalized)
  3. Computes the "entropy attractor" (centroid of 3 IT models)
  4. Runs the same Q1/Q2/Q3 + attractor + LOO analysis as the D version
  5. Reports side-by-side: does entropy show the same pattern as D?

Output: atlas/analysis/atlas_entropy_v0.2.1.json
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
}


def load_capture(model_name: str) -> dict:
    cap_dir = ATLAS / "captures"
    slug = model_name.replace("/", "__").replace(".", "_")
    matches = sorted(cap_dir.glob(f"{slug}__v*.json"))
    if not matches:
        raise FileNotFoundError(model_name)
    with open(matches[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def entropy_signature(capture: dict) -> np.ndarray:
    """
    Build a 6-vector of per-category mean entropy for this model.

    Entropy is captured per token in the trajectories; we take the
    mean across all tokens in a prompt, then the mean across all
    prompts in a category.
    """
    sig = []
    for cat_id in CATEGORY_ORDER:
        cat_prompts = capture["probes"].get(cat_id, {})
        per_prompt_means = []
        for pid, rec in cat_prompts.items():
            if "error" in rec:
                continue
            ent_summary = rec.get("summaries", {}).get("entropy")
            if ent_summary and ent_summary.get("mean") is not None:
                per_prompt_means.append(ent_summary["mean"])
        sig.append(float(np.mean(per_prompt_means)) if per_prompt_means else 0.0)
    return np.array(sig, dtype=float)


def entropy_signature_early(capture: dict) -> np.ndarray:
    """Per-category mean of 'early_mean' entropy — tokens 0-6 window."""
    sig = []
    for cat_id in CATEGORY_ORDER:
        cat_prompts = capture["probes"].get(cat_id, {})
        vals = []
        for pid, rec in cat_prompts.items():
            if "error" in rec:
                continue
            ent_summary = rec.get("summaries", {}).get("entropy")
            if ent_summary and ent_summary.get("early_mean") is not None:
                vals.append(ent_summary["early_mean"])
        sig.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(sig, dtype=float)


def shape_normalize(s: np.ndarray) -> np.ndarray:
    return s - s.mean()


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a @ b) / (na * nb)) if na and nb else 0.0


def euclidean(a, b):
    return float(np.linalg.norm(a - b))


def main():
    captures = {}
    for family, vs in MODEL_GROUPS.items():
        captures[(family, "base")] = load_capture(vs["base"])
        captures[(family, "instruct")] = load_capture(vs["instruct"])

    # Raw and shape-normalized entropy signatures
    raw = {k: entropy_signature(v) for k, v in captures.items()}
    early_raw = {k: entropy_signature_early(v) for k, v in captures.items()}
    shaped = {k: shape_normalize(v) for k, v in raw.items()}
    shaped_early = {k: shape_normalize(v) for k, v in early_raw.items()}

    # Attractor (full trajectory, shape-normalized)
    it_sigs = [shaped[(f, "instruct")] for f in MODEL_GROUPS]
    attractor = np.mean(it_sigs, axis=0)
    it_sigs_early = [shaped_early[(f, "instruct")] for f in MODEL_GROUPS]
    attractor_early = np.mean(it_sigs_early, axis=0)

    # RDS per model
    rds = {f"{f}/{t}": euclidean(shaped[(f, t)], attractor)
            for (f, t) in shaped}
    rds_early = {f"{f}/{t}": euclidean(shaped_early[(f, t)], attractor_early)
                  for (f, t) in shaped_early}

    # LOO direction test on full trajectory
    loo = {}
    for f_held in MODEL_GROUPS:
        other_its = [shaped[(f, "instruct")]
                     for f in MODEL_GROUPS if f != f_held]
        attractor_loo = np.mean(other_its, axis=0)
        s_base = shaped[(f_held, "base")]
        s_it = shaped[(f_held, "instruct")]
        d_loo = attractor_loo - s_base
        d_true = s_it - s_base
        loo[f_held] = {
            "cos_loo_vs_true": cosine(d_loo, d_true),
            "norm_d_loo": float(np.linalg.norm(d_loo)),
            "norm_d_true": float(np.linalg.norm(d_true)),
        }

    # LOO direction test on early window
    loo_early = {}
    for f_held in MODEL_GROUPS:
        other_its = [shaped_early[(f, "instruct")]
                     for f in MODEL_GROUPS if f != f_held]
        attractor_loo = np.mean(other_its, axis=0)
        s_base = shaped_early[(f_held, "base")]
        s_it = shaped_early[(f_held, "instruct")]
        d_loo = attractor_loo - s_base
        d_true = s_it - s_base
        loo_early[f_held] = {
            "cos_loo_vs_true": cosine(d_loo, d_true),
            "norm_d_loo": float(np.linalg.norm(d_loo)),
            "norm_d_true": float(np.linalg.norm(d_true)),
        }

    # Entropy vs D correlation: for each model, compute the Pearson r
    # between its D signature (from existing atlas_v0.2.1 report) and
    # its entropy signature. High correlation = entropy is a valid proxy.
    d_report_path = ATLAS / "analysis" / "atlas_v0.2.1_attractor.json"
    entropy_vs_d = None
    if d_report_path.exists():
        with open(d_report_path, "r") as f:
            d_report = json.load(f)
        d_sigs = d_report["signatures"]  # these are shape-normalized
        entropy_vs_d = {}
        for f in MODEL_GROUPS:
            for t in ("base", "instruct"):
                key = f"{f}/{t}"
                if key in d_sigs:
                    d_vec = np.array(d_sigs[key])
                    e_vec = shaped[(f, t)]
                    # Correlation between per-category D profile and per-category entropy profile
                    if np.std(d_vec) > 0 and np.std(e_vec) > 0:
                        r = float(np.corrcoef(d_vec, e_vec)[0, 1])
                    else:
                        r = 0.0
                    entropy_vs_d[key] = r

    # Pairwise shape-distance matrix (1 - Pearson r) on entropy
    models = sorted(shaped.keys())
    dist = {}
    for m1 in models:
        dist[f"{m1[0]}/{m1[1]}"] = {}
        for m2 in models:
            s1, s2 = shaped[m1], shaped[m2]
            if np.std(s1) == 0 or np.std(s2) == 0:
                dist[f"{m1[0]}/{m1[1]}"][f"{m2[0]}/{m2[1]}"] = 0.0
            else:
                r = float(np.corrcoef(s1, s2)[0, 1])
                dist[f"{m1[0]}/{m1[1]}"][f"{m2[0]}/{m2[1]}"] = 1 - r

    report = {
        "measurement": "entropy",
        "atlas_version": "0.2.1-entropy",
        "raw_signatures": {f"{f}/{t}": raw[(f, t)].tolist() for (f, t) in raw},
        "shape_signatures": {f"{f}/{t}": shaped[(f, t)].tolist() for (f, t) in shaped},
        "raw_signatures_early": {f"{f}/{t}": early_raw[(f, t)].tolist() for (f, t) in early_raw},
        "shape_signatures_early": {f"{f}/{t}": shaped_early[(f, t)].tolist() for (f, t) in shaped_early},
        "attractor_full": attractor.tolist(),
        "attractor_early": attractor_early.tolist(),
        "rds_full": rds,
        "rds_early": rds_early,
        "loo_full": loo,
        "loo_early": loo_early,
        "loo_mean_cosine_full": float(np.mean([loo[f]["cos_loo_vs_true"]
                                                  for f in MODEL_GROUPS])),
        "loo_mean_cosine_early": float(np.mean([loo_early[f]["cos_loo_vs_true"]
                                                    for f in MODEL_GROUPS])),
        "entropy_vs_D_correlation": entropy_vs_d,
        "distance_matrix": dist,
    }

    out_path = ATLAS / "analysis" / "atlas_entropy_v0.2.1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("ENTROPY-SIGNATURE ATLAS (bridge to closed frontier models)")
    print("=" * 72)
    print()
    print("Raw entropy signatures (per category mean, full trajectory):")
    print(f"{'Model':<28s}", end="")
    for c in CATEGORY_ORDER:
        print(f"{c[:8]:>9s}", end="")
    print()
    print("-" * 80)
    for (f, t), vec in raw.items():
        label = f"{f}/{t}"
        print(f"{label:<28s}", end="")
        for v in vec:
            print(f"{v:>9.3f}", end="")
        print()
    print()
    print("Entropy Attractor (shape-normalized, full trajectory):")
    for cat, val in zip(CATEGORY_ORDER, attractor):
        print(f"  {cat:<14s} {val:+.4f}")
    print()
    print("Entropy RDS (Euclidean distance to attractor, shape-normalized):")
    print(f"{'Model':<28s}{'RDS full':>12s}{'RDS early':>12s}")
    print("-" * 52)
    for key, v in rds.items():
        print(f"{key:<28s}{v:>12.4f}{rds_early.get(key, 0):>12.4f}")
    print()
    print("LOO direction prediction (full trajectory):")
    for f, r in loo.items():
        print(f"  {f:<18s} cos = {r['cos_loo_vs_true']:+.3f}  "
              f"|d_loo| = {r['norm_d_loo']:.4f}  "
              f"|d_true| = {r['norm_d_true']:.4f}")
    print(f"  mean cosine (full)  = {report['loo_mean_cosine_full']:+.3f}")
    print()
    print("LOO direction prediction (early window, tokens 0-6):")
    for f, r in loo_early.items():
        print(f"  {f:<18s} cos = {r['cos_loo_vs_true']:+.3f}")
    print(f"  mean cosine (early) = {report['loo_mean_cosine_early']:+.3f}")
    print()
    if entropy_vs_d is not None:
        print("Entropy vs D per-category Pearson r (per model):")
        print("(How well does per-category entropy profile match per-category D profile?)")
        for key, r in entropy_vs_d.items():
            print(f"  {key:<32s} r = {r:+.3f}")
        mean_r = float(np.mean(list(entropy_vs_d.values())))
        print(f"  mean r = {mean_r:+.3f}")
        print()
        if mean_r > 0.7:
            verdict = ("STRONG: entropy signatures track D signatures tightly. "
                        "Entropy is a valid proxy for extending the atlas to "
                        "closed models.")
        elif mean_r > 0.3:
            verdict = ("MODERATE: entropy signatures share structure with D. "
                        "Usable as a secondary measurement with caveats.")
        elif mean_r > -0.3:
            verdict = ("WEAK/NULL: entropy and D measure different aspects of "
                        "cognitive state on this probe set. Entropy atlas is "
                        "its own thing, not a D proxy.")
        else:
            verdict = ("ANTI-CORRELATED: entropy and D move in opposite "
                        "directions per category. Needs investigation.")
        print(f"Verdict: {verdict}")

    print()
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
