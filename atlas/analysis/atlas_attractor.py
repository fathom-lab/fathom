# -*- coding: utf-8 -*-
"""
atlas_attractor.py -- RLHF Attractor analysis
===============================================

Formalizes an observation from the v0.2 cross-model analysis:
the three instruct models in the atlas (Gemma-IT, Llama-IT, Qwen-IT)
cluster within pairwise cognitive distance 0.09-0.14, forming a
tight region in 6-dimensional per-category signature space, while
their three base counterparts are 0.67-1.50 apart.

This script defines the RLHF Attractor as a first-class object,
computes derived quantities (RLHF Deviation Score, direction
prediction test, leave-one-out robustness), and writes outputs
for the FINDINGS_v0.2_attractor memo.

Definitions
-----------
Signature vector:   s_M in R^6 of (retrieval, reasoning, refusal,
                                    creative, adversarial, hallucination)
                    mean-D values for model M.

Attractor point:    A_RLHF := (1/3)(s_Gemma-IT + s_Llama-IT + s_Qwen-IT)
                    The centroid of the three RLHF-trained instruct
                    models in signature space.

RLHF Deviation Score:
                    RDS(M) := ||s_M - A_RLHF||   (Euclidean in R^6)

LOO attractor for family F:
                    A_LOO(F) := centroid of IT models NOT in family F
                    Used for leave-one-out prediction tests.

Direction prediction test:
                    For each family F, compute
                      d_LOO(F) := A_LOO(F) - s_F,base
                      d_true(F) := s_F,IT - s_F,base
                    alignment(F) := cos(d_LOO, d_true)

                    If alignment(F) > 0 across all 3 families, the
                    attractor direction (computed WITHOUT using F's IT)
                    still predicts where F's actual RLHF goes.

All analysis is CPU, pure Python + numpy.
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


def load_capture_signature(model_name: str) -> np.ndarray:
    """Load the per-category D mean signature vector from a capture file."""
    cap_dir = ATLAS / "captures"
    # The runner writes files with '/' replaced by '__' and '.' replaced by '_'.
    # Capture files are named "<slug>__v<version>.json", so the exact match
    # uses "<slug>__v*.json" — this prevents "gemma-2-2b" from matching
    # "gemma-2-2b-it" via a prefix collision.
    slug = model_name.replace("/", "__").replace(".", "_")
    matches = list(cap_dir.glob(f"{slug}__v*.json"))
    if not matches:
        raise FileNotFoundError(f"No capture file for {model_name}")
    # Read the most recent version (lexicographic sort suffices for semver)
    matches.sort()
    with open(matches[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    sig = []
    for cat in CATEGORY_ORDER:
        sig.append(float(data["signatures"][cat]["D_mean"]))
    return np.array(sig, dtype=float)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


def shape_normalize(s: np.ndarray) -> np.ndarray:
    """
    Remove per-model magnitude: subtract the mean so what remains is
    the relative category profile. Models with different overall D
    magnitudes (e.g. Gemma base at ~0.14 vs Llama IT at ~0.23) become
    comparable in terms of their per-category DEVIATIONS from their
    own mean.
    """
    return s - s.mean()


def pearson_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Pearson r across the 6 categories. Shape-invariant."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0
    r = float(np.corrcoef(a, b)[0, 1])
    return 1.0 - r


def analyze():
    # Load all 6 raw signatures
    sigs_raw = {}
    for family, vs in MODEL_GROUPS.items():
        sigs_raw[(family, "base")] = load_capture_signature(vs["base"])
        sigs_raw[(family, "instruct")] = load_capture_signature(vs["instruct"])

    # Shape-normalize: subtract per-model mean so signatures are in
    # pure "category deviation" space, magnitude removed.
    sigs = {k: shape_normalize(v) for k, v in sigs_raw.items()}

    # ── Attractor: mean of the 3 SHAPE-NORMALIZED IT signatures ──
    it_signatures = [sigs[(f, "instruct")] for f in MODEL_GROUPS]
    attractor = np.mean(it_signatures, axis=0)

    # ── RDS for every model ──
    rds = {}
    for (family, training), s in sigs.items():
        rds[(family, training)] = euclidean(s, attractor)

    # ── Base → IT shift magnitudes ──
    shift_magnitude = {}
    for f in MODEL_GROUPS:
        shift = sigs[(f, "instruct")] - sigs[(f, "base")]
        shift_magnitude[f] = {
            "vector": shift.tolist(),
            "norm": float(np.linalg.norm(shift)),
        }

    # ── Leave-one-out direction prediction ──
    # For each family F, compute attractor from OTHER 2 IT models,
    # then test whether that attractor direction predicts F's actual shift.
    loo_results = {}
    for f_held in MODEL_GROUPS:
        other_its = [sigs[(f, "instruct")]
                     for f in MODEL_GROUPS if f != f_held]
        attractor_loo = np.mean(other_its, axis=0)
        s_base = sigs[(f_held, "base")]
        s_it = sigs[(f_held, "instruct")]
        d_loo = attractor_loo - s_base
        d_true = s_it - s_base
        loo_results[f_held] = {
            "attractor_loo": attractor_loo.tolist(),
            "d_loo": d_loo.tolist(),
            "d_true": d_true.tolist(),
            "cosine_loo_vs_true": cosine(d_loo, d_true),
            "norm_d_loo": float(np.linalg.norm(d_loo)),
            "norm_d_true": float(np.linalg.norm(d_true)),
            "rds_base_loo": euclidean(s_base, attractor_loo),
            "rds_it_loo": euclidean(s_it, attractor_loo),
            "rds_shrinkage_loo": (euclidean(s_base, attractor_loo) -
                                    euclidean(s_it, attractor_loo)),
        }

    # ── Full-attractor sanity check ──
    # Using the full attractor (all 3 IT), check how much closer IT
    # is than base. This is a biased test (attractor contains the IT)
    # but useful for context.
    sanity = {}
    for f in MODEL_GROUPS:
        sanity[f] = {
            "rds_base": rds[(f, "base")],
            "rds_it": rds[(f, "instruct")],
            "shrinkage": rds[(f, "base")] - rds[(f, "instruct")],
        }

    # Summary cosine across all 3 LOO tests
    mean_cosine = float(np.mean([loo_results[f]["cosine_loo_vs_true"]
                                  for f in MODEL_GROUPS]))

    report = {
        "atlas_version": "0.2.1",
        "attractor_signature": {
            "vector": attractor.tolist(),
            "categories": CATEGORY_ORDER,
            "per_category": {c: float(v)
                              for c, v in zip(CATEGORY_ORDER, attractor)},
        },
        "signatures": {
            f"{f}/{t}": sigs[(f, t)].tolist()
            for (f, t) in sigs
        },
        "rds": {
            f"{f}/{t}": rds[(f, t)]
            for (f, t) in rds
        },
        "base_to_it_shifts": shift_magnitude,
        "loo_direction_test": loo_results,
        "loo_mean_cosine": mean_cosine,
        "full_attractor_sanity": sanity,
    }

    out_path = ATLAS / "analysis" / "atlas_v0.2.1_attractor.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Print ──
    print("=" * 72)
    print("RLHF ATTRACTOR ANALYSIS (atlas v0.2.1)")
    print("=" * 72)
    print()
    print("Attractor signature (mean of 3 IT models):")
    for cat, val in zip(CATEGORY_ORDER, attractor):
        print(f"  {cat:<14s} {val:+.4f}")
    print()

    print("RLHF Deviation Score (RDS, Euclidean distance to attractor):")
    print(f"{'Model':<28s} {'Training':<10s} {'RDS':>10s}")
    print("-" * 52)
    for (family, training) in sigs:
        label = f"{family}/{training}"
        print(f"{label:<28s} {training:<10s} {rds[(family, training)]:>10.4f}")
    print()

    print("Full-attractor base-to-IT shrinkage (BIASED -- uses own IT):")
    print(f"{'Family':<18s} {'RDS(base)':>12s} {'RDS(IT)':>12s} {'shrink':>12s}")
    print("-" * 58)
    for f in MODEL_GROUPS:
        s = sanity[f]
        print(f"{f:<18s} {s['rds_base']:>12.4f} {s['rds_it']:>12.4f} "
              f"{s['shrinkage']:>+12.4f}")
    print()

    print("LEAVE-ONE-OUT direction prediction test:")
    print("(attractor computed WITHOUT the held-out family's IT)")
    print()
    print(f"{'Held-out':<18s} {'|d_loo|':>10s} {'|d_true|':>10s} "
          f"{'cos(loo,true)':>14s} {'rds shrink':>12s}")
    print("-" * 74)
    for f in MODEL_GROUPS:
        r = loo_results[f]
        print(f"{f:<18s} {r['norm_d_loo']:>10.4f} {r['norm_d_true']:>10.4f} "
              f"{r['cosine_loo_vs_true']:>+14.3f} "
              f"{r['rds_shrinkage_loo']:>+12.4f}")
    print("-" * 74)
    print(f"Mean LOO cosine across 3 families: {mean_cosine:+.3f}")
    print()

    # ── Verdict ──
    if mean_cosine > 0.7:
        verdict = ("STRONG: the attractor direction predicts actual RLHF "
                    "direction even under leave-one-out. The attractor is "
                    "not a centroid artifact.")
    elif mean_cosine > 0.3:
        verdict = ("MODERATE: directional prediction above chance. "
                    "Attractor structure is real but noisy at n=3 families.")
    elif mean_cosine > -0.3:
        verdict = ("NULL: no directional prediction above noise. The "
                    "attractor is likely just a centroid artifact at "
                    "this n.")
    else:
        verdict = ("REVERSED: attractor direction is anti-correlated "
                    "with actual RLHF direction. The hypothesis is "
                    "directionally wrong.")
    print(f"Verdict: {verdict}")
    print()
    print(f"[SAVED] {out_path}")
    return report


if __name__ == "__main__":
    analyze()
