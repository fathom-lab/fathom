# -*- coding: utf-8 -*-
"""
atlas_analyze.py -- First-pass cross-model atlas analysis
==========================================================

Loads all available atlas captures and answers three questions:

  Q1. Does the D-axis distinguish categories within a model?
      If yes: per-category D distributions separate visibly, and a
      simple classifier reaches non-chance accuracy.

  Q2. Does the D-axis distinguish models on the same category?
      If yes: per-model D-signatures cluster separately and a
      model classifier reaches non-chance accuracy.

  Q3. Is the cross-model category ordering consistent?
      If yes: when models agree that category A has lower D than
      category B, they agree on WHICH categories rank high/low.
      This would validate a universal "cognitive ranking" hypothesis.

All analysis is cheap and runs on CPU from saved JSON captures.
Outputs:
  atlas/analysis/atlas_v0.1_summary.json
  atlas/analysis/figures/cat_distribution_<model>.png
  atlas/analysis/figures/cross_model_category_means.png
  atlas/analysis/figures/model_distance_matrix.png

Usage:
  python atlas/analysis/atlas_analyze.py
"""

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ATLAS = ROOT / "atlas"


def load_captures() -> dict:
    """Find and load every *.json in atlas/captures/ that looks like a capture."""
    cap_dir = ATLAS / "captures"
    out = {}
    for p in sorted(cap_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "model" in data and "signatures" in data:
                out[data["model"]] = data
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
    return out


def per_prompt_D_means(capture: dict) -> dict:
    """Return {category: [D_mean per prompt]} from a capture."""
    out = {}
    for cat_id, cat in capture.get("probes", {}).items():
        d_means = []
        for pid, rec in cat.items():
            if "error" in rec:
                continue
            dm = rec.get("summaries", {}).get("D", {}).get("mean")
            if dm is not None:
                d_means.append(dm)
        out[cat_id] = d_means
    return out


def per_prompt_D_early_means(capture: dict) -> dict:
    """Return {category: [D_early_mean per prompt]} — the first-7-tokens window."""
    out = {}
    for cat_id, cat in capture.get("probes", {}).items():
        d_earlies = []
        for pid, rec in cat.items():
            if "error" in rec:
                continue
            de = rec.get("summaries", {}).get("D", {}).get("early_mean")
            if de is not None:
                d_earlies.append(de)
        out[cat_id] = d_earlies
    return out


# ── Q1: Within-model category separation ─────────────────────────────

def q1_category_separation(captures: dict) -> dict:
    """For each model, measure how well D separates the 6 categories."""
    report = {}
    for model, cap in captures.items():
        d_by_cat = per_prompt_D_means(cap)
        d_by_cat = {k: v for k, v in d_by_cat.items() if v}
        if len(d_by_cat) < 2:
            continue
        # One-way ANOVA-style F statistic by hand (no scipy dep at runtime)
        all_vals = []
        group_means = {}
        for cat, vals in d_by_cat.items():
            all_vals.extend(vals)
            group_means[cat] = float(np.mean(vals))
        grand = float(np.mean(all_vals))
        ss_between = sum(len(v) * (np.mean(v) - grand) ** 2
                          for v in d_by_cat.values())
        ss_within = sum(sum((x - np.mean(v)) ** 2 for x in v)
                         for v in d_by_cat.values())
        k = len(d_by_cat)
        n = len(all_vals)
        df_between = k - 1
        df_within = n - k
        if df_within > 0 and ss_within > 0:
            ms_b = ss_between / df_between
            ms_w = ss_within / df_within
            F = ms_b / ms_w
            # eta^2
            eta2 = ss_between / (ss_between + ss_within)
        else:
            F = 0.0
            eta2 = 0.0

        # Best/worst category pair (largest mean distance)
        pairs = list(combinations(d_by_cat.keys(), 2))
        max_gap = max(pairs,
                       key=lambda p: abs(group_means[p[0]] - group_means[p[1]]))
        max_gap_value = abs(group_means[max_gap[0]] - group_means[max_gap[1]])

        report[model] = {
            "group_means": group_means,
            "F_statistic": float(F),
            "eta_squared": float(eta2),
            "max_category_gap": {
                "categories": list(max_gap),
                "gap": float(max_gap_value),
            },
            "n_per_cat": {k: len(v) for k, v in d_by_cat.items()},
        }
    return report


# ── Q2: Cross-model signature distance ───────────────────────────────

def q2_model_distance_matrix(captures: dict) -> dict:
    """
    Compute pairwise cosine distance between model signatures.
    Signature = concatenation of per-category D_mean values.
    """
    models = sorted(captures.keys())
    signatures = {}
    for m in models:
        d_by_cat = per_prompt_D_means(captures[m])
        # Use a canonical category order (from the probe set)
        canon = ["retrieval", "reasoning", "refusal", "creative",
                 "adversarial", "hallucination"]
        sig = []
        for cat in canon:
            if cat in d_by_cat and d_by_cat[cat]:
                sig.append(float(np.mean(d_by_cat[cat])))
            else:
                sig.append(None)
        signatures[m] = sig

    # Pairwise cosine distances (ignoring None categories)
    dist = {}
    for m1 in models:
        dist[m1] = {}
        for m2 in models:
            s1 = np.array([x for x in signatures[m1] if x is not None])
            s2 = np.array([x for x in signatures[m2] if x is not None])
            if len(s1) == 0 or len(s2) == 0 or len(s1) != len(s2):
                dist[m1][m2] = None
                continue
            # Use 1 - pearson correlation as a shape-invariant distance
            if np.std(s1) == 0 or np.std(s2) == 0:
                dist[m1][m2] = 0.0
            else:
                r = float(np.corrcoef(s1, s2)[0, 1])
                dist[m1][m2] = 1.0 - r

    return {"signatures": signatures, "distances": dist}


# ── Q3: Cross-model category ordering ───────────────────────────────

def q3_category_ordering(captures: dict) -> dict:
    """
    Rank categories by D_mean per model, then check ordering consistency.
    Uses Kendall-tau-style agreement across models.
    """
    models = sorted(captures.keys())
    rankings = {}
    for m in models:
        d_by_cat = per_prompt_D_means(captures[m])
        cat_means = {k: float(np.mean(v)) for k, v in d_by_cat.items() if v}
        if len(cat_means) < 2:
            continue
        ranked = sorted(cat_means, key=cat_means.get)
        rankings[m] = {
            "sorted_low_to_high": ranked,
            "means": cat_means,
        }

    # Pairwise Kendall-tau-ish: fraction of category pairs that agree
    # on direction across pairs of models
    agreement = {}
    model_list = list(rankings.keys())
    for i, m1 in enumerate(model_list):
        for m2 in model_list[i+1:]:
            cats = sorted(set(rankings[m1]["means"]) & set(rankings[m2]["means"]))
            if len(cats) < 2:
                continue
            pairs = list(combinations(cats, 2))
            agree = 0
            for a, b in pairs:
                sign1 = rankings[m1]["means"][a] - rankings[m1]["means"][b]
                sign2 = rankings[m2]["means"][a] - rankings[m2]["means"][b]
                if sign1 * sign2 > 0:
                    agree += 1
            frac = agree / max(len(pairs), 1)
            agreement[f"{m1} vs {m2}"] = {
                "agreement_fraction": float(frac),
                "n_pairs_compared": len(pairs),
            }

    return {"rankings": rankings, "cross_model_agreement": agreement}


# ── Figures ──────────────────────────────────────────────────────────

def build_figures(captures: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "serif", "font.size": 9})

    # Figure 1: per-category D-mean, one bar per category per model
    canon_cats = ["retrieval", "reasoning", "refusal", "creative",
                  "adversarial", "hallucination"]
    models = sorted(captures.keys())
    if not models:
        return

    fig, ax = plt.subplots(figsize=(11, 4.5))
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(len(canon_cats))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_models, 10)))
    for i, m in enumerate(models):
        d_by_cat = per_prompt_D_means(captures[m])
        means = []
        sems = []
        for cat in canon_cats:
            vals = d_by_cat.get(cat, [])
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                means.append(np.nan)
                sems.append(0)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=sems, capsize=2,
               label=m.split("/")[-1], color=colors[i],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(canon_cats)
    ax.set_ylabel("mean $D$ per category")
    ax.set_title("Fathom Atlas v0.1 — per-category cognitive signatures")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_model_category_means.png",
                dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "cross_model_category_means.pdf",
                bbox_inches="tight")
    plt.close(fig)

    # Figure 2: per-model category distribution (one subplot per model)
    fig, axes = plt.subplots(1, n_models,
                              figsize=(3 * n_models, 3.5),
                              sharey=True, squeeze=False)
    for i, m in enumerate(models):
        ax = axes[0, i]
        d_by_cat = per_prompt_D_means(captures[m])
        data = [d_by_cat.get(cat, []) for cat in canon_cats]
        bp = ax.boxplot(data, tick_labels=canon_cats, showfliers=True,
                        patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(m.split("/")[-1], fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("$D$ per prompt")
    fig.tight_layout()
    fig.savefig(out_dir / "per_model_category_distributions.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    captures = load_captures()
    if not captures:
        print("[!] No atlas captures found.")
        return
    print(f"Loaded {len(captures)} atlas captures:")
    for m in captures:
        print(f"  - {m}")
    print()

    q1 = q1_category_separation(captures)
    q2 = q2_model_distance_matrix(captures)
    q3 = q3_category_ordering(captures)

    print("=" * 70)
    print("Q1: Within-model category separation")
    print("=" * 70)
    for m, r in q1.items():
        print(f"\n{m}")
        print(f"  F = {r['F_statistic']:.2f}, eta^2 = {r['eta_squared']:.3f}")
        gap = r["max_category_gap"]
        print(f"  max category gap: {gap['categories'][0]} <-> "
              f"{gap['categories'][1]} = {gap['gap']:.4f}")
        print(f"  group means:")
        for cat, mean in sorted(r["group_means"].items(),
                                 key=lambda x: x[1]):
            print(f"    {cat:<14s} {mean:+.4f}")

    print()
    print("=" * 70)
    print("Q2: Cross-model distance matrix (1 - Pearson r on signatures)")
    print("=" * 70)
    models = sorted(q2["signatures"].keys())
    if len(models) >= 2:
        for m in models:
            row = " ".join(f"{q2['distances'][m].get(m2, 0) or 0:6.3f}"
                            for m2 in models)
            print(f"  {m.split('/')[-1]:<26s}  {row}")

    print()
    print("=" * 70)
    print("Q3: Cross-model category ordering agreement")
    print("=" * 70)
    for pair, info in q3["cross_model_agreement"].items():
        print(f"  {pair:<55s}  "
              f"{info['agreement_fraction']:.2%}  "
              f"(n={info['n_pairs_compared']})")

    # Save report
    out_path = ATLAS / "analysis" / "atlas_v0.1_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "atlas_version": "0.1",
            "n_models": len(captures),
            "models": list(captures.keys()),
            "q1_within_model": q1,
            "q2_cross_model": q2,
            "q3_ordering": q3,
        }, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    # Figures
    try:
        fig_dir = ATLAS / "analysis" / "figures"
        build_figures(captures, fig_dir)
        print(f"[SAVED] figures in {fig_dir}/")
    except Exception as e:
        print(f"[figures] ERROR: {e}")


if __name__ == "__main__":
    main()
