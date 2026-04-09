# -*- coding: utf-8 -*-
"""
Cognitive Divergence Detector + Autopsy + Predictive Lock-In
=============================================================

Three unsolved problems, one toolkit:

1. COGNITIVE DIVERGENCE: Detect the honesty gap between internal
   computation (C_delta/Gini) and output confidence (entropy).
   High C_delta + low entropy = "confidently wrong" = DANGER.

2. COGNITIVE AUTOPSY: Per-layer fault localization. When a model
   hallucinates, identify WHICH layers concentrated and by how much.

3. PREDICTIVE LOCK-IN: Use the Gini trajectory slope in early tokens
   to predict hallucination BEFORE the wrong content appears.

Data sources:
  - S_baselines_n200.json: per-token entropy + logprob + c_delta (gemma-2-2b-it, n=200)
  - coherence_n200_final.json: per-layer profiles (gemma-2-2b, n=200)
  - S_truthfulqa_n200_labels.json: hallucination labels
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "figures"
RESULTS_DIR = REPO / "truthfulqa_results"


# =================================================================
# DATA LOADING
# =================================================================

def load_baselines_with_labels():
    """Load S_baselines (entropy+c_delta per token) with hallucination labels."""
    baselines_path = RESULTS_DIR / "S_baselines_n200.json"
    labels_path = RESULTS_DIR / "S_truthfulqa_n200_labels.json"

    with open(baselines_path, encoding="utf-8") as f:
        baselines = json.load(f)
    with open(labels_path, encoding="utf-8") as f:
        labels_data = json.load(f)

    # Build label lookup by question
    label_map = {}
    for item in labels_data.get("items", labels_data.get("results", [])):
        q = item.get("question", "")
        lab = item.get("label", item.get("is_correct", None))
        if q and lab is not None:
            label_map[q] = lab

    items = []
    for item in baselines["results"]:
        q = item["question"]
        gen = item["generation"]

        # Get label (try multiple field names)
        label = label_map.get(q)
        if label is None:
            # Try semantic matching from correct_answers
            correct = item.get("correct_answers", [])
            incorrect = item.get("incorrect_answers", [])
            text = gen["text"].strip().lower()
            # Simple heuristic: check if any correct answer substring matches
            is_correct = any(ca.lower()[:30] in text[:100] for ca in correct) if correct else None
            if is_correct is not None:
                label = "correct" if is_correct else "hallucinated"

        if label is None:
            continue

        c_delta_traj = gen.get("c_delta_trajectory", [])
        entropy_traj = gen.get("entropy_trajectory", [])
        logprob_traj = gen.get("logprob_trajectory", [])

        if not c_delta_traj or not entropy_traj:
            continue

        items.append({
            "question": q,
            "text": gen["text"],
            "label": label,
            "c_delta_trajectory": c_delta_traj,
            "entropy_trajectory": entropy_traj,
            "logprob_trajectory": logprob_traj,
            "category": item.get("category", ""),
        })

    return items


def load_coherence_profiles():
    """Load n=200 per-layer coherence profiles."""
    path = RESULTS_DIR / "coherence_n200_final.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for item in data["results"]:
        items.append({
            "question": item["question"],
            "correct_profile": np.array([item["correct"]["layer_coherence"].get(str(i), 0)
                                         for i in range(26)]),
            "halluc_profile": np.array([item["hallucinated"]["layer_coherence"].get(str(i), 0)
                                        for i in range(26)]),
            "K_correct": item["correct"]["K"],
            "K_halluc": item["hallucinated"]["K"],
            "C_delta_correct": item["correct"]["C_delta"],
            "C_delta_halluc": item["hallucinated"]["C_delta"],
        })
    return items


# =================================================================
# METRIC FUNCTIONS
# =================================================================

def gini(vals):
    s = np.sort(vals)
    n = len(s)
    t = s.sum()
    if t == 0 or n == 0: return 0.0
    idx = np.arange(1, n + 1)
    return (2 * (idx * s).sum()) / (n * t) - (n + 1) / n


def cognitive_divergence(c_delta_traj, entropy_traj, window=None):
    """
    Compute the divergence between internal commitment (C_delta)
    and output uncertainty (entropy).

    High C_delta + low entropy = model is internally committed
    but outputting confidently = potential confident hallucination.

    Low C_delta + high entropy = genuinely uncertain = honest uncertainty.

    Returns: divergence score (higher = more divergent = more suspicious)
    """
    if window:
        cd = np.array(c_delta_traj[:window])
        ent = np.array(entropy_traj[:window])
    else:
        cd = np.array(c_delta_traj)
        ent = np.array(entropy_traj)

    # Filter Nones
    valid = [(c, e) for c, e in zip(cd, ent) if c is not None and e is not None]
    if len(valid) < 3:
        return None

    cd_arr = np.array([v[0] for v in valid])
    ent_arr = np.array([v[1] for v in valid])

    # Normalize to [0,1] range
    cd_norm = (cd_arr - cd_arr.min()) / max(cd_arr.max() - cd_arr.min(), 1e-10)
    ent_norm = (ent_arr - ent_arr.min()) / max(ent_arr.max() - ent_arr.min(), 1e-10)

    # Divergence = high C_delta coinciding with low entropy
    # When C_delta is high AND entropy is low, the model is committed but confident
    divergence_per_token = cd_norm * (1 - ent_norm)

    return {
        "mean_divergence": float(np.mean(divergence_per_token)),
        "max_divergence": float(np.max(divergence_per_token)),
        "divergence_trajectory": divergence_per_token.tolist(),
        # Also: correlation between C_delta and entropy
        # Negative correlation = divergent (committed but confident)
        "cd_entropy_corr": float(np.corrcoef(cd_arr, ent_arr)[0, 1]) if len(cd_arr) > 2 else 0,
        # Fraction of tokens in "danger zone" (top 25% C_delta AND bottom 25% entropy)
        "danger_zone_frac": float(np.mean(
            (cd_arr > np.percentile(cd_arr, 75)) &
            (ent_arr < np.percentile(ent_arr, 25))
        )),
        # Mean C_delta when entropy is low (bottom quartile)
        "cd_when_confident": float(np.mean(cd_arr[ent_arr < np.percentile(ent_arr, 25)]))
        if np.sum(ent_arr < np.percentile(ent_arr, 25)) > 0 else 0,
    }


def cognitive_autopsy(correct_profile, halluc_profile, n_layers=26):
    """
    Per-layer fault localization: identify which layers concentrated
    during hallucination vs correct response.

    Returns per-layer delta, fault score, and primary fault layers.
    """
    delta = halluc_profile - correct_profile
    # Normalize by correct profile to get relative change
    rel_delta = delta / np.clip(correct_profile, 1e-10, None)

    # Fault score: how much each layer deviated from healthy baseline
    fault_scores = np.abs(delta) / max(np.abs(delta).max(), 1e-10)

    # Find primary fault layers (top contributors to the hallucination)
    total_excess = np.sum(np.maximum(delta, 0))  # total positive excess
    if total_excess > 0:
        layer_contribution = np.maximum(delta, 0) / total_excess
    else:
        layer_contribution = np.zeros(n_layers)

    # Identify fault zone
    threshold = 0.1  # layers contributing >10% of excess
    primary_faults = [int(i) for i in range(n_layers) if layer_contribution[i] > threshold]

    # Gini of both profiles
    g_correct = gini(correct_profile)
    g_halluc = gini(halluc_profile)

    return {
        "delta_per_layer": delta.tolist(),
        "relative_delta": rel_delta.tolist(),
        "fault_scores": fault_scores.tolist(),
        "layer_contribution": layer_contribution.tolist(),
        "primary_fault_layers": primary_faults,
        "gini_correct": g_correct,
        "gini_halluc": g_halluc,
        "gini_delta": g_halluc - g_correct,
        "concentration_shift": float(np.sum(layer_contribution[9:17])),  # mid-layer concentration
        "early_depletion": float(np.sum(delta[:9])),  # early layer change
        "late_elevation": float(np.sum(delta[17:])),   # late layer change
    }


def predictive_lockin(c_delta_traj, entropy_traj, max_prefix=7):
    """
    Predict hallucination from early-token trajectory slope.

    If C_delta is RISING in tokens 1-5, the model is locking in.
    If entropy is FALLING simultaneously, it's committing confidently.
    The slope of early C_delta predicts eventual hallucination.
    """
    results = {}

    for prefix_len in range(3, max_prefix + 1):
        cd = [c for c in c_delta_traj[:prefix_len] if c is not None]
        ent = [e for e in entropy_traj[:prefix_len] if e is not None]

        if len(cd) < 3:
            continue

        x = np.arange(len(cd))
        # C_delta slope (linear regression)
        cd_slope, cd_intercept = np.polyfit(x, cd, 1) if len(cd) > 1 else (0, 0)
        # Entropy slope
        ent_slope, ent_intercept = np.polyfit(x, ent, 1) if len(ent) > 1 else (0, 0)
        # Divergence: C_delta rising AND entropy falling
        lockin_score = cd_slope - ent_slope  # positive = locking in

        results[f"prefix_{prefix_len}"] = {
            "cd_slope": float(cd_slope),
            "ent_slope": float(ent_slope),
            "lockin_score": float(lockin_score),
            "cd_mean": float(np.mean(cd)),
            "ent_mean": float(np.mean(ent)),
        }

    return results


# =================================================================
# MAIN ANALYSIS
# =================================================================

def main():
    print("=" * 80)
    print("COGNITIVE DIVERGENCE + AUTOPSY + PREDICTIVE LOCK-IN")
    print("=" * 80)

    # ─── PART 1: COGNITIVE DIVERGENCE (S_baselines data) ───────────
    print("\n[1/3] COGNITIVE DIVERGENCE DETECTOR")
    print("-" * 60)

    items = load_baselines_with_labels()
    print(f"  Loaded {len(items)} items with labels")

    # Separate correct vs hallucinated
    correct_items = [i for i in items if i["label"] in ("correct", True)]
    halluc_items = [i for i in items if i["label"] in ("hallucinated", False)]
    print(f"  Correct: {len(correct_items)}, Hallucinated: {len(halluc_items)}")

    if len(correct_items) > 0 and len(halluc_items) > 0:
        correct_divs = [cognitive_divergence(i["c_delta_trajectory"], i["entropy_trajectory"])
                        for i in correct_items]
        halluc_divs = [cognitive_divergence(i["c_delta_trajectory"], i["entropy_trajectory"])
                       for i in halluc_items]

        correct_divs = [d for d in correct_divs if d is not None]
        halluc_divs = [d for d in halluc_divs if d is not None]

        if correct_divs and halluc_divs:
            metrics = ["mean_divergence", "max_divergence", "danger_zone_frac",
                       "cd_entropy_corr", "cd_when_confident"]

            print(f"\n  {'METRIC':<25} {'CORRECT':>10} {'HALLUC':>10} {'d':>8} {'p':>12} {'AUC':>6}")
            print("  " + "-" * 75)

            div_results = {}
            for metric in metrics:
                c_vals = np.array([d[metric] for d in correct_divs])
                h_vals = np.array([d[metric] for d in halluc_divs])

                diff = h_vals.mean() - c_vals.mean()
                pooled_std = np.sqrt((c_vals.var() + h_vals.var()) / 2)
                d = diff / pooled_std if pooled_std > 0 else 0
                t, p = stats.ttest_ind(h_vals, c_vals)

                # AUC
                all_vals = np.concatenate([c_vals, h_vals])
                all_labels = np.concatenate([np.zeros(len(c_vals)), np.ones(len(h_vals))])
                sorted_idx = np.argsort(all_vals)[::-1]
                sorted_labels = all_labels[sorted_idx]
                n_pos = sorted_labels.sum()
                n_neg = len(sorted_labels) - n_pos
                tp = np.cumsum(sorted_labels)
                fp = np.arange(1, len(sorted_labels) + 1) - tp
                tpr = tp / max(n_pos, 1)
                fpr = fp / max(n_neg, 1)
                auc = float(np.trapz(tpr, fpr))
                auc = max(auc, 1 - auc)

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {metric:<25} {c_vals.mean():>10.4f} {h_vals.mean():>10.4f} "
                      f"{d:>+8.3f} {p:>12.2e} {auc:>6.3f} {sig}")

                div_results[metric] = {
                    "correct_mean": float(c_vals.mean()),
                    "halluc_mean": float(h_vals.mean()),
                    "cohen_d": float(d),
                    "p_value": float(p),
                    "auc": float(auc),
                }
    else:
        print("  WARNING: Could not separate correct/hallucinated. Checking label format...")
        div_results = {}

    # ─── PART 2: COGNITIVE AUTOPSY (coherence_n200 data) ───────────
    print(f"\n\n[2/3] COGNITIVE AUTOPSY")
    print("-" * 60)

    profiles = load_coherence_profiles()
    print(f"  Loaded {len(profiles)} paired profiles")

    autopsies = [cognitive_autopsy(p["correct_profile"], p["halluc_profile"]) for p in profiles]

    # Aggregate fault layer statistics
    fault_counts = np.zeros(26)
    for a in autopsies:
        for fl in a["primary_fault_layers"]:
            fault_counts[fl] += 1

    print(f"\n  PRIMARY FAULT LAYERS (>10% contribution to excess coherence):")
    print(f"  {'LAYER':>6} {'FREQUENCY':>10} {'PCT':>8}")
    print("  " + "-" * 30)
    for i in range(26):
        if fault_counts[i] > 0:
            print(f"  {i:>6} {fault_counts[i]:>10.0f} {100*fault_counts[i]/len(autopsies):>7.1f}%")

    # Aggregate gini delta
    gini_deltas = [a["gini_delta"] for a in autopsies]
    conc_shifts = [a["concentration_shift"] for a in autopsies]
    early_deps = [a["early_depletion"] for a in autopsies]

    print(f"\n  AUTOPSY SUMMARY (n={len(autopsies)}):")
    print(f"    Gini delta (halluc-correct):  mean={np.mean(gini_deltas):+.4f}, "
          f"median={np.median(gini_deltas):+.4f}")
    print(f"    Mid-layer concentration:      mean={np.mean(conc_shifts):.4f}")
    print(f"    Early-layer depletion:         mean={np.mean(early_deps):+.6f}")

    # Mean contribution map
    mean_contrib = np.mean([a["layer_contribution"] for a in autopsies], axis=0)
    print(f"\n  MEAN FAULT CONTRIBUTION BY LAYER:")
    for i in range(26):
        bar = "#" * int(mean_contrib[i] * 100)
        print(f"    Layer {i:>2}: {mean_contrib[i]:>6.3f} {bar}")

    # ─── PART 3: PREDICTIVE LOCK-IN (S_baselines data) ────────────
    print(f"\n\n[3/3] PREDICTIVE LOCK-IN")
    print("-" * 60)

    if len(correct_items) > 0 and len(halluc_items) > 0:
        lockin_results = {}
        for prefix in range(3, 8):
            c_scores = []
            h_scores = []
            for item in correct_items:
                pl = predictive_lockin(item["c_delta_trajectory"], item["entropy_trajectory"], prefix)
                key = f"prefix_{prefix}"
                if key in pl:
                    c_scores.append(pl[key]["lockin_score"])
            for item in halluc_items:
                pl = predictive_lockin(item["c_delta_trajectory"], item["entropy_trajectory"], prefix)
                key = f"prefix_{prefix}"
                if key in pl:
                    h_scores.append(pl[key]["lockin_score"])

            if c_scores and h_scores:
                c_arr = np.array(c_scores)
                h_arr = np.array(h_scores)
                diff = h_arr.mean() - c_arr.mean()
                pooled = np.sqrt((c_arr.var() + h_arr.var()) / 2)
                d = diff / pooled if pooled > 0 else 0
                t, p = stats.ttest_ind(h_arr, c_arr)

                # Quick AUC
                all_v = np.concatenate([c_arr, h_arr])
                all_l = np.concatenate([np.zeros(len(c_arr)), np.ones(len(h_arr))])
                n_greater = sum(1 for h in h_arr for c in c_arr if h > c)
                auc = n_greater / (len(h_arr) * len(c_arr))
                auc = max(auc, 1 - auc)

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  Prefix {prefix} tokens: d={d:+.3f}, p={p:.4f}, AUC={auc:.3f} {sig}")
                print(f"    Correct lock-in: {c_arr.mean():+.6f}")
                print(f"    Halluc lock-in:  {h_arr.mean():+.6f}")

                lockin_results[f"prefix_{prefix}"] = {
                    "d": float(d), "p": float(p), "auc": float(auc),
                    "correct_mean": float(c_arr.mean()),
                    "halluc_mean": float(h_arr.mean()),
                }

    # ─── FIGURES ───────────────────────────────────────────────────
    print(f"\n\nGenerating figures...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 1: Cognitive Autopsy — fault layer map
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Cognitive Autopsy: Where Hallucinations Go Wrong (n=200)",
                    fontsize=13, fontweight="bold")

        # Panel A: Mean fault contribution
        ax = axes[0]
        colors = ["#cc3333" if c > 0.05 else "#666666" for c in mean_contrib]
        ax.barh(range(26), mean_contrib, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(26))
        ax.set_yticklabels([str(i) for i in range(26)], fontsize=7)
        ax.set_xlabel("Mean Fault Contribution")
        ax.set_title("A. Fault Contribution by Layer")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        # Panel B: Fault frequency histogram
        ax = axes[1]
        ax.barh(range(26), fault_counts, color="#ff6644", edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(26))
        ax.set_yticklabels([str(i) for i in range(26)], fontsize=7)
        ax.set_xlabel("# Items Where Layer is Primary Fault")
        ax.set_title("B. Fault Frequency (>10% contribution)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        # Panel C: Gini delta distribution
        ax = axes[2]
        ax.hist(gini_deltas, bins=30, color="#ff6644", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(np.mean(gini_deltas), color="#cc0000", linewidth=2, label=f"mean={np.mean(gini_deltas):+.4f}")
        ax.set_xlabel("Gini Delta (halluc - correct)")
        ax.set_title("C. Coherence Concentration Shift")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES / f"cognitive_autopsy.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> cognitive_autopsy.pdf/png")

        # Figure 2: Mean layer profile overlay with fault zones highlighted
        fig, ax = plt.subplots(figsize=(12, 5))
        layers = np.arange(26)
        c_mean = np.mean([p["correct_profile"] for p in profiles], axis=0)
        h_mean = np.mean([p["halluc_profile"] for p in profiles], axis=0)

        ax.fill_between(layers, c_mean, h_mean,
                        where=h_mean > c_mean, alpha=0.3, color="#cc3333",
                        label="Hallucination excess")
        ax.fill_between(layers, c_mean, h_mean,
                        where=h_mean <= c_mean, alpha=0.3, color="#00aa44",
                        label="Correct excess")
        ax.plot(layers, c_mean, "o-", color="#00aa44", markersize=4, label="Correct")
        ax.plot(layers, h_mean, "s-", color="#cc3333", markersize=4, label="Hallucinated")

        # Highlight primary fault layers
        for i in range(26):
            if fault_counts[i] > len(autopsies) * 0.3:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="red")

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Mean Coherence", fontsize=12)
        ax.set_title("Cognitive Autopsy: Hallucination Excess by Layer (n=200, Gemma-2-2B)",
                    fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES / f"cognitive_autopsy_overlay.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> cognitive_autopsy_overlay.pdf/png")

    except ImportError:
        print("  matplotlib not available")

    # ─── SAVE ──────────────────────────────────────────────────────
    output = {
        "divergence_results": div_results,
        "autopsy_summary": {
            "mean_gini_delta": float(np.mean(gini_deltas)),
            "mean_concentration_shift": float(np.mean(conc_shifts)),
            "mean_fault_contribution": mean_contrib.tolist(),
            "fault_counts": fault_counts.tolist(),
            "n_profiles": len(profiles),
        },
        "predictive_lockin": lockin_results if 'lockin_results' in dir() else {},
    }

    out_path = RESULTS_DIR / "cognitive_divergence_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results -> {out_path.name}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
