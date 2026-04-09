# -*- coding: utf-8 -*-
"""
Layer Spectrum Analyzer -- Fathom Frontier 5
============================================
Goes beyond mean depth (K) to analyze the SHAPE of per-layer coherence
distributions. Correct vs hallucinated responses may have fundamentally
different spectral signatures that the scalar K washes out.

Data source: coherence_google_gemma-2-2b_n50.json (50 TruthfulQA pairs,
26-layer coherence profiles for each correct & hallucinated response).

Metrics computed per profile:
  - Peak layer & peak prominence (where does coherence spike?)
  - Spectral centroid (weighted mean — same as K but on coherence, not features)
  - Spectral spread (std of coherence distribution)
  - Skewness & kurtosis (asymmetry and tail weight)
  - Bimodality coefficient (Sarle's BC)
  - Number of peaks (scipy find_peaks)
  - Peak concentration ratio (top-3 layers / total coherence)
  - Early/mid/late coherence mass ratios
  - Gini coefficient (inequality of coherence distribution)
  - Spectral entropy (Shannon entropy of normalized profile)

Statistical tests: paired t-test + Wilcoxon signed-rank on each metric,
with Holm-Bonferroni correction for multiple comparisons.
"""

import json
import sys
import os
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "truthfulqa_results" / "coherence_google_gemma-2-2b_n50.json"
FIGURES = REPO / "figures"
RESULTS = REPO / "truthfulqa_results" / "layer_spectrum_analysis.json"

N_LAYERS = 26
EARLY = range(0, 9)       # layers 0-8
MID = range(9, 17)        # layers 9-16
LATE = range(17, 26)      # layers 17-25


def load_profiles(path):
    """Load per-layer coherence profiles from the n=50 coherence dataset."""
    with open(path) as f:
        data = json.load(f)

    correct_profiles = []
    halluc_profiles = []
    questions = []

    for item in data["results"]:
        c_lc = item["correct"]["layer_coherence"]
        h_lc = item["hallucinated"]["layer_coherence"]

        # Build 26-element arrays (some layers might be missing -> 0)
        c_arr = np.array([c_lc.get(str(i), 0.0) for i in range(N_LAYERS)])
        h_arr = np.array([h_lc.get(str(i), 0.0) for i in range(N_LAYERS)])

        correct_profiles.append(c_arr)
        halluc_profiles.append(h_arr)
        questions.append(item["question"])

    return np.array(correct_profiles), np.array(halluc_profiles), questions


# ── shape metrics ──────────────────────────────────────────────────────

def spectral_centroid(profile):
    """Weighted mean layer index (coherence-weighted)."""
    layers = np.arange(N_LAYERS)
    total = profile.sum()
    if total == 0:
        return N_LAYERS / 2
    return (layers * profile).sum() / total


def spectral_spread(profile):
    """Std of coherence distribution around centroid."""
    layers = np.arange(N_LAYERS)
    total = profile.sum()
    if total == 0:
        return 0.0
    mu = (layers * profile).sum() / total
    return np.sqrt(((layers - mu) ** 2 * profile).sum() / total)


def spectral_skewness(profile):
    """Skewness of the coherence distribution."""
    layers = np.arange(N_LAYERS)
    total = profile.sum()
    if total == 0:
        return 0.0
    p = profile / total
    mu = (layers * p).sum()
    sigma = np.sqrt(((layers - mu) ** 2 * p).sum())
    if sigma == 0:
        return 0.0
    return (((layers - mu) ** 3 * p).sum()) / sigma ** 3


def spectral_kurtosis(profile):
    """Excess kurtosis of the coherence distribution."""
    layers = np.arange(N_LAYERS)
    total = profile.sum()
    if total == 0:
        return 0.0
    p = profile / total
    mu = (layers * p).sum()
    sigma = np.sqrt(((layers - mu) ** 2 * p).sum())
    if sigma == 0:
        return 0.0
    return (((layers - mu) ** 4 * p).sum()) / sigma ** 4 - 3


def bimodality_coefficient(profile):
    """Sarle's bimodality coefficient. BC > 5/9 ≈ 0.555 suggests bimodality."""
    layers = np.arange(N_LAYERS)
    total = profile.sum()
    if total == 0:
        return 0.0
    p = profile / total
    mu = (layers * p).sum()
    sigma = np.sqrt(((layers - mu) ** 2 * p).sum())
    if sigma == 0:
        return 0.0
    skew = (((layers - mu) ** 3 * p).sum()) / sigma ** 3
    kurt = (((layers - mu) ** 4 * p).sum()) / sigma ** 4 - 3
    # n is effectively the number of layers
    n = N_LAYERS
    bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return bc


def count_peaks(profile, prominence_frac=0.1):
    """Count peaks with prominence > prominence_frac * max coherence."""
    min_prom = prominence_frac * profile.max() if profile.max() > 0 else 0
    peaks, props = find_peaks(profile, prominence=min_prom)
    return len(peaks), peaks.tolist()


def peak_concentration(profile, top_k=3):
    """Fraction of total coherence in the top-k layers."""
    total = profile.sum()
    if total == 0:
        return 0.0
    top_k_sum = np.sort(profile)[-top_k:].sum()
    return top_k_sum / total


def band_ratios(profile):
    """Coherence mass in early/mid/late bands."""
    total = profile.sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    early = profile[list(EARLY)].sum() / total
    mid = profile[list(MID)].sum() / total
    late = profile[list(LATE)].sum() / total
    return early, mid, late


def gini_coefficient(profile):
    """Gini coefficient — 0 = perfectly uniform, 1 = all in one layer."""
    sorted_p = np.sort(profile)
    n = len(sorted_p)
    index = np.arange(1, n + 1)
    total = sorted_p.sum()
    if total == 0:
        return 0.0
    return (2 * (index * sorted_p).sum()) / (n * total) - (n + 1) / n


def spectral_entropy(profile):
    """Shannon entropy of normalized coherence profile."""
    total = profile.sum()
    if total == 0:
        return 0.0
    p = profile / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def peak_layer(profile):
    """Layer index with maximum coherence."""
    return int(np.argmax(profile))


def compute_all_metrics(profile):
    """Compute all shape metrics for a single layer profile."""
    n_peaks, peak_locs = count_peaks(profile)
    early, mid, late = band_ratios(profile)
    return {
        "peak_layer": peak_layer(profile),
        "peak_value": float(profile.max()),
        "centroid": spectral_centroid(profile),
        "spread": spectral_spread(profile),
        "skewness": spectral_skewness(profile),
        "kurtosis": spectral_kurtosis(profile),
        "bimodality_coeff": bimodality_coefficient(profile),
        "n_peaks": n_peaks,
        "peak_locations": peak_locs,
        "peak_concentration_top3": peak_concentration(profile, 3),
        "early_mass": early,
        "mid_mass": mid,
        "late_mass": late,
        "late_early_ratio": late / early if early > 0 else float("inf"),
        "gini": gini_coefficient(profile),
        "spectral_entropy": spectral_entropy(profile),
    }


# ── statistical tests ──────────────────────────────────────────────────

SCALAR_METRICS = [
    "peak_layer", "peak_value", "centroid", "spread", "skewness",
    "kurtosis", "bimodality_coeff", "n_peaks", "peak_concentration_top3",
    "early_mass", "mid_mass", "late_mass", "late_early_ratio",
    "gini", "spectral_entropy",
]


def paired_tests(correct_metrics, halluc_metrics):
    """Run paired t-test and Wilcoxon on each scalar metric."""
    results = {}
    p_values = []

    for metric in SCALAR_METRICS:
        c_vals = np.array([m[metric] for m in correct_metrics])
        h_vals = np.array([m[metric] for m in halluc_metrics])
        diff = h_vals - c_vals

        mean_c = float(np.mean(c_vals))
        mean_h = float(np.mean(h_vals))
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        # Cohen's d (paired)
        d = mean_diff / std_diff if std_diff > 0 else 0.0

        # Paired t-test
        t_stat, p_t = stats.ttest_rel(h_vals, c_vals)

        # Wilcoxon signed-rank
        try:
            w_stat, p_w = stats.wilcoxon(diff)
        except ValueError:
            w_stat, p_w = float("nan"), 1.0

        results[metric] = {
            "mean_correct": mean_c,
            "mean_halluc": mean_h,
            "mean_delta": mean_diff,
            "cohen_d": d,
            "t_stat": float(t_stat),
            "p_ttest": float(p_t),
            "w_stat": float(w_stat),
            "p_wilcoxon": float(p_w),
        }
        p_values.append((metric, float(p_t)))

    # Holm-Bonferroni correction
    p_values.sort(key=lambda x: x[1])
    n_tests = len(p_values)
    for rank, (metric, p_raw) in enumerate(p_values):
        adjusted = min(p_raw * (n_tests - rank), 1.0)
        results[metric]["p_holm"] = adjusted
        results[metric]["significant_holm"] = adjusted < 0.05

    return results


# ── visualization ──────────────────────────────────────────────────────

def make_figures(correct_profiles, halluc_profiles, test_results, questions):
    """Generate spectrum analysis figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not available — skipping figures")
        return

    layers = np.arange(N_LAYERS)

    # ── Figure 1: Mean spectrum overlay ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Layer Coherence Spectrum — Correct vs Hallucinated (n=50)",
                 fontsize=14, fontweight="bold")

    # Panel A: Mean profiles with CI
    ax = axes[0, 0]
    c_mean = correct_profiles.mean(axis=0)
    h_mean = halluc_profiles.mean(axis=0)
    c_se = correct_profiles.std(axis=0) / np.sqrt(len(correct_profiles))
    h_se = halluc_profiles.std(axis=0) / np.sqrt(len(halluc_profiles))

    ax.fill_between(layers, c_mean - 1.96 * c_se, c_mean + 1.96 * c_se,
                    alpha=0.2, color="#00aa44")
    ax.fill_between(layers, h_mean - 1.96 * h_se, h_mean + 1.96 * h_se,
                    alpha=0.2, color="#cc3333")
    ax.plot(layers, c_mean, "o-", color="#00aa44", label="Correct", markersize=4)
    ax.plot(layers, h_mean, "s-", color="#cc3333", label="Hallucinated", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Coherence")
    ax.set_title("A. Mean Coherence Spectrum (±95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Difference spectrum (halluc - correct) per layer
    ax = axes[0, 1]
    diff = halluc_profiles - correct_profiles  # (50, 26)
    diff_mean = diff.mean(axis=0)
    diff_se = diff.std(axis=0) / np.sqrt(len(diff))

    colors = ["#cc3333" if d > 0 else "#00aa44" for d in diff_mean]
    ax.bar(layers, diff_mean, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.errorbar(layers, diff_mean, yerr=1.96 * diff_se, fmt="none",
                ecolor="black", capsize=2, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Delta Coherence (halluc - correct)")
    ax.set_title("B. Per-Layer Coherence Difference")
    ax.grid(True, alpha=0.3)

    # Per-layer paired t-test significance markers
    for i in range(N_LAYERS):
        t, p = stats.ttest_rel(halluc_profiles[:, i], correct_profiles[:, i])
        if p < 0.001:
            ax.text(i, diff_mean[i] + 1.96 * diff_se[i] + 0.0003, "***",
                    ha="center", fontsize=8, fontweight="bold")
        elif p < 0.01:
            ax.text(i, diff_mean[i] + 1.96 * diff_se[i] + 0.0003, "**",
                    ha="center", fontsize=8)
        elif p < 0.05:
            ax.text(i, diff_mean[i] + 1.96 * diff_se[i] + 0.0003, "*",
                    ha="center", fontsize=8)

    # Panel C: Heatmap of all 50 correct profiles
    ax = axes[1, 0]
    # Normalize each profile to sum to 1 for shape comparison
    c_norm = correct_profiles / correct_profiles.sum(axis=1, keepdims=True)
    # Sort by centroid for visual clarity
    centroids_c = np.array([spectral_centroid(p) for p in correct_profiles])
    sort_idx = np.argsort(centroids_c)
    im = ax.imshow(c_norm[sort_idx], aspect="auto", cmap="YlGn",
                   extent=[0, N_LAYERS, len(correct_profiles), 0])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Item (sorted by centroid)")
    ax.set_title("C. Correct — Normalized Coherence Spectra")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel D: Heatmap of all 50 hallucinated profiles
    ax = axes[1, 1]
    h_norm = halluc_profiles / halluc_profiles.sum(axis=1, keepdims=True)
    centroids_h = np.array([spectral_centroid(p) for p in halluc_profiles])
    sort_idx_h = np.argsort(centroids_h)
    im = ax.imshow(h_norm[sort_idx_h], aspect="auto", cmap="YlOrRd",
                   extent=[0, N_LAYERS, len(halluc_profiles), 0])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Item (sorted by centroid)")
    ax.set_title("D. Hallucinated — Normalized Coherence Spectra")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"layer_spectrum_overview.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  -> layer_spectrum_overview.pdf/png")

    # ── Figure 2: Shape metric comparisons ─────────────────────────────
    # Pick the most interesting metrics to visualize
    show_metrics = ["spread", "skewness", "kurtosis", "gini",
                    "peak_concentration_top3", "late_early_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Spectral Shape Metrics — Correct vs Hallucinated",
                 fontsize=14, fontweight="bold")

    for ax, metric in zip(axes.flat, show_metrics):
        c_vals = np.array([compute_all_metrics(p)[metric] for p in correct_profiles])
        h_vals = np.array([compute_all_metrics(p)[metric] for p in halluc_profiles])

        info = test_results[metric]
        p = info["p_ttest"]
        d = info["cohen_d"]

        bp = ax.boxplot([c_vals, h_vals], labels=["Correct", "Halluc"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#00aa44")
        bp["boxes"][0].set_alpha(0.4)
        bp["boxes"][1].set_facecolor("#cc3333")
        bp["boxes"][1].set_alpha(0.4)
        # Overlay individual points
        jitter = 0.05 * np.random.randn(len(c_vals))
        ax.scatter(np.ones(len(c_vals)) + jitter, c_vals, alpha=0.3,
                   color="#00aa44", s=15, zorder=3)
        ax.scatter(2 * np.ones(len(h_vals)) + jitter, h_vals, alpha=0.3,
                   color="#cc3333", s=15, zorder=3)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{metric}\np={p:.4f}, d={d:.3f} {sig}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"layer_spectrum_metrics.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  -> layer_spectrum_metrics.pdf/png")

    # ── Figure 3: Individual extreme cases ─────────────────────────────
    # Find the pair with largest and smallest spectral difference
    spread_diffs = []
    for i in range(len(correct_profiles)):
        sc = spectral_spread(correct_profiles[i])
        sh = spectral_spread(halluc_profiles[i])
        spread_diffs.append(sh - sc)
    spread_diffs = np.array(spread_diffs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Individual Spectrum Examples", fontsize=13, fontweight="bold")

    for ax, idx, label in [
        (axes[0], np.argmax(spread_diffs), "Largest spread Δ"),
        (axes[1], np.argmin(spread_diffs), "Smallest spread Δ"),
        (axes[2], np.argmin(np.abs(spread_diffs)), "Near-zero spread Δ"),
    ]:
        c_p = correct_profiles[idx] / correct_profiles[idx].sum()
        h_p = halluc_profiles[idx] / halluc_profiles[idx].sum()
        ax.plot(layers, c_p, "o-", color="#00aa44", label="Correct", markersize=4)
        ax.plot(layers, h_p, "s-", color="#cc3333", label="Halluc", markersize=4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized Coherence")
        q = questions[idx][:50] + "..." if len(questions[idx]) > 50 else questions[idx]
        ax.set_title(f"{label}\n\"{q}\"", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"layer_spectrum_examples.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  -> layer_spectrum_examples.pdf/png")


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LAYER SPECTRUM ANALYZER -- Fathom Frontier 5")
    print("=" * 70)

    if not DATA.exists():
        print(f"ERROR: Data not found at {DATA}")
        sys.exit(1)

    print(f"\nLoading {DATA.name}...")
    correct_profiles, halluc_profiles, questions = load_profiles(DATA)
    n = len(correct_profiles)
    print(f"  {n} pairs × {N_LAYERS} layers loaded")

    # Compute metrics for each profile
    print("\nComputing shape metrics...")
    correct_metrics = [compute_all_metrics(p) for p in correct_profiles]
    halluc_metrics = [compute_all_metrics(p) for p in halluc_profiles]

    # Statistical tests
    print("\nRunning paired statistical tests...")
    test_results = paired_tests(correct_metrics, halluc_metrics)

    # ── Print results table ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'METRIC':<25} {'CORRECT':>10} {'HALLUC':>10} {'DELTA':>10} {'d':>8} {'p(t)':>10} {'p(Holm)':>10} {'SIG':>5}")
    print("-" * 100)

    sig_count = 0
    for metric in SCALAR_METRICS:
        r = test_results[metric]
        sig = "***" if r["p_holm"] < 0.001 else "**" if r["p_holm"] < 0.01 else \
              "*" if r["p_holm"] < 0.05 else ""
        if r["p_holm"] < 0.05:
            sig_count += 1
        print(f"{metric:<25} {r['mean_correct']:>10.4f} {r['mean_halluc']:>10.4f} "
              f"{r['mean_delta']:>+10.4f} {r['cohen_d']:>+8.3f} "
              f"{r['p_ttest']:>10.6f} {r['p_holm']:>10.6f} {sig:>5}")

    print("-" * 100)
    print(f"{sig_count}/{len(SCALAR_METRICS)} metrics significant after Holm-Bonferroni correction")

    # ── Per-layer analysis ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-LAYER COHERENCE DIFFERENCES (paired t-test)")
    print("-" * 70)
    print(f"{'LAYER':>6} {'C_correct':>12} {'C_halluc':>12} {'DELTA':>12} {'t':>8} {'p':>10}")
    print("-" * 70)

    layer_results = {}
    for i in range(N_LAYERS):
        c_vals = correct_profiles[:, i]
        h_vals = halluc_profiles[:, i]
        t, p = stats.ttest_rel(h_vals, c_vals)
        diff = h_vals.mean() - c_vals.mean()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{i:>6} {c_vals.mean():>12.6f} {h_vals.mean():>12.6f} "
              f"{diff:>+12.6f} {t:>8.3f} {p:>10.6f} {sig}")
        layer_results[str(i)] = {
            "mean_correct": float(c_vals.mean()),
            "mean_halluc": float(h_vals.mean()),
            "delta": float(diff),
            "t_stat": float(t),
            "p_value": float(p),
        }

    # ── Peak layer distribution ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PEAK LAYER DISTRIBUTION")
    print("-" * 70)
    c_peaks = [np.argmax(p) for p in correct_profiles]
    h_peaks = [np.argmax(p) for p in halluc_profiles]
    print(f"Correct peak layers:  mean={np.mean(c_peaks):.1f}, "
          f"mode={stats.mode(c_peaks, keepdims=False).mode}, "
          f"std={np.std(c_peaks):.2f}")
    print(f"Halluc  peak layers:  mean={np.mean(h_peaks):.1f}, "
          f"mode={stats.mode(h_peaks, keepdims=False).mode}, "
          f"std={np.std(h_peaks):.2f}")

    from collections import Counter
    c_counts = Counter(c_peaks)
    h_counts = Counter(h_peaks)
    print("\n  Layer | Correct peaks | Halluc peaks")
    print("  ------|---------------|-------------")
    for layer in range(N_LAYERS):
        cc = c_counts.get(layer, 0)
        hc = h_counts.get(layer, 0)
        if cc > 0 or hc > 0:
            bar_c = "#" * cc
            bar_h = "#" * hc
            print(f"  {layer:>5} | {cc:>2} {bar_c:<15} | {hc:>2} {bar_h}")

    # ── Bimodality check ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BIMODALITY ANALYSIS")
    print("-" * 70)
    c_bcs = [compute_all_metrics(p)["bimodality_coeff"] for p in correct_profiles]
    h_bcs = [compute_all_metrics(p)["bimodality_coeff"] for p in halluc_profiles]
    threshold = 5 / 9  # ≈ 0.555
    c_bimodal = sum(1 for bc in c_bcs if bc > threshold)
    h_bimodal = sum(1 for bc in h_bcs if bc > threshold)
    print(f"Bimodality threshold (Sarle's BC > {threshold:.3f}):")
    print(f"  Correct:      {c_bimodal}/{n} ({100*c_bimodal/n:.0f}%) bimodal")
    print(f"  Hallucinated: {h_bimodal}/{n} ({100*h_bimodal/n:.0f}%) bimodal")

    # Multi-peak count
    c_npeaks = [compute_all_metrics(p)["n_peaks"] for p in correct_profiles]
    h_npeaks = [compute_all_metrics(p)["n_peaks"] for p in halluc_profiles]
    print(f"\nPeak counts (prominence > 10% of max):")
    print(f"  Correct:      mean={np.mean(c_npeaks):.2f}, "
          f"1-peak={c_npeaks.count(1)}, 2-peak={c_npeaks.count(2)}, 3+={sum(1 for x in c_npeaks if x >= 3)}")
    print(f"  Hallucinated: mean={np.mean(h_npeaks):.2f}, "
          f"1-peak={h_npeaks.count(1)}, 2-peak={h_npeaks.count(2)}, 3+={sum(1 for x in h_npeaks if x >= 3)}")

    # ── Discriminative power: which shape metric best separates? ───────
    print("\n" + "=" * 70)
    print("DISCRIMINATIVE POWER (AUC-ROC per metric)")
    print("-" * 70)

    auc_results = {}
    for metric in SCALAR_METRICS:
        c_vals = np.array([m[metric] for m in correct_metrics])
        h_vals = np.array([m[metric] for m in halluc_metrics])
        # Simple AUC: fraction of (halluc, correct) pairs where halluc > correct
        n_greater = sum(1 for h, c in zip(h_vals, c_vals) if h > c)
        n_equal = sum(1 for h, c in zip(h_vals, c_vals) if h == c)
        n_total = len(h_vals)
        auc = (n_greater + 0.5 * n_equal) / n_total
        # If AUC < 0.5, flip direction
        direction = "halluc > correct" if auc >= 0.5 else "correct > halluc"
        auc_display = max(auc, 1 - auc)
        auc_results[metric] = {"auc": float(auc_display), "direction": direction}

    # Sort by AUC descending
    sorted_aucs = sorted(auc_results.items(), key=lambda x: x[1]["auc"], reverse=True)
    for metric, info in sorted_aucs:
        bar = "#" * int(info["auc"] * 40)
        print(f"  {metric:<25} AUC={info['auc']:.3f} {bar} ({info['direction']})")

    # ── Save results ───────────────────────────────────────────────────
    output = {
        "experiment": "layer_spectrum_analysis",
        "data_source": str(DATA.name),
        "n_pairs": n,
        "n_layers": N_LAYERS,
        "shape_metrics": test_results,
        "per_layer_tests": layer_results,
        "peak_distribution": {
            "correct_mean": float(np.mean(c_peaks)),
            "halluc_mean": float(np.mean(h_peaks)),
        },
        "bimodality": {
            "correct_bimodal_count": c_bimodal,
            "halluc_bimodal_count": h_bimodal,
            "threshold": threshold,
        },
        "auc_roc": auc_results,
    }

    with open(RESULTS, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved -> {RESULTS.name}")

    # ── Figures ────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    make_figures(correct_profiles, halluc_profiles, test_results, questions)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
