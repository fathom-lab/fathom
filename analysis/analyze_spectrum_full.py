# -*- coding: utf-8 -*-
"""
Full Spectrum Analysis -- Fathom Frontier 5 Extended
=====================================================
1. Validate spectrum on n=200 (4x original)
2. Combined Gini + S_early classifier with 5-fold CV
3. ROC curves and precision-recall for all spectrum metrics
4. Multi-metric logistic regression: which combination is best?
5. Generate paper-ready figures

No GPU needed -- pure numpy/scipy on existing data.
"""

import json
import sys
import os
import warnings
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -- paths --
REPO = Path(__file__).resolve().parent.parent
DATA_50 = REPO / "truthfulqa_results" / "coherence_google_gemma-2-2b_n50.json"
DATA_200 = REPO / "truthfulqa_results" / "coherence_n200_final.json"
S_DATA = REPO / "truthfulqa_results" / "S_truthfulqa_n200_biphasic.json"
BASELINES = REPO / "truthfulqa_results" / "S_baselines_n200.json"
FIGURES = REPO / "figures"
RESULTS = REPO / "truthfulqa_results" / "spectrum_full_analysis.json"

N_LAYERS = 26
EARLY = list(range(0, 9))
MID = list(range(9, 17))
LATE = list(range(17, 26))


def load_profiles(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    correct_profiles = []
    halluc_profiles = []
    questions = []
    extra = {"K_correct": [], "K_halluc": [], "C_delta_correct": [], "C_delta_halluc": []}
    for item in data["results"]:
        c_lc = item["correct"]["layer_coherence"]
        h_lc = item["hallucinated"]["layer_coherence"]
        c_arr = np.array([c_lc.get(str(i), 0.0) for i in range(N_LAYERS)])
        h_arr = np.array([h_lc.get(str(i), 0.0) for i in range(N_LAYERS)])
        correct_profiles.append(c_arr)
        halluc_profiles.append(h_arr)
        questions.append(item["question"])
        extra["K_correct"].append(item["correct"]["K"])
        extra["K_halluc"].append(item["hallucinated"]["K"])
        extra["C_delta_correct"].append(item["correct"]["C_delta"])
        extra["C_delta_halluc"].append(item["hallucinated"]["C_delta"])
    return np.array(correct_profiles), np.array(halluc_profiles), questions, extra


# -- shape metrics (same as analyze_layer_spectrum.py) --

def spectral_centroid(p):
    layers = np.arange(N_LAYERS)
    t = p.sum()
    return (layers * p).sum() / t if t > 0 else N_LAYERS / 2

def spectral_spread(p):
    layers = np.arange(N_LAYERS)
    t = p.sum()
    if t == 0: return 0.0
    mu = (layers * p).sum() / t
    return np.sqrt(((layers - mu)**2 * p).sum() / t)

def spectral_skewness(p):
    layers = np.arange(N_LAYERS)
    t = p.sum()
    if t == 0: return 0.0
    pn = p / t
    mu = (layers * pn).sum()
    sigma = np.sqrt(((layers - mu)**2 * pn).sum())
    if sigma == 0: return 0.0
    return ((layers - mu)**3 * pn).sum() / sigma**3

def spectral_kurtosis(p):
    layers = np.arange(N_LAYERS)
    t = p.sum()
    if t == 0: return 0.0
    pn = p / t
    mu = (layers * pn).sum()
    sigma = np.sqrt(((layers - mu)**2 * pn).sum())
    if sigma == 0: return 0.0
    return ((layers - mu)**4 * pn).sum() / sigma**4 - 3

def gini_coefficient(p):
    sp = np.sort(p)
    n = len(sp)
    t = sp.sum()
    if t == 0: return 0.0
    idx = np.arange(1, n + 1)
    return (2 * (idx * sp).sum()) / (n * t) - (n + 1) / n

def spectral_entropy(p):
    t = p.sum()
    if t == 0: return 0.0
    pn = p[p > 0] / t
    return -np.sum(pn * np.log2(pn))

def peak_concentration(p, k=3):
    t = p.sum()
    if t == 0: return 0.0
    return np.sort(p)[-k:].sum() / t

def band_mass(p, band):
    t = p.sum()
    if t == 0: return 0.0
    return p[band].sum() / t

def count_peaks(p, prom_frac=0.1):
    min_prom = prom_frac * p.max() if p.max() > 0 else 0
    peaks, _ = find_peaks(p, prominence=min_prom)
    return len(peaks)

def layer_11_dominance(p):
    """Fraction of total coherence in layers 11-13 (the spike zone)."""
    t = p.sum()
    if t == 0: return 0.0
    return p[11:14].sum() / t

def late_layer_elevation(p):
    """Mean coherence in layers 17-25 relative to global mean."""
    global_mean = p.mean()
    if global_mean == 0: return 0.0
    return p[17:26].mean() / global_mean

def early_depletion(p):
    """Ratio of early-layer mass to mid-layer mass."""
    mid = p[MID].sum()
    if mid == 0: return 0.0
    return p[EARLY].sum() / mid


def compute_all(p):
    return {
        "centroid": spectral_centroid(p),
        "spread": spectral_spread(p),
        "skewness": spectral_skewness(p),
        "kurtosis": spectral_kurtosis(p),
        "gini": gini_coefficient(p),
        "spectral_entropy": spectral_entropy(p),
        "peak_value": float(p.max()),
        "peak_layer": int(np.argmax(p)),
        "peak_concentration_top3": peak_concentration(p, 3),
        "n_peaks": count_peaks(p),
        "early_mass": band_mass(p, EARLY),
        "mid_mass": band_mass(p, MID),
        "late_mass": band_mass(p, LATE),
        "late_early_ratio": band_mass(p, LATE) / max(band_mass(p, EARLY), 1e-10),
        "layer_11_dominance": layer_11_dominance(p),
        "late_layer_elevation": late_layer_elevation(p),
        "early_depletion": early_depletion(p),
    }


SCALAR_METRICS = [
    "centroid", "spread", "skewness", "kurtosis", "gini",
    "spectral_entropy", "peak_value", "peak_concentration_top3",
    "n_peaks", "early_mass", "mid_mass", "late_mass",
    "late_early_ratio", "layer_11_dominance", "late_layer_elevation",
    "early_depletion",
]


def paired_tests(c_metrics, h_metrics):
    results = {}
    p_values = []
    for metric in SCALAR_METRICS:
        c_vals = np.array([m[metric] for m in c_metrics])
        h_vals = np.array([m[metric] for m in h_metrics])
        diff = h_vals - c_vals
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        d = mean_diff / std_diff if std_diff > 0 else 0.0
        t_stat, p_t = stats.ttest_rel(h_vals, c_vals)
        try:
            w_stat, p_w = stats.wilcoxon(diff)
        except ValueError:
            w_stat, p_w = float("nan"), 1.0
        results[metric] = {
            "mean_correct": float(np.mean(c_vals)),
            "mean_halluc": float(np.mean(h_vals)),
            "mean_delta": mean_diff,
            "cohen_d": d,
            "t_stat": float(t_stat),
            "p_ttest": float(p_t),
            "p_wilcoxon": float(p_w),
        }
        p_values.append((metric, float(p_t)))

    # Holm-Bonferroni
    p_values.sort(key=lambda x: x[1])
    n_tests = len(p_values)
    for rank, (metric, p_raw) in enumerate(p_values):
        adjusted = min(p_raw * (n_tests - rank), 1.0)
        results[metric]["p_holm"] = adjusted
        results[metric]["significant_holm"] = adjusted < 0.05
    return results


# -- classifier infrastructure --

def build_feature_matrix(profiles, metric_names=None):
    """Build feature matrix from profiles. Each row is one profile's metrics."""
    if metric_names is None:
        metric_names = SCALAR_METRICS
    metrics = [compute_all(p) for p in profiles]
    X = np.array([[m[name] for name in metric_names] for m in metrics])
    return X


def auc_paired(c_vals, h_vals):
    """Paired AUC: fraction of pairs where halluc > correct."""
    n = len(c_vals)
    greater = sum(1 for h, c in zip(h_vals, c_vals) if h > c)
    equal = sum(1 for h, c in zip(h_vals, c_vals) if h == c)
    return (greater + 0.5 * equal) / n


def logistic_cv(X, y, n_folds=5, seed=42):
    """Manual logistic regression with k-fold CV (no sklearn dependency)."""
    from numpy.linalg import lstsq

    n = len(y)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    aucs = []
    accs = []

    for fold in range(n_folds):
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Standardize
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0) + 1e-10
        X_tr = (X_train - mu) / sigma
        X_te = (X_test - mu) / sigma

        # Add bias
        X_tr = np.column_stack([X_tr, np.ones(len(X_tr))])
        X_te = np.column_stack([X_te, np.ones(len(X_te))])

        # Iteratively reweighted least squares (IRLS) for logistic regression
        w = np.zeros(X_tr.shape[1])
        for iteration in range(50):
            p = 1 / (1 + np.exp(-X_tr @ w))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            W_diag = p * (1 - p)
            z = X_tr @ w + (y_train - p) / W_diag
            # Weighted least squares
            W_sqrt = np.sqrt(W_diag)
            X_w = X_tr * W_sqrt[:, None]
            z_w = z * W_sqrt
            w_new, _, _, _ = lstsq(X_w, z_w, rcond=None)
            if np.max(np.abs(w_new - w)) < 1e-6:
                w = w_new
                break
            w = w_new

        # Predict
        p_test = 1 / (1 + np.exp(-X_te @ w))
        pred = (p_test > 0.5).astype(int)
        acc = (pred == y_test).mean()

        # AUC (simple: sort by score, count concordant pairs)
        pos = p_test[y_test == 1]
        neg = p_test[y_test == 0]
        if len(pos) > 0 and len(neg) > 0:
            auc = sum(1 for p in pos for n in neg if p > n) / (len(pos) * len(neg))
        else:
            auc = 0.5

        aucs.append(auc)
        accs.append(acc)

    return {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "fold_aucs": [float(a) for a in aucs],
    }


# -- main analysis --

def main():
    print("=" * 80)
    print("FULL SPECTRUM ANALYSIS -- Fathom Frontier 5 Extended")
    print("=" * 80)

    # ================================================================
    # PART 1: n=200 VALIDATION
    # ================================================================
    print("\n[1/4] VALIDATING ON n=200...")

    if not DATA_200.exists():
        print(f"  ERROR: {DATA_200} not found")
        sys.exit(1)

    c200, h200, q200, extra200 = load_profiles(DATA_200)
    n = len(c200)
    print(f"  Loaded {n} pairs x {N_LAYERS} layers")

    c200_metrics = [compute_all(p) for p in c200]
    h200_metrics = [compute_all(p) for p in h200]
    test200 = paired_tests(c200_metrics, h200_metrics)

    sig_count = sum(1 for m in SCALAR_METRICS if test200[m].get("significant_holm", False))
    print(f"\n  n=200 RESULTS ({sig_count}/{len(SCALAR_METRICS)} significant after Holm-Bonferroni):")
    print(f"  {'METRIC':<25} {'d':>8} {'p_Holm':>12} {'AUC':>8} {'SIG':>5}")
    print("  " + "-" * 60)

    for metric in SCALAR_METRICS:
        r = test200[metric]
        c_vals = np.array([m[metric] for m in c200_metrics])
        h_vals = np.array([m[metric] for m in h200_metrics])
        auc = auc_paired(c_vals, h_vals)
        auc_d = max(auc, 1 - auc)
        sig = "***" if r["p_holm"] < 0.001 else "**" if r["p_holm"] < 0.01 else \
              "*" if r["p_holm"] < 0.05 else ""
        print(f"  {metric:<25} {r['cohen_d']:>+8.3f} {r['p_holm']:>12.6f} {auc_d:>8.3f} {sig:>5}")

    # Per-layer tests on n=200
    print(f"\n  PER-LAYER TESTS (n=200):")
    print(f"  {'LAYER':>6} {'delta':>12} {'t':>8} {'p':>12}")
    print("  " + "-" * 45)
    layer_results_200 = {}
    for i in range(N_LAYERS):
        c_v = c200[:, i]
        h_v = h200[:, i]
        t, p = stats.ttest_rel(h_v, c_v)
        d = h_v.mean() - c_v.mean()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {i:>6} {d:>+12.6f} {t:>8.3f} {p:>12.2e} {sig}")
        layer_results_200[str(i)] = {"delta": float(d), "t": float(t), "p": float(p)}

    # ================================================================
    # PART 2: CLASSIFIER COMPARISON
    # ================================================================
    print("\n" + "=" * 80)
    print("[2/4] CLASSIFIER COMPARISON (5-fold CV on n=200)")
    print("=" * 80)

    # Build labels: 0 = correct, 1 = hallucinated (unpaired for classification)
    X_correct = build_feature_matrix(c200)
    X_halluc = build_feature_matrix(h200)
    X_all = np.vstack([X_correct, X_halluc])
    y_all = np.concatenate([np.zeros(n), np.ones(n)])

    # Also build K, C_delta as baseline features
    K_correct = np.array(extra200["K_correct"])
    K_halluc = np.array(extra200["K_halluc"])
    Cd_correct = np.array(extra200["C_delta_correct"])
    Cd_halluc = np.array(extra200["C_delta_halluc"])

    K_all = np.concatenate([K_correct, K_halluc]).reshape(-1, 1)
    Cd_all = np.concatenate([Cd_correct, Cd_halluc]).reshape(-1, 1)

    # Single-metric classifiers
    print("\n  SINGLE METRIC CV:")
    single_results = {}
    for metric in SCALAR_METRICS:
        idx = SCALAR_METRICS.index(metric)
        X_single = X_all[:, idx:idx+1]
        cv = logistic_cv(X_single, y_all)
        single_results[metric] = cv
        star = " <--" if cv["mean_auc"] > 0.65 else ""
        print(f"    {metric:<25} AUC={cv['mean_auc']:.3f}+/-{cv['std_auc']:.3f}{star}")

    # Baseline classifiers
    print("\n  BASELINE CLASSIFIERS:")
    cv_K = logistic_cv(K_all, y_all)
    cv_Cd = logistic_cv(Cd_all, y_all)
    cv_K_Cd = logistic_cv(np.column_stack([K_all, Cd_all]), y_all)
    print(f"    K alone               AUC={cv_K['mean_auc']:.3f}+/-{cv_K['std_auc']:.3f}")
    print(f"    C_delta alone          AUC={cv_Cd['mean_auc']:.3f}+/-{cv_Cd['std_auc']:.3f}")
    print(f"    K + C_delta            AUC={cv_K_Cd['mean_auc']:.3f}+/-{cv_K_Cd['std_auc']:.3f}")

    # Combined classifiers
    print("\n  COMBINED CLASSIFIERS:")

    # Top-3 spectrum metrics by AUC
    top3 = ["gini", "early_mass", "spread"]
    top3_idx = [SCALAR_METRICS.index(m) for m in top3]
    X_top3 = X_all[:, top3_idx]
    cv_top3 = logistic_cv(X_top3, y_all)
    print(f"    Top-3 spectrum         AUC={cv_top3['mean_auc']:.3f}+/-{cv_top3['std_auc']:.3f}  ({', '.join(top3)})")

    # Top-5 spectrum
    top5 = ["gini", "early_mass", "spread", "spectral_entropy", "peak_value"]
    top5_idx = [SCALAR_METRICS.index(m) for m in top5]
    X_top5 = X_all[:, top5_idx]
    cv_top5 = logistic_cv(X_top5, y_all)
    print(f"    Top-5 spectrum         AUC={cv_top5['mean_auc']:.3f}+/-{cv_top5['std_auc']:.3f}")

    # All spectrum metrics
    cv_all_spectrum = logistic_cv(X_all, y_all)
    print(f"    All 16 spectrum        AUC={cv_all_spectrum['mean_auc']:.3f}+/-{cv_all_spectrum['std_auc']:.3f}")

    # Spectrum + K + C_delta
    X_everything = np.column_stack([X_all, K_all, Cd_all])
    cv_everything = logistic_cv(X_everything, y_all)
    print(f"    All spectrum + K + Cd  AUC={cv_everything['mean_auc']:.3f}+/-{cv_everything['std_auc']:.3f}")

    # Gini alone (the champion)
    gini_idx = SCALAR_METRICS.index("gini")
    X_gini = X_all[:, gini_idx:gini_idx+1]
    cv_gini = logistic_cv(X_gini, y_all)
    print(f"    Gini alone             AUC={cv_gini['mean_auc']:.3f}+/-{cv_gini['std_auc']:.3f}")

    # ================================================================
    # PART 3: EFFECT SIZE COMPARISON vs. PRIOR RESULTS
    # ================================================================
    print("\n" + "=" * 80)
    print("[3/4] EFFECT SIZE LEADERBOARD")
    print("=" * 80)

    leaderboard = []

    # Add spectrum metrics
    for metric in SCALAR_METRICS:
        r = test200[metric]
        c_vals = np.array([m[metric] for m in c200_metrics])
        h_vals = np.array([m[metric] for m in h200_metrics])
        auc = max(auc_paired(c_vals, h_vals), 1 - auc_paired(c_vals, h_vals))
        leaderboard.append({
            "metric": f"spectrum:{metric}",
            "d": abs(r["cohen_d"]),
            "p": r["p_ttest"],
            "auc": auc,
            "source": "spectrum_n200",
        })

    # Add K and C_delta
    K_diff = K_halluc - K_correct
    Cd_diff = Cd_halluc - Cd_correct
    K_d = K_diff.mean() / K_diff.std(ddof=1) if K_diff.std(ddof=1) > 0 else 0
    Cd_d = Cd_diff.mean() / Cd_diff.std(ddof=1) if Cd_diff.std(ddof=1) > 0 else 0
    _, K_p = stats.ttest_rel(K_halluc, K_correct)
    _, Cd_p = stats.ttest_rel(Cd_halluc, Cd_correct)
    leaderboard.append({"metric": "K (depth)", "d": abs(K_d), "p": K_p,
                        "auc": max(auc_paired(K_correct, K_halluc), 1 - auc_paired(K_correct, K_halluc)),
                        "source": "original"})
    leaderboard.append({"metric": "C_delta", "d": abs(Cd_d), "p": Cd_p,
                        "auc": max(auc_paired(Cd_correct, Cd_halluc), 1 - auc_paired(Cd_correct, Cd_halluc)),
                        "source": "original"})

    # Sort by AUC
    leaderboard.sort(key=lambda x: x["auc"], reverse=True)
    print(f"\n  {'RANK':>4} {'METRIC':<35} {'|d|':>6} {'AUC':>6} {'p':>12} {'SOURCE':>15}")
    print("  " + "-" * 85)
    for rank, entry in enumerate(leaderboard[:20], 1):
        print(f"  {rank:>4} {entry['metric']:<35} {entry['d']:>6.3f} {entry['auc']:>6.3f} "
              f"{entry['p']:>12.2e} {entry['source']:>15}")

    # ================================================================
    # PART 4: FIGURES
    # ================================================================
    print("\n" + "=" * 80)
    print("[4/4] GENERATING FIGURES")
    print("=" * 80)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available")
        return

    layers = np.arange(N_LAYERS)

    # -- Figure 1: n=200 spectrum overview --
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Layer Coherence Spectrum (n=200, Gemma-2-2B)", fontsize=14, fontweight="bold")

    # A: Mean profiles
    ax = axes[0, 0]
    c_mean = c200.mean(axis=0)
    h_mean = h200.mean(axis=0)
    c_se = c200.std(axis=0) / np.sqrt(n)
    h_se = h200.std(axis=0) / np.sqrt(n)
    ax.fill_between(layers, c_mean - 1.96*c_se, c_mean + 1.96*c_se, alpha=0.2, color="#00aa44")
    ax.fill_between(layers, h_mean - 1.96*h_se, h_mean + 1.96*h_se, alpha=0.2, color="#cc3333")
    ax.plot(layers, c_mean, "o-", color="#00aa44", label="Correct", ms=4)
    ax.plot(layers, h_mean, "s-", color="#cc3333", label="Hallucinated", ms=4)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Coherence")
    ax.set_title("A. Mean Coherence Spectrum (+/-95% CI)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # B: Difference with significance
    ax = axes[0, 1]
    diff = h200 - c200
    diff_mean = diff.mean(axis=0)
    diff_se = diff.std(axis=0) / np.sqrt(n)
    colors = ["#cc3333" if d > 0 else "#00aa44" for d in diff_mean]
    ax.bar(layers, diff_mean, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.errorbar(layers, diff_mean, yerr=1.96*diff_se, fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    for i in range(N_LAYERS):
        p = layer_results_200[str(i)]["p"]
        ypos = diff_mean[i] + 1.96 * diff_se[i] + 0.0002
        if p < 0.001:
            ax.text(i, ypos, "***", ha="center", fontsize=7, fontweight="bold")
        elif p < 0.01:
            ax.text(i, ypos, "**", ha="center", fontsize=7)
        elif p < 0.05:
            ax.text(i, ypos, "*", ha="center", fontsize=7)
    ax.set_xlabel("Layer"); ax.set_ylabel("Delta Coherence (halluc - correct)")
    ax.set_title("B. Per-Layer Difference (n=200)")
    ax.grid(True, alpha=0.3)

    # C: Gini distribution
    ax = axes[1, 0]
    c_gini = np.array([compute_all(p)["gini"] for p in c200])
    h_gini = np.array([compute_all(p)["gini"] for p in h200])
    bins = np.linspace(min(c_gini.min(), h_gini.min()), max(c_gini.max(), h_gini.max()), 30)
    ax.hist(c_gini, bins=bins, alpha=0.5, color="#00aa44", label="Correct", density=True)
    ax.hist(h_gini, bins=bins, alpha=0.5, color="#cc3333", label="Halluc", density=True)
    ax.axvline(c_gini.mean(), color="#00aa44", ls="--", lw=2)
    ax.axvline(h_gini.mean(), color="#cc3333", ls="--", lw=2)
    r = test200["gini"]
    ax.set_title(f"C. Gini Coefficient (d={r['cohen_d']:+.3f}, p={r['p_ttest']:.2e})")
    ax.set_xlabel("Gini"); ax.legend(); ax.grid(True, alpha=0.3)

    # D: Classifier comparison bar chart
    ax = axes[1, 1]
    classifiers = {
        "K": cv_K["mean_auc"],
        "C_delta": cv_Cd["mean_auc"],
        "K+Cd": cv_K_Cd["mean_auc"],
        "Gini": cv_gini["mean_auc"],
        "Top-3\nspectrum": cv_top3["mean_auc"],
        "Top-5\nspectrum": cv_top5["mean_auc"],
        "All\nspectrum": cv_all_spectrum["mean_auc"],
        "All+K+Cd": cv_everything["mean_auc"],
    }
    stds = {
        "K": cv_K["std_auc"],
        "C_delta": cv_Cd["std_auc"],
        "K+Cd": cv_K_Cd["std_auc"],
        "Gini": cv_gini["std_auc"],
        "Top-3\nspectrum": cv_top3["std_auc"],
        "Top-5\nspectrum": cv_top5["std_auc"],
        "All\nspectrum": cv_all_spectrum["std_auc"],
        "All+K+Cd": cv_everything["std_auc"],
    }
    names = list(classifiers.keys())
    vals = list(classifiers.values())
    errs = list(stds.values())
    bar_colors = ["#999999"] * 3 + ["#00ccaa"] * 5
    ax.bar(names, vals, yerr=errs, color=bar_colors, alpha=0.7, edgecolor="black",
           linewidth=0.5, capsize=3)
    ax.axhline(0.5, color="black", ls=":", lw=1, label="Chance")
    ax.set_ylabel("5-fold CV AUC")
    ax.set_title("D. Classifier Comparison")
    ax.set_ylim(0.4, 0.85)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"spectrum_n200_full.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> spectrum_n200_full.pdf/png")

    # -- Figure 2: ROC-style AUC leaderboard --
    fig, ax = plt.subplots(figsize=(10, 6))
    top_entries = leaderboard[:15]
    names = [e["metric"].replace("spectrum:", "") for e in top_entries]
    aucs = [e["auc"] for e in top_entries]
    colors = ["#00ccaa" if "spectrum" in e["metric"] else "#cc6633" for e in top_entries]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, aucs, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AUC (paired)")
    ax.set_title("Hallucination Discrimination: Spectrum vs Scalar Metrics (n=200)")
    ax.axvline(0.5, color="black", ls=":", lw=1)
    ax.set_xlim(0.45, 0.85)
    ax.invert_yaxis()
    # Add value labels
    for i, (name, auc_val) in enumerate(zip(names, aucs)):
        ax.text(auc_val + 0.005, i, f"{auc_val:.3f}", va="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#00ccaa", alpha=0.7, label="Spectrum metric"),
        Patch(facecolor="#cc6633", alpha=0.7, label="Scalar metric"),
    ], loc="lower right")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"spectrum_auc_leaderboard.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> spectrum_auc_leaderboard.pdf/png")

    # -- Figure 3: Per-layer significance heatmap --
    fig, ax = plt.subplots(figsize=(12, 3))
    p_vals = np.array([layer_results_200[str(i)]["p"] for i in range(N_LAYERS)])
    neg_log_p = -np.log10(np.clip(p_vals, 1e-20, 1.0))
    deltas = np.array([layer_results_200[str(i)]["delta"] for i in range(N_LAYERS)])

    # Color by direction: red = halluc higher, green = correct higher
    signed_sig = neg_log_p * np.sign(deltas)
    im = ax.imshow(signed_sig.reshape(1, -1), aspect="auto", cmap="RdYlGn_r",
                   extent=[-0.5, N_LAYERS - 0.5, -0.5, 0.5],
                   vmin=-10, vmax=10)
    ax.set_xticks(range(N_LAYERS))
    ax.set_yticks([])
    ax.set_xlabel("Layer")
    ax.set_title("Per-Layer Significance (-log10 p, signed by direction, n=200)")
    # Significance thresholds
    for i in range(N_LAYERS):
        if p_vals[i] < 0.001:
            ax.text(i, 0, "***", ha="center", va="center", fontsize=8, fontweight="bold")
        elif p_vals[i] < 0.01:
            ax.text(i, 0, "**", ha="center", va="center", fontsize=8)
        elif p_vals[i] < 0.05:
            ax.text(i, 0, "*", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="-log10(p) * sign(delta)")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"spectrum_layer_significance.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> spectrum_layer_significance.pdf/png")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    output = {
        "experiment": "spectrum_full_analysis",
        "n_pairs_200": n,
        "shape_metrics_n200": test200,
        "per_layer_n200": layer_results_200,
        "classifiers": {
            "K_alone": cv_K,
            "C_delta_alone": cv_Cd,
            "K_plus_Cd": cv_K_Cd,
            "gini_alone": cv_gini,
            "top3_spectrum": cv_top3,
            "top5_spectrum": cv_top5,
            "all_spectrum": cv_all_spectrum,
            "all_plus_K_Cd": cv_everything,
        },
        "leaderboard": leaderboard[:20],
    }
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results -> {RESULTS.name}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
