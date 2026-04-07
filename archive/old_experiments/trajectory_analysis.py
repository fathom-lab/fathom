"""
trajectory_analysis.py — Divergence Analysis for Cognitive Trajectories
=======================================================================

Day 2: Analyze token-by-token (K, C_delta) trajectories from
trajectory_experiment.py. Find divergence signatures between
truthful and hallucinated generations.

Key questions:
  1. Do trajectories diverge before hallucination appears in output?
  2. At which token does divergence become statistically detectable?
  3. Can a simple classifier predict hallucination from trajectory prefix?

Output:
  - Per-token mean trajectory plots (K and C_delta)
  - Divergence detection with N-token-early measurement
  - Logistic regression classifier on trajectory prefixes
  - Bootstrap confidence intervals

Patent pending: US 64/020,489 & 64/021,113
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_trajectories(path: str = "trajectory_results/trajectories_final.json"):
    """Load trajectory data from experiment output."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[+] Loaded {len(data)} trajectory pairs")
    return data


def extract_timeseries(pairs, metric="k_fast", max_tokens=30):
    """
    Extract aligned time series for correct and hallucinated generations.

    Returns:
        correct_matrix: (n_pairs, max_tokens) — NaN padded
        halluc_matrix: (n_pairs, max_tokens) — NaN padded
    """
    correct_series = []
    halluc_series = []

    for pair in pairs:
        c = pair.get("correct")
        h = pair.get("hallucinated")
        if not c or not h:
            continue

        c_vals = [s[metric] for s in c["states"][:max_tokens]]
        h_vals = [s[metric] for s in h["states"][:max_tokens]]

        # Pad to max_tokens with NaN
        c_padded = c_vals + [None] * (max_tokens - len(c_vals))
        h_padded = h_vals + [None] * (max_tokens - len(h_vals))

        correct_series.append([v if v is not None else float("nan") for v in c_padded])
        halluc_series.append([v if v is not None else float("nan") for v in h_padded])

    return np.array(correct_series), np.array(halluc_series)


def per_token_stats(correct_matrix, halluc_matrix):
    """
    Compute per-token statistics: mean, CI, and paired t-test at each position.

    Returns list of dicts with stats at each token position.
    """
    n_tokens = correct_matrix.shape[1]
    results = []

    for t in range(n_tokens):
        c_vals = correct_matrix[:, t]
        h_vals = halluc_matrix[:, t]

        # Drop NaN pairs
        valid = ~(np.isnan(c_vals) | np.isnan(h_vals))
        c_clean = c_vals[valid]
        h_clean = h_vals[valid]
        n = len(c_clean)

        if n < 3:
            results.append({"t": t, "n": n, "significant": False})
            continue

        c_mean = np.mean(c_clean)
        h_mean = np.mean(h_clean)
        delta = h_mean - c_mean

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(h_clean, c_clean)

        # Cohen's d (paired)
        diffs = h_clean - c_clean
        d = np.mean(diffs) / max(np.std(diffs, ddof=1), 1e-10)

        # Bootstrap CI for delta
        boot_deltas = []
        rng = np.random.RandomState(42 + t)
        for _ in range(1000):
            idx = rng.choice(n, size=n, replace=True)
            boot_deltas.append(np.mean(h_clean[idx]) - np.mean(c_clean[idx]))
        ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])

        results.append({
            "t": t,
            "n": n,
            "correct_mean": float(c_mean),
            "halluc_mean": float(h_mean),
            "delta": float(delta),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "significant": p_val < 0.05,
        })

    return results


def find_divergence_point(token_stats, p_threshold=0.05, consecutive=2):
    """
    Find the first token where trajectories significantly diverge.

    Requires `consecutive` consecutive significant tokens to confirm divergence.
    Returns token index or None.
    """
    count = 0
    for s in token_stats:
        if s.get("significant", False):
            count += 1
            if count >= consecutive:
                return s["t"] - consecutive + 1
        else:
            count = 0
    return None


def trajectory_classifier(pairs, max_prefix=None):
    """
    Train logistic regression on trajectory prefixes to predict hallucination.

    For each prefix length from 1 to max_prefix, train a classifier and
    measure accuracy. This answers: how many tokens early can we predict?

    Uses leave-one-out cross-validation for small samples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    if max_prefix is None:
        max_prefix = 20

    # Build feature matrix: for each generation, take the first N tokens'
    # (K_fast, C_delta) as features
    results_by_prefix = []

    for prefix_len in range(1, max_prefix + 1):
        X = []
        y = []

        for pair in pairs:
            c = pair.get("correct")
            h = pair.get("hallucinated")
            if not c or not h:
                continue

            for gen, label in [(c, 0), (h, 1)]:
                states = gen["states"][:prefix_len]
                if len(states) < prefix_len:
                    continue

                # Features: mean K, mean C_delta, last K, last C_delta,
                # K slope, C_delta slope over prefix
                ks = [s["k_fast"] for s in states if s["k_fast"] is not None]
                cds = [s["c_delta"] for s in states if s["c_delta"] is not None]

                if not ks or not cds:
                    continue

                # darkflobi's feature spec: slope and variance, not means
                # The signal is dynamic, not static
                feats = [
                    (ks[-1] - ks[0]) if len(ks) > 1 else 0,             # dK/dt (K slope)
                    (cds[-1] - cds[0]) if len(cds) > 1 else 0,          # dC/dt (C_delta slope)
                    ks[0] if ks else 0,                                   # K at first gen token
                    cds[0] if cds else 0,                                 # C_delta at first gen token
                    np.var(ks) if len(ks) > 1 else 0,                    # K variance
                    np.var(cds) if len(cds) > 1 else 0,                  # C_delta variance
                ]
                X.append(feats)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        if len(X) < 6:
            results_by_prefix.append({
                "prefix_len": prefix_len,
                "n_samples": len(X),
                "accuracy": None,
            })
            continue

        # Leave-one-out CV
        loo = LeaveOneOut()
        correct_preds = 0
        total = 0

        for train_idx, test_idx in loo.split(X):
            clf = LogisticRegression(max_iter=1000, C=1.0)
            try:
                clf.fit(X[train_idx], y[train_idx])
                pred = clf.predict(X[test_idx])
                if pred[0] == y[test_idx[0]]:
                    correct_preds += 1
                total += 1
            except Exception:
                continue

        accuracy = correct_preds / max(total, 1)
        results_by_prefix.append({
            "prefix_len": prefix_len,
            "n_samples": len(X),
            "accuracy": float(accuracy),
            "correct_preds": correct_preds,
            "total": total,
        })

    return results_by_prefix


def print_analysis(k_stats, cd_stats, k_diverge, cd_diverge, classifier_results):
    """Print full analysis report."""
    print(f"\n{'='*70}")
    print(f"PREDICTIVE COGNITIVE TRAJECTORY ANALYSIS")
    print(f"{'='*70}")

    print(f"\n--- K_fast (depth) per-token ---")
    print(f"{'Token':>5} {'N':>3} {'Correct':>8} {'Halluc':>8} {'Delta':>8} {'p':>8} {'d':>6} {'Sig':>4}")
    for s in k_stats:
        if s["n"] < 3:
            continue
        sig = "***" if s.get("p_value", 1) < 0.001 else "**" if s.get("p_value", 1) < 0.01 else "*" if s.get("significant") else ""
        print(f"{s['t']:>5} {s['n']:>3} {s.get('correct_mean', 0):>8.3f} {s.get('halluc_mean', 0):>8.3f} "
              f"{s.get('delta', 0):>+8.3f} {s.get('p_value', 1):>8.4f} {s.get('cohens_d', 0):>+6.2f} {sig:>4}")

    if k_diverge is not None:
        print(f"\n  >>> K DIVERGENCE at token {k_diverge}")
    else:
        print(f"\n  >>> No sustained K divergence detected")

    print(f"\n--- C_delta (coherence shift) per-token ---")
    print(f"{'Token':>5} {'N':>3} {'Correct':>8} {'Halluc':>8} {'Delta':>8} {'p':>8} {'d':>6} {'Sig':>4}")
    for s in cd_stats:
        if s["n"] < 3:
            continue
        sig = "***" if s.get("p_value", 1) < 0.001 else "**" if s.get("p_value", 1) < 0.01 else "*" if s.get("significant") else ""
        print(f"{s['t']:>5} {s['n']:>3} {s.get('correct_mean', 0):>8.4f} {s.get('halluc_mean', 0):>8.4f} "
              f"{s.get('delta', 0):>+8.4f} {s.get('p_value', 1):>8.4f} {s.get('cohens_d', 0):>+6.2f} {sig:>4}")

    if cd_diverge is not None:
        print(f"\n  >>> C_delta DIVERGENCE at token {cd_diverge}")
    else:
        print(f"\n  >>> No sustained C_delta divergence detected")

    print(f"\n--- Predictive Classifier (LOO-CV accuracy by prefix length) ---")
    print(f"{'Prefix':>6} {'N':>4} {'Accuracy':>8} {'Verdict':>20}")
    for r in classifier_results:
        if r["accuracy"] is None:
            continue
        acc = r["accuracy"]
        verdict = ""
        if acc >= 0.80:
            verdict = "STRONG SIGNAL"
        elif acc >= 0.65:
            verdict = "moderate signal"
        elif acc >= 0.55:
            verdict = "weak signal"
        else:
            verdict = "chance"
        print(f"{r['prefix_len']:>6} {r['n_samples']:>4} {acc:>8.1%} {verdict:>20}")

    # Find earliest prefix with >65% accuracy
    for r in classifier_results:
        if r["accuracy"] is not None and r["accuracy"] >= 0.65:
            print(f"\n  >>> EARLIEST PREDICTIVE PREFIX: {r['prefix_len']} tokens ({r['accuracy']:.1%} accuracy)")
            break
    else:
        print(f"\n  >>> No prefix achieved >65% accuracy")

    # The money question
    if k_diverge is not None or cd_diverge is not None:
        div_token = min(x for x in [k_diverge, cd_diverge] if x is not None)
        print(f"\n{'='*70}")
        print(f"RESULT: Cognitive trajectory diverges at token {div_token}")
        print(f"This is PRECOGNITIVE HALLUCINATION DETECTION.")
        print(f"{'='*70}")


def run_analysis(data_path: str = "trajectory_results/trajectories_final.json"):
    """Run full divergence analysis pipeline."""
    pairs = load_trajectories(data_path)

    # Extract time series
    k_correct, k_halluc = extract_timeseries(pairs, "k_fast")
    cd_correct, cd_halluc = extract_timeseries(pairs, "c_delta")

    print(f"[+] Extracted {k_correct.shape[0]} pairs, {k_correct.shape[1]} tokens max")

    # Per-token stats
    k_stats = per_token_stats(k_correct, k_halluc)
    cd_stats = per_token_stats(cd_correct, cd_halluc)

    # Find divergence points
    k_diverge = find_divergence_point(k_stats)
    cd_diverge = find_divergence_point(cd_stats)

    # Classifier
    try:
        classifier_results = trajectory_classifier(pairs, max_prefix=min(20, k_correct.shape[1]))
    except ImportError:
        print("[!] sklearn not available, skipping classifier")
        classifier_results = []

    # Report
    print_analysis(k_stats, cd_stats, k_diverge, cd_diverge, classifier_results)

    # Save analysis
    out_path = Path(data_path).parent / "analysis_results.json"
    analysis = {
        "k_stats": k_stats,
        "cd_stats": cd_stats,
        "k_divergence_token": k_diverge,
        "cd_divergence_token": cd_diverge,
        "classifier_results": classifier_results,
    }
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)  # fallback for any other non-serializable types

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=convert)
    print(f"\n[+] Analysis saved to {out_path}")

    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="trajectory_results/trajectories_final.json")
    args = parser.parse_args()
    run_analysis(args.data)
