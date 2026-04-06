"""
fathom_pertoken_roc.py — Per-Token C_delta ROC Analysis
=========================================================

The experiment darkflobi called for: measure C_delta at EVERY token
during generation, extract spike features, build ROC on the spike
instead of the aggregate.

"the spike is the signal. you've been measuring the echo."

Selects 50 questions from n=200 baseline:
  - 25 where model hallucinates (highest C_delta gap)
  - 25 where model is correct (lowest C_delta gap / negative gap)

Runs coherence steerer in monitor-only mode on each.
Extracts per-trace features:
  1. max C_delta (peak value)
  2. max slope (sharpest single-token rise)
  3. peak position (fraction through generation: 0=early, 1=late)

Builds ROC on each feature + combined classifier.

Usage:
    python -u fathom_pertoken_roc.py [--n-per-class 25] [--max-tokens 40]

Patent pending: US 64/020,489 & 64/021,113
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
from scipy import stats as scipy_stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "truthfulqa_results")
BASELINE_PATH = os.path.join(RESULTS_DIR, "n200_checkpoint.json")


def select_balanced_set(n_per_class: int):
    """Select top-N hallucinators and top-N correct from baseline."""
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    gaps = []
    for i, r in enumerate(results):
        gap = r["hallucinated"]["C_delta"] - r["correct"]["C_delta"]
        gaps.append((i, gap, r))

    gaps.sort(key=lambda x: x[1], reverse=True)

    # Top N hallucinators (biggest positive gap)
    halluc_set = [(g[0], g[2], "hallucinated") for g in gaps[:n_per_class]]

    # Top N correct (biggest negative gap = model most clearly NOT hallucinating)
    correct_set = [(g[0], g[2], "correct") for g in gaps[-n_per_class:]]

    print(f"[Selection] {n_per_class} hallucinators (gap range: {gaps[0][1]:+.6f} to {gaps[n_per_class-1][1]:+.6f})")
    print(f"[Selection] {n_per_class} correct (gap range: {gaps[-n_per_class][1]:+.6f} to {gaps[-1][1]:+.6f})")

    return halluc_set + correct_set


def extract_trace_features(c_delta_trajectory):
    """Extract spike features from per-token C_delta trace."""
    valid = [(i, v) for i, v in enumerate(c_delta_trajectory) if v is not None]
    if len(valid) < 3:
        return None

    indices = [v[0] for v in valid]
    values = np.array([v[1] for v in valid])

    # Feature 1: max C_delta (peak)
    max_cdelta = float(np.max(values))

    # Feature 2: max slope (sharpest single-token rise)
    if len(values) >= 2:
        diffs = np.diff(values)
        max_slope = float(np.max(diffs))
    else:
        max_slope = 0.0

    # Feature 3: peak position (fraction through generation)
    peak_idx = np.argmax(values)
    peak_position = float(peak_idx / max(len(values) - 1, 1))

    # Bonus features
    mean_cdelta = float(np.mean(values))
    std_cdelta = float(np.std(values))
    n_above_005 = int(np.sum(values > 0.005))
    n_above_010 = int(np.sum(values > 0.010))

    return {
        "max_cdelta": max_cdelta,
        "max_slope": max_slope,
        "peak_position": peak_position,
        "mean_cdelta": mean_cdelta,
        "std_cdelta": std_cdelta,
        "n_above_005": n_above_005,
        "n_above_010": n_above_010,
        "n_tokens": len(values),
    }


def compute_roc(y_true, scores, label=""):
    """Compute ROC + AUC with bootstrap CI."""
    from sklearn.metrics import roc_curve, roc_auc_score

    y_true = np.array(y_true)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    # Bootstrap CI
    boot_aucs = []
    for _ in range(2000):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(y_true[idx], scores[idx]))
        except:
            pass

    ci_low = np.percentile(boot_aucs, 2.5) if boot_aucs else 0
    ci_high = np.percentile(boot_aucs, 97.5) if boot_aucs else 0

    return {
        "auc": float(auc),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "label": label,
    }


def main():
    parser = argparse.ArgumentParser(description="Per-Token C_delta ROC")
    parser.add_argument("--n-per-class", type=int, default=25, help="Questions per class")
    parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens per generation")
    args = parser.parse_args()

    n_total = args.n_per_class * 2
    print(f"\n{'='*70}")
    print(f"FATHOM PER-TOKEN ROC — {n_total} questions ({args.n_per_class} per class)")
    print(f"{'='*70}\n")

    # Select balanced set
    question_set = select_balanced_set(args.n_per_class)

    # Initialize steerer in monitor-only mode
    from coherence_steerer import CoherenceSteerer
    steerer = CoherenceSteerer(model_name="google/gemma-2-2b")

    # Run monitor-only on each question
    all_results = []
    y_true = []  # 1 = hallucinated, 0 = correct
    all_features = []

    t_start = time.time()

    for i, (idx, r, label) in enumerate(question_set):
        q = r["question"]
        is_halluc = label == "hallucinated"

        # Build prompt with the answer (correct or incorrect based on label)
        if is_halluc:
            answer = r["incorrect_answer"]
        else:
            answer = r["correct_answer"]

        prompt = f"Q: {q}\nA: {answer}"

        print(f"\n[{i+1}/{n_total}] {'HALLUC' if is_halluc else 'CORRECT'} | {q[:50]}")

        result = steerer.generate_monitored(prompt, max_tokens=args.max_tokens)
        features = extract_trace_features(result.c_delta_trajectory)

        if features is None:
            print(f"  SKIP — insufficient trace data")
            continue

        y_true.append(1 if is_halluc else 0)
        all_features.append(features)

        safe_text = result.text.encode('ascii', 'replace').decode()[:50]
        print(f"  max_cd={features['max_cdelta']:.6f} slope={features['max_slope']:.6f} peak_pos={features['peak_position']:.2f} n_spikes={features['n_above_005']}")
        print(f"  \"{safe_text}\"")

        all_results.append({
            "question": q,
            "label": label,
            "answer_fed": answer[:80],
            "generated_text": result.text,
            "c_delta_trajectory": result.c_delta_trajectory,
            "features": features,
        })

    elapsed = time.time() - t_start

    # Compute ROCs for each feature
    print(f"\n{'='*70}")
    print(f"ROC ANALYSIS")
    print(f"{'='*70}")

    roc_results = {}
    feature_names = ["max_cdelta", "max_slope", "peak_position", "mean_cdelta", "std_cdelta", "n_above_005"]

    for feat in feature_names:
        scores = [f[feat] for f in all_features]
        roc = compute_roc(y_true, scores, label=feat)
        roc_results[feat] = roc
        print(f"  {feat:20s}: AUC = {roc['auc']:.3f}  95% CI [{roc['ci_low']:.3f}, {roc['ci_high']:.3f}]")

    # Combined: simple logistic-style = max_cdelta + max_slope (normalize first)
    max_cd = np.array([f["max_cdelta"] for f in all_features])
    max_sl = np.array([f["max_slope"] for f in all_features])
    combined = (max_cd - max_cd.mean()) / max(max_cd.std(), 1e-8) + (max_sl - max_sl.mean()) / max(max_sl.std(), 1e-8)
    roc_combined = compute_roc(y_true, combined.tolist(), label="combined_peak+slope")
    roc_results["combined"] = roc_combined
    print(f"  {'combined':20s}: AUC = {roc_combined['auc']:.3f}  95% CI [{roc_combined['ci_low']:.3f}, {roc_combined['ci_high']:.3f}]")

    # Save results
    output = {
        "experiment": "pertoken_roc",
        "n_per_class": args.n_per_class,
        "n_total": len(y_true),
        "max_tokens": args.max_tokens,
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed,
        "roc_summary": {k: {"auc": v["auc"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]} for k, v in roc_results.items()},
        "results": all_results,
    }

    out_path = os.path.join(RESULTS_DIR, "pertoken_roc.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Plot ROC curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')

        colors = {'max_cdelta': '#00ff88', 'max_slope': '#00ccff', 'mean_cdelta': '#ffaa00', 'combined': '#ff44ff', 'n_above_005': '#ff8844'}
        for feat in ["max_cdelta", "max_slope", "mean_cdelta", "combined", "n_above_005"]:
            if feat in roc_results:
                r = roc_results[feat]
                c = colors.get(feat, '#ffffff')
                lw = 2.5 if feat in ("max_cdelta", "combined") else 1.5
                ax.plot(r["fpr"], r["tpr"], color=c, linewidth=lw, label=f'{feat} (AUC={r["auc"]:.3f})')

        ax.plot([0,1],[0,1],'--', color='gray', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate', color='#cccccc', fontfamily='monospace', fontsize=11)
        ax.set_ylabel('True Positive Rate', color='#cccccc', fontfamily='monospace', fontsize=11)
        ax.set_title(f'Fathom Per-Token Hallucination Detection (n={len(y_true)})', color='#00ffcc', fontfamily='monospace', fontweight='bold', fontsize=12)
        ax.legend(facecolor='#1a1a1a', edgecolor='#333333', labelcolor='#cccccc', fontsize=9)
        ax.tick_params(colors='#666666')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        plt.tight_layout()

        fig_path = os.path.join(os.path.dirname(__file__), "paper", "figures", "pertoken_roc.pdf")
        fig_path_png = fig_path.replace('.pdf', '.png')
        plt.savefig(fig_path, dpi=300, facecolor='#0d0d0d', bbox_inches='tight')
        plt.savefig(fig_path_png, dpi=150, facecolor='#0d0d0d', bbox_inches='tight')
        print(f"\nROC figure saved: {fig_path}")
    except Exception as e:
        print(f"\nFigure error: {e}")

    print(f"\nResults saved: {out_path}")
    print(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
