"""
analyze_S_triplet.py — pre-registered analysis of S-axis triplet
=================================================================
Runs the analysis plan locked in osf_prereg_S_triplet.md.

All tests:
  H1 (primary):   paired t-test S(C-) vs S(C+)
  H2 (secondary): paired t-test S(C+) vs S(H)
  H3 (tertiary):  Friedman + Wilcoxon post-hoc (Bonferroni)

Gates (pre-registered):
  p < 0.05 (two-sided)
  d > 0.40
  AUC > 0.65
  Bonferroni alpha: 0.017

Sensitivity: threshold in [0.003, 0.005, 0.008]
Length check: if conditions differ by >5 tokens, retest with trunc.

Usage:
    python analyze_S_triplet.py [path_to_results.json]
"""

import json
import sys
import os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

THR_PRIMARY = 0.005
THR_SENSITIVITY = [0.003, 0.005, 0.008]
GATES = {"p": 0.05, "d": 0.40, "AUC": 0.65}
BONFERRONI_ALPHA = 0.017


def compute_S(traj, threshold=THR_PRIMARY):
    if not traj:
        return 0.0
    spikes = [c for c in traj if c > threshold]
    if len(spikes) == 0:
        return 0.0
    return max(traj) / len(spikes)


def check_gates(p, d, auc):
    return (p < GATES["p"]) and (abs(d) > GATES["d"]) and (auc > GATES["AUC"])


def paired_test(a, b, label_a="A", label_b="B"):
    """Paired t-test, d, AUC, Wilcoxon. a should be <, b should be >."""
    a, b = np.array(a), np.array(b)
    assert len(a) == len(b)
    n = len(a)
    diff = b - a
    t, p = stats.ttest_rel(a, b)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    auc = roc_auc_score([0]*n + [1]*n, list(a) + list(b))
    w, wp = stats.wilcoxon(a, b)
    dir_acc = np.mean(b > a)
    return {
        "n": n, "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b)),
        "ratio": float(np.mean(b) / np.mean(a)) if np.mean(a) > 0 else None,
        "t": float(t), "p": float(p), "d": float(d), "auc": float(auc),
        "wilcoxon_W": float(w), "wilcoxon_p": float(wp),
        "direction": float(dir_acc),
        "gates_passed": check_gates(p, d, auc),
    }


def analyze(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 70)
    print("S-axis triplet analysis (pre-registered)")
    print("=" * 70)
    print(f"Data:       {path}")
    print(f"Model:      {data.get('model')}")
    print(f"N topics:   {len(data['results'])}")
    print(f"Mode:       {data.get('mode')}")
    print(f"Timestamp:  {data.get('timestamp')}")
    print(f"Threshold:  {THR_PRIMARY} (primary, locked)")
    print(f"Gates:      p<{GATES['p']}, d>{GATES['d']}, AUC>{GATES['AUC']}")
    print(f"Bonferroni: alpha={BONFERRONI_ALPHA}")
    print()

    results = data["results"]

    # --- Length check ---
    lens_H = [r["H"]["total_tokens"] for r in results]
    lens_Cp = [r["C_plus"]["total_tokens"] for r in results]
    lens_Cm = [r["C_minus"]["total_tokens"] for r in results]
    means = [np.mean(lens_H), np.mean(lens_Cp), np.mean(lens_Cm)]
    spread = max(means) - min(means)
    print("Length check:")
    print(f"  H  mean={np.mean(lens_H):.1f}  C+ mean={np.mean(lens_Cp):.1f}  "
          f"C- mean={np.mean(lens_Cm):.1f}")
    print(f"  inter-condition spread: {spread:.1f} tokens "
          f"({'OK' if spread <= 5 else 'FLAGGED — will rerun with truncation'})")
    print()

    # --- Compute S per condition ---
    S_H = [compute_S(r["H"]["c_delta_trajectory"]) for r in results]
    S_Cp = [compute_S(r["C_plus"]["c_delta_trajectory"]) for r in results]
    S_Cm = [compute_S(r["C_minus"]["c_delta_trajectory"]) for r in results]

    print("-" * 70)
    print("Descriptive statistics (primary threshold = 0.005)")
    print("-" * 70)
    for label, vals in [("S(H)  hedge       ", S_H),
                       ("S(C+) correct-commit", S_Cp),
                       ("S(C-) wrong-commit  ", S_Cm)]:
        arr = np.array(vals)
        print(f"  {label}: mean={arr.mean():.6f}  std={arr.std(ddof=1):.6f}  "
              f"median={np.median(arr):.6f}")
    print()

    # --- H1: primary ---
    print("=" * 70)
    print("H1 (PRIMARY): S(C-) > S(C+)  — confident-wrongness detector")
    print("=" * 70)
    h1 = paired_test(S_Cp, S_Cm, "C+", "C-")
    print(f"  n={h1['n']}  C+ mean={h1['a_mean']:.6f}  C- mean={h1['b_mean']:.6f}  "
          f"ratio={h1['ratio']:.3f}x")
    print(f"  paired t={h1['t']:.3f}  p={h1['p']:.4f}  d={h1['d']:.3f}  "
          f"AUC={h1['auc']:.3f}  dir={h1['direction']*100:.1f}%")
    print(f"  Wilcoxon W={h1['wilcoxon_W']:.0f}  p={h1['wilcoxon_p']:.4f}")
    print(f"  Bonferroni (alpha={BONFERRONI_ALPHA}): {'PASS' if h1['p'] < BONFERRONI_ALPHA else 'FAIL'}")
    print(f"  Gates (pre-reg): {'PASS' if h1['gates_passed'] else 'FAIL'}")
    if h1['d'] < 0:
        print(f"  ** DIRECTION REVERSED: S(C+) > S(C-), opposite of H1 **")
    print()

    # --- H2: secondary ---
    print("=" * 70)
    print("H2 (SECONDARY): S(C+) > S(H)  — replicate prior commit vs hedge")
    print("=" * 70)
    h2 = paired_test(S_H, S_Cp, "H", "C+")
    print(f"  n={h2['n']}  H mean={h2['a_mean']:.6f}  C+ mean={h2['b_mean']:.6f}  "
          f"ratio={h2['ratio']:.3f}x")
    print(f"  paired t={h2['t']:.3f}  p={h2['p']:.4f}  d={h2['d']:.3f}  "
          f"AUC={h2['auc']:.3f}  dir={h2['direction']*100:.1f}%")
    print(f"  Wilcoxon W={h2['wilcoxon_W']:.0f}  p={h2['wilcoxon_p']:.4f}")
    print(f"  Bonferroni: {'PASS' if h2['p'] < BONFERRONI_ALPHA else 'FAIL'}")
    print(f"  Gates: {'PASS' if h2['gates_passed'] else 'FAIL'}")
    print()

    # --- H3: tertiary (monotonic ordering) ---
    print("=" * 70)
    print("H3 (TERTIARY): S(H) < S(C+) < S(C-)  — monotonic ordering")
    print("=" * 70)
    fstat, fp = stats.friedmanchisquare(S_H, S_Cp, S_Cm)
    print(f"  Friedman: chi2={fstat:.3f}  p={fp:.4f}")
    # post-hoc Wilcoxon
    _, p_HC = stats.wilcoxon(S_H, S_Cm)
    _, p_HCp = stats.wilcoxon(S_H, S_Cp)
    _, p_CpCm = stats.wilcoxon(S_Cp, S_Cm)
    print(f"  Post-hoc Wilcoxon (Bonferroni alpha={BONFERRONI_ALPHA}):")
    print(f"    H vs C-:  p={p_HC:.4f} {'*' if p_HC < BONFERRONI_ALPHA else ''}")
    print(f"    H vs C+:  p={p_HCp:.4f} {'*' if p_HCp < BONFERRONI_ALPHA else ''}")
    print(f"    C+ vs C-: p={p_CpCm:.4f} {'*' if p_CpCm < BONFERRONI_ALPHA else ''}")
    # ordering per pair
    n_ordered = sum(1 for i in range(len(S_H))
                    if S_H[i] < S_Cp[i] < S_Cm[i])
    print(f"  Exact monotonic ordering: {n_ordered}/{len(S_H)} pairs "
          f"({n_ordered/len(S_H)*100:.1f}%)")
    print()

    # --- Sensitivity across thresholds ---
    print("=" * 70)
    print("Sensitivity analysis (H1: S(C-) vs S(C+) across thresholds)")
    print("=" * 70)
    print(f"{'threshold':>10} {'ratio':>8} {'p':>10} {'d':>8} {'AUC':>8}")
    for thr in THR_SENSITIVITY:
        sS_Cp = [compute_S(r["C_plus"]["c_delta_trajectory"], thr) for r in results]
        sS_Cm = [compute_S(r["C_minus"]["c_delta_trajectory"], thr) for r in results]
        out = paired_test(sS_Cp, sS_Cm)
        print(f"{thr:>10.3f} {out['ratio']:>8.2f} {out['p']:>10.4f} "
              f"{out['d']:>8.3f} {out['auc']:>8.3f}")
    print()

    # --- Length-controlled retest (if needed) ---
    if spread > 5.0:
        min_len = int(min(means))
        print("=" * 70)
        print(f"Length-controlled retest (trunc={min_len})")
        print("=" * 70)
        sS_H, sS_Cp, sS_Cm = [], [], []
        kept = 0
        for r in results:
            h = r["H"]["c_delta_trajectory"]
            cp = r["C_plus"]["c_delta_trajectory"]
            cm = r["C_minus"]["c_delta_trajectory"]
            if min(len(h), len(cp), len(cm)) < min_len:
                continue
            sS_H.append(compute_S(h[:min_len]))
            sS_Cp.append(compute_S(cp[:min_len]))
            sS_Cm.append(compute_S(cm[:min_len]))
            kept += 1
        print(f"  pairs surviving: {kept}/{len(results)}")
        if kept >= 3:
            tc = paired_test(sS_Cp, sS_Cm)
            print(f"  H1 length-controlled: ratio={tc['ratio']:.2f}x "
                  f"p={tc['p']:.4f} d={tc['d']:.3f} AUC={tc['auc']:.3f}")
            print(f"  Gates: {'PASS' if tc['gates_passed'] else 'FAIL'}")
        print()

    # --- Final verdict ---
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  H1 (S(C-)>S(C+)): {'SUPPORTED' if h1['gates_passed'] and h1['d'] > 0 else 'NOT supported'}")
    print(f"  H2 (S(C+)>S(H)):  {'SUPPORTED' if h2['gates_passed'] and h2['d'] > 0 else 'NOT supported'}")
    if h1['gates_passed'] and h1['d'] > 0 and h2['gates_passed'] and h2['d'] > 0:
        print("  => S is a confident-wrongness detector (H1) AND commitment detector (H2)")
    elif h2['gates_passed'] and h2['d'] > 0:
        print("  => S is only a commitment detector (not truth-sensitive)")
    elif h1['gates_passed'] and h1['d'] > 0:
        print("  => S differentiates correct from wrong but not hedge from commit (unexpected)")
    else:
        print("  => S did not survive in this experiment")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join("truthfulqa_results", "S_triplet_n30.json")
    if not os.path.exists(path):
        pilot = path.replace(".json", "_pilot.json")
        if os.path.exists(pilot):
            print(f"Main file not found, using pilot: {pilot}")
            path = pilot
        else:
            print(f"ERROR: Results file not found: {path}")
            sys.exit(1)
    analyze(path)
