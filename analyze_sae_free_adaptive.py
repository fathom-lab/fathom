"""
analyze_sae_free_adaptive.py — SAE-free S_early with adaptive threshold
===========================================================================
SAE-based C_delta values are ~0.005-0.015, so threshold τ=0.005 works.
SAE-free IPR values are 0-1 scaled, need different thresholds.

This analyzer:
 1. Finds the right threshold per-model (at a given percentile of
    the C_delta distribution)
 2. Recomputes S_early with the adaptive threshold
 3. Tests discrimination vs v2.2 labels

Usage:
    python analyze_sae_free_adaptive.py [sae_free_file.json]
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_S_range_at(traj, start, end, threshold):
    """Compute S on traj[start:end] at given threshold."""
    slice_ = traj[start:end]
    if not slice_: return 0.0
    spikes = [c for c in slice_ if c > threshold]
    if not spikes: return 0.0
    return max(slice_) / len(spikes)


def find_threshold(all_trajs, percentile=50):
    """Pick threshold at given percentile of all C_delta values."""
    vals = []
    for traj in all_trajs:
        vals.extend(traj)
    return float(np.percentile(vals, percentile))


def main():
    sae_free_path = sys.argv[1] if len(sys.argv) > 1 else None
    if sae_free_path is None:
        # Find most recent SAE-free result
        import glob
        candidates = glob.glob("truthfulqa_results/sae_free_ipr_*.json")
        candidates = [c for c in candidates if "checkpoint" not in c and "agreement" not in c]
        candidates.sort()
        if not candidates:
            print("No SAE-free files found")
            sys.exit(1)
        sae_free_path = candidates[-1]

    print("=" * 72)
    print("SAE-FREE IPR with ADAPTIVE THRESHOLD")
    print("=" * 72)
    print(f"Data: {sae_free_path}")

    with open(sae_free_path) as f:
        data = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a['v22_label'] for a in audit_data['details']}

    print(f"Model: {data.get('model')}")
    print(f"N items: {len(data['results'])}")
    print()

    # Gather trajectories and labels
    items = []
    for r in data['results']:
        idx = r['hf_index']
        lbl = audit.get(idx)
        if lbl in ('correct', 'correct_semantic'): label = 0
        elif lbl in ('hallucinated', 'hallucinated_semantic'): label = 1
        else: continue
        traj = r['generation']['c_delta_trajectory']
        if not traj: continue
        items.append({'label': label, 'traj': traj, 'hf_index': idx})

    n_c = sum(1 for i in items if i['label'] == 0)
    n_h = sum(1 for i in items if i['label'] == 1)
    print(f"Labeled: {n_c} correct, {n_h} hallucinated")

    if n_c < 3 or n_h < 3:
        print("[!] Insufficient labeled items")
        sys.exit(0)

    all_trajs = [i['traj'] for i in items]

    # Distribution summary
    all_vals = [v for t in all_trajs for v in t]
    print(f"\nC_delta distribution (IPR divergence):")
    print(f"  min={min(all_vals):+.4f} max={max(all_vals):+.4f}")
    print(f"  mean={np.mean(all_vals):+.4f} median={np.median(all_vals):+.4f}")
    print(f"  p25={np.percentile(all_vals,25):+.4f} p75={np.percentile(all_vals,75):+.4f}")
    print()

    # Test S_early at multiple thresholds
    print("-" * 72)
    print("S_early[0,7) DISCRIMINATION AT VARIOUS THRESHOLDS")
    print("-" * 72)
    print(f"{'percentile':>12} {'threshold':>12} {'d':>8} {'AUC':>8} {'p':>8}")

    best = None
    for pct in [10, 25, 40, 50, 60, 70, 80, 90]:
        thr = find_threshold(all_trajs, pct)
        S_c = [compute_S_range_at(i['traj'], 0, 7, thr)
               for i in items if i['label']==0]
        S_h = [compute_S_range_at(i['traj'], 0, 7, thr)
               for i in items if i['label']==1]
        S_c, S_h = np.array(S_c), np.array(S_h)
        if S_c.var(ddof=1) + S_h.var(ddof=1) < 1e-12: continue
        _, p = stats.ttest_ind(S_c, S_h, equal_var=False)
        pool = np.sqrt((S_c.var(ddof=1) + S_h.var(ddof=1))/2)
        d = (S_h.mean() - S_c.mean())/pool if pool>0 else 0
        auc = roc_auc_score([0]*len(S_c)+[1]*len(S_h), list(S_c)+list(S_h))
        print(f"   p{pct:>2}        {thr:>12.5f} {d:>+8.3f} {auc:>8.3f} {p:>8.4f}")
        if best is None or abs(d) > abs(best['d']):
            best = {'pct': pct, 'thr': thr, 'd': d, 'auc': auc, 'p': p}

    print(f"\nBest threshold: p{best['pct']} (thr={best['thr']:.5f}): "
          f"d={best['d']:+.3f}, AUC={best['auc']:.3f}, p={best['p']:.4f}")

    # Also compute max-alone and mean-alone for window [0,7)
    print()
    print("-" * 72)
    print("ALTERNATIVE METRICS (window [0,7))")
    print("-" * 72)
    print(f"{'metric':<20} {'d':>8} {'AUC':>8} {'p':>8}")
    for name, fn in [
        ("max early", lambda t: max(t[:7]) if t[:7] else 0),
        ("mean early", lambda t: float(np.mean(t[:7])) if t[:7] else 0),
        ("max[:7] - min[:7]", lambda t: max(t[:7]) - min(t[:7]) if t[:7] else 0),
    ]:
        c_vals = [fn(i['traj']) for i in items if i['label']==0]
        h_vals = [fn(i['traj']) for i in items if i['label']==1]
        if np.var(c_vals, ddof=1) + np.var(h_vals, ddof=1) < 1e-12: continue
        _, p = stats.ttest_ind(c_vals, h_vals, equal_var=False)
        pool = np.sqrt((np.var(c_vals,ddof=1)+np.var(h_vals,ddof=1))/2)
        d = (np.mean(h_vals)-np.mean(c_vals))/pool if pool>0 else 0
        auc = roc_auc_score([0]*len(c_vals)+[1]*len(h_vals), list(c_vals)+list(h_vals))
        print(f"{name:<20} {d:>+8.3f} {auc:>8.3f} {p:>8.4f}")

    # Save
    out = {
        'source': sae_free_path,
        'model': data.get('model'),
        'n_correct': n_c,
        'n_hallucinated': n_h,
        'adaptive_threshold': best,
    }
    out_path = sae_free_path.replace(".json", "_adaptive.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
