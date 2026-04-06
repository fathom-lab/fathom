"""
verify_s_axis.py — reproduce all claims in findings_2026_04_05_S_axis_thorough.md
=================================================================================

Runs 14 statistical assertions. Exits non-zero on any mismatch.
No GPU needed — reads existing JSON results.

Usage:
    python verify_s_axis.py
"""
import json
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

THR = 0.005


def compute_S(traj, threshold=THR):
    if not traj:
        return 0.0
    spikes = [c for c in traj if c > threshold]
    if len(spikes) == 0:
        return 0.0
    return max(traj) / len(spikes)


def get_layer_traj(layer_dict):
    return np.array([v for _, v in sorted(layer_dict.items(), key=lambda kv: int(kv[0]))])


def S_layer(tr, thr=0.002):
    tr = np.abs(tr)
    sp = tr[tr > thr]
    return tr.max() / len(sp) if len(sp) > 0 else 0.0


def main():
    print("=" * 70)
    print("S-axis verification pass (findings_2026_04_05_S_axis_thorough.md)")
    print("=" * 70)

    with open('truthfulqa_results/commitment_IT_replication.json') as f:
        dIT = json.load(f)
    with open('truthfulqa_results/commitment_phase_transition.json') as f:
        dBASE = json.load(f)
    with open('truthfulqa_results/coherence_n200_final.json') as f:
        d200 = json.load(f)

    assert len(dIT['results']) == 30
    assert len(dBASE['results']) == 30
    assert len(d200['results']) == 200
    print(f"[OK 1/14] Data loaded: IT={len(dIT['results'])} BASE={len(dBASE['results'])} n200={len(d200['results'])}")
    print(f"[NOTE]    dBASE JSON mislabels model as IT; script used base (see findings doc)")

    # 2. S formula
    r0 = dIT['results'][0]
    h0 = r0['hedging']['c_delta_trajectory']
    h0_manual = max(h0) / sum(1 for c in h0 if c > 0.005)
    h0_func = compute_S(h0)
    assert abs(h0_manual - h0_func) < 1e-12
    print(f"[OK 2/14] S formula: max/count verified on first pair = {h0_func:.6f}")

    # 3. Primary IT result
    hS = np.array([compute_S(r['hedging']['c_delta_trajectory']) for r in dIT['results']])
    cS = np.array([compute_S(r['commitment']['c_delta_trajectory']) for r in dIT['results']])
    t, p = stats.ttest_rel(hS, cS)
    diff = cS - hS
    d_eff = np.mean(diff) / np.std(diff, ddof=1)
    auc = roc_auc_score([0] * 30 + [1] * 30, list(hS) + list(cS))
    ratio = np.mean(cS) / np.mean(hS)
    dir_acc = np.mean(cS > hS)
    assert abs(ratio - 1.946) < 0.01
    assert abs(p - 0.0021) < 0.001
    assert abs(d_eff - 0.615) < 0.01
    assert abs(auc - 0.769) < 0.01
    assert abs(dir_acc - 0.867) < 0.01
    print(f"[OK 3/14] Primary IT: ratio={ratio:.3f}x p={p:.4f} d={d_eff:.3f} AUC={auc:.3f}")

    # 4. Wilcoxon
    w_stat, w_p = stats.wilcoxon(hS, cS)
    assert w_p < 0.001
    print(f"[OK 4/14] Wilcoxon: W={w_stat:.0f} p={w_p:.6f}")

    # 5. Bootstrap CI
    np.random.seed(42)
    boot = [np.mean((cS - hS)[np.random.choice(30, 30, replace=True)]) for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    assert ci_lo > 0
    print(f"[OK 5/14] Bootstrap 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}] excludes 0")

    # 6. LOO
    loo_ps = []
    for i in range(30):
        mask = np.arange(30) != i
        _, p_loo = stats.ttest_rel(hS[mask], cS[mask])
        loo_ps.append(p_loo)
    assert all(p < 0.05 for p in loo_ps)
    print(f"[OK 6/14] LOO: 30/30 refits p<0.05, range [{min(loo_ps):.4f}, {max(loo_ps):.4f}]")

    # 7. Length confound
    h_n = np.array([len(r['hedging']['c_delta_trajectory']) for r in dIT['results']])
    c_n = np.array([len(r['commitment']['c_delta_trajectory']) for r in dIT['results']])
    all_S = np.concatenate([hS, cS])
    all_n = np.concatenate([h_n, c_n])
    r_len, _ = stats.pearsonr(all_S, all_n)
    assert abs(r_len + 0.898) < 0.01
    print(f"[OK 7/14] Length confound: r(S, n_tokens) = {r_len:.3f}")

    # 8. Length-controlled trunc=25
    TRUNC = 25
    hSt, cSt = [], []
    for r in dIT['results']:
        h, c = r['hedging']['c_delta_trajectory'], r['commitment']['c_delta_trajectory']
        if len(h) >= TRUNC and len(c) >= TRUNC:
            hSt.append(compute_S(h[:TRUNC]))
            cSt.append(compute_S(c[:TRUNC]))
    hSt, cSt = np.array(hSt), np.array(cSt)
    assert len(hSt) == 20
    _, p_t = stats.ttest_rel(hSt, cSt)
    d_t = np.mean(cSt - hSt) / np.std(cSt - hSt, ddof=1)
    auc_t = roc_auc_score([0] * 20 + [1] * 20, list(hSt) + list(cSt))
    ratio_t = np.mean(cSt) / np.mean(hSt)
    assert p_t < 0.001
    assert d_t > 1.0
    assert auc_t > 0.8
    print(f"[OK 8/14] Trunc=25 (n=20): ratio={ratio_t:.2f}x p={p_t:.4f} d={d_t:.3f} AUC={auc_t:.3f}")

    # 9. Threshold sweep (full length)
    thresholds = [0.003, 0.005, 0.008, 0.010]
    thr_results = []
    for thr in thresholds:
        hS_ = np.array([compute_S(r['hedging']['c_delta_trajectory'], thr) for r in dIT['results']])
        cS_ = np.array([compute_S(r['commitment']['c_delta_trajectory'], thr) for r in dIT['results']])
        _, p_ = stats.ttest_rel(hS_, cS_)
        thr_results.append(p_)
    assert all(p < 0.01 for p in thr_results)
    print(f"[OK 9/14] Threshold sweep (full length): all {len(thresholds)} thresholds p<0.01")

    # 10. Orthogonality
    Ks, Cs, CDs, Ss = [], [], [], []
    for r in d200['results']:
        for side in ['correct', 'hallucinated']:
            v = r[side]
            Ks.append(v['K'])
            Cs.append(v['C'])
            CDs.append(v['C_delta'])
            Ss.append(S_layer(get_layer_traj(v['layer_coherence'])))
    Ks, Cs, CDs, Ss = map(np.array, [Ks, Cs, CDs, Ss])
    assert len(Ks) == 400
    pairs = [('K', 'S', Ks, Ss), ('C', 'S', Cs, Ss), ('S', 'C_delta', Ss, CDs),
             ('K', 'C', Ks, Cs), ('K', 'C_delta', Ks, CDs), ('C', 'C_delta', Cs, CDs)]
    max_r = 0.0
    for a, b, x, y in pairs:
        r_val, _ = stats.pearsonr(x, y)
        max_r = max(max_r, abs(r_val))
        assert abs(r_val) < 0.30, f"{a} vs {b} r={r_val}"
    print(f"[OK 10/14] Orthogonality: all 6 pairs |r|<0.30 (max |r|={max_r:.3f})")

    # 11. Layer-S discrimination
    S_corr = np.array([S_layer(get_layer_traj(r['correct']['layer_coherence'])) for r in d200['results']])
    S_hall = np.array([S_layer(get_layer_traj(r['hallucinated']['layer_coherence'])) for r in d200['results']])
    _, p_200 = stats.ttest_rel(S_corr, S_hall)
    d_200 = np.mean(S_hall - S_corr) / np.std(S_hall - S_corr, ddof=1)
    auc_200 = roc_auc_score([0] * 200 + [1] * 200, list(S_corr) + list(S_hall))
    assert p_200 < 0.001
    assert abs(d_200 - 0.343) < 0.02
    assert abs(auc_200 - 0.582) < 0.02
    print(f"[OK 11/14] Layer-S n=200: p={p_200:.2e} d={d_200:.3f} AUC={auc_200:.3f}")

    # 12. Reference metrics on n=200
    K_corr = np.array([r['correct']['K'] for r in d200['results']])
    K_hall = np.array([r['hallucinated']['K'] for r in d200['results']])
    _, pK = stats.ttest_rel(K_corr, K_hall)
    assert pK > 0.05  # K is NULL on n=200 TruthfulQA
    auc_K = roc_auc_score([0] * 200 + [1] * 200, list(K_corr) + list(K_hall))
    print(f"[OK 12/14] K null on n=200 (confirms prior finding): p={pK:.3f} AUC={auc_K:.3f}")

    # 13. C_delta on n=200 matches prior
    CD_corr = np.array([r['correct']['C_delta'] for r in d200['results']])
    CD_hall = np.array([r['hallucinated']['C_delta'] for r in d200['results']])
    _, pCD = stats.ttest_rel(CD_corr, CD_hall)
    auc_CD = roc_auc_score([0] * 200 + [1] * 200, list(CD_corr) + list(CD_hall))
    assert pCD < 0.001
    print(f"[OK 13/14] C_delta n=200: p={pCD:.2e} AUC={auc_CD:.3f}")

    # 14. Layer-S beats C_delta on same data
    assert auc_200 > auc_CD
    print(f"[OK 14/14] Layer-S AUC ({auc_200:.3f}) > C_delta AUC ({auc_CD:.3f}) on same data")

    print()
    print("=" * 70)
    print("ALL 14 ASSERTIONS PASSED")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
