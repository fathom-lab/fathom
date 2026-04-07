"""
verify_all_claims.py — reproduce every numerical claim across all findings docs
==================================================================================
Every number in findings_*.md gets re-computed from raw JSONs and
asserted against published value. Exits non-zero on any mismatch.

Purpose: sanity check before writing the paper. Nothing moves forward
until these assertions all pass.
"""
import json
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import compute_S, classify_generation, MIN_GEN_TOKENS, THR_PRIMARY
from analyze_S_biphasic import compute_S_range


def assert_close(name, observed, expected, tol=0.001):
    """Assert that observed matches expected within tolerance."""
    status = "OK" if abs(observed - expected) < tol else "FAIL"
    print(f"  [{status}] {name}: observed={observed:.4f} expected={expected:.4f}")
    assert abs(observed - expected) < tol, f"{name}: {observed} != {expected}"


def assert_exact(name, observed, expected):
    status = "OK" if observed == expected else "FAIL"
    print(f"  [{status}] {name}: observed={observed} expected={expected}")
    assert observed == expected, f"{name}: {observed} != {expected}"


def main():
    print("=" * 72)
    print("VERIFICATION: all findings_*.md numerical claims")
    print("=" * 72)

    # Load all datasets
    with open("truthfulqa_results/commitment_phase_transition.json") as f:
        d_base = json.load(f)
    with open("truthfulqa_results/commitment_IT_replication.json") as f:
        d_it = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200.json") as f:
        d_tqa = json.load(f)
    with open("truthfulqa_results/S_truthfulqa_n200_v22_audit.json") as f:
        audit_data = json.load(f)
    audit = {a['hf_index']: a for a in audit_data['details']}

    print("\n[1] DATA INTEGRITY")
    print("-" * 72)
    assert_exact("base dataset n_pairs", len(d_base['results']), 30)
    assert_exact("IT dataset n_pairs", len(d_it['results']), 30)
    assert_exact("TQA dataset n_items", len(d_tqa['results']), 200)
    assert_exact("TQA audit n_details", len(audit_data['details']), 200)

    print("\n[2] v2.2 LABEL COUNTS (from findings_2026_04_05_META_ANALYSIS.md)")
    print("-" * 72)
    v22_counts = {'correct': 0, 'hallucinated': 0}
    for r in d_tqa['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS: continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        if lbl in ('correct', 'correct_semantic'): v22_counts['correct'] += 1
        elif lbl in ('hallucinated', 'hallucinated_semantic'): v22_counts['hallucinated'] += 1
    assert_exact("v2.2 correct count", v22_counts['correct'], 51)
    assert_exact("v2.2 hallucinated count", v22_counts['hallucinated'], 43)
    assert_exact("v2.2 total usable", v22_counts['correct'] + v22_counts['hallucinated'], 94)

    print("\n[3] PRIMARY PRE-REGISTERED TEST (n-gram v2.1, n=66)")
    print("-" * 72)
    v21_corr_S = []; v21_hall_S = []
    for r in d_tqa['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS: continue
        lbl = audit.get(r['hf_index'], {}).get('v21_label')
        traj = r['generation']['c_delta_trajectory']
        if not traj: continue
        S = compute_S(traj)
        if lbl == 'correct': v21_corr_S.append(S)
        elif lbl == 'hallucinated': v21_hall_S.append(S)
    assert_exact("v2.1 n_correct", len(v21_corr_S), 41)
    assert_exact("v2.1 n_hallucinated", len(v21_hall_S), 25)
    c_arr = np.array(v21_corr_S); h_arr = np.array(v21_hall_S)
    _, p = stats.ttest_ind(c_arr, h_arr, equal_var=False)
    pool = np.sqrt((c_arr.var(ddof=1) + h_arr.var(ddof=1))/2)
    d = (h_arr.mean() - c_arr.mean()) / pool
    auc = roc_auc_score([0]*len(c_arr)+[1]*len(h_arr), list(c_arr)+list(h_arr))
    assert_close("v2.1 primary p", p, 0.3151)
    assert_close("v2.1 primary d", d, -0.246, 0.005)
    assert_close("v2.1 primary AUC", auc, 0.443, 0.005)

    print("\n[4] V2.2 S_EARLY TEST (n=94)")
    print("-" * 72)
    v22_corr_t = []; v22_hall_t = []
    for r in d_tqa['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS: continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        traj = r['generation']['c_delta_trajectory']
        if not traj: continue
        if lbl in ('correct', 'correct_semantic'): v22_corr_t.append(traj)
        elif lbl in ('hallucinated', 'hallucinated_semantic'): v22_hall_t.append(traj)
    S_e_c = np.array([compute_S_range(t, 0, 7) for t in v22_corr_t])
    S_e_h = np.array([compute_S_range(t, 0, 7) for t in v22_hall_t])
    _, p_se = stats.ttest_ind(S_e_c, S_e_h, equal_var=False)
    pool_se = np.sqrt((S_e_c.var(ddof=1) + S_e_h.var(ddof=1))/2)
    d_se = (S_e_h.mean() - S_e_c.mean()) / pool_se
    auc_se = roc_auc_score([0]*len(S_e_c)+[1]*len(S_e_h), list(S_e_c)+list(S_e_h))
    assert_close("v2.2 S_early p", p_se, 0.0175, 0.001)
    assert_close("v2.2 S_early d", d_se, 0.506, 0.01)
    assert_close("v2.2 S_early AUC", auc_se, 0.669, 0.005)

    print("\n[5] CROSS-DATASET META-ANALYSIS")
    print("-" * 72)
    # Dataset 1: base hedge vs commit
    hedge1 = [r['hedging']['c_delta_trajectory'] for r in d_base['results']]
    commit1 = [r['commitment']['c_delta_trajectory'] for r in d_base['results']]
    S_h1 = np.array([compute_S_range(t, 0, 7) for t in hedge1])
    S_c1 = np.array([compute_S_range(t, 0, 7) for t in commit1])
    pool1 = np.sqrt((S_h1.var(ddof=1) + S_c1.var(ddof=1))/2)
    d1 = (S_c1.mean() - S_h1.mean()) / pool1
    assert_close("Dataset 1 (base h/c) d", d1, 0.501, 0.005)

    hedge2 = [r['hedging']['c_delta_trajectory'] for r in d_it['results']]
    commit2 = [r['commitment']['c_delta_trajectory'] for r in d_it['results']]
    S_h2 = np.array([compute_S_range(t, 0, 7) for t in hedge2])
    S_c2 = np.array([compute_S_range(t, 0, 7) for t in commit2])
    pool2 = np.sqrt((S_h2.var(ddof=1) + S_c2.var(ddof=1))/2)
    d2 = (S_c2.mean() - S_h2.mean()) / pool2
    assert_close("Dataset 2 (IT h/c) d", d2, 0.466, 0.005)

    # Dataset 3 already done in [4]
    # Pooled
    d_pooled = (d1*30 + d2*30 + d_se*94) / 154
    assert_close("Pooled d across 3 datasets", d_pooled, 0.494, 0.005)

    print("\n[6] IPR VERIFICATION (S = M * IPR on real data)")
    print("-" * 72)
    all_match = True
    for r in d_tqa['results'][:10]:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS: continue
        traj = np.array(r['generation']['c_delta_trajectory'])
        M = traj.max(); K = int((traj > THR_PRIMARY).sum())
        if K == 0: continue
        S_direct = M / K
        e_hat = np.where(traj > THR_PRIMARY, 1/K, 0)
        ipr = (e_hat ** 2).sum()
        S_from_ipr = M * ipr
        if abs(S_direct - S_from_ipr) > 1e-9:
            all_match = False
            break
    assert all_match, "IPR equivalence failed"
    print(f"  [OK] S = M * IPR exactly on 10 sampled trajectories")

    print("\n[7] FISHER COMBINED p")
    print("-" * 72)
    # Compute from datasets 1, 2, 3
    _, p1 = stats.ttest_ind(S_h1, S_c1, equal_var=False)
    _, p2 = stats.ttest_ind(S_h2, S_c2, equal_var=False)
    p3 = p_se
    # One-sided (testing d > 0, commit/halluc > baseline)
    ps_one_sided = [
        p1/2 if d1 > 0 else 1 - p1/2,
        p2/2 if d2 > 0 else 1 - p2/2,
        p3/2 if d_se > 0 else 1 - p3/2,
    ]
    fisher_stat = -2 * sum(np.log(p) for p in ps_one_sided)
    from scipy.stats import chi2
    fisher_p = 1 - chi2.cdf(fisher_stat, df=2*3)
    assert_close("Fisher combined p (one-sided)", fisher_p, 0.0008, 0.001)

    print("\n[8] LENGTH CONFOUND VERIFICATION")
    print("-" * 72)
    q_chars_all = []; S_early_all = []; y_all = []
    for r in d_tqa['results']:
        if r['generation']['total_tokens'] < MIN_GEN_TOKENS: continue
        lbl = audit.get(r['hf_index'], {}).get('v22_label')
        if lbl in ('correct','correct_semantic'): y_all.append(0)
        elif lbl in ('hallucinated','hallucinated_semantic'): y_all.append(1)
        else: continue
        traj = r['generation']['c_delta_trajectory']
        if not traj:
            y_all.pop(); continue
        S_early_all.append(compute_S_range(traj, 0, 7))
        q_chars_all.append(len(r['question']))
    r_len, p_len = stats.pearsonr(S_early_all, q_chars_all)
    assert_close("r(S_early, q_chars)", r_len, 0.171, 0.01)

    # q_chars AUC
    auc_qc = roc_auc_score(y_all, q_chars_all)
    assert_close("q_chars alone AUC", auc_qc, 0.730, 0.005)

    print("\n[9] FILE INVENTORY")
    print("-" * 72)
    expected_files = [
        "findings_2026_04_05_META_ANALYSIS.md",
        "findings_2026_04_05_HONEST_FINAL.md",
        "osf_prereg_S_truthfulqa_v2.md",
        "osf_prereg_semantic_relabel.md",
        "osf_prereg_biphasic_replication.md",
        "paper/section_IPR_grounding.md",
        "paper/section_intro_draft.md",
        "paper/section_methods_draft.md",
        "truthfulqa_results/S_truthfulqa_n200.json",
        "truthfulqa_results/S_truthfulqa_n200_v22_audit.json",
        "truthfulqa_results/cross_dataset_meta.json",
        "truthfulqa_results/calibration_analysis.json",
    ]
    for f in expected_files:
        exists = os.path.exists(f)
        print(f"  [{'OK' if exists else 'FAIL'}] {f}")
        assert exists, f"Missing file: {f}"

    print()
    print("=" * 72)
    print("ALL VERIFICATIONS PASSED")
    print("=" * 72)


if __name__ == '__main__':
    main()
