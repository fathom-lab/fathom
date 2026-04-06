"""
make_executive_summary.py — one-page markdown summary from all analysis outputs
=================================================================================
Reads all the *_labels.json, *_horizon.json, *_features.json, *_predictor.json,
*_categories.json, *_horizon_perm.json produced by run_all_S_analyses.py
and writes a single executive summary.

Run AFTER run_all_S_analyses.py completes.

Usage:
    python make_executive_summary.py [results_file.json]
"""

import json
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(ROOT, "truthfulqa_results", "S_truthfulqa_n200.json")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt(v, fmt_spec=".3f"):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        try:
            return format(v, fmt_spec)
        except Exception:
            return str(v)
    return str(v)


def make_summary(data_path):
    stem = data_path.replace('.json', '')
    labels_data = load_json(stem + '_labels.json')
    horizon_data = load_json(stem + '_horizon.json')
    features_data = load_json(stem + '_features.json')
    categories_data = load_json(stem + '_categories.json')
    predictor_data = load_json(stem + '_predictor.json')
    perm_data = load_json(stem + '_horizon_perm.json')

    out = ["# S-Axis × TruthfulQA Natural Hallucination — Executive Summary",
           "",
           f"**Data:** `{data_path}`",
           ""]

    # Primary result
    if labels_data:
        counts = labels_data.get('label_counts', {})
        primary = labels_data.get('primary_test', {})
        out.append("## Primary pre-registered test (Welch's t-test)")
        out.append("")
        out.append(f"- **n_correct:** {counts.get('correct', 0)}")
        out.append(f"- **n_hallucinated:** {counts.get('hallucinated', 0)}")
        out.append(f"- **excluded:** {counts.get('ambiguous', 0)} ambiguous + "
                   f"{counts.get('unclassified', 0)} unclassified + "
                   f"{counts.get('refusal', 0)} refusal + "
                   f"{counts.get('too_short', 0)} too_short")
        out.append("")
        if primary:
            out.append(f"- **S(correct) mean:** {fmt(primary.get('correct_mean'), '.6f')}")
            out.append(f"- **S(hallucinated) mean:** {fmt(primary.get('hallucinated_mean'), '.6f')}")
            out.append(f"- **ratio:** {fmt(primary.get('ratio'))}")
            out.append(f"- **Welch t:** {fmt(primary.get('t'))}")
            out.append(f"- **p-value:** {fmt(primary.get('p'), '.4f')}")
            out.append(f"- **Cohen's d:** {fmt(primary.get('d'))}")
            out.append(f"- **AUC:** {fmt(primary.get('auc'))}")
            out.append(f"- **Mann-Whitney p:** {fmt(primary.get('mann_whitney_p'), '.4f')}")
            out.append(f"- **Gates passed:** {primary.get('gates_passed')}")
        out.append("")

    # Horizon
    if horizon_data:
        out.append("## Predictive horizon curve (stream-time detection)")
        out.append("")
        out.append("| N | AUC | d | n_correct | n_hallucinated |")
        out.append("|---|-----|---|-----------|----------------|")
        for e in horizon_data.get('curve', []):
            if e.get('AUC') is not None:
                out.append(f"| {e['N']} | {e['AUC']:.3f} | {fmt(e.get('d'))} | "
                           f"{e.get('n_correct', '-')} | {e.get('n_hallucinated', '-')} |")
        out.append("")
        horizons = horizon_data.get('horizons', {})
        out.append("**Earliest token crossing AUC threshold:**")
        for thr, N in horizons.items():
            out.append(f"- AUC ≥ {thr}: token {N if N is not None else 'NOT REACHED'}")
        out.append("")

    # Permutation test
    if perm_data:
        out.append("## Horizon permutation test (null distribution)")
        out.append("")
        out.append(f"- **observed max AUC:** {fmt(perm_data.get('observed_max_auc'))}")
        out.append(f"- **null max AUC p95:** {fmt(perm_data.get('null_max_auc_p95'))}")
        out.append(f"- **global permutation p-value:** {fmt(perm_data.get('global_p_value'), '.4f')}")
        out.append(f"- **n permutations:** {perm_data.get('n_permutations')}")
        out.append("")

    # Multi-feature classifier
    if predictor_data:
        out.append("## Cross-validated calibrated predictor (5-fold CV)")
        out.append("")
        out.append(f"- **n_samples:** {predictor_data.get('n_samples')}")
        out.append(f"- **n_hallucinated:** {predictor_data.get('n_hallucinated')}")
        out.append(f"- **n_features:** {predictor_data.get('n_features')}")
        cv = predictor_data.get('cv_full', {})
        if cv:
            out.append(f"- **CV AUC (all features):** {fmt(cv.get('mean'))} ± {fmt(cv.get('std'))}")
            ci = cv.get('ci_95', [None, None])
            out.append(f"- **95% CI:** [{fmt(ci[0])}, {fmt(ci[1])}]")
        out.append("")
        out.append("**Baseline comparisons:**")
        for name, cv_bl in predictor_data.get('baseline_comparisons', {}).items():
            if cv_bl:
                out.append(f"- {name}: {fmt(cv_bl.get('mean'))} ± {fmt(cv_bl.get('std'))}")
        out.append("")

    # Top features
    if features_data:
        out.append("## Top discriminating features")
        out.append("")
        ranking = features_data.get('single_feature_ranking', [])[:10]
        if ranking:
            out.append("| rank | feature | AUC_dir | d | p |")
            out.append("|------|---------|---------|---|---|")
            for i, r in enumerate(ranking, 1):
                out.append(f"| {i} | {r['name']} | {fmt(r['auc_directional'])} | "
                           f"{fmt(r['d'])} | {fmt(r['p'], '.4f')} |")
        out.append("")

    # Category breakdown
    if categories_data:
        out.append("## TruthfulQA categories × hallucination rate")
        out.append("")
        cats = sorted(categories_data.get('category_stats', []),
                      key=lambda x: -x.get('halluc_rate', 0))[:10]
        if cats:
            out.append("| category | n | halluc_rate | S_mean |")
            out.append("|----------|---|-------------|--------|")
            for c in cats:
                out.append(f"| {c['category']} | {c['n']} | {c['halluc_rate']*100:.1f}% | "
                           f"{fmt(c['S_mean'], '.6f')} |")
        out.append("")

    # Verdict
    out.append("## Verdict")
    out.append("")
    if labels_data and labels_data.get('primary_test'):
        primary = labels_data['primary_test']
        p = primary.get('p')
        d = primary.get('d')
        auc = primary.get('auc')
        if primary.get('gates_passed'):
            if d > 0:
                out.append("**H1 SUPPORTED**: S(hallucinated) > S(correct). S discriminates")
                out.append("natural hallucinations at the circuit level on Gemma-2-2B-IT n=200.")
            else:
                out.append("**H1 REVERSED**: S(correct) > S(hallucinated). Model is more")
                out.append("geometrically committed to truths than falsehoods.")
        else:
            out.append(f"**H1 NOT supported**: gates failed (p={fmt(p, '.4f')}, d={fmt(d)}, AUC={fmt(auc)})")
    out.append("")

    return "\n".join(out)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        pilot = path.replace('n200.json', 'pilot.json')
        if os.path.exists(pilot):
            path = pilot
        else:
            print(f"ERROR: {path} not found")
            sys.exit(1)
    summary = make_summary(path)
    out_path = path.replace('.json', '_SUMMARY.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary written: {out_path}")
    print()
    try:
        print(summary)
    except UnicodeEncodeError:
        # Windows console can't encode ≥ etc., write as ASCII fallback
        print(summary.encode('ascii', 'replace').decode('ascii'))
