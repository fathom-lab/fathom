#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
"""
RLHF DEPTH SUPPRESSION — SAE Attribution Method
=================================================
The definitive test. Uses the SAME circuit-tracer SAE method that produced
K=1.0302 (p=0.000051) on Gemma-2-2B base — now run on Gemma-2-2B-it.

Compares against existing base results in results_power_replication.json.

If K_base > K_instruct: RLHF suppresses depth (confirmed).
If K_base <= K_instruct: hypothesis not supported at this scale.

Usage:
    python experiment_rlhf_sae.py --smoke     # 3 pairs
    python experiment_rlhf_sae.py             # full 30 pairs
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

# ── JSON Encoder ─────────────────────────────────────────────────────────────
import json as _json

class NumpyEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def json_dump(obj, f, **kwargs):
    kwargs.setdefault('indent', 2)
    _json.dump(obj, f, cls=NumpyEncoder, **kwargs)

# ── Configuration ────────────────────────────────────────────────────────────

INSTRUCT_MODEL = "google/gemma-2-2b-it"
TRANSCODER_SET = "gemma"  # same transcoders as base — same architecture
N_LAYERS = 26
BAND_BOUNDARIES = (8, 16)

# Base model reference (from results_power_replication.json)
BASE_REFERENCE = {
    "model": "google/gemma-2-2b",
    "surface_mean": 8.3592,
    "insight_mean": 8.1025,
    "delta": 0.2568,
    "ratio": 1.0302,
    "p_value": 0.000051,
    "n_pairs": 30,
}

SIGNIFICANCE_THRESHOLD = 0.05

BASE_DIR = Path(__file__).parent
PAIRS_FILE = BASE_DIR / ".." / "research" / "experiment_12_power_pairs.json"
BASE_RESULTS_FILE = BASE_DIR / "results_power_replication.json"
INSTRUCT_RESULTS_FILE = BASE_DIR / "gemma_it_experiment_results.json"
COMPARISON_FILE = BASE_DIR / "rlhf_sae_comparison.json"
REPORT_FILE = BASE_DIR / "RLHF_SAE_REPORT.md"

# ── Load Pairs ───────────────────────────────────────────────────────────────

def load_pairs():
    for candidate in [PAIRS_FILE, BASE_DIR / "experiment_12_power_pairs.json"]:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            if isinstance(data, dict) and "pairs" in data:
                raw_pairs = data["pairs"]
            elif isinstance(data, list):
                raw_pairs = data
            else:
                raise ValueError(f"Unexpected format in {candidate}")
            normalized = []
            for p in raw_pairs:
                s = p.get("surface", "")
                i = p.get("insight", "")
                if isinstance(s, str):
                    normalized.append({
                        "id": p.get("topic", p.get("id", "unknown")),
                        "surface": {"prompt": s, "target": p.get("surface_target", "")},
                        "insight": {"prompt": i, "target": p.get("insight_target", "")},
                    })
                else:
                    normalized.append({
                        "id": p.get("id", p.get("topic", "unknown")),
                        "surface": {"prompt": s.get("prompt", ""), "target": s.get("target", "")},
                        "insight": {"prompt": i.get("prompt", ""), "target": i.get("target", "")},
                    })
            return normalized
    raise FileNotFoundError("Pairs file not found")


def load_base_results():
    """Load existing base model results keyed by topic."""
    if not BASE_RESULTS_FILE.exists():
        print(f"[WARN] Base results not found: {BASE_RESULTS_FILE}")
        return {}
    with open(BASE_RESULTS_FILE) as f:
        data = json.load(f)
    results = {}
    for pair in data.get("pairs", []):
        results[pair["topic"]] = {
            "surface_depth": pair["surface_mean_layer"],
            "insight_depth": pair["insight_mean_layer"],
            "delta": pair["delta"],
            "surface_n": pair.get("surface_n"),
            "insight_n": pair.get("insight_n"),
        }
    return results


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model():
    """Load Gemma-2-2B-it with circuit-tracer transcoders."""
    import torch
    from circuit_tracer import ReplacementModel

    print(f"[MODEL] Loading {INSTRUCT_MODEL}...")
    print(f"[MODEL] Transcoder set: {TRANSCODER_SET} (trained on base, applied to instruct)")
    print(f"[MODEL] Layers: {N_LAYERS}, Bands: {BAND_BOUNDARIES}")

    model = ReplacementModel.from_pretrained(
        INSTRUCT_MODEL,
        TRANSCODER_SET,
        dtype=torch.bfloat16,
        backend="transformerlens",
    )

    print(f"[MODEL] Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


# ── Depth Scoring (same as original experiment) ─────────────────────────────

def score_depth(model, prompt, target, max_features=500):
    import torch
    from circuit_tracer import attribute

    full_prompt = prompt + target

    try:
        graph = attribute(
            full_prompt,
            model,
            attribution_targets=[target],
            max_feature_nodes=max_features,
            batch_size=16,
        )

        layers = graph.active_features[:, 0].float().cpu().numpy()

        if len(layers) == 0:
            print(f"[WARN] No features activated for: {prompt[:50]}...")
            return None

        depth = float(np.mean(layers))
        n_features = len(layers)

        layer_counts = {}
        for l in layers:
            k = int(l)
            layer_counts[k] = layer_counts.get(k, 0) + 1

        probs = np.array(list(layer_counts.values())) / n_features
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        early = sum(1 for l in layers if l < BAND_BOUNDARIES[0])
        mid = sum(1 for l in layers if BAND_BOUNDARIES[0] <= l < BAND_BOUNDARIES[1])
        late = sum(1 for l in layers if l >= BAND_BOUNDARIES[1])

        return {
            "depth": depth,
            "n_features": n_features,
            "entropy": entropy,
            "early_ratio": early / n_features,
            "mid_ratio": mid / n_features,
            "late_ratio": late / n_features,
        }

    except Exception as e:
        print(f"[ERROR] Attribution failed: {e}")
        return None


# ── Experiment Runner ────────────────────────────────────────────────────────

def run_experiment(model, pairs, smoke_test=False):
    if smoke_test:
        pairs = pairs[:3]
        print("[SMOKE TEST] Running 3 pairs only")

    results = []
    n = len(pairs)

    for i, pair in enumerate(pairs):
        pair_id = pair["id"]
        print(f"\n[PAIR {i+1}/{n}] {pair_id}")

        sp = pair["surface"]["prompt"]
        st = pair["surface"]["target"]
        ip = pair["insight"]["prompt"]
        it = pair["insight"]["target"]

        print(f"  Surface: ...{sp[-40:]} -> '{st}'")
        print(f"  Insight: ...{ip[-40:]} -> '{it}'")

        t0 = time.time()
        surface_result = score_depth(model, sp, st)
        t_surface = time.time() - t0

        t0 = time.time()
        insight_result = score_depth(model, ip, it)
        t_insight = time.time() - t0

        if surface_result is None or insight_result is None:
            print(f"  [SKIP] Attribution failed for pair {pair_id}")
            continue

        delta = surface_result["depth"] - insight_result["depth"]
        direction = "surface_deeper" if delta > 0 else "insight_deeper"

        result = {
            "pair_id": pair_id,
            "pair_index": i,
            "surface_depth": surface_result["depth"],
            "surface_features": surface_result["n_features"],
            "surface_entropy": surface_result["entropy"],
            "insight_depth": insight_result["depth"],
            "insight_features": insight_result["n_features"],
            "insight_entropy": insight_result["entropy"],
            "delta": delta,
            "direction": direction,
            "time_surface_s": round(t_surface, 1),
            "time_insight_s": round(t_insight, 1),
        }

        results.append(result)
        print(f"  Surface: {surface_result['depth']:.4f} ({surface_result['n_features']} features)")
        print(f"  Insight: {insight_result['depth']:.4f} ({insight_result['n_features']} features)")
        print(f"  Delta: {delta:+.4f} ({direction}) [{t_surface + t_insight:.1f}s]")

    return results


# ── Comparison Analysis ──────────────────────────────────────────────────────

def compare_with_base(instruct_results, base_results):
    """Compare instruct results against base results pair by pair."""
    inst_by_id = {r["pair_id"]: r for r in instruct_results}
    common = sorted(set(inst_by_id.keys()) & set(base_results.keys()))

    if len(common) < 3:
        print("[WARN] Too few common pairs")
        return None

    n = len(common)

    base_surface = np.array([base_results[pid]["surface_depth"] for pid in common])
    base_insight = np.array([base_results[pid]["insight_depth"] for pid in common])
    base_deltas = base_surface - base_insight

    inst_surface = np.array([inst_by_id[pid]["surface_depth"] for pid in common])
    inst_insight = np.array([inst_by_id[pid]["insight_depth"] for pid in common])
    inst_deltas = inst_surface - inst_insight

    # K ratios
    K_base = float(np.mean(base_surface)) / float(np.mean(base_insight))
    K_inst = float(np.mean(inst_surface)) / float(np.mean(inst_insight))
    K_delta = K_base - K_inst

    # Paired test on deltas
    delta_diff = base_deltas - inst_deltas
    t_stat, p_value = stats.ttest_1samp(delta_diff, 0, alternative="greater")

    # Also test each model independently
    _, p_base = stats.ttest_rel(base_surface, base_insight, alternative="greater")
    _, p_inst = stats.ttest_rel(inst_surface, inst_insight, alternative="greater")

    # Direction counts
    base_pct = sum(1 for d in base_deltas if d > 0) / n * 100
    inst_pct = sum(1 for d in inst_deltas if d > 0) / n * 100

    # Effect size
    dd_mean = float(np.mean(delta_diff))
    dd_std = float(np.std(delta_diff, ddof=1))
    cohens_d = dd_mean / dd_std if dd_std > 0 else 0

    # Verdict
    if p_value < 0.01 and K_delta > 0:
        verdict = "CONFIRMED: RLHF significantly suppresses reasoning depth (SAE method)"
    elif p_value < 0.05 and K_delta > 0:
        verdict = "SUPPORTED: RLHF reduces depth (SAE method, p < 0.05)"
    elif K_delta > 0 and p_value < 0.1:
        verdict = "TREND: Direction supports hypothesis, marginal significance"
    elif K_delta > 0:
        verdict = "DIRECTION CORRECT: Base shows more depth, but not significant"
    elif K_delta < 0:
        verdict = "REVERSED: Instruct model shows more depth differentiation"
    else:
        verdict = "NULL: No difference detected"

    return {
        "method": "SAE_attribution",
        "base_model": "google/gemma-2-2b",
        "instruct_model": INSTRUCT_MODEL,
        "n_pairs": n,
        "timestamp": datetime.now().isoformat(),

        # K ratios
        "K_base": round(K_base, 6),
        "K_instruct": round(K_inst, 6),
        "K_delta": round(K_delta, 6),

        # Means
        "base_surface_mean": round(float(np.mean(base_surface)), 4),
        "base_insight_mean": round(float(np.mean(base_insight)), 4),
        "base_delta_mean": round(float(np.mean(base_deltas)), 4),
        "inst_surface_mean": round(float(np.mean(inst_surface)), 4),
        "inst_insight_mean": round(float(np.mean(inst_insight)), 4),
        "inst_delta_mean": round(float(np.mean(inst_deltas)), 4),

        # Independent tests
        "base_p_value": round(float(p_base), 6),
        "inst_p_value": round(float(p_inst), 6),

        # Direction
        "base_pct_expected": round(base_pct, 1),
        "inst_pct_expected": round(inst_pct, 1),

        # Paired comparison
        "suppression_t": round(float(t_stat), 4),
        "suppression_p": round(float(p_value), 6),
        "significant": p_value < SIGNIFICANCE_THRESHOLD,
        "cohens_d": round(cohens_d, 4),

        # Verdict
        "verdict": verdict,

        # Reference
        "base_reference": BASE_REFERENCE,
    }


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report(instruct_results, comparison):
    c = comparison
    lines = [
        "# RLHF DEPTH SUPPRESSION — SAE Attribution Method",
        "",
        "## The Definitive Test",
        "",
        "Same method (circuit-tracer SAE attribution) that produced K=1.0302 (p=0.000051)",
        f"on Gemma-2-2B base, now run on {INSTRUCT_MODEL}.",
        "",
        "---",
        "",
        "## Results",
        "",
        f"| | Base (gemma-2-2b) | Instruct (gemma-2-2b-it) |",
        f"|--|---|---|",
        f"| **K ratio** | **{c['K_base']:.4f}** | **{c['K_instruct']:.4f}** |",
        f"| Surface mean | {c['base_surface_mean']:.4f} | {c['inst_surface_mean']:.4f} |",
        f"| Insight mean | {c['base_insight_mean']:.4f} | {c['inst_insight_mean']:.4f} |",
        f"| Delta mean | {c['base_delta_mean']:+.4f} | {c['inst_delta_mean']:+.4f} |",
        f"| p-value (independent) | {c['base_p_value']:.6f} | {c['inst_p_value']:.6f} |",
        f"| % expected direction | {c['base_pct_expected']:.0f}% | {c['inst_pct_expected']:.0f}% |",
        "",
        f"**K delta**: {c['K_delta']:+.6f}",
        f"**Suppression test p**: {c['suppression_p']:.6f}",
        f"**Cohen's d**: {c['cohens_d']:.4f}",
        "",
        f"## VERDICT: {c['verdict']}",
        "",
        "---",
        "",
        f"*Generated at {c['timestamp'][:19]}*",
        f"*Fathom -- US Patent Pending 64/020,489 & 64/021,113*",
    ]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RLHF Depth — SAE Method")
    parser.add_argument("--smoke", action="store_true", help="3 pairs only")
    parser.add_argument("--analyze-only", action="store_true", help="Re-analyze existing")
    parser.add_argument("--max-features", type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print("  RLHF DEPTH SUPPRESSION — SAE Attribution")
    print("  Gemma-2-2B (base) vs Gemma-2-2B-it (instruct)")
    print("  Same method that gave K=1.0302, p=0.000051")
    print("=" * 60)

    pairs = load_pairs()
    base_results = load_base_results()
    print(f"[PAIRS] {len(pairs)} calibrated pairs")
    print(f"[BASE] {len(base_results)} base results loaded from prior experiment")

    if args.analyze_only:
        if not INSTRUCT_RESULTS_FILE.exists():
            print("[ERROR] No instruct results found. Run experiment first.")
            sys.exit(1)
        with open(INSTRUCT_RESULTS_FILE) as f:
            instruct_results = json.load(f)
    else:
        model = load_model()
        instruct_results = run_experiment(model, pairs, smoke_test=args.smoke)

        with open(INSTRUCT_RESULTS_FILE, "w") as f:
            json_dump(instruct_results, f)
        print(f"\n[SAVED] Instruct results -> {INSTRUCT_RESULTS_FILE}")

        del model
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    if args.smoke:
        if instruct_results:
            print(f"\n{'=' * 60}")
            print("  SMOKE TEST RESULTS")
            for r in instruct_results:
                print(f"  {r['pair_id']}: S={r['surface_depth']:.4f} I={r['insight_depth']:.4f} d={r['delta']:+.4f}")
            print(f"{'=' * 60}")
        return

    # Compare
    comparison = compare_with_base(instruct_results, base_results)

    if comparison:
        with open(COMPARISON_FILE, "w") as f:
            json_dump(comparison, f)
        print(f"[SAVED] Comparison -> {COMPARISON_FILE}")

        report = generate_report(instruct_results, comparison)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[SAVED] Report -> {REPORT_FILE}")

        print(f"\n{'=' * 60}")
        print(f"  K_base     = {comparison['K_base']:.6f}")
        print(f"  K_instruct = {comparison['K_instruct']:.6f}")
        print(f"  K_delta    = {comparison['K_delta']:+.6f}")
        print(f"  p-value    = {comparison['suppression_p']:.6f}")
        print(f"  VERDICT: {comparison['verdict']}")
        print(f"{'=' * 60}")

        if comparison['significant'] and comparison['K_delta'] > 0:
            print("\n  *** RLHF DEPTH SUPPRESSION CONFIRMED (SAE METHOD) ***")
            print("  *** THE DEFAULT MODE NETWORK HYPOTHESIS HOLDS ***\n")


if __name__ == "__main__":
    main()
