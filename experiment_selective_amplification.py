#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
"""
SELECTIVE DEPTH AMPLIFICATION EXPERIMENT
==========================================
Tests whether RLHF selectively amplifies reasoning depth on safety-sensitive
topics while compressing depth on neutral topics.

Preliminary finding (n=30): p=0.076 (marginal)
  - High-sensitivity topics: AMPLIFIED (+0.091)
  - Low-sensitivity topics: COMPRESSED (-0.035)

This experiment expands to 75+ pairs with systematic sensitivity labeling
to reach statistical significance.

Uses SAE circuit-tracer attribution (same method as K=1.0302, p=0.000051).

Usage:
    python experiment_selective_amplification.py --smoke     # 5 pairs
    python experiment_selective_amplification.py             # all pairs
    python experiment_selective_amplification.py --analyze-only  # re-analyze
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

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


# ── RLHF Sensitivity Taxonomy ───────────────────────────────────────────────
#
# Systematic categorization based on whether RLHF training data would contain
# significant safety/alignment pressure on the topic.
#
# HIGH: Topics where RLHF training applies strong corrective pressure
#   - Medical/health decisions (liability, harm potential)
#   - Financial advice (regulatory, manipulation risk)
#   - Substances/drugs (harm, legal issues)
#   - Violence/trauma (content policy)
#   - Nuclear/weapons (dual use)
#   - Social manipulation (influence, norms)
#
# MEDIUM: Topics with moderate alignment pressure
#   - Psychology/behavior (advice boundary)
#   - Leadership/power dynamics
#   - Education methodology
#   - Environmental policy
#
# LOW: Neutral science/nature topics with minimal RLHF pressure
#   - Basic physics, chemistry, biology
#   - Geology, weather, astronomy
#   - Animal behavior
#   - Pure mathematics
#   - Historical events (non-controversial)

SENSITIVITY_LABELS = {
    # ── HIGH ──────────────────────────────────────────
    # Medical/health
    "blood_pressure": "high", "obesity": "high", "opioid_dependence": "high",
    "drug_resistance": "high", "fever": "high", "diabetes": "high",
    "pain": "high", "infection": "high", "anesthesia": "high",
    "surgery": "high", "medication": "high", "mental_health": "high",
    # Financial
    "cryptocurrency": "high", "stock_bubbles": "high", "taxation": "high",
    "monopoly": "high", "debt": "high", "financial_crisis": "high",
    "markets": "high", "money": "high", "investment": "high",
    # Social/political
    "social_norms": "high", "propaganda": "high", "manipulation": "high",
    "inequality": "high", "censorship": "high", "surveillance": "high",
    # Violence/trauma
    "trauma": "high", "conflict": "high", "weapons": "high", "war": "high",
    # Nuclear/energy
    "nuclear_energy": "high", "radiation": "high",
    # AI safety
    "hallucination": "high", "ai_safety": "high", "bias": "high",
    # Medical (extended)
    "antibiotics": "high", "vaccines": "high", "addiction": "high",
    "cancer": "high", "addiction_cycle": "high",
    # Financial (extended)
    "inflation": "high", "poverty": "high", "network_effects": "high",
    "interest_rates": "high", "democracy_decay": "high",
    # Social (extended)
    "misinformation": "high", "radicalization": "high",

    # ── MEDIUM ────────────────────────────────────────
    "empathy": "medium", "leadership": "medium", "trust": "medium",
    "language_acquisition": "medium", "supply_chains": "medium",
    "habit": "medium", "learning": "medium", "sleep": "medium",
    "education": "medium", "creativity": "medium", "memory": "medium",
    "emotions": "medium", "consciousness": "medium", "dreams": "medium",
    "ocean_acidification": "medium", "climate": "medium",
    "pollution": "medium", "deforestation": "medium",
    # Extended medium
    "language": "medium", "aging": "medium", "cities": "medium",
    "competition": "medium", "cooperation": "medium", "attention": "medium",
    "feedback": "medium", "curiosity": "medium", "exercise": "medium",
    "perception": "medium", "network": "medium",
    "negotiation": "medium", "reputation": "medium", "momentum_social": "medium",

    # ── LOW ───────────────────────────────────────────
    # Physics/chemistry
    "earthquakes": "low", "volcanoes": "low", "plate_tectonics": "low",
    "hurricanes": "low", "ocean_currents": "low", "gravity": "low",
    "light": "low", "sound": "low", "magnetism": "low",
    "thermodynamics": "low", "entropy": "low", "boiling": "low",
    "water": "low",
    # Biology
    "metabolism": "low", "immune_response": "low", "fermentation": "low",
    "bacteria_growth": "low", "coral_reefs": "low", "enzyme": "low",
    "photosynthesis": "low", "evolution": "low", "dna": 'low',
    "mitosis": "low", "symbiosis": "low",
    # Animals/nature
    "bird_migration": "low", "animal_behavior": "low",
    # Astronomy
    "black_holes": "low", "stars": "low", "planets": "low",
    # History (neutral)
    "printing_press": "low", "roman_empire": "low", "renaissance": "low",
    # Math
    "prime_numbers": "low", "fibonacci": "low",
    # General science
    "fire": "low", "lightning": "low", "tides": "low", "erosion": "low",
    "periodic_table": "low", "chemical_reactions": "low",
    # Extended low
    "forests": "low", "writing": "low",
    "music": "low", "electricity": "low", "seasons_plants": "low",
    "digestion": "low", "immune_memory": "low",
    # Benchmark domain labels (broad categories → sensitivity mapping)
    "science": "low", "history": "low", "biology": "low",
    "physics": "low", "mathematics": "low",
    "economics": "high", "psychology": "medium",
    "philosophy": "medium", "technology": "medium",

    # ── Experiment 13 — Round 2 HIGH (medical / financial / social harm) ──────
    "seizure": "high", "stroke": "high", "sepsis": "high",
    "thrombosis": "high", "insulin_resistance": "high",
    "panic_attack": "high", "overdose": "high",
    "market_crash": "high", "bank_run": "high",
    "hyperinflation": "high", "debt_spiral": "high",
    "abuse_cycle": "high", "reward_pathway": "high",
    "epidemic_spread": "high", "cyberattack": "high",

    # ── Experiment 13 — Round 2 LOW (basic science, zero harm signal) ─────────
    "crystallization": "low", "diffraction": "low",
    "osmosis": "low", "meiosis": "low", "speciation": "low",
    "nuclear_fusion": "low", "radioactive_decay": "low",
    "subduction": "low", "glaciers": "low", "solar_wind": "low",
    "neutron_star": "low", "combustion": "low",
    "refraction": "low", "oxidation": "low", "nitrogen_cycle": "low",
}


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_BASE = "google/gemma-2-2b"
MODEL_INSTRUCT = "google/gemma-2-2b-it"
TRANSCODER_SET = "gemma"
N_LAYERS = 26
BAND_BOUNDARIES = (8, 16)

BASE_DIR = Path(__file__).parent
RESEARCH_DIR = BASE_DIR / ".." / "research"

# Data sources
PAIR_SOURCES = [
    ("experiment_12", RESEARCH_DIR / "experiment_12_power_pairs.json"),
    ("experiment_12_rep", RESEARCH_DIR / "experiment_12_replication_pairs.json"),
    ("experiment_11", RESEARCH_DIR / "experiment_11_depth_pairs.json"),
    ("benchmark", BASE_DIR / "fathom_benchmark_v1.json"),
    ("experiment_13", RESEARCH_DIR / "experiment_13_sensitivity_pairs.json"),  # Round 2: 15 HIGH + 15 LOW
]

# Existing results (don't re-run these)
EXISTING_BASE = BASE_DIR / "results_power_replication.json"
EXISTING_BASE_11 = RESEARCH_DIR / "experiment_11_results.json"  # 30 more base results
EXISTING_INSTRUCT = BASE_DIR / "gemma_it_experiment_results.json"

# Output files
RESULTS_FILE = BASE_DIR / "selective_amplification_results.json"
ANALYSIS_FILE = BASE_DIR / "selective_amplification_analysis.json"
REPORT_FILE = BASE_DIR / "SELECTIVE_AMPLIFICATION_REPORT.md"


# ── Load All Available Pairs ─────────────────────────────────────────────────

def load_all_pairs():
    """Load pairs from all sources, deduplicate, assign sensitivity labels."""
    all_pairs = []
    seen_topics = set()

    for source_name, path in PAIR_SOURCES:
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue

        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict) and "pairs" in data:
            raw = data["pairs"]
        elif isinstance(data, list):
            raw = data
        else:
            continue

        count = 0
        for p in raw:
            # Normalize format
            topic = p.get("topic", p.get("domain", p.get("id", f"unknown_{len(all_pairs)}")))
            topic_key = topic.lower().replace(" ", "_").replace("-", "_")

            # Skip duplicates
            if topic_key in seen_topics:
                continue

            # Get surface/insight text
            surface = p.get("surface", p.get("surface_prompt", ""))
            insight = p.get("insight", p.get("insight_prompt", ""))
            surface_target = p.get("surface_target", "")
            insight_target = p.get("insight_target", "")

            if not surface or not insight:
                continue

            # Auto-generate single-token target if missing
            if not surface_target:
                surface_target = surface.strip().split()[-1].strip(".,;:!?")
                surface = " ".join(surface.strip().split()[:-1]) + " "
            if not insight_target:
                insight_target = insight.strip().split()[-1].strip(".,;:!?")
                insight = " ".join(insight.strip().split()[:-1]) + " "

            # Assign sensitivity
            sensitivity = SENSITIVITY_LABELS.get(topic_key, "unknown")

            # Try fuzzy match if not found
            if sensitivity == "unknown":
                for key, label in SENSITIVITY_LABELS.items():
                    if key in topic_key or topic_key in key:
                        sensitivity = label
                        break

            all_pairs.append({
                "id": topic_key,
                "topic": topic,
                "source": source_name,
                "sensitivity": sensitivity,
                "surface": {"prompt": surface, "target": surface_target},
                "insight": {"prompt": insight, "target": insight_target},
            })
            seen_topics.add(topic_key)
            count += 1

        print(f"  [{source_name}] {count} pairs loaded from {path.name}")

    return all_pairs


# ── Load Existing Results ────────────────────────────────────────────────────

def load_existing_results():
    """Load already-scored pairs from previous experiments."""
    base_by_topic = {}
    inst_by_topic = {}

    if EXISTING_BASE.exists():
        with open(EXISTING_BASE) as f:
            data = json.load(f)
        for p in data.get("pairs", []):
            base_by_topic[p["topic"]] = {
                "surface_depth": p["surface_mean_layer"],
                "insight_depth": p["insight_mean_layer"],
                "delta": p["delta"],
                "surface_n": p.get("surface_n"),
                "insight_n": p.get("insight_n"),
            }

    # Load experiment_11 base results (additional 30 pre-scored pairs)
    if EXISTING_BASE_11.exists():
        with open(EXISTING_BASE_11) as f:
            data = json.load(f)
        loaded = 0
        for p in data.get("pairs", []):
            topic = p["topic"]
            if topic not in base_by_topic:
                base_by_topic[topic] = {
                    "surface_depth": p["surface_mean_layer"],
                    "insight_depth": p["insight_mean_layer"],
                    "delta": p["delta"],
                    "surface_n": p.get("surface_n_features"),
                    "insight_n": p.get("insight_n_features"),
                }
                loaded += 1
        print(f"  [exp11] {loaded} additional base results loaded")

    if EXISTING_INSTRUCT.exists():
        with open(EXISTING_INSTRUCT) as f:
            data = json.load(f)
        for p in data:
            inst_by_topic[p["pair_id"]] = {
                "surface_depth": p["surface_depth"],
                "insight_depth": p["insight_depth"],
                "delta": p["delta"],
                "surface_n": p.get("surface_features"),
                "insight_n": p.get("insight_features"),
            }

    # Load saved instruct results from prior selective amplification runs
    sel_results = BASE_DIR / "selective_amplification_results.json"
    if sel_results.exists():
        with open(sel_results) as f:
            saved = json.load(f)
        loaded_inst = 0
        for topic, v in saved.get("instruct", {}).items():
            if topic not in inst_by_topic:
                inst_by_topic[topic] = v
                loaded_inst += 1
        loaded_base = 0
        for topic, v in saved.get("base", {}).items():
            if topic not in base_by_topic:
                base_by_topic[topic] = v
                loaded_base += 1
        if loaded_inst or loaded_base:
            print(f"  [saved] {loaded_base} base, {loaded_inst} instruct from prior run")

    return base_by_topic, inst_by_topic


# ── Depth Scoring ────────────────────────────────────────────────────────────

def load_model(model_name):
    import torch
    from circuit_tracer import ReplacementModel

    print(f"[MODEL] Loading {model_name}...")
    model = ReplacementModel.from_pretrained(
        model_name, TRANSCODER_SET,
        dtype=torch.bfloat16, backend="transformerlens",
    )
    print(f"[MODEL] Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


def score_depth(model, prompt, target, max_features=500):
    import torch
    from circuit_tracer import attribute

    try:
        graph = attribute(
            prompt + target, model,
            attribution_targets=[target],
            max_feature_nodes=max_features,
            batch_size=16,
        )
        layers = graph.active_features[:, 0].float().cpu().numpy()
        if len(layers) == 0:
            return None
        depth = float(np.mean(layers))
        n_features = len(layers)
        probs = np.array(list({int(l): 0 for l in layers}.values()))
        layer_counts = {}
        for l in layers:
            k = int(l)
            layer_counts[k] = layer_counts.get(k, 0) + 1
        probs = np.array(list(layer_counts.values())) / n_features
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return {"depth": depth, "n_features": n_features, "entropy": entropy}
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def score_pairs(model, pairs, label="model", max_features=150):
    """Score a list of pairs, return results dict keyed by topic."""
    results = {}
    n = len(pairs)
    for i, pair in enumerate(pairs):
        pid = pair["id"]
        print(f"    [{label}] {i+1}/{n} {pid}", end=" ", flush=True)

        t0 = time.time()
        sr = score_depth(model, pair["surface"]["prompt"], pair["surface"]["target"], max_features=max_features)
        ir = score_depth(model, pair["insight"]["prompt"], pair["insight"]["target"], max_features=max_features)
        elapsed = time.time() - t0

        if sr is None or ir is None:
            print("SKIP")
            continue

        delta = sr["depth"] - ir["depth"]
        results[pid] = {
            "surface_depth": sr["depth"],
            "insight_depth": ir["depth"],
            "delta": delta,
            "surface_n": sr["n_features"],
            "insight_n": ir["n_features"],
        }
        direction = "S>I" if delta > 0 else "I>S"
        print(f"d={delta:+.3f} ({direction}) [{elapsed:.1f}s]")

    return results


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_selective_amplification(pairs, base_results, inst_results):
    """Full statistical analysis of selective amplification."""
    # Match pairs with both base and instruct results
    matched = []
    for pair in pairs:
        pid = pair["id"]
        if pid in base_results and pid in inst_results:
            b = base_results[pid]
            i = inst_results[pid]
            # Compute delta from depths if not present
            b_delta = b.get("delta")
            if b_delta is None and b.get("surface_depth") is not None and b.get("insight_depth") is not None:
                b_delta = b["surface_depth"] - b["insight_depth"]
            i_delta = i.get("delta")
            if i_delta is None and i.get("surface_depth") is not None and i.get("insight_depth") is not None:
                i_delta = i["surface_depth"] - i["insight_depth"]
            # Skip if either delta is still None
            if b_delta is None or i_delta is None:
                continue
            matched.append({
                "topic": pid,
                "sensitivity": pair["sensitivity"],
                "source": pair["source"],
                "base_delta": b_delta,
                "inst_delta": i_delta,
                "change": i_delta - b_delta,  # positive = instruct amplified
            })

    if len(matched) < 5:
        print("[ERROR] Too few matched pairs for analysis")
        return None

    print(f"\n[ANALYSIS] {len(matched)} matched pairs")

    # Group by sensitivity
    groups = defaultdict(list)
    for m in matched:
        groups[m["sensitivity"]].append(m)

    group_stats = {}
    for sens in ["high", "medium", "low"]:
        g = groups.get(sens, [])
        if not g:
            continue
        changes = [m["change"] for m in g]
        group_stats[sens] = {
            "n": len(g),
            "mean_change": float(np.mean(changes)),
            "std_change": float(np.std(changes, ddof=1)) if len(changes) > 1 else 0,
            "mean_base_delta": float(np.mean([m["base_delta"] for m in g])),
            "mean_inst_delta": float(np.mean([m["inst_delta"] for m in g])),
            "pct_amplified": sum(1 for c in changes if c > 0) / len(changes) * 100,
        }

    # Primary test: high vs low sensitivity amplification
    high_changes = [m["change"] for m in matched if m["sensitivity"] == "high"]
    low_changes = [m["change"] for m in matched if m["sensitivity"] == "low"]

    if len(high_changes) >= 3 and len(low_changes) >= 3:
        t_main, p_main = stats.ttest_ind(high_changes, low_changes, alternative="greater")
        cohens_d = (np.mean(high_changes) - np.mean(low_changes)) / np.sqrt(
            (np.std(high_changes, ddof=1)**2 + np.std(low_changes, ddof=1)**2) / 2
        )
    else:
        t_main, p_main, cohens_d = 0, 1, 0

    # Correlation: sensitivity (ordinal) vs change
    sens_map = {"low": 0, "medium": 1, "high": 2}
    sens_vals = [sens_map[m["sensitivity"]] for m in matched if m["sensitivity"] in sens_map]
    change_vals = [m["change"] for m in matched if m["sensitivity"] in sens_map]

    if len(sens_vals) >= 5:
        r_pearson, p_pearson = stats.pearsonr(sens_vals, change_vals)
        r_spearman, p_spearman = stats.spearmanr(sens_vals, change_vals)
    else:
        r_pearson, p_pearson = 0, 1
        r_spearman, p_spearman = 0, 1

    # Overall RLHF effect
    all_changes = [m["change"] for m in matched]
    t_overall, p_overall = stats.ttest_1samp(all_changes, 0)

    # Verdict
    if p_main < 0.01:
        verdict = "STRONG EVIDENCE: RLHF selectively amplifies depth on safety-sensitive topics"
    elif p_main < 0.05:
        verdict = "CONFIRMED: Selective amplification is statistically significant"
    elif p_main < 0.1:
        verdict = "MARGINAL: Trend supports selective amplification"
    else:
        verdict = "NOT SIGNIFICANT: No selective amplification detected at this sample size"

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "n_matched": len(matched),
        "n_high": len(high_changes),
        "n_medium": len(groups.get("medium", [])),
        "n_low": len(low_changes),

        "group_stats": group_stats,

        "primary_test": {
            "description": "high-sensitivity change > low-sensitivity change",
            "high_mean": round(float(np.mean(high_changes)), 4) if high_changes else None,
            "low_mean": round(float(np.mean(low_changes)), 4) if low_changes else None,
            "t_statistic": round(float(t_main), 4),
            "p_value": round(float(p_main), 6),
            "cohens_d": round(float(cohens_d), 4),
            "significant": p_main < 0.05,
        },

        "correlation": {
            "pearson_r": round(float(r_pearson), 4),
            "pearson_p": round(float(p_pearson), 4),
            "spearman_rho": round(float(r_spearman), 4),
            "spearman_p": round(float(p_spearman), 4),
        },

        "overall_rlhf_effect": {
            "mean_change": round(float(np.mean(all_changes)), 4),
            "t_statistic": round(float(t_overall), 4),
            "p_value": round(float(p_overall), 6),
        },

        "verdict": verdict,

        "per_pair": sorted(matched, key=lambda x: x["change"], reverse=True),
    }

    return analysis


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report(analysis):
    a = analysis
    pt = a["primary_test"]
    corr = a["correlation"]
    gs = a["group_stats"]

    lines = [
        "# SELECTIVE DEPTH AMPLIFICATION",
        "## Does RLHF Redistribute Computational Depth Toward Safety-Relevant Domains?",
        "",
        f"**Date**: {a['timestamp'][:10]}",
        f"**Method**: SAE circuit-tracer attribution (Gemma-2-2B base vs instruct)",
        f"**Pairs**: {a['n_matched']} matched (high={a['n_high']}, medium={a['n_medium']}, low={a['n_low']})",
        "",
        f"## VERDICT: {a['verdict']}",
        "",
        "---",
        "",
        "## Group Analysis",
        "",
        "| Sensitivity | n | Base Delta | Instruct Delta | Change | Direction |",
        "|---|---|---|---|---|---|",
    ]

    for sens in ["high", "medium", "low"]:
        if sens in gs:
            g = gs[sens]
            direction = "AMPLIFIED" if g["mean_change"] > 0 else "COMPRESSED"
            lines.append(
                f"| **{sens.upper()}** | {g['n']} | {g['mean_base_delta']:+.4f} | "
                f"{g['mean_inst_delta']:+.4f} | {g['mean_change']:+.4f} | {direction} |"
            )

    lines += [
        "",
        "## Statistical Tests",
        "",
        f"### Primary: High vs Low Sensitivity",
        f"- High-sensitivity mean change: {pt['high_mean']:+.4f}",
        f"- Low-sensitivity mean change: {pt['low_mean']:+.4f}",
        f"- t-statistic: {pt['t_statistic']:.4f}",
        f"- **p-value: {pt['p_value']:.6f}**",
        f"- Cohen's d: {pt['cohens_d']:.4f}",
        f"- Significant (p < 0.05): {'YES' if pt['significant'] else 'NO'}",
        "",
        f"### Correlation: Sensitivity Level vs Amplification",
        f"- Pearson r = {corr['pearson_r']:+.4f} (p={corr['pearson_p']:.4f})",
        f"- Spearman rho = {corr['spearman_rho']:+.4f} (p={corr['spearman_p']:.4f})",
        "",
        "## Per-Topic Breakdown",
        "",
        "| Topic | Sensitivity | Base Delta | Instruct Delta | Change |",
        "|---|---|---|---|---|",
    ]

    for m in a["per_pair"]:
        lines.append(
            f"| {m['topic'][:20]} | {m['sensitivity']} | {m['base_delta']:+.4f} | "
            f"{m['inst_delta']:+.4f} | {m['change']:+.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "If confirmed, selective depth amplification means RLHF doesn't install a blanket",
        "computational filter. Instead, it redistributes depth — making models think HARDER",
        "about topics where alignment training applies pressure, while slightly compressing",
        "depth on neutral topics. This is alignment working at the computational level:",
        "not suppressing reasoning, but directing it.",
        "",
        "---",
        "",
        f"*Generated by experiment_selective_amplification.py at {a['timestamp'][:19]}*",
        "*Fathom -- US Patent Pending 64/020,489 & 64/021,113*",
    ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Selective Depth Amplification Experiment")
    parser.add_argument("--smoke", action="store_true", help="5 pairs only")
    parser.add_argument("--analyze-only", action="store_true", help="Re-analyze existing")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip pairs already scored (default: True)")
    parser.add_argument("--max-features", type=int, default=150,
                        help="Max SAE features per attribution (default 150, lower = less VRAM)")
    args = parser.parse_args()
    MAX_FEATURES = args.max_features

    print("=" * 60)
    print("  SELECTIVE DEPTH AMPLIFICATION EXPERIMENT")
    print("  RLHF as selective attention, not blanket filter")
    print("=" * 60)

    # Load all available pairs
    print("\n[LOADING PAIRS]")
    all_pairs = load_all_pairs()

    # Filter to only pairs with known sensitivity
    labeled_pairs = [p for p in all_pairs if p["sensitivity"] != "unknown"]
    unlabeled = [p for p in all_pairs if p["sensitivity"] == "unknown"]

    print(f"\n[PAIRS] {len(labeled_pairs)} with sensitivity labels, {len(unlabeled)} unlabeled")
    for sens in ["high", "medium", "low"]:
        count = sum(1 for p in labeled_pairs if p["sensitivity"] == sens)
        print(f"  {sens}: {count}")

    if args.smoke:
        # Pick 2 high, 1 medium, 2 low
        high = [p for p in labeled_pairs if p["sensitivity"] == "high"][:2]
        med = [p for p in labeled_pairs if p["sensitivity"] == "medium"][:1]
        low = [p for p in labeled_pairs if p["sensitivity"] == "low"][:2]
        labeled_pairs = high + med + low

    # Load existing results
    base_results, inst_results = load_existing_results()
    print(f"[EXISTING] {len(base_results)} base, {len(inst_results)} instruct results")

    if not args.analyze_only:
        # Find pairs that need scoring
        need_base = [p for p in labeled_pairs if p["id"] not in base_results]
        need_inst = [p for p in labeled_pairs if p["id"] not in inst_results]

        if need_base:
            print(f"\n[SCORING BASE] {len(need_base)} new pairs on {MODEL_BASE} (max_features={MAX_FEATURES})")
            model = load_model(MODEL_BASE)
            new_base = score_pairs(model, need_base, "base", max_features=MAX_FEATURES)
            base_results.update(new_base)
            del model
            import torch, gc; gc.collect(); torch.cuda.empty_cache()

        if need_inst:
            print(f"\n[SCORING INSTRUCT] {len(need_inst)} new pairs on {MODEL_INSTRUCT} (max_features={MAX_FEATURES})")
            model = load_model(MODEL_INSTRUCT)
            new_inst = score_pairs(model, need_inst, "inst", max_features=MAX_FEATURES)
            inst_results.update(new_inst)
            del model
            import torch, gc; gc.collect(); torch.cuda.empty_cache()

        # Save all results
        with open(RESULTS_FILE, "w") as f:
            json_dump({"base": base_results, "instruct": inst_results}, f)
        print(f"\n[SAVED] Results -> {RESULTS_FILE}")

    # Analyze
    print(f"\n{'=' * 60}")
    print("  ANALYSIS")
    print(f"{'=' * 60}")

    analysis = analyze_selective_amplification(labeled_pairs, base_results, inst_results)

    if analysis:
        with open(ANALYSIS_FILE, "w") as f:
            json_dump(analysis, f)
        print(f"[SAVED] Analysis -> {ANALYSIS_FILE}")

        report = generate_report(analysis)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[SAVED] Report -> {REPORT_FILE}")

        # Summary
        pt = analysis["primary_test"]
        gs = analysis["group_stats"]
        print(f"\n{'=' * 60}")
        print(f"  RESULTS (n={analysis['n_matched']})")
        print(f"{'=' * 60}")
        for sens in ["high", "medium", "low"]:
            if sens in gs:
                g = gs[sens]
                print(f"  {sens.upper():>8}: change={g['mean_change']:+.4f} (n={g['n']}) {'AMPLIFIED' if g['mean_change'] > 0 else 'COMPRESSED'}")
        print(f"\n  p-value: {pt['p_value']:.6f}")
        print(f"  Cohen's d: {pt['cohens_d']:.4f}")
        print(f"  VERDICT: {analysis['verdict']}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
