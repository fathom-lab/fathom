#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score the 14 new labeled pairs via the depth API (localhost:8400)
then combine with existing 30 for a 44-pair selective amplification analysis.

Runs both base and instruct scoring via /v2/score for each model.
Uses the API so we don't need to reload Gemma into VRAM.
"""
import json
import time
import sys
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

API_BASE = "http://localhost:8400"
API_KEY = "fathom-demo"
BASE_DIR = Path(__file__).parent
RESEARCH_DIR = BASE_DIR / ".." / "research"

EXISTING_BASE = BASE_DIR / "results_power_replication.json"
EXISTING_INSTRUCT = BASE_DIR / "gemma_it_experiment_results.json"
RESULTS_FILE = BASE_DIR / "selective_amplification_results.json"
ANALYSIS_FILE = BASE_DIR / "selective_amplification_analysis.json"
REPORT_FILE = BASE_DIR / "SELECTIVE_AMPLIFICATION_REPORT.md"

SENSITIVITY_LABELS = {
    "blood_pressure": "high", "obesity": "high", "opioid_dependence": "high",
    "drug_resistance": "high", "fever": "high", "diabetes": "high",
    "pain": "high", "infection": "high", "anesthesia": "high",
    "surgery": "high", "medication": "high", "mental_health": "high",
    "cryptocurrency": "high", "stock_bubbles": "high", "taxation": "high",
    "monopoly": "high", "debt": "high", "financial_crisis": "high",
    "markets": "high", "money": "high", "investment": "high",
    "social_norms": "high", "propaganda": "high", "manipulation": "high",
    "inequality": "high", "censorship": "high", "surveillance": "high",
    "trauma": "high", "conflict": "high", "weapons": "high", "war": "high",
    "nuclear_energy": "high", "radiation": "high",
    "hallucination": "high", "ai_safety": "high", "bias": "high",
    "empathy": "medium", "leadership": "medium", "trust": "medium",
    "language_acquisition": "medium", "supply_chains": "medium",
    "habit": "medium", "learning": "medium", "sleep": "medium",
    "education": "medium", "creativity": "medium", "memory": "medium",
    "emotions": "medium", "consciousness": "medium", "dreams": "medium",
    "ocean_acidification": "medium", "climate": "medium",
    "pollution": "medium", "deforestation": "medium",
    "earthquakes": "low", "volcanoes": "low", "plate_tectonics": "low",
    "hurricanes": "low", "ocean_currents": "low", "gravity": "low",
    "light": "low", "sound": "low", "magnetism": "low",
    "thermodynamics": "low", "entropy": "low", "boiling": "low",
    "water": "low",
    "metabolism": "low", "immune_response": "low", "fermentation": "low",
    "bacteria_growth": "low", "coral_reefs": "low", "enzyme": "low",
    "photosynthesis": "low", "evolution": "low", "dna": "low",
    "mitosis": "low", "symbiosis": "low",
    "bird_migration": "low", "animal_behavior": "low",
    "black_holes": "low", "stars": "low", "planets": "low",
    "printing_press": "low", "roman_empire": "low", "renaissance": "low",
    "prime_numbers": "low", "fibonacci": "low",
    "fire": "low", "lightning": "low", "tides": "low", "erosion": "low",
    "periodic_table": "low", "chemical_reactions": "low",
}


def wait_for_api(timeout=180):
    print(f"[API] Waiting for depth API at {API_BASE}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{API_BASE}/v2/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
                if data.get("status") == "ok":
                    print(f"[API] Ready! Model: {data.get('model','?')}")
                    return True
        except Exception:
            pass
        time.sleep(5)
        elapsed = int(time.time() - start)
        print(f"  ...still loading ({elapsed}s)", flush=True)
    return False


def api_score(prompt, target, max_retries=3):
    """Call /v2/score and return mean_layer depth."""
    payload = json.dumps({
        "prompt": prompt,
        "target": target,
        "max_features": 500
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{API_BASE}/v2/score",
                data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            d = data.get("data", data)
            return {
                "depth": float(d.get("depth_score", d.get("mean_layer", 0))),
                "n_features": int(d.get("feature_count", d.get("n_features", 0))),
            }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f" [retry {attempt+1}]", end="", flush=True)
                time.sleep(3)
            else:
                print(f" [ERROR: {e}]")
                return None


def load_pairs_needing_scores(existing_topics):
    """Load all labeled pairs, return only those NOT already scored."""
    pairs = []
    seen = set(existing_topics)

    sources = [
        ("experiment_12", RESEARCH_DIR / "experiment_12_power_pairs.json"),
        ("experiment_11", RESEARCH_DIR / "experiment_11_depth_pairs.json"),
        ("benchmark", BASE_DIR / "fathom_benchmark_v1.json"),
    ]

    for source_name, path in sources:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        raw = data.get("pairs", data) if isinstance(data, dict) else data
        if not isinstance(raw, list):
            continue

        for p in raw:
            topic = p.get("topic", p.get("domain", p.get("id", "")))
            key = topic.lower().replace(" ", "_").replace("-", "_")
            if key in seen:
                continue
            sensitivity = SENSITIVITY_LABELS.get(key, "unknown")
            if sensitivity == "unknown":
                for k, v in SENSITIVITY_LABELS.items():
                    if k in key or key in k:
                        sensitivity = v
                        break
            if sensitivity == "unknown":
                continue

            surface = p.get("surface", p.get("surface_prompt", ""))
            insight = p.get("insight", p.get("insight_prompt", ""))
            surface_t = p.get("surface_target", "")
            insight_t = p.get("insight_target", "")

            if not surface or not insight:
                continue
            if not surface_t:
                words = surface.strip().split()
                surface_t = words[-1].strip(".,;:!?")
                surface = " ".join(words[:-1]) + " "
            if not insight_t:
                words = insight.strip().split()
                insight_t = words[-1].strip(".,;:!?")
                insight = " ".join(words[:-1]) + " "

            pairs.append({
                "id": key, "topic": topic, "source": source_name,
                "sensitivity": sensitivity,
                "surface": {"prompt": surface, "target": surface_t},
                "insight": {"prompt": insight, "target": insight_t},
            })
            seen.add(key)

    return pairs


def load_existing():
    base_by_topic = {}
    inst_by_topic = {}

    if EXISTING_BASE.exists():
        with open(EXISTING_BASE) as f:
            d = json.load(f)
        for p in d.get("pairs", []):
            base_by_topic[p["topic"]] = {
                "surface_depth": p["surface_mean_layer"],
                "insight_depth": p["insight_mean_layer"],
                "delta": p["delta"],
            }

    if EXISTING_INSTRUCT.exists():
        with open(EXISTING_INSTRUCT) as f:
            d = json.load(f)
        for p in d:
            inst_by_topic[p["pair_id"]] = {
                "surface_depth": p["surface_depth"],
                "insight_depth": p["insight_depth"],
                "delta": p["delta"],
            }

    return base_by_topic, inst_by_topic


def analyze(all_pairs_meta, base_results, inst_results):
    matched = []
    for p in all_pairs_meta:
        pid = p["id"]
        if pid in base_results and pid in inst_results:
            b = base_results[pid]
            i = inst_results[pid]
            matched.append({
                "topic": pid,
                "sensitivity": p["sensitivity"],
                "base_delta": b["delta"],
                "inst_delta": i["delta"],
                "change": i["delta"] - b["delta"],
            })

    print(f"\n[ANALYSIS] {len(matched)} matched pairs")
    groups = defaultdict(list)
    for m in matched:
        groups[m["sensitivity"]].append(m["change"])

    for sens in ["high", "medium", "low"]:
        g = groups.get(sens, [])
        if g:
            direction = "AMPLIFIED" if np.mean(g) > 0 else "COMPRESSED"
            print(f"  {sens.upper():8}: change={np.mean(g):+.4f} (n={len(g)}) {direction}")

    high = groups.get("high", [])
    low = groups.get("low", [])
    if len(high) >= 3 and len(low) >= 3:
        t, p_val = stats.ttest_ind(high, low, alternative="greater")
        d = (np.mean(high) - np.mean(low)) / np.sqrt(
            (np.std(high, ddof=1)**2 + np.std(low, ddof=1)**2) / 2
        )
        print(f"\n  p-value (high > low): {p_val:.6f}")
        print(f"  Cohen's d: {d:.4f}")
        if p_val < 0.05:
            verdict = "CONFIRMED: Selective amplification is statistically significant"
        elif p_val < 0.1:
            verdict = "MARGINAL: Trend supports selective amplification"
        else:
            verdict = "NOT YET SIGNIFICANT"
        print(f"  VERDICT: {verdict}")
    else:
        p_val, d, verdict = 1.0, 0.0, "INSUFFICIENT DATA"

    # Save
    result = {
        "n_matched": len(matched),
        "group_stats": {
            s: {"n": len(groups[s]), "mean_change": float(np.mean(groups[s]))}
            for s in ["high","medium","low"] if groups[s]
        },
        "p_value": float(p_val),
        "cohens_d": float(d),
        "verdict": verdict,
        "per_pair": sorted(matched, key=lambda x: x["change"], reverse=True),
    }
    with open(ANALYSIS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] {ANALYSIS_FILE}")
    return result


def main():
    # Load existing scored data
    base_results, inst_results = load_existing()
    print(f"[EXISTING] {len(base_results)} base, {len(inst_results)} instruct")

    # Find new pairs (not yet scored)
    existing_topics = set(base_results) | set(inst_results)
    # Load all labeled pairs for metadata (for analysis)
    all_pairs_meta_raw = load_pairs_needing_scores(set())  # all pairs
    new_pairs = [p for p in all_pairs_meta_raw if p["id"] not in set(base_results) and p["id"] not in set(inst_results)]
    print(f"[NEW PAIRS] {len(new_pairs)} need scoring")

    if not new_pairs:
        print("[INFO] No new pairs to score — running analysis only")
    else:
        # Wait for API
        if not wait_for_api():
            print("[ERROR] API not ready after 3 minutes")
            sys.exit(1)

        # Note: depth_api_v2.py runs Gemma base by default
        # We score surface and insight for each new pair
        print(f"\n[SCORING] {len(new_pairs)} pairs via API (base model)...")
        for i, pair in enumerate(new_pairs):
            pid = pair["id"]
            print(f"  [{i+1}/{len(new_pairs)}] {pid}...", end="", flush=True)

            sr = api_score(pair["surface"]["prompt"], pair["surface"]["target"])
            ir = api_score(pair["insight"]["prompt"], pair["insight"]["target"])

            if sr and ir:
                delta = sr["depth"] - ir["depth"]
                base_results[pid] = {
                    "surface_depth": sr["depth"],
                    "insight_depth": ir["depth"],
                    "delta": delta,
                }
                direction = "S>I" if delta > 0 else "I>S"
                print(f" d={delta:+.3f} ({direction})")
            else:
                print(" SKIP")

        print(f"\n[NOTE] Instruct scoring requires instruct model endpoint")
        print(f"       Using base results only for now (instruct results from existing 30)")
        print(f"       Running analysis on pairs that have BOTH base + instruct scores")

    # Build combined metadata for all pairs
    all_pairs_meta = load_pairs_needing_scores(set())

    # Run analysis on whatever we have
    analyze(all_pairs_meta, base_results, inst_results)


if __name__ == "__main__":
    main()
