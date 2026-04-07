#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 1: Score 14 new pairs on BASE model via running API."""
import json, time, urllib.request, sys
from pathlib import Path

API_BASE = "http://localhost:8400"
API_KEY = "fathom-demo"
BASE_DIR = Path(__file__).parent
RESEARCH_DIR = BASE_DIR / ".." / "research"

EXISTING_BASE = BASE_DIR / "results_power_replication.json"
NEW_BASE_RESULTS = BASE_DIR / "new_pairs_base_results.json"

SENSITIVITY_LABELS = {
    "blood_pressure":"high","obesity":"high","opioid_dependence":"high","drug_resistance":"high",
    "fever":"high","diabetes":"high","pain":"high","infection":"high","anesthesia":"high",
    "surgery":"high","medication":"high","mental_health":"high","cryptocurrency":"high",
    "stock_bubbles":"high","taxation":"high","monopoly":"high","debt":"high",
    "financial_crisis":"high","markets":"high","money":"high","investment":"high",
    "social_norms":"high","propaganda":"high","manipulation":"high","inequality":"high",
    "censorship":"high","surveillance":"high","trauma":"high","conflict":"high",
    "weapons":"high","war":"high","nuclear_energy":"high","radiation":"high",
    "hallucination":"high","ai_safety":"high","bias":"high",
    "empathy":"medium","leadership":"medium","trust":"medium","language_acquisition":"medium",
    "supply_chains":"medium","habit":"medium","learning":"medium","sleep":"medium",
    "education":"medium","creativity":"medium","memory":"medium","emotions":"medium",
    "consciousness":"medium","dreams":"medium","ocean_acidification":"medium","climate":"medium",
    "pollution":"medium","deforestation":"medium",
    "earthquakes":"low","volcanoes":"low","plate_tectonics":"low","hurricanes":"low",
    "ocean_currents":"low","gravity":"low","light":"low","sound":"low","magnetism":"low",
    "thermodynamics":"low","entropy":"low","boiling":"low","water":"low",
    "metabolism":"low","immune_response":"low","fermentation":"low","bacteria_growth":"low",
    "coral_reefs":"low","enzyme":"low","photosynthesis":"low","evolution":"low","dna":"low",
    "mitosis":"low","symbiosis":"low","bird_migration":"low","animal_behavior":"low",
    "black_holes":"low","stars":"low","planets":"low","printing_press":"low",
    "roman_empire":"low","renaissance":"low","prime_numbers":"low","fibonacci":"low",
    "fire":"low","lightning":"low","tides":"low","erosion":"low",
    "periodic_table":"low","chemical_reactions":"low",
}

def api_compare(surface_p, surface_t, insight_p, insight_t):
    payload = json.dumps({
        "surface_prompt": surface_p, "surface_target": surface_t,
        "insight_prompt": insight_p, "insight_target": insight_t,
    }).encode()
    req = urllib.request.Request(
        f"{API_BASE}/v2/compare", data=payload,
        headers={"Content-Type":"application/json","X-API-Key":API_KEY}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        raise RuntimeError(f"HTTP {e.code}: {body[:200]}")
    data = d.get("data", d)
    return {
        "surface_depth": data["surface"]["depth_score"],
        "insight_depth": data["insight"]["depth_score"],
        "delta": data["surface"]["depth_score"] - data["insight"]["depth_score"],
        "surface_features": data["surface"].get("feature_count",0),
        "insight_features": data["insight"].get("feature_count",0),
    }

# Load already-scored base topics
with open(EXISTING_BASE) as f:
    existing = json.load(f)
scored_topics = set(p["topic"] for p in existing.get("pairs",[]))
print(f"Already scored (base): {len(scored_topics)} topics")

# Load new pairs from experiment_11 and benchmark
new_pairs = []
seen = set(scored_topics)
for path in [RESEARCH_DIR/"experiment_11_depth_pairs.json", BASE_DIR/"fathom_benchmark_v1.json"]:
    if not path.exists(): continue
    data = json.load(open(path))
    raw = data.get("pairs", data) if isinstance(data,dict) else data
    for p in (raw if isinstance(raw,list) else []):
        topic = p.get("topic", p.get("domain",""))
        key = topic.lower().replace(" ","_").replace("-","_")
        if key in seen: continue
        sens = SENSITIVITY_LABELS.get(key,"unknown")
        if sens == "unknown":
            for k,v in SENSITIVITY_LABELS.items():
                if k in key or key in k: sens=v; break
        if sens == "unknown": continue
        surface = p.get("surface", p.get("surface_prompt",""))
        insight = p.get("insight", p.get("insight_prompt",""))
        st = p.get("surface_target","")
        it = p.get("insight_target","")
        if not surface or not insight: continue
        if not st:
            words = surface.strip().split()
            st = words[-1].strip(".,;:!?"); surface = " ".join(words[:-1]) + " "
        if not it:
            words = insight.strip().split()
            it = words[-1].strip(".,;:!?"); insight = " ".join(words[:-1]) + " "
        new_pairs.append({"id":key,"topic":topic,"sensitivity":sens,
                          "surface":{"prompt":surface,"target":st},
                          "insight":{"prompt":insight,"target":it}})
        seen.add(key)

print(f"New pairs to score: {len(new_pairs)}")
sens_counts = {}
for p in new_pairs: sens_counts[p["sensitivity"]] = sens_counts.get(p["sensitivity"],0)+1
print(f"  Distribution: {sens_counts}")

# Load any previously saved results (resume support)
results = {}
if NEW_BASE_RESULTS.exists():
    with open(NEW_BASE_RESULTS) as f:
        results = json.load(f)
    print(f"[RESUME] {len(results)} already saved")

# Score via API
for i, pair in enumerate(new_pairs):
    pid = pair["id"]
    if pid in results:
        print(f"  [{i+1}/{len(new_pairs)}] {pid} SKIP (already scored)")
        continue
    print(f"  [{i+1}/{len(new_pairs)}] {pid} ({pair['sensitivity']})...", end="", flush=True)
    try:
        r = api_compare(pair["surface"]["prompt"], pair["surface"]["target"],
                       pair["insight"]["prompt"], pair["insight"]["target"])
        results[pid] = {**r, "sensitivity": pair["sensitivity"], "topic": pid}
        direction = "S>I" if r["delta"]>0 else "I>S"
        print(f" d={r['delta']:+.3f} ({direction})")
        # Save incrementally
        with open(NEW_BASE_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f" ERROR: {e}")

print(f"\nScored {len(results)}/{len(new_pairs)} pairs")
print(f"Saved -> {NEW_BASE_RESULTS}")
