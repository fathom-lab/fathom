#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Score new pairs on INSTRUCT model via circuit-tracer directly.
Run AFTER stopping the base API to free VRAM.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json, time, gc
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
NEW_BASE_RESULTS = BASE_DIR / "new_pairs_base_results.json"
NEW_INSTRUCT_RESULTS = BASE_DIR / "new_pairs_instruct_results.json"

MODEL_INSTRUCT = "google/gemma-2-2b-it"
TRANSCODER_SET = "gemma"

if not NEW_BASE_RESULTS.exists():
    print(f"[ERROR] Run score_base_via_api.py first — {NEW_BASE_RESULTS} missing")
    sys.exit(1)

with open(NEW_BASE_RESULTS) as f:
    base_data = json.load(f)

pairs_to_score = list(base_data.keys())
print(f"[INSTRUCT] Scoring {len(pairs_to_score)} pairs on {MODEL_INSTRUCT}")

# Load all pair metadata (prompts) from base results file includes sensitivity
# We need the actual prompts — reload from source
RESEARCH_DIR = BASE_DIR / ".." / "research"
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

pair_prompts = {}
target_set = set(pairs_to_score)
for path in [RESEARCH_DIR/"experiment_11_depth_pairs.json", BASE_DIR/"fathom_benchmark_v1.json"]:
    if not path.exists(): continue
    data = json.load(open(path))
    raw = data.get("pairs", data) if isinstance(data,dict) else data
    for p in (raw if isinstance(raw,list) else []):
        topic = p.get("topic", p.get("domain",""))
        key = topic.lower().replace(" ","_").replace("-","_")
        if key not in target_set: continue
        surface = p.get("surface", p.get("surface_prompt",""))
        insight = p.get("insight", p.get("insight_prompt",""))
        st = p.get("surface_target","")
        it = p.get("insight_target","")
        if not st:
            words = surface.strip().split()
            st = words[-1].strip(".,;:!?"); surface = " ".join(words[:-1]) + " "
        if not it:
            words = insight.strip().split()
            it = words[-1].strip(".,;:!?"); insight = " ".join(words[:-1]) + " "
        pair_prompts[key] = {"surface":{"prompt":surface,"target":st},
                             "insight":{"prompt":insight,"target":it}}

print(f"[PROMPTS] Found {len(pair_prompts)}/{len(pairs_to_score)} prompts")

# Load instruct model
import torch
from circuit_tracer import ReplacementModel, attribute

print(f"[MODEL] Loading {MODEL_INSTRUCT}...")
t0 = time.time()
model = ReplacementModel.from_pretrained(
    MODEL_INSTRUCT, TRANSCODER_SET,
    dtype=torch.bfloat16, backend="transformerlens",
)
print(f"[MODEL] Loaded in {time.time()-t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

def score_depth(prompt, target, max_features=500):
    try:
        graph = attribute(
            prompt + target, model,
            attribution_targets=[target],
            max_feature_nodes=max_features, batch_size=16,
        )
        layers = graph.active_features[:, 0].float().cpu().numpy()
        if len(layers) == 0: return None
        depth = float(np.mean(layers))
        n_features = len(layers)
        layer_counts = {}
        for l in layers:
            k = int(l); layer_counts[k] = layer_counts.get(k,0)+1
        return {"depth": depth, "n_features": n_features}
    except Exception as e:
        print(f" [ERR: {e}]"); return None

results = {}
n = len(pair_prompts)
for i, (pid, pair) in enumerate(pair_prompts.items()):
    print(f"  [{i+1}/{n}] {pid}...", end="", flush=True)
    sr = score_depth(pair["surface"]["prompt"], pair["surface"]["target"])
    ir = score_depth(pair["insight"]["prompt"], pair["insight"]["target"])
    if sr and ir:
        delta = sr["depth"] - ir["depth"]
        results[pid] = {
            "pair_id": pid,
            "surface_depth": sr["depth"],
            "insight_depth": ir["depth"],
            "delta": delta,
            "surface_features": sr["n_features"],
            "insight_features": ir["n_features"],
        }
        print(f" d={delta:+.3f} ({'S>I' if delta>0 else 'I>S'})")
    else:
        print(" SKIP")

del model; gc.collect(); torch.cuda.empty_cache()

print(f"\nScored {len(results)}/{n} pairs")
with open(NEW_INSTRUCT_RESULTS, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved -> {NEW_INSTRUCT_RESULTS}")
