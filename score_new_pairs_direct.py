#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, gc, json, time
if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    sys.stderr.reconfigure(encoding='utf-8',errors='replace')
"""
Direct scorer for 14 new pairs — base then instruct.
Max features = 150 (safe for 8GB VRAM).
Saves after every pair.
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

BASE_DIR    = Path(__file__).parent
RESEARCH_DIR = BASE_DIR / ".." / "research"

EXISTING_BASE    = BASE_DIR / "results_power_replication.json"
EXISTING_INSTRUCT = BASE_DIR / "gemma_it_experiment_results.json"
NEW_BASE_FILE    = BASE_DIR / "new_pairs_base_results.json"
NEW_INST_FILE    = BASE_DIR / "new_pairs_inst_results.json"
REPORT_FILE      = BASE_DIR / "SELECTIVE_AMPLIFICATION_REPORT.md"

MODEL_BASE    = "google/gemma-2-2b"
MODEL_INSTRUCT = "google/gemma-2-2b-it"
TRANSCODER_SET = "gemma"
MAX_FEATURES   = 150   # safe ceiling for 8GB VRAM

SENSITIVITY = {
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

# ── Load pairs ────────────────────────────────────────────────────────────────

def load_existing_topics():
    topics = set()
    for path in [EXISTING_BASE, EXISTING_INSTRUCT]:
        if not path.exists(): continue
        d = json.load(open(path))
        if isinstance(d, dict):
            for p in d.get("pairs",[]): topics.add(p.get("topic",""))
        elif isinstance(d, list):
            for p in d: topics.add(p.get("pair_id",""))
    return topics

def load_new_pairs(skip_topics):
    pairs, seen = [], set(skip_topics)
    for path in [RESEARCH_DIR/"experiment_11_depth_pairs.json",
                 BASE_DIR/"fathom_benchmark_v1.json"]:
        if not path.exists(): continue
        d = json.load(open(path))
        raw = d.get("pairs",d) if isinstance(d,dict) else d
        for p in (raw if isinstance(raw,list) else []):
            topic = p.get("topic",p.get("domain",""))
            key = topic.lower().replace(" ","_").replace("-","_")
            if key in seen: continue
            sens = SENSITIVITY.get(key,"unknown")
            if sens == "unknown":
                for k,v in SENSITIVITY.items():
                    if k in key or key in k: sens=v; break
            if sens == "unknown": continue
            s = p.get("surface",p.get("surface_prompt",""))
            i = p.get("insight",p.get("insight_prompt",""))
            st = p.get("surface_target","")
            it = p.get("insight_target","")
            if not s or not i: continue
            if not st:
                ws=s.strip().split(); st=ws[-1].strip(".,;:!?"); s=" ".join(ws[:-1])+" "
            if not it:
                wi=i.strip().split(); it=wi[-1].strip(".,;:!?"); i=" ".join(wi[:-1])+" "
            pairs.append({"id":key,"sensitivity":sens,
                          "surface":{"prompt":s,"target":st},
                          "insight":{"prompt":i,"target":it}})
            seen.add(key)
    return pairs

# ── Score one model ────────────────────────────────────────────────────────────

def score_with_model(model_name, pairs, out_file):
    import torch
    from circuit_tracer import ReplacementModel, attribute

    # Load from checkpoint if any
    done = json.load(open(out_file)) if out_file.exists() else {}
    remaining = [p for p in pairs if p["id"] not in done]
    if not remaining:
        print(f"  [{model_name}] All already scored.")
        return done

    free_vram, total_vram = torch.cuda.mem_get_info()
    print(f"\n[LOAD] {model_name} ... (VRAM free: {free_vram/1e9:.2f}GB / {total_vram/1e9:.2f}GB)")
    if free_vram < 6e9:
        print(f"  [WARN] Only {free_vram/1e9:.1f}GB VRAM free — may OOM. Continuing anyway.")
    t0 = time.time()
    try:
        model = ReplacementModel.from_pretrained(
            model_name, TRANSCODER_SET,
            dtype=torch.bfloat16, backend="transformerlens",
        )
    except Exception as e:
        print(f"  [FATAL] Model load failed: {e}")
        raise
    print(f"  Loaded in {time.time()-t0:.1f}s  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    n = len(remaining)
    for idx, pair in enumerate(remaining):
        pid = pair["id"]
        print(f"  [{idx+1}/{n}] {pid} ({pair['sensitivity']})...", end="", flush=True)
        t1 = time.time()
        try:
            sg = attribute(pair["surface"]["prompt"]+pair["surface"]["target"], model,
                           attribution_targets=[pair["surface"]["target"]],
                           max_feature_nodes=MAX_FEATURES, batch_size=16)
            ig = attribute(pair["insight"]["prompt"]+pair["insight"]["target"], model,
                           attribution_targets=[pair["insight"]["target"]],
                           max_feature_nodes=MAX_FEATURES, batch_size=16)
            sl = sg.active_features[:,0].float().cpu().numpy()
            il = ig.active_features[:,0].float().cpu().numpy()
            if len(sl)==0 or len(il)==0:
                print(" SKIP (empty)"); continue
            sd = float(np.mean(sl)); id_ = float(np.mean(il))
            delta = sd - id_
            done[pid] = {"surface_depth":sd,"insight_depth":id_,"delta":delta,
                         "surface_n":len(sl),"insight_n":len(il),"sensitivity":pair["sensitivity"]}
            direction = "S>I" if delta>0 else "I>S"
            print(f" d={delta:+.3f} ({direction}) [{time.time()-t1:.1f}s]")
        except Exception as e:
            print(f" ERR: {str(e)[:60]}")
        finally:
            # Flush VRAM between calls
            if 'sg' in dir(): del sg
            if 'ig' in dir(): del ig
            torch.cuda.empty_cache()
        # Save after every pair
        with open(out_file,"w") as f: json.dump(done,f,indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"  Done. {len(done)} scored.")
    return done

# ── Final analysis ─────────────────────────────────────────────────────────────

def analyze_combined(new_base, new_inst, existing_base_path, existing_inst_path):
    # Load existing
    base_all = {}
    if existing_base_path.exists():
        d = json.load(open(existing_base_path))
        for p in d.get("pairs",[]):
            base_all[p["topic"]] = {"delta":p["delta"],"sensitivity":
                SENSITIVITY.get(p["topic"].lower().replace(" ","_"),"low")}
    inst_all = {}
    if existing_inst_path.exists():
        d = json.load(open(existing_inst_path))
        for p in (d if isinstance(d,list) else []):
            inst_all[p["pair_id"]] = {"delta":p["delta"]}

    # Merge new results
    for pid,v in new_base.items():
        base_all[pid] = {"delta":v["delta"],"sensitivity":v.get("sensitivity","low")}
    for pid,v in new_inst.items():
        inst_all[pid] = {"delta":v["delta"]}

    # Match pairs
    matched = []
    for pid, b in base_all.items():
        if pid in inst_all:
            i = inst_all[pid]
            matched.append({"topic":pid,"sensitivity":b["sensitivity"],
                            "base_delta":b["delta"],"inst_delta":i["delta"],
                            "change":i["delta"]-b["delta"]})

    print(f"\n{'='*55}")
    print(f"  SELECTIVE AMPLIFICATION — FINAL ({len(matched)} pairs)")
    print(f"{'='*55}")

    groups = defaultdict(list)
    for m in matched: groups[m["sensitivity"]].append(m["change"])

    for sens in ["high","medium","low"]:
        g = groups.get(sens,[])
        if g:
            d = "AMPLIFIED" if np.mean(g)>0 else "COMPRESSED"
            print(f"  {sens.upper():8}: change={np.mean(g):+.4f} (n={len(g)}) {d}")

    high, low = groups.get("high",[]), groups.get("low",[])
    if len(high)>=3 and len(low)>=3:
        t, p = stats.ttest_ind(high, low, alternative="greater")
        cd = (np.mean(high)-np.mean(low)) / np.sqrt(
            (np.std(high,ddof=1)**2+np.std(low,ddof=1)**2)/2)
        print(f"\n  p-value (high > low): {p:.6f}")
        print(f"  Cohen's d:            {cd:.4f}")
        if p<0.05:   verdict = "CONFIRMED p<0.05"
        elif p<0.1:  verdict = "MARGINAL  p<0.10"
        else:        verdict = f"not yet significant (p={p:.3f})"
        print(f"  VERDICT: {verdict}")

        # Write report
        lines = [f"# SELECTIVE DEPTH AMPLIFICATION\n",
                 f"**Pairs**: {len(matched)} (high={len(high)}, medium={len(groups.get('medium',[]))}, low={len(low)})\n",
                 f"**Verdict**: {verdict}\n\n## Group Stats\n",
                 "| Sensitivity | n | Mean Change | Direction |",
                 "|---|---|---|---|"]
        for s in ["high","medium","low"]:
            g2 = groups.get(s,[])
            if g2:
                lines.append(f"| {s.upper()} | {len(g2)} | {np.mean(g2):+.4f} | {'AMPLIFIED' if np.mean(g2)>0 else 'COMPRESSED'} |")
        lines += [f"\n**p-value**: {p:.6f} | **Cohen's d**: {cd:.4f}",
                  "\n\n---\n*Fathom -- US Patent Pending 64/020,489 & 64/021,113*"]
        with open(REPORT_FILE,"w",encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n  Saved -> {REPORT_FILE}")
    else:
        print("  Not enough data for primary test")

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  SELECTIVE AMPLIFICATION — DIRECT SCORER")
    print("="*55)

    existing = load_existing_topics()
    pairs = load_new_pairs(existing)
    print(f"\nNew pairs: {len(pairs)}")
    for s in ["high","medium","low"]:
        print(f"  {s}: {sum(1 for p in pairs if p['sensitivity']==s)}")

    if not pairs:
        print("Nothing new to score. Running analysis only.")
    else:
        # Phase 1: Base
        base_new = score_with_model(MODEL_BASE, pairs, NEW_BASE_FILE)
        # Phase 2: Instruct
        inst_new = score_with_model(MODEL_INSTRUCT, pairs, NEW_INST_FILE)

    base_new = json.load(open(NEW_BASE_FILE)) if NEW_BASE_FILE.exists() else {}
    inst_new = json.load(open(NEW_INST_FILE)) if NEW_INST_FILE.exists() else {}
    analyze_combined(base_new, inst_new, EXISTING_BASE, EXISTING_INSTRUCT)
