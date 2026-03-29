"""
FATHOM × RLHF — Does Alignment Training Reduce Reasoning Depth?
================================================================
Hypothesis: RLHF/instruction tuning compresses reasoning depth (phi).
Method: Compare base vs instruction-tuned versions of same architecture.
Prediction: instruct model scores LOWER depth than base on reasoning tasks.

If confirmed: alignment reduces phi. Nature paper.

Models tested:
- Gemma-2-2B (base) vs Gemma-2-2B-IT (instruction tuned)
- Uses raw activation hooks (no transcoders needed)
"""
import json
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = Path(__file__).parent / "rlhf_depth_results.json"

# 30 reasoning pairs — same as calibration set
PAIRS = [
    {"surface": "What is the capital of France?",
     "insight": "Why did Paris become the dominant capital rather than other French cities?"},
    {"surface": "What causes inflation?",
     "insight": "Why does moderate inflation benefit economies while hyperinflation destroys them?"},
    {"surface": "What is photosynthesis?",
     "insight": "Why is the light-dependent reaction spatially separated from the Calvin cycle?"},
    {"surface": "What is DNA?",
     "insight": "Why does the double-helix make DNA particularly suited for information storage?"},
    {"surface": "What is gravity?",
     "insight": "Why does general relativity describe gravity more accurately than Newtonian mechanics?"},
    {"surface": "What is the immune system?",
     "insight": "Why does adaptive immunity take days despite speed being critical to survival?"},
    {"surface": "What is consciousness?",
     "insight": "Why might integrated information be a necessary condition for consciousness?"},
    {"surface": "What is evolution?",
     "insight": "Why does natural selection favor cooperation in certain evolutionary environments?"},
    {"surface": "What is machine learning?",
     "insight": "Why does gradient descent find good solutions despite the existence of many local minima?"},
    {"surface": "What is the mitochondria?",
     "insight": "Why does the endosymbiotic theory explain mitochondrial origin better than alternatives?"},
    {"surface": "What causes recessions?",
     "insight": "Why do recessions become self-reinforcing once consumer confidence drops below a threshold?"},
    {"surface": "What is the sun made of?",
     "insight": "Why does nuclear fusion produce exponentially more energy than chemical reactions?"},
    {"surface": "What are neurons?",
     "insight": "Why do some neurons have myelin sheaths and how does this change information processing speed?"},
    {"surface": "What is compound interest?",
     "insight": "Why does compound interest produce exponential rather than linear wealth accumulation?"},
    {"surface": "What is quantum mechanics?",
     "insight": "Why does observation collapse the quantum wave function and what does this mean for reality?"},
    {"surface": "What is democracy?",
     "insight": "Why do democracies statistically avoid wars with each other — the democratic peace theory?"},
    {"surface": "What is sleep?",
     "insight": "Why is sleep essential for memory consolidation at a neurological level?"},
    {"surface": "What is language?",
     "insight": "Why does language acquisition become significantly harder after early childhood?"},
    {"surface": "What is a black hole?",
     "insight": "Why does time slow down near a black hole and what does this reveal about spacetime?"},
    {"surface": "What is entropy?",
     "insight": "Why does entropy always increase in isolated systems and what does this imply for causality?"},
    {"surface": "What is free will?",
     "insight": "Why does compatibilism argue free will is consistent with a deterministic universe?"},
    {"surface": "What is Gödel's theorem?",
     "insight": "Why does Gödel's incompleteness theorem apply to any sufficiently powerful formal system?"},
    {"surface": "What is the nitrogen cycle?",
     "insight": "Why is nitrogen fixation the rate-limiting step in ecosystem productivity despite atmospheric abundance?"},
    {"surface": "What is a mirror?",
     "insight": "Why does a mirror reverse front-to-back but we perceive it as reversing left-right?"},
    {"surface": "What is stochastic resonance?",
     "insight": "Why does moderate noise improve signal detection in nonlinear biological systems?"},
    {"surface": "What is the placebo effect?",
     "insight": "Why does belief in treatment produce measurable physiological changes independent of the treatment itself?"},
    {"surface": "What is emergence?",
     "insight": "Why do complex systems exhibit properties that cannot be predicted from their components?"},
    {"surface": "What is time?",
     "insight": "Why does time appear to flow only in one direction despite physical laws being time-symmetric?"},
    {"surface": "What is anaesthesia?",
     "insight": "Why do anaesthetics reversibly suppress consciousness and what does this reveal about its neural basis?"},
    {"surface": "What is intelligence?",
     "insight": "Why does general intelligence correlate across diverse cognitive domains despite their apparent independence?"},
]

def get_depth_score(model, tokenizer, text, max_len=300):
    """Compute mid-layer activation ratio using hooks."""
    inputs = tokenizer(text, return_tensors="pt", max_length=max_len, 
                       truncation=True, padding=False).to(DEVICE)
    
    layer_acts = []
    hooks = []
    
    def make_hook(i):
        def hook(module, inp, out):
            if isinstance(out, tuple): out = out[0]
            layer_acts.append(out.detach().float().abs().mean().item())
        return hook
    
    # Get transformer layers (works for most HF models)
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers  # Llama, Gemma style
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h  # GPT-2 style
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        layers = model.model.decoder.layers  # OPT style
    
    if layers is None:
        return None
    
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))
    
    with torch.no_grad():
        model(**inputs)
    
    for h in hooks:
        h.remove()
    
    if len(layer_acts) < 3:
        return None
    
    n = len(layer_acts)
    t = n // 3
    early = np.mean(layer_acts[:t])
    mid = np.mean(layer_acts[t:2*t])
    late = np.mean(layer_acts[2*t:])
    total = early + mid + late
    if total == 0:
        return None
    
    return (mid / total) / (late / total) if late > 0 else 1.0

def score_model(model_name, model, tokenizer, pairs):
    """Score all pairs on a model, return surface and insight depths."""
    print(f"\n[MODEL] {model_name}")
    surface_depths = []
    insight_depths = []
    
    for i, pair in enumerate(pairs):
        s = get_depth_score(model, tokenizer, pair["surface"])
        ins = get_depth_score(model, tokenizer, pair["insight"])
        
        if s is not None and ins is not None:
            surface_depths.append(s)
            insight_depths.append(ins)
            delta = ins - s
            print(f"  [{i+1:02d}] surface={s:.4f} insight={ins:.4f} delta={delta:+.4f}")
    
    K = np.mean(insight_depths) / np.mean(surface_depths) if surface_depths else 0
    pct_deeper = sum(1 for s, ins in zip(surface_depths, insight_depths) if ins > s) / len(surface_depths)
    
    return {
        "model": model_name,
        "n_pairs": len(surface_depths),
        "surface_mean": round(float(np.mean(surface_depths)), 4),
        "insight_mean": round(float(np.mean(insight_depths)), 4),
        "K": round(float(K), 4),
        "pct_insight_deeper": round(float(pct_deeper), 3),
        "delta_mean": round(float(np.mean(np.array(insight_depths) - np.array(surface_depths))), 4),
    }

def main():
    print("=" * 65)
    print("FATHOM × RLHF — BASE vs INSTRUCTION-TUNED DEPTH COMPARISON")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print("Hypothesis: RLHF reduces reasoning depth (phi)")
    print("=" * 65)
    
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    results = []
    
    model_pairs = [
        ("google/gemma-2-2b", "google/gemma-2-2b-it"),
    ]
    
    for base_name, instruct_name in model_pairs:
        print(f"\n{'='*65}")
        print(f"COMPARING: {base_name.split('/')[-1]} vs {instruct_name.split('/')[-1]}")
        print(f"{'='*65}")
        
        # Load and score base model
        print(f"\n[LOAD] {base_name}...")
        t0 = time.time()
        tok_base = AutoTokenizer.from_pretrained(base_name)
        mod_base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype, 
                                                         device_map="auto", 
                                                         low_cpu_mem_usage=True)
        mod_base.eval()
        print(f"  Loaded in {time.time()-t0:.1f}s")
        
        base_result = score_model(base_name, mod_base, tok_base, PAIRS)
        results.append({**base_result, "type": "base"})
        
        # Free GPU memory
        del mod_base
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
        import gc; gc.collect()
        
        # Load and score instruct model
        print(f"\n[LOAD] {instruct_name}...")
        t0 = time.time()
        tok_it = AutoTokenizer.from_pretrained(instruct_name)
        mod_it = AutoModelForCausalLM.from_pretrained(instruct_name, dtype=dtype,
                                                        device_map="auto",
                                                        low_cpu_mem_usage=True)
        mod_it.eval()
        print(f"  Loaded in {time.time()-t0:.1f}s")
        
        instruct_result = score_model(instruct_name, mod_it, tok_it, PAIRS)
        results.append({**instruct_result, "type": "instruct"})
        
        del mod_it
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
        import gc; gc.collect()
        
        # Compare
        depth_delta = base_result["K"] - instruct_result["K"]
        print(f"\n{'='*65}")
        print(f"RESULT: BASE K={base_result['K']:.4f} vs INSTRUCT K={instruct_result['K']:.4f}")
        print(f"DEPTH REDUCTION FROM RLHF: {depth_delta:+.4f}")
        if depth_delta > 0.05:
            print("✓ HYPOTHESIS CONFIRMED: RLHF reduces reasoning depth")
        elif depth_delta < -0.05:
            print("✗ HYPOTHESIS REJECTED: RLHF increases depth")
        else:
            print("~ INCONCLUSIVE: small effect size")
        print(f"{'='*65}")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "RLHF reduces reasoning depth (phi)",
        "results": results,
        "conclusion": {
            "base_K": results[0]["K"] if results else None,
            "instruct_K": results[1]["K"] if len(results) > 1 else None,
            "depth_reduction": (results[0]["K"] - results[1]["K"]) if len(results) > 1 else None,
        }
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[SAVED] {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
