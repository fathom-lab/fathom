"""
FATHOM CONSTANT — GPT-2 Direct Activation Test
==============================================
Tests the Fathom depth ratio on GPT-2 using raw activation hooks
(no transcoders needed). Measures layer activation distribution
directly, same method as the depth_api_v2.py scorer.

If K≈1.0302 holds → the constant is architecture-independent.
"""
import json
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE = Path(__file__).parent / "fathom_gpt2_results.json"

# Standard 30 pairs from the paper
PAIRS = [
    {"surface": "What is the capital of France?", "insight": "Why did Paris become the capital of France rather than another city?"},
    {"surface": "What year did World War II end?", "insight": "What were the key factors that caused the Axis powers to collapse in 1945?"},
    {"surface": "What is the speed of light?", "insight": "Why is the speed of light the universal speed limit in physics?"},
    {"surface": "Name three types of clouds.", "insight": "How do cumulus clouds form and why do they sometimes produce thunderstorms?"},
    {"surface": "Who wrote Hamlet?", "insight": "How does Hamlet's soliloquy 'To be or not to be' reflect existentialist philosophy?"},
    {"surface": "What is photosynthesis?", "insight": "Why is the light-dependent reaction of photosynthesis spatially separated from the Calvin cycle?"},
    {"surface": "What is the boiling point of water?", "insight": "Why does water's boiling point change with altitude and what does this reveal about intermolecular forces?"},
    {"surface": "What are the primary colors?", "insight": "Why are red, green, and blue the primary colors for light but red, yellow, and blue for pigments?"},
    {"surface": "What is DNA?", "insight": "Why does the double-helix structure of DNA make it particularly suited for information storage and replication?"},
    {"surface": "Name the planets in our solar system.", "insight": "Why do all planets in our solar system orbit in roughly the same plane?"},
    {"surface": "What is the capital of Japan?", "insight": "How did Tokyo become Japan's capital and what does this reflect about Japanese modernization?"},
    {"surface": "What is gravity?", "insight": "Why does Einstein's general relativity explain gravity more accurately than Newton's law of gravitation?"},
    {"surface": "What is the mitochondria?", "insight": "Why is the endosymbiotic theory the most accepted explanation for mitochondrial origin?"},
    {"surface": "How many sides does a hexagon have?", "insight": "Why do honeybees build hexagonal cells rather than circular or square ones?"},
    {"surface": "What language do Brazilians speak?", "insight": "Why does Brazil speak Portuguese rather than Spanish despite being surrounded by Spanish-speaking countries?"},
    {"surface": "What is inflation?", "insight": "Why does moderate inflation generally benefit economic growth while hyperinflation destroys it?"},
    {"surface": "What is the largest ocean?", "insight": "Why does the Pacific Ocean have more seismic activity than the Atlantic?"},
    {"surface": "What is a black hole?", "insight": "Why does time slow down near a black hole and what does this reveal about the nature of spacetime?"},
    {"surface": "What is democracy?", "insight": "Why do democracies tend to avoid wars with each other — the democratic peace theory?"},
    {"surface": "What is the immune system?", "insight": "Why does the adaptive immune system take days to respond to new pathogens when speed is critical?"},
    {"surface": "What is machine learning?", "insight": "Why does gradient descent find good solutions in high-dimensional loss landscapes despite the existence of many local minima?"},
    {"surface": "What is a recession?", "insight": "Why do recessions tend to be self-reinforcing once consumer confidence drops below a critical threshold?"},
    {"surface": "What is the sun made of?", "insight": "Why does nuclear fusion in the sun's core produce so much energy relative to chemical reactions?"},
    {"surface": "What is consciousness?", "insight": "Why does Integrated Information Theory propose that consciousness is substrate-independent and how might this apply to AI systems?"},
    {"surface": "What is evolution?", "insight": "Why does natural selection favor cooperation in certain environments despite individual competitive pressures?"},
    {"surface": "What is quantum mechanics?", "insight": "Why does observation collapse the quantum wave function and what does this imply about the role of information in physics?"},
    {"surface": "What is the nitrogen cycle?", "insight": "Why is nitrogen fixation the rate-limiting step in ecosystem productivity despite nitrogen being 78% of the atmosphere?"},
    {"surface": "What is a neuron?", "insight": "Why do some neurons have myelin sheaths and how does myelination fundamentally change information processing speed?"},
    {"surface": "What is compound interest?", "insight": "Why does compound interest have exponential rather than linear growth and what are the implications for long-term wealth?"},
    {"surface": "What is the theory of relativity?", "insight": "Why did Einstein's special relativity require abandoning the concept of absolute simultaneity?"},
]

def get_layer_activations(model, tokenizer, text, max_len=200):
    """Extract mean activation magnitude per layer via hooks."""
    inputs = tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True).to(DEVICE)
    
    layer_acts = []
    hooks = []
    
    def make_hook(idx):
        def hook(module, input, output):
            # output is (batch, seq, hidden) for GPT-2
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            act = out.detach().float().abs().mean().item()
            layer_acts.append((idx, act))
        return hook
    
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_hook(make_hook(i)))
    
    with torch.no_grad():
        model(**inputs)
    
    for h in hooks:
        h.remove()
    
    layer_acts.sort(key=lambda x: x[0])
    return [a for _, a in layer_acts]

def compute_depth_ratio(acts):
    """
    Compute the mid-layer / output-layer activation ratio.
    Same formula as depth_scorer.py: ratio of mid-third to final-third.
    """
    n = len(acts)
    if n < 3:
        return 1.0
    
    third = n // 3
    early = np.mean(acts[:third]) if third > 0 else acts[0]
    mid = np.mean(acts[third:2*third])
    late = np.mean(acts[2*third:])
    
    total = early + mid + late
    if total == 0:
        return 1.0
    
    # Depth = mid-layer concentration relative to output
    mid_weight = mid / total
    late_weight = late / total
    
    if late_weight == 0:
        return 1.0
    
    return mid_weight / late_weight

def main():
    print("=" * 60)
    print("FATHOM CONSTANT — GPT-2 DIRECT ACTIVATION TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    print("\n[LOAD] Loading GPT-2 Medium...")
    t0 = time.time()
    model_name = "gpt2-medium"  # 24 layers, 355M params — fits easily
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
    model.eval()
    model.to(DEVICE)
    print(f"[LOAD] Model loaded in {time.time()-t0:.1f}s — {len(model.transformer.h)} layers")
    
    results = []
    surface_ratios = []
    insight_ratios = []
    
    for i, pair in enumerate(PAIRS):
        print(f"\n[{i+1:02d}/30] Testing pair...")
        
        s_acts = get_layer_activations(model, tokenizer, pair["surface"])
        i_acts = get_layer_activations(model, tokenizer, pair["insight"])
        
        s_ratio = compute_depth_ratio(s_acts)
        i_ratio = compute_depth_ratio(i_acts)
        
        delta = i_ratio - s_ratio
        surface_ratios.append(s_ratio)
        insight_ratios.append(i_ratio)
        
        results.append({
            "pair_id": i,
            "surface_ratio": round(s_ratio, 4),
            "insight_ratio": round(i_ratio, 4),
            "delta": round(delta, 4),
            "deeper": delta > 0,
        })
        
        print(f"  surface={s_ratio:.4f}  insight={i_ratio:.4f}  delta={delta:+.4f}  {'DEEP✓' if delta > 0 else 'shallow'}")
    
    # Compute the Fathom constant
    mean_surface = np.mean(surface_ratios)
    mean_insight = np.mean(insight_ratios)
    constant = mean_insight / mean_surface if mean_surface > 0 else 0
    pct_correct = sum(1 for r in results if r["deeper"]) / len(results)
    
    gemma_constant = 1.0302
    convergence = abs(constant - gemma_constant) < 0.15
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: {model_name} ({len(model.transformer.h)} layers)")
    print(f"Pairs tested: {len(PAIRS)}")
    print(f"Mean surface ratio: {mean_surface:.4f}")
    print(f"Mean insight ratio: {mean_insight:.4f}")
    print(f"Fathom constant (K): {constant:.4f}")
    print(f"Gemma constant:      {gemma_constant:.4f}")
    print(f"Delta:               {abs(constant - gemma_constant):.4f}")
    print(f"Convergent (<0.15):  {'YES ✓' if convergence else 'NO ✗'}")
    print(f"% insight deeper:    {pct_correct:.1%}")
    print("=" * 60)
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "n_layers": len(model.transformer.h),
        "n_pairs": len(PAIRS),
        "mean_surface_ratio": round(mean_surface, 4),
        "mean_insight_ratio": round(mean_insight, 4),
        "fathom_constant": round(constant, 4),
        "gemma_constant": gemma_constant,
        "delta_from_gemma": round(abs(constant - gemma_constant), 4),
        "convergent": convergence,
        "pct_insight_deeper": round(pct_correct, 3),
        "pair_results": results,
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[SAVED] {RESULTS_FILE}")
    
    if convergence:
        print("\n🔬 FATHOM CONSTANT CONFIRMED CROSS-ARCHITECTURE")
        print(f"   K={constant:.4f} converges with Gemma K=1.0302")
        print("   This supports the universality claim in the patent.")
    else:
        print(f"\n⚠️  Constant did NOT converge: K={constant:.4f} vs {gemma_constant}")
        print("   Delta too large — may indicate different architecture family")

if __name__ == "__main__":
    main()
