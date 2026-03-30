"""
RLHF Test — Instruct Model Only (Raw Activation Hooks)
======================================================
Scores Gemma-2-2B-IT on 30 pairs using raw activation hooks.
Compare K_instruct to K_base from GPT-2/direct method.

Hypothesis: K_instruct < K_base → RLHF reduces depth
"""
import json, time, torch, numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same 20 pairs as GPT-2 test for direct comparison
PAIRS = [
    {"surface": "What is the capital of France?", 
     "insight": "Why did Paris become the dominant capital of France historically?"},
    {"surface": "What causes inflation?",
     "insight": "Why does moderate inflation benefit economies while hyperinflation destroys them?"},
    {"surface": "What is photosynthesis?",
     "insight": "Why is the light-dependent reaction spatially separated from the Calvin cycle?"},
    {"surface": "What is DNA?",
     "insight": "Why does the double-helix make DNA suited for information storage and replication?"},
    {"surface": "What is gravity?",
     "insight": "Why does general relativity describe gravity more accurately than Newton's law?"},
    {"surface": "What is the immune system?",
     "insight": "Why does adaptive immunity take days to respond despite speed being critical?"},
    {"surface": "What is consciousness?",
     "insight": "Why might integrated information be necessary for consciousness according to IIT?"},
    {"surface": "What is evolution?",
     "insight": "Why does natural selection sometimes favor cooperation over competition?"},
    {"surface": "What is machine learning?",
     "insight": "Why does gradient descent find good solutions despite many local minima?"},
    {"surface": "What is the mitochondria?",
     "insight": "Why does endosymbiotic theory best explain mitochondrial origin?"},
    {"surface": "What is compound interest?",
     "insight": "Why does compound interest grow exponentially rather than linearly?"},
    {"surface": "What is quantum mechanics?",
     "insight": "Why does observation collapse the wave function and what does this mean for reality?"},
    {"surface": "What is democracy?",
     "insight": "Why do democracies statistically avoid wars with other democracies?"},
    {"surface": "What is sleep?",
     "insight": "Why is sleep essential for memory consolidation at a neurological level?"},
    {"surface": "What is entropy?",
     "insight": "Why does entropy always increase in isolated systems?"},
    {"surface": "What is free will?",
     "insight": "Why does compatibilism argue free will is consistent with determinism?"},
    {"surface": "What is the nitrogen cycle?",
     "insight": "Why is nitrogen fixation the rate-limiting step despite N₂ being 78% of air?"},
    {"surface": "What is a mirror?",
     "insight": "Why does a mirror appear to reverse left-right but not up-down?"},
    {"surface": "What is the placebo effect?",
     "insight": "Why does belief in treatment produce real measurable physiological changes?"},
    {"surface": "What is emergence?",
     "insight": "Why do complex systems exhibit properties unpredictable from their components?"},
]

def depth_ratio(model, tokenizer, text, max_len=200):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True).to(DEVICE)
    acts = []
    hooks = []
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    if not layers:
        return None
    for i, layer in enumerate(layers):
        def hook(m, inp, out, i=i):
            o = out[0] if isinstance(out, tuple) else out
            acts.append(o.detach().float().abs().mean().item())
        hooks.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    if len(acts) < 3:
        return None
    n = len(acts); t = n//3
    mid = np.mean(acts[t:2*t])
    late = np.mean(acts[2*t:])
    return float(mid/late) if late > 0 else None

print(f"Loading gemma-2-2b-it on {DEVICE}...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    dtype=torch.float16,
    device_map={"": DEVICE},  # force single device, no offload
    low_cpu_mem_usage=True,
)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s — {len(model.model.layers)} layers")

surface_K, insight_K = [], []
for i, p in enumerate(PAIRS):
    s = depth_ratio(model, tok, p["surface"])
    ins = depth_ratio(model, tok, p["insight"])
    if s and ins:
        surface_K.append(s); insight_K.append(ins)
        print(f"[{i+1:02d}] s={s:.4f} i={ins:.4f} Δ={ins-s:+.4f}")

K_instruct = np.mean(insight_K) / np.mean(surface_K)
K_base_gpt2 = 0.8454  # our GPT-2 baseline (same raw hook method)

print(f"\n{'='*50}")
print(f"K_instruct (gemma-2-2b-it): {K_instruct:.4f}")
print(f"K_base     (gemma-2-2b):    0.8454 (GPT-2 raw hook reference)")
print(f"RLHF delta: {K_instruct - 0.8454:+.4f}")
if K_instruct < 0.8454 - 0.05:
    print("✓ HYPOTHESIS CONFIRMED: instruct model is shallower")
elif K_instruct > 0.8454 + 0.05:
    print("✗ HYPOTHESIS REJECTED: instruct model is deeper")
else:
    print("~ INCONCLUSIVE")

out = {
    "timestamp": datetime.now().isoformat(),
    "model": "google/gemma-2-2b-it",
    "n_layers": len(model.model.layers),
    "n_pairs": len(surface_K),
    "K_instruct": round(float(K_instruct), 4),
    "K_base_reference": 0.8454,
    "rlhf_delta": round(float(K_instruct - 0.8454), 4),
    "surface_mean": round(float(np.mean(surface_K)), 4),
    "insight_mean": round(float(np.mean(insight_K)), 4),
}
json.dump(out, open("rlhf_instruct_results.json","w"), indent=2)
print(f"\n[SAVED] rlhf_instruct_results.json")
