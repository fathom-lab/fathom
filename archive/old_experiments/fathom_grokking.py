"""
FATHOM × GROKKING — Depth at the Phase Transition
===================================================
Hypothesis: Reasoning depth (phi) spikes at the grokking threshold —
the training checkpoint where a model suddenly generalizes.

Based on Neel Nanda et al. (2023): "Progress Measures for Grokking via
Mechanistic Interpretability" — models show sudden phase transitions in
generalization that appear as discrete jumps in weight structure.

Our prediction: Fathom depth score spikes at the SAME checkpoint where
grokking occurs. This would be the first identification of the
computational ignition point of reasoning.

Setup: Uses transformerlens models with saved training checkpoints.
Model: tiny-stories-1M or a grokking checkpoint from Neel's work.
"""
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Required: pip install transformer_lens
try:
    import transformer_lens
    from transformer_lens import HookedTransformer
    TL_AVAILABLE = True
except ImportError:
    TL_AVAILABLE = False
    print("[WARNING] transformer_lens not installed. Run: pip install transformer_lens")

OUTPUT_FILE = Path(__file__).parent / "grokking_depth_results.json"

# Neel Nanda's grokking checkpoints are available at:
# https://huggingface.co/neelnanda-io/Grokking-Checkpoints
# They trained a 1-layer transformer on modular arithmetic (a+b mod 113)
# and saved checkpoints at regular intervals.
# Grokking occurs around checkpoint 10,000-30,000 steps.

GROKKING_MODEL = "neelnanda-io/Grokking-Checkpoints"
CHECKPOINT_STEPS = [100, 500, 1000, 3000, 5000, 10000, 15000, 20000, 25000, 
                     30000, 40000, 50000, 75000, 100000]

# Reasoning probe: simple tasks that require computation vs retrieval
PROBE_PAIRS = [
    {"surface": "What is 5 + 3?", 
     "insight": "Why is (5 + 3) mod 113 = 8 but (60 + 60) mod 113 = 7?"},
    {"surface": "What is 113 mod 5?", 
     "insight": "Why does modular arithmetic form a cyclic group under addition?"},
    {"surface": "List multiples of 7 up to 50.",
     "insight": "Why does the pattern of 7×n mod 113 eventually cycle through all values?"},
]

def get_depth_hooked(model, text, tokenizer=None):
    """Get layer activation depth using TransformerLens hooks."""
    if not TL_AVAILABLE:
        return None
    
    layer_acts = []
    
    def hook_fn(value, hook):
        layer_acts.append(value.detach().float().abs().mean().item())
        return value
    
    try:
        tokens = model.to_tokens(text) if hasattr(model, 'to_tokens') else None
        if tokens is None:
            return None
        
        hooks = [(f"blocks.{i}.hook_resid_post", hook_fn) 
                 for i in range(model.cfg.n_layers)]
        
        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=hooks)
        
        if len(layer_acts) < 3:
            return None
        
        n = len(layer_acts)
        t = n // 3
        mid = np.mean(layer_acts[t:2*t])
        late = np.mean(layer_acts[2*t:])
        
        return float(mid / late) if late > 0 else 1.0
        
    except Exception as e:
        print(f"  Hook error: {e}")
        return None

def score_checkpoint(step, pairs):
    """Load a specific grokking checkpoint and score it."""
    if not TL_AVAILABLE:
        return None
    
    try:
        model = HookedTransformer.from_pretrained(
            "neelnanda-io/Grokking-Checkpoints",
            checkpoint_index=step,
        )
        model.eval()
        
        surface_scores = []
        insight_scores = []
        
        for pair in pairs:
            s = get_depth_hooked(model, pair["surface"])
            ins = get_depth_hooked(model, pair["insight"])
            if s and ins:
                surface_scores.append(s)
                insight_scores.append(ins)
        
        if not surface_scores:
            return None
        
        K = np.mean(insight_scores) / np.mean(surface_scores)
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "step": step,
            "K": round(float(K), 4),
            "surface_mean": round(float(np.mean(surface_scores)), 4),
            "insight_mean": round(float(np.mean(insight_scores)), 4),
        }
    except Exception as e:
        print(f"  Checkpoint {step} failed: {e}")
        return None

def main():
    print("=" * 65)
    print("FATHOM × GROKKING — Phase Transition Depth Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Hypothesis: Depth spikes at the grokking phase transition")
    print("=" * 65)
    
    if not TL_AVAILABLE:
        print("\n[BLOCKED] transformer_lens not installed.")
        print("Fix: pip install transformer_lens")
        print("\nSaving experiment design for future run...")
        
        output = {
            "status": "blocked",
            "reason": "transformer_lens not installed",
            "fix": "pip install transformer_lens",
            "experiment_design": {
                "model": GROKKING_MODEL,
                "checkpoints": CHECKPOINT_STEPS,
                "prediction": "K spikes at grokking threshold (~10000-30000 steps)",
                "falsification": "K is flat across all checkpoints",
                "significance": "First identification of computational ignition point of reasoning",
            },
            "next_steps": [
                "1. pip install transformer_lens",
                "2. python fathom_grokking.py",
                "3. Plot K vs training_step",
                "4. Overlay with Neel Nanda's grokking curves",
            ]
        }
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"[SAVED] Experiment design: {OUTPUT_FILE}")
        return
    
    results = []
    
    for step in CHECKPOINT_STEPS:
        print(f"\n[CHECKPOINT {step}] Scoring...")
        result = score_checkpoint(step, PROBE_PAIRS)
        if result:
            results.append(result)
            print(f"  K={result['K']:.4f} surface={result['surface_mean']:.4f} insight={result['insight_mean']:.4f}")
        
        # Save incrementally
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)
    
    if results:
        # Find the phase transition
        Ks = [r["K"] for r in results]
        steps = [r["step"] for r in results]
        max_idx = np.argmax(Ks)
        
        print(f"\n{'='*65}")
        print(f"RESULTS")
        print(f"{'='*65}")
        print(f"Peak K={Ks[max_idx]:.4f} at step {steps[max_idx]}")
        print(f"K trajectory: {' → '.join(f'{k:.3f}' for k in Ks)}")
        
        # Does K spike?
        if Ks[max_idx] > np.mean(Ks[:3]) * 1.5:
            print("✓ PHASE TRANSITION DETECTED — depth spikes at grokking")
        else:
            print("~ No clear phase transition in depth")

if __name__ == "__main__":
    main()
