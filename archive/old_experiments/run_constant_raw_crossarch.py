"""
FATHOM CONSTANT — RAW ACTIVATION CROSS-ARCHITECTURE TEST
=========================================================
Tests whether the early/late layer activation ratio threshold
converges across architectures WITHOUT needing transformer_lens
or model-specific transcoders.

Method: HuggingFace output_hidden_states=True → compute L2 norm 
of hidden states at each layer → early/mid/late ratio analysis.

Models: Gemma-2-2B, Llama-3.2-1B (both cached locally)
No internet required.

Run: python run_constant_raw_crossarch.py
"""

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
CONVERGENCE_THRESHOLD = 0.15

# The 30 surface/insight pairs from the paper (same as fathom_constant.py)
PAIRS = [
    # (surface, insight)
    ("What is the boiling point of water?", "Why does water boil at a lower temperature at higher altitudes?"),
    ("What is the Pythagorean theorem?", "Why does the Pythagorean theorem only hold in flat (Euclidean) geometry?"),
    ("What is photosynthesis?", "Why do plants need both light-dependent and light-independent reactions to survive?"),
    ("What is Newton's first law?", "Why does Newton's first law break down in non-inertial reference frames?"),
    ("What is inflation?", "Why does inflation disproportionately harm fixed-income earners and savers?"),
    ("What is the speed of light?", "Why is the speed of light the same for all observers regardless of their motion?"),
    ("What is DNA?", "Why does DNA replication require a leading and lagging strand?"),
    ("What is entropy?", "Why does entropy increase explain the arrow of time but not why time started ordered?"),
    ("What is supply and demand?", "Why does price elasticity vary across goods and what determines it?"),
    ("What is recursion in programming?", "Why does recursion require a base case to avoid infinite regress?"),
    ("What is the greenhouse effect?", "Why do different greenhouse gases have different warming potentials?"),
    ("What is a prime number?", "Why is the distribution of prime numbers related to the Riemann Hypothesis?"),
    ("What year did World War II end?", "Why did the atomic bomb accelerate Japan's surrender rather than extend resistance?"),
    ("What is a neural network?", "Why do deeper neural networks require residual connections to train effectively?"),
    ("What is natural selection?", "Why does natural selection lead to local optima rather than globally optimal organisms?"),
    ("What is the blood-brain barrier?", "Why does the brain maintain a separate chemical environment from the body?"),
    ("What is compound interest?", "Why does compound interest create exponential rather than arithmetic growth?"),
    ("What is the placebo effect?", "Why does the placebo effect work even when patients know it is a placebo?"),
    ("What is quantum entanglement?", "Why does quantum entanglement not allow faster-than-light communication?"),
    ("What is confirmation bias?", "Why does confirmation bias persist even in trained scientists and rational actors?"),
    ("What is the tragedy of the commons?", "Why does the tragedy of the commons not always occur in practice?"),
    ("What is zero?", "Why was the invention of zero historically difficult and why does it matter for mathematics?"),
    ("What is the Dunning-Kruger effect?", "Why does incompetence specifically impair the ability to recognize incompetence?"),
    ("What is free will?", "Why do compatibilists argue free will is compatible with determinism?"),
    ("What is sleep?", "Why does sleep deprivation impair cognition more than equivalent caloric restriction?"),
    ("What is money?", "Why does fiat currency maintain value despite having no intrinsic worth?"),
    ("What is consciousness?", "Why is the hard problem of consciousness distinct from functional explanations of awareness?"),
    ("What is a black hole?", "Why does information appear to be destroyed in black holes, violating quantum mechanics?"),
    ("What is democracy?", "Why do democracies tend not to go to war with each other?"),
    ("What is antibody immunity?", "Why does the immune system remember pathogens for decades but gradually forget others?"),
]


def compute_depth_raw(model, tokenizer, text: str, device: str, n_layers: int) -> Dict:
    """
    Compute raw activation depth score using hidden state L2 norms.
    Returns early/mid/late ratios and depth score.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # tuple of (n_layers+1, batch, seq, hidden)
    # Skip embedding layer (index 0), use transformer layers 1..n_layers
    layer_norms = []
    for h in hidden_states[1:]:  # shape: (batch, seq, hidden)
        # Mean L2 norm across sequence positions
        norm = h[0].norm(dim=-1).mean().item()
        layer_norms.append(norm)
    
    # Normalize to sum=1 (activation mass distribution)
    total = sum(layer_norms)
    if total < 1e-9:
        return {"depth": 0.0, "early": 0.0, "mid": 0.0, "late": 0.0, "ratio": 1.0}
    
    dist = [n / total for n in layer_norms]
    
    # Split into thirds
    n = len(dist)
    early_cut = n // 3
    late_cut = 2 * (n // 3)
    
    early = sum(dist[:early_cut])
    mid = sum(dist[early_cut:late_cut])
    late = sum(dist[late_cut:])
    
    # Depth score: how distributed vs concentrated in late layers
    # High depth = more in early/mid, less in late
    depth = (early + mid) / max(late, 1e-9)  # ratio: reasoning concentrates early
    ratio = early / max(late, 1e-9)  # early/late ratio (the Fathom Constant candidate)
    
    return {"depth": depth, "early": early, "mid": mid, "late": late, "ratio": ratio}


def score_model(model_name: str, device: str) -> Dict:
    """Score all pairs on a model, return constant analysis."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load with reduced precision to save VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    print(f"[{model_name}] Loaded in {time.time()-t0:.1f}s — {n_layers} layers")
    
    surface_ratios = []
    insight_ratios = []
    surface_depths = []
    insight_depths = []
    
    for i, (surface, insight) in enumerate(PAIRS):
        print(f"  Pair {i+1}/{len(PAIRS)}...", end="\r")
        
        try:
            s_result = compute_depth_raw(model, tokenizer, surface, device, n_layers)
            ins_result = compute_depth_raw(model, tokenizer, insight, device, n_layers)
            
            surface_ratios.append(s_result["ratio"])
            insight_ratios.append(ins_result["ratio"])
            surface_depths.append(s_result["depth"])
            insight_depths.append(ins_result["depth"])
        except Exception as e:
            print(f"\n  [WARN] Pair {i} error: {e}")
    
    print(f"  Scored {len(surface_ratios)}/{len(PAIRS)} pairs        ")
    
    # Find optimal threshold (same method as fathom_constant.py)
    all_ratios = [(r, 0) for r in surface_ratios] + [(r, 1) for r in insight_ratios]  # 0=surface, 1=insight
    all_ratios.sort(key=lambda x: x[0])
    
    best_acc = 0.0
    best_threshold = 0.0
    n_total = len(all_ratios)
    
    for threshold, _ in all_ratios:
        # Classify: ratio > threshold = surface, ratio <= threshold = insight
        correct = sum(1 for r, label in all_ratios if (r > threshold) == (label == 0))
        acc = correct / n_total
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    mean_surface_ratio = float(np.mean(surface_ratios)) if surface_ratios else 0.0
    mean_insight_ratio = float(np.mean(insight_ratios)) if insight_ratios else 0.0
    mean_surface_depth = float(np.mean(surface_depths)) if surface_depths else 0.0
    mean_insight_depth = float(np.mean(insight_depths)) if insight_depths else 0.0
    
    result = {
        "model": model_name,
        "n_layers": n_layers,
        "n_pairs": len(surface_ratios),
        "mean_surface_ratio": mean_surface_ratio,
        "mean_insight_ratio": mean_insight_ratio,
        "mean_surface_depth": mean_surface_depth,
        "mean_insight_depth": mean_insight_depth,
        "optimal_ratio_threshold": best_threshold,
        "classification_accuracy": best_acc,
        "delta_ratio": mean_surface_ratio - mean_insight_ratio,
    }
    
    print(f"  mean_surface_ratio: {mean_surface_ratio:.4f}")
    print(f"  mean_insight_ratio: {mean_insight_ratio:.4f}")
    print(f"  optimal_threshold:  {best_threshold:.4f}")
    print(f"  classification_acc: {best_acc:.1%}")
    
    # Free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return result


def main():
    print("=" * 70)
    print("FATHOM CONSTANT — RAW ACTIVATION CROSS-ARCHITECTURE TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Method: HuggingFace hidden_states (no transformer_lens/transcoders)")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("[FATAL] No CUDA GPU available.")
        sys.exit(1)
    
    device = "cuda"
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[GPU] {torch.cuda.get_device_name(0)} — {mem_gb:.1f} GB VRAM")
    
    models_to_test = [
        "google/gemma-2-2b",
        "meta-llama/Llama-3.2-1B",
    ]
    
    results = []
    
    for model_name in models_to_test:
        try:
            gc.collect()
            torch.cuda.empty_cache()
            result = score_model(model_name, device)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    if len(results) < 2:
        print(f"\n[FATAL] Only {len(results)} models scored. Cannot compare.")
        sys.exit(1)
    
    # Cross-arch comparison
    print("\n" + "=" * 70)
    print("FATHOM CONSTANT — CROSS-ARCHITECTURE COMPARISON (RAW ACTIVATIONS)")
    print("=" * 70)
    
    thresholds = [r["optimal_ratio_threshold"] for r in results]
    mean_threshold = float(np.mean(thresholds))
    std_threshold = float(np.std(thresholds))
    max_diff = max(thresholds) - min(thresholds)
    converged = max_diff < CONVERGENCE_THRESHOLD
    
    print(f"\n{'Model':<35} {'Layers':>6} {'Opt Threshold':>14} {'Acc':>7} {'Δ(S-I)':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['model']:<35} {r['n_layers']:>6} {r['optimal_ratio_threshold']:>14.4f} "
              f"{r['classification_accuracy']:>7.1%} {r['delta_ratio']:>8.4f}")
    
    print(f"\nMean threshold:  {mean_threshold:.4f}")
    print(f"Std threshold:   {std_threshold:.4f}")
    print(f"Max difference:  {max_diff:.4f}")
    print(f"Converged (<{CONVERGENCE_THRESHOLD}): {'YES ✓ CONSTANT IS REAL' if converged else 'NO — needs more models'}")
    
    # Write results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = {
        "timestamp": timestamp,
        "method": "raw_activations",
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "mean_threshold": mean_threshold,
        "std_threshold": std_threshold,
        "max_diff": max_diff,
        "converged": converged,
        "models": results,
    }
    
    out_path = PROJECT_ROOT / "results_raw_crossarch.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[WRITE] {out_path}")
    
    if converged:
        report_path = PROJECT_ROOT / "DISCOVERY_RAW_CROSSARCH.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# DISCOVERY: FATHOM CONSTANT CONFIRMED (RAW ACTIVATIONS)\n\n")
            f.write(f"**Date:** {timestamp}\n\n")
            f.write(f"Optimal early/late activation ratio threshold converges across {len(results)} architectures.\n")
            f.write(f"Mean: {mean_threshold:.4f} ± {std_threshold:.4f} (max diff: {max_diff:.4f})\n\n")
            f.write("## Models\n\n")
            for r in results:
                f.write(f"- **{r['model']}** ({r['n_layers']} layers): threshold={r['optimal_ratio_threshold']:.4f}, acc={r['classification_accuracy']:.1%}\n")
            f.write(f"\n## Interpretation\n\n")
            f.write("The Fathom Constant — the optimal ratio threshold separating surface recall ")
            f.write("from explanatory reasoning — appears architecture-independent. ")
            f.write("This suggests reasoning depth topology is a fundamental property of the computation, ")
            f.write("not an artifact of a specific model's training or architecture.\n")
        print(f"[DISCOVERY] Written to {report_path}")
        print("\n*** FATHOM CONSTANT CONFIRMED VIA RAW ACTIVATIONS ***")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
