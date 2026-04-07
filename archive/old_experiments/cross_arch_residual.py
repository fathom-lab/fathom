#!/usr/bin/env python3
"""
Cross-Architecture Residual Stream Depth Test
==============================================
Tests whether the surface/insight depth effect generalizes across architectures
using TransformerLens residual stream norms (no SAE transcoders needed).

For each model, hooks capture the residual stream norm at every layer.
The "depth" metric is the mean layer index of maximum residual stream norm.
Surface prompts should show higher mean-layer (output-proximal concentration);
insight prompts should show lower mean-layer (more distributed, earlier processing).

Models: google/gemma-2-2b, Qwen/Qwen3-0.6B, gpt2-xl
"""

import gc
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

PAIRS = [
    ("Gravity pulls objects toward the center of the Earth, making them",
     "Gravity pulls objects toward the center of the Earth, which means that"),
    ("Vaccines train the immune system to recognize pathogens, preventing",
     "Vaccines train the immune system to recognize pathogens, which works because"),
    ("Evolution occurs when organisms with favorable traits survive and pass those traits to their",
     "Evolution occurs when organisms with favorable traits survive, which explains how"),
    ("Antibiotics kill bacteria by targeting their cell walls and preventing them from",
     "Antibiotics kill bacteria by targeting their cell walls, which causes resistance because"),
    ("DNA contains the genetic instructions for building proteins, encoding",
     "DNA contains the genetic instructions for building proteins, which explains how"),
    ("Climate change is caused by greenhouse gas emissions that trap heat in the",
     "Climate change accelerates because warming releases frozen methane and melts ice that was reflecting"),
    ("Memory forms when experiences create new connections between neurons and repeated activation makes those connections",
     "Memory consolidation requires sleep because the hippocampus replays experiences to the cortex to transfer them into"),
    ("Fire needs oxygen, fuel, and heat, and all three are present when something",
     "Fire needs oxygen, fuel, and heat, and each one feeds the others when something"),
    ("Tectonic plates shift and collide beneath the surface, causing",
     "Tectonic plates are driven by convection in the mantle where heat differences create currents that drag the surface into"),
    ("Black holes are regions where gravity is so strong that nothing can",
     "Black holes form when stellar mass collapses below the Schwarzschild radius and spacetime curvature becomes sufficient to trap"),
]

MODELS = [
    {"name": "google/gemma-2-2b", "trust_remote_code": False},
    {"name": "Qwen/Qwen3-0.6B", "trust_remote_code": True},
    {"name": "gpt2-xl", "trust_remote_code": False},
]


def score_model(model_name: str, trust_remote_code: bool = False) -> dict:
    """Score all pairs on a single model using residual stream norms."""
    import torch
    from transformer_lens import HookedTransformer

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return {"model": model_name, "error": str(e), "pairs": []}

    n_layers = model.cfg.n_layers
    device = model.cfg.device
    print(f"  Loaded: {n_layers} layers, device={device}, time={time.time()-t0:.1f}s")

    pair_results = []
    surface_depths = []
    insight_depths = []

    for i, (surface, insight) in enumerate(PAIRS):
        try:
            # Score surface prompt
            s_depth = get_max_norm_layer(model, surface, n_layers)
            # Score insight prompt
            i_depth = get_max_norm_layer(model, insight, n_layers)

            delta = s_depth - i_depth
            direction = "+" if delta > 0 else "-"

            pair_results.append({
                "pair_idx": i,
                "surface_depth": round(s_depth, 4),
                "insight_depth": round(i_depth, 4),
                "delta": round(delta, 4),
                "direction": direction,
            })
            surface_depths.append(s_depth)
            insight_depths.append(i_depth)

            print(f"  [{i+1:2d}/10] s={s_depth:.4f} i={i_depth:.4f} delta={delta:+.4f} {direction}")

        except Exception as e:
            print(f"  [{i+1:2d}/10] ERROR: {e}")
            pair_results.append({
                "pair_idx": i, "error": str(e)
            })

    # Compute summary statistics
    if surface_depths and insight_depths:
        s_mean = float(np.mean(surface_depths))
        i_mean = float(np.mean(insight_depths))
        mean_delta = s_mean - i_mean
        ratio = s_mean / i_mean if i_mean > 0 else 0
        pct_expected = sum(1 for p in pair_results if p.get("delta", 0) > 0) / len(pair_results) * 100

        # Paired t-test
        from scipy import stats
        deltas = [p["delta"] for p in pair_results if "delta" in p]
        if len(deltas) >= 3:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            # One-tailed (we predict surface > insight)
            p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        else:
            t_stat, p_one = 0, 1.0

        summary = {
            "surface_mean": round(s_mean, 4),
            "insight_mean": round(i_mean, 4),
            "mean_delta": round(mean_delta, 4),
            "ratio": round(ratio, 4),
            "pct_expected_direction": round(pct_expected, 1),
            "t_statistic": round(float(t_stat), 4),
            "p_value_one_tailed": round(float(p_one), 6),
            "n_pairs": len(deltas),
        }
        print(f"\n  SUMMARY: surface_mean={s_mean:.4f}, insight_mean={i_mean:.4f}")
        print(f"  delta={mean_delta:+.4f}, ratio={ratio:.4f}, pct_dir={pct_expected:.1f}%")
        print(f"  t={t_stat:.4f}, p={p_one:.6f}")
    else:
        summary = {"error": "No valid pairs scored"}

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    # Free memory
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "model": model_name,
        "n_layers": n_layers,
        "method": "residual_stream_max_norm_layer",
        "n_pairs": len(PAIRS),
        "summary": summary,
        "pairs": pair_results,
        "elapsed_seconds": round(elapsed, 1),
    }


def get_max_norm_layer(model, prompt: str, n_layers: int) -> float:
    """Get the mean layer index of maximum residual stream norm.

    Hooks capture the L2 norm of residual stream at each layer.
    Returns the layer index (normalized to 0-n_layers) where norm is maximized,
    averaged across all token positions.
    """
    import torch

    layer_norms = []

    def hook_fn(activation, hook):
        # activation shape: (batch, seq_len, d_model)
        norms = activation.float().norm(dim=-1).mean(dim=-1)  # mean across seq positions
        layer_norms.append(norms.item())

    # Register hooks on residual stream at each layer
    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

    with torch.no_grad():
        try:
            model.run_with_hooks(
                prompt,
                fwd_hooks=[(name, hook_fn) for name in hook_names],
            )
        except Exception:
            # Fallback: try without hooks and use logits cache
            layer_norms = []
            _, cache = model.run_with_cache(prompt, names_filter=lambda n: "resid_post" in n)
            for i in range(n_layers):
                key = f"blocks.{i}.hook_resid_post"
                if key in cache:
                    norms = cache[key].float().norm(dim=-1).mean().item()
                    layer_norms.append(norms)
            del cache

    if not layer_norms:
        return n_layers / 2.0  # fallback: middle

    # Weighted mean layer (each layer weighted by its norm)
    total = sum(layer_norms)
    if total == 0:
        return n_layers / 2.0
    weighted_mean = sum(i * n for i, n in enumerate(layer_norms)) / total
    return weighted_mean


def main():
    print("FATHOM Cross-Architecture Residual Stream Depth Test")
    print("=" * 60)
    print(f"Pairs: {len(PAIRS)}")
    print(f"Models: {[m['name'] for m in MODELS]}")
    print(f"Method: mean layer index of max residual stream L2 norm")

    all_results = []
    for model_config in MODELS:
        result = score_model(model_config["name"], model_config.get("trust_remote_code", False))
        all_results.append(result)

    # Save results
    output = {
        "test": "cross_architecture_residual_stream_depth",
        "method": "TransformerLens residual stream L2 norm, weighted mean layer",
        "n_pairs": len(PAIRS),
        "hypothesis": "Surface prompts concentrate activation at higher (output-proximal) layers; insight prompts show more distributed earlier-layer activation.",
        "results": all_results,
    }

    out_path = Path(__file__).parent / "cross_arch_residual_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"CROSS-ARCHITECTURE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Layers':>6} {'S Mean':>8} {'I Mean':>8} {'Delta':>8} {'Ratio':>7} {'%Dir':>6} {'p':>10}")
    print("-" * 80)
    for r in all_results:
        if "error" in r and not r.get("pairs"):
            print(f"{r['model']:<25} {'ERROR':>6}  {r.get('error','')[:40]}")
        else:
            s = r.get("summary", {})
            print(f"{r['model']:<25} {r.get('n_layers','?'):>6} "
                  f"{s.get('surface_mean',0):>8.4f} {s.get('insight_mean',0):>8.4f} "
                  f"{s.get('mean_delta',0):>+8.4f} {s.get('ratio',0):>7.4f} "
                  f"{s.get('pct_expected_direction',0):>5.1f}% "
                  f"{s.get('p_value_one_tailed',1):>10.6f}")


if __name__ == "__main__":
    main()
