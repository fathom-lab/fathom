"""
FATHOM CONSTANT — GPT-2-XL Cross-Model Test
=============================================
Tests whether the optimal early/late ratio threshold (3.538 from Gemma-2-2B)
holds on GPT-2-XL (48 layers). If |difference| < 0.15 -> the constant is real.

GPT-2-XL weights and transcoder set already cached locally.
Run: python run_constant_gpt2.py
"""
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Gemma baseline from experiment 2
GEMMA_BASELINE = {
    "model": "google/gemma-2-2b",
    "optimal_ratio": 3.5381,
    "classification_accuracy": 0.625,
    "mean_surface_depth": 8.0609,
    "mean_insight_depth": 8.1921,
    "n_layers": 26,
}

# GPT-2-XL config
GPT2_CONFIG = {
    "model_name": "gpt2-xl",
    "transcoder_set": "jbloom/GPT2-Small-SAEs-Reformatted",
    "n_layers": 48,
}

CONVERGENCE_THRESHOLD = 0.15

# Standard pairs
from fathom_constant import STANDARD_PAIRS
from run_experiments import EXTRA_CONSTANT_PAIRS


def main():
    t_start = time.time()

    print("=" * 70)
    print("FATHOM CONSTANT — GPT-2-XL CROSS-MODEL TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Gemma baseline ratio: {GEMMA_BASELINE['optimal_ratio']:.4f}")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD}")
    print(f"GPT-2-XL layers: {GPT2_CONFIG['n_layers']}")
    print("=" * 70)

    # GPU check
    if not torch.cuda.is_available():
        print("[FATAL] No CUDA GPU.")
        sys.exit(1)
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    mem = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
    print(f"[GPU] {name} — {mem:.1f} GB")
    print(f"[GPU] CUDA {torch.version.cuda}, PyTorch {torch.__version__}")

    # Pre-load cleanup
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"[INIT] VRAM free before load: {free_gb:.2f} GB")

    # Load model
    print(f"\n[INIT] Loading DepthScorer ({GPT2_CONFIG['model_name']})...")
    t_load = time.time()

    from depth_scorer import DepthScorer

    scorer = None
    try:
        scorer = DepthScorer(
            model_name=GPT2_CONFIG["model_name"],
            model_type=GPT2_CONFIG["transcoder_set"],
            max_feature_nodes=150,
        )
    except OSError as e:
        print(f"[INIT] First attempt failed: {e}")
        print("[INIT] Retrying with flash SDP disabled...")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.backends.cuda.enable_flash_sdp(False)
        except Exception:
            pass
        try:
            scorer = DepthScorer(
                model_name=GPT2_CONFIG["model_name"],
                model_type=GPT2_CONFIG["transcoder_set"],
                max_feature_nodes=150,
            )
        except Exception as e2:
            print(f"[FATAL] Cannot load {GPT2_CONFIG['model_name']}: {e2}")
            sys.exit(1)

    if scorer is None:
        print("[FATAL] No model loaded.")
        sys.exit(1)

    load_time = time.time() - t_load
    print(f"[INIT] Model loaded in {load_time:.1f}s")

    # Score pairs
    from fathom_constant import FathomConstantCalculator

    calculator = FathomConstantCalculator()
    all_pairs = list(STANDARD_PAIRS) + list(EXTRA_CONSTANT_PAIRS)

    print(f"\n[RUN] Scoring {len(all_pairs)} pairs on {GPT2_CONFIG['model_name']}...")
    result = calculator.compute_universal_ratio(
        scorer, pairs=all_pairs, model_id=GPT2_CONFIG["model_name"]
    )

    # Compare to Gemma baseline
    ratio_diff = abs(result.optimal_ratio - GEMMA_BASELINE["optimal_ratio"])
    converged = ratio_diff < CONVERGENCE_THRESHOLD

    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("FATHOM CONSTANT — CROSS-MODEL COMPARISON")
    print("=" * 70)
    print(f"Gemma-2-2B ratio:   {GEMMA_BASELINE['optimal_ratio']:.4f}  "
          f"(acc: {GEMMA_BASELINE['classification_accuracy']:.1%}, "
          f"{GEMMA_BASELINE['n_layers']} layers)")
    print(f"GPT-2-XL ratio:     {result.optimal_ratio:.4f}  "
          f"(acc: {result.classification_accuracy:.1%}, "
          f"{GPT2_CONFIG['n_layers']} layers)")
    print(f"Difference:         {ratio_diff:.4f}")
    print(f"Converged (<{CONVERGENCE_THRESHOLD}):  "
          f"{'YES' if converged else 'NO'}")
    print(f"Total time:         {total_time:.1f}s")

    # Write report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"""## Cross-Model Comparison

| Model | Layers | Optimal Ratio | Accuracy | Mean Surface Depth | Mean Insight Depth |
|-------|--------|--------------|----------|-------------------|-------------------|
| google/gemma-2-2b | {GEMMA_BASELINE['n_layers']} | {GEMMA_BASELINE['optimal_ratio']:.4f} | {GEMMA_BASELINE['classification_accuracy']:.1%} | {GEMMA_BASELINE['mean_surface_depth']:.4f} | {GEMMA_BASELINE['mean_insight_depth']:.4f} |
| gpt2-xl | {GPT2_CONFIG['n_layers']} | {result.optimal_ratio:.4f} | {result.classification_accuracy:.1%} | {result.mean_surface_depth:.4f} | {result.mean_insight_depth:.4f} |

**Ratio difference:** {ratio_diff:.4f}
**Convergence threshold:** {CONVERGENCE_THRESHOLD}
**Converged:** {'YES' if converged else 'NO'}

## Architecture Notes

- Gemma-2-2B: 26 layers, ~2B parameters, modern architecture
- GPT-2-XL: 48 layers, ~1.5B parameters, original GPT-2 architecture
- Different training data, tokenizers, and attention patterns
- If the ratio converges despite these differences, it is architecture-independent

## Interpretation

{'The Fathom Constant holds across architectures. Gemma-2-2B (26 layers) and GPT-2-XL (48 layers) independently arrive at an optimal early/late ratio within ' + f'{ratio_diff:.4f}' + ' of each other — below the ' + f'{CONVERGENCE_THRESHOLD}' + ' convergence threshold. The ratio that separates surface recall from explanatory reasoning is not a model-specific artifact. It is a property of the reasoning task itself.' if converged else f'Ratios differ by {ratio_diff:.4f} — above the {CONVERGENCE_THRESHOLD} convergence threshold. The constant may require more models, refined prompt design, or layer-count normalization to confirm across architectures with very different depth profiles.'}

## Timing

Total run time: {total_time:.1f}s ({total_time / 60:.1f} min)

## Raw Data

```json
{json.dumps({
    "model_id": result.model_id,
    "n_surface": result.n_surface,
    "n_insight": result.n_insight,
    "mean_surface_depth": result.mean_surface_depth,
    "mean_insight_depth": result.mean_insight_depth,
    "mean_surface_early_late": result.mean_surface_early_late,
    "mean_insight_early_late": result.mean_insight_early_late,
    "optimal_ratio": result.optimal_ratio,
    "classification_accuracy": result.classification_accuracy,
    "delta": result.delta,
}, indent=2)}
```
"""

    report_path = PROJECT_ROOT / "EXPERIMENT_CONSTANT_GPT2.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Fathom Constant — GPT-2-XL Cross-Model Test\n\n"
                f"**Date:** {timestamp}\n"
                f"**Runner:** run_constant_gpt2.py\n\n{body}")
    print(f"\n[WRITE] {report_path}")

    # Discovery file if converged
    if converged:
        discovery_path = PROJECT_ROOT / "DISCOVERY_CONSTANT_GPT2.md"
        with open(discovery_path, "w", encoding="utf-8") as f:
            f.write(f"# DISCOVERY: Fathom Constant Holds on GPT-2-XL\n\n")
            f.write(f"**Date:** {timestamp}\n\n")
            f.write(f"## Finding\n\n")
            f.write(f"The optimal early/late ratio threshold converges across "
                    f"Gemma-2-2B ({GEMMA_BASELINE['optimal_ratio']:.4f}, "
                    f"{GEMMA_BASELINE['n_layers']} layers) and "
                    f"GPT-2-XL ({result.optimal_ratio:.4f}, "
                    f"{GPT2_CONFIG['n_layers']} layers). "
                    f"Difference: {ratio_diff:.4f} (below {CONVERGENCE_THRESHOLD} threshold).\n\n")
            f.write(f"## Significance\n\n")
            f.write(f"These models differ in architecture (Gemma vs GPT-2), scale "
                    f"({GEMMA_BASELINE['n_layers']} vs {GPT2_CONFIG['n_layers']} layers), "
                    f"training data, and tokenization. The convergence of the optimal "
                    f"ratio despite these differences suggests the reasoning depth "
                    f"threshold is a property of the computational task, not the model.\n\n")
            f.write(f"## Evidence\n\n{body}")
        print(f"[DISCOVERY] Written to {discovery_path}")
        print("\n" + "*" * 70)
        print("*** FATHOM CONSTANT CONFIRMED ON GPT-2-XL ***")
        print("*" * 70)
    else:
        print("\nBelow convergence threshold. Constant not confirmed on this pair.")

    print("\nDone.")


if __name__ == "__main__":
    main()
