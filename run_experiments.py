"""
FATHOM EXPERIMENT RUNNER
========================

Runs all 4 experiments in sequence on the local GPU.
Single command: python run_experiments.py

Experiments:
  1. HallucinationCorrelator — TruthfulQA depth vs accuracy correlation
  2. FathomConstantCalculator — universal early/late ratio on gemma-2-2b
  3. PhiApproximator — phi_proxy distribution across shallow-to-deep prompts
  4. UniversalCognitionMapper — transformer vs neuroscience signature comparison

All results written to EXPERIMENT_*.md files in the project root.
Discovery files auto-generated if significance thresholds are met.
Nothing leaves the machine.
"""

import os
# ── Memory / GPU env vars — MUST be set before any torch/safetensors import ──
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path

# ── Resolve project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


# ── Dual logger: console + file ──────────────────────────────────────

class TeeLogger:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, log_path: Path):
        self._stdout = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        self._log.close()


# ── GPU diagnostics ───────────────────────────────────────────────────

def gpu_check():
    """Verify CUDA is available and print GPU info."""
    import torch
    if not torch.cuda.is_available():
        print("[FATAL] No CUDA GPU detected. Experiments require GPU.")
        sys.exit(1)
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    mem = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
    print(f"[GPU] {name} — {mem:.1f} GB")
    print(f"[GPU] CUDA {torch.version.cuda}, PyTorch {torch.__version__}")
    return name


def gpu_flush():
    """Aggressive GPU memory cleanup between experiments."""
    import torch
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


# ── Markdown report writers ───────────────────────────────────────────

def write_experiment_md(filename: str, title: str, body: str):
    """Write an EXPERIMENT_*.md file to project root."""
    path = PROJECT_ROOT / filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"# {title}\n\n**Date:** {timestamp}\n**Runner:** run_experiments.py\n\n{body}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[WRITE] {path}")


# ── Prompt batteries ─────────────────────────────────────────────────

# 20 prompts spanning shallow recall to deep reasoning for phi analysis
PHI_PROMPTS = [
    # Shallow / recall-heavy
    "The capital of France is",
    "Water freezes at zero degrees Celsius, turning into",
    "The speed of light is approximately 300,000 kilometers per",
    "Abraham Lincoln was the 16th president of the",
    "The chemical formula for water is",
    "The Great Wall of China was built to protect against",
    "DNA stands for deoxyribonucleic",
    # Moderate complexity
    "Antibiotics work by targeting bacterial cell structures, which means",
    "The greenhouse effect traps heat in the atmosphere because",
    "Supply and demand determine market prices when",
    "Neural networks learn patterns by adjusting weights through",
    "Democracy requires separation of powers to prevent",
    "The Doppler effect causes the pitch of a siren to change because",
    # Deep reasoning / explanatory
    "The second law of thermodynamics implies that entropy increases, which fundamentally means",
    "Consciousness remains difficult to define scientifically because",
    "Godel's incompleteness theorems demonstrate that formal systems cannot",
    "The measurement problem in quantum mechanics reveals that observation",
    "Free will is difficult to reconcile with determinism because",
    "The hard problem of consciousness differs from easy problems because",
    "Emergent properties arise in complex systems because the interactions between components",
]

# 40 matched pairs for the Fathom Constant (20 standard + 20 additional)
EXTRA_CONSTANT_PAIRS = [
    ("Mitosis divides a cell into two identical daughter cells, producing",
     "Mitosis divides a cell into two identical daughter cells, which matters because"),
    ("The speed of sound in air is about 343 meters per second, reaching",
     "The speed of sound in air is about 343 meters per second, which depends on"),
    ("Batteries store chemical energy and convert it to electrical energy, powering",
     "Batteries store chemical energy and convert it to electrical energy, which works because"),
    ("Ocean currents distribute heat across the planet, affecting",
     "Ocean currents distribute heat across the planet, which happens because"),
    ("Metamorphic rocks form under extreme heat and pressure, becoming",
     "Metamorphic rocks form under extreme heat and pressure, which explains why"),
    ("Insulin regulates blood sugar by signaling cells to absorb glucose, maintaining",
     "Insulin regulates blood sugar by signaling cells to absorb glucose, which works because"),
    ("Satellites orbit Earth by balancing velocity and gravity, staying",
     "Satellites orbit Earth by balancing velocity and gravity, which works because"),
    ("Plate tectonics causes earthquakes along fault lines, releasing",
     "Plate tectonics causes earthquakes along fault lines, which happens because"),
    ("Photons have both wave and particle properties, behaving as",
     "Photons have both wave and particle properties, which reveals that"),
    ("The water cycle moves water between the ocean, atmosphere, and land, sustaining",
     "The water cycle moves water between the ocean, atmosphere, and land, which matters because"),
    ("Catalysts speed up chemical reactions without being consumed, enabling",
     "Catalysts speed up chemical reactions without being consumed, which works because"),
    ("Neurons transmit signals through electrical impulses and neurotransmitters, connecting",
     "Neurons transmit signals through electrical impulses and neurotransmitters, which explains how"),
    ("Erosion shapes landscapes by wearing away rock and soil, creating",
     "Erosion shapes landscapes by wearing away rock and soil, which happens because"),
    ("Symbiosis describes organisms living together for mutual benefit, maintaining",
     "Symbiosis describes organisms living together for mutual benefit, which evolved because"),
    ("Greenhouse gases trap infrared radiation in the atmosphere, warming",
     "Greenhouse gases trap infrared radiation in the atmosphere, which matters because"),
    ("The doppler effect shifts the frequency of waves from moving sources, changing",
     "The doppler effect shifts the frequency of waves from moving sources, which happens because"),
    ("Enzymes lower activation energy for biochemical reactions, accelerating",
     "Enzymes lower activation energy for biochemical reactions, which works because"),
    ("Tidal forces from the moon stretch the Earth's oceans, causing",
     "Tidal forces from the moon stretch the Earth's oceans, which happens because"),
    ("Superconductors carry current with zero resistance below a critical temperature, enabling",
     "Superconductors carry current with zero resistance below a critical temperature, which occurs because"),
    ("Natural selection favors traits that increase survival, shaping",
     "Natural selection favors traits that increase survival, which explains why"),
]


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Hallucination Correlator V2 (model-generated answers)
# ══════════════════════════════════════════════════════════════════════

def run_hallucination_experiment(scorer):
    """V2: Generate model's own answers, score depth, correlate with correctness."""
    from fathom_hallucination_v2 import HallucinationCorrelatorV2

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: HALLUCINATION CORRELATOR V2")
    print("  (model-generated answers, not pre-supplied prefixes)")
    print("=" * 70)

    correlator = HallucinationCorrelatorV2()

    # Run the full study — handles checkpoint/resume internally
    result = correlator.run_study(
        scorer,
        n=200,
        max_new_tokens=20,
        checkpoint_every=10,
    )

    # Write report (auto-generates DISCOVERY file if significant)
    correlator.write_report(result)

    return result


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Fathom Constant
# ══════════════════════════════════════════════════════════════════════

def run_constant_experiment(scorer):
    """Score 40 balanced surface/insight pairs, compute universal ratio."""
    from fathom_constant import FathomConstantCalculator, STANDARD_PAIRS

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: FATHOM CONSTANT CALCULATOR")
    print("=" * 70)

    calculator = FathomConstantCalculator()

    # Combine standard pairs (20) + extra pairs (20) = 40 total
    all_pairs = list(STANDARD_PAIRS) + EXTRA_CONSTANT_PAIRS

    # Score on gemma-2-2b
    model_result = calculator.compute_universal_ratio(
        scorer, pairs=all_pairs, model_id=scorer.model_name
    )

    # With only one model, we can't call find_constant() (needs >= 2).
    # Document the single-model baseline.
    body = f"""## Hypothesis

There exists a universal early_ratio / late_ratio threshold (the Fathom Constant)
that separates surface recall from explanatory reasoning across all transformer
architectures.

## Method

- Model: {model_result.model_id}
- Pairs: {model_result.n_surface + model_result.n_insight} scored ({model_result.n_surface} surface, {model_result.n_insight} insight)
- Sweep early/late ratio thresholds, find optimal classification boundary
- Report classification accuracy at optimal threshold

## Results — Single Model Baseline

| Metric | Value |
|--------|-------|
| Model | {model_result.model_id} |
| Mean surface depth | {model_result.mean_surface_depth:.4f} |
| Mean insight depth | {model_result.mean_insight_depth:.4f} |
| Depth delta (surface - insight) | {model_result.delta:.4f} |
| Mean surface early/late ratio | {model_result.mean_surface_early_late:.4f} |
| Mean insight early/late ratio | {model_result.mean_insight_early_late:.4f} |
| **Optimal ratio threshold** | **{model_result.optimal_ratio:.4f}** |
| **Classification accuracy** | **{model_result.classification_accuracy:.1%}** |

## Interpretation

The optimal early/late ratio threshold for {model_result.model_id} is **{model_result.optimal_ratio:.4f}**,
achieving **{model_result.classification_accuracy:.1%}** classification accuracy on {model_result.n_surface + model_result.n_insight} prompts.

{'The depth delta is positive (' + f'{model_result.delta:.4f}' + '), confirming that surface prompts have higher mean layer indices (shallower processing) than insight prompts.' if model_result.delta > 0 else 'The depth delta is negative or zero — unexpected direction. Investigate prompt quality.'}

**Next step:** Run on Mistral-7B and other models. If optimal_ratio converges
(std < 0.05 across models), the Fathom Constant is real.

## Raw Data

```json
{json.dumps({
    "model_id": model_result.model_id,
    "n_surface": model_result.n_surface,
    "n_insight": model_result.n_insight,
    "mean_surface_depth": model_result.mean_surface_depth,
    "mean_insight_depth": model_result.mean_insight_depth,
    "mean_surface_early_late": model_result.mean_surface_early_late,
    "mean_insight_early_late": model_result.mean_insight_early_late,
    "optimal_ratio": model_result.optimal_ratio,
    "classification_accuracy": model_result.classification_accuracy,
    "delta": model_result.delta,
}, indent=2)}
```
"""
    write_experiment_md("EXPERIMENT_CONSTANT.md", "Fathom Constant — Single Model Baseline", body)
    return model_result


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Phi Approximator
# ══════════════════════════════════════════════════════════════════════

def run_phi_experiment(scorer):
    """Compute phi_proxy distribution across 20 shallow-to-deep prompts."""
    from fathom_phi import PhiApproximator, CONSCIOUSNESS_CAVEAT

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: PHI APPROXIMATOR")
    print("=" * 70)

    phi = PhiApproximator(max_feature_nodes=scorer.max_feature_nodes)

    # phi_distribution scores all prompts and auto-writes PHI_ANALYSIS.md
    dist = phi.phi_distribution(PHI_PROMPTS, scorer)

    # Write additional experiment report with per-prompt breakdown
    per_prompt_rows = ""
    for pr in dist.per_prompt:
        label = pr.prompt[:55] + "..." if len(pr.prompt) > 55 else pr.prompt
        per_prompt_rows += (
            f"| {label} | {pr.phi_proxy:.4f} | {pr.integration_level} | "
            f"{pr.layer_entropy:.3f} | {pr.concentration_coefficient:.3f} | "
            f"{pr.feature_engagement:.3f} |\n"
        )

    body = f"""## Caveat

{CONSCIOUSNESS_CAVEAT}

## Hypothesis

Prompts that demand deeper reasoning produce higher information integration
(phi_proxy) than recall prompts — consistent with IIT's prediction that
consciousness correlates with integrated information.

## Method

- 20 prompts: 7 shallow recall, 6 moderate, 7 deep reasoning
- Each scored with DepthScorer, then phi_proxy computed from:
  phi_proxy = layer_entropy_norm * (1 - concentration_coeff) * feature_engagement
- Distribution statistics computed across all prompts

## Distribution

| Metric | Value |
|--------|-------|
| n prompts | {dist.n_prompts} |
| mean phi_proxy | {dist.mean_phi:.4f} |
| std phi_proxy | {dist.std_phi:.4f} |
| median phi_proxy | {dist.median_phi:.4f} |
| min phi_proxy | {dist.min_phi:.4f} |
| max phi_proxy | {dist.max_phi:.4f} |
| % high/exceptional integration | {dist.pct_high_integration:.1%} |

## Per-Prompt Breakdown

| Prompt | phi_proxy | Integration | Entropy | Concentration | Engagement |
|--------|-----------|-------------|---------|---------------|------------|
{per_prompt_rows}

## Interpretation

{'Prompts span a range of phi_proxy values, suggesting the metric captures meaningful variation in information integration across cognitive demands.' if dist.std_phi > 0.01 else 'Low variance in phi_proxy — metric may not differentiate well across these prompts.'}

Mean phi_proxy = {dist.mean_phi:.4f} ({phi.interpret_phi(dist.mean_phi)['integration_level']} integration).
{f'{dist.pct_high_integration:.0%} of prompts show high or exceptional integration.' if dist.pct_high_integration > 0 else 'No prompts reached high integration threshold.'}
"""
    write_experiment_md("EXPERIMENT_PHI.md", "Phi Proxy — Information Integration Distribution", body)
    return dist


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Universal Cognition Mapper
# ══════════════════════════════════════════════════════════════════════

def run_universal_experiment(scorer):
    """Compare transformer signatures to neuroscience reference data."""
    from fathom_universal import UniversalCognitionMapper, NEURO_REFERENCE_SIGNATURES
    from fathom_constant import STANDARD_PAIRS
    import torch

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: UNIVERSAL COGNITION MAPPER")
    print("=" * 70)

    mapper = UniversalCognitionMapper()

    # Score standard pairs to get transformer signatures
    print("[Universal] Scoring standard pairs for transformer signatures...")
    surface_results = []
    insight_results = []

    pairs_to_score = list(STANDARD_PAIRS)[:20]

    for i, (surface_prompt, insight_prompt) in enumerate(pairs_to_score):
        try:
            sr = scorer.score(surface_prompt)
            if not math.isnan(sr.depth_score):
                surface_results.append(sr)
        except Exception as e:
            print(f"  Surface {i} failed: {e}")

        gpu_flush()

        try:
            ir = scorer.score(insight_prompt)
            if not math.isnan(ir.depth_score):
                insight_results.append(ir)
        except Exception as e:
            print(f"  Insight {i} failed: {e}")

        gpu_flush()

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(pairs_to_score)}] scored...")

    print(f"[Universal] Got {len(surface_results)} surface, {len(insight_results)} insight results")

    if len(surface_results) < 5 or len(insight_results) < 5:
        raise ValueError(f"Insufficient results for universality analysis")

    # Collect signatures: transformer data + neuro references
    all_results = surface_results + insight_results
    labels = (["surface"] * len(surface_results)) + (["insight"] * len(insight_results))

    sources = [
        {
            "name": f"gemma-2-2b",
            "type": "transformer",
            "depth_results": all_results,
            "labels": labels,
        },
        {
            "name": "neuro_reference",
            "type": "neuro_reference",
        },
    ]

    mapper.collect_signatures(sources)

    # Compute universality
    result = mapper.compute_universality_score()

    # Generate report (auto-writes DISCOVERY_UNIVERSAL.md if coefficient > 0.8)
    report_content = mapper.universality_report(result)

    # Build signature table for our report
    sig_rows = ""
    for s in result.signatures:
        sig_rows += (
            f"| {s.source_name} | {s.source_type} | {s.condition} | "
            f"{s.early_ratio:.3f} | {s.mid_ratio:.3f} | {s.late_ratio:.3f} | {s.n_samples} |\n"
        )

    body = f"""## Hypothesis

Reasoning depth signatures are universal across information processing systems.
Transformer models and biological neural systems share a common topology:
surface recall concentrates in late/output layers, reasoning distributes across
early/mid layers.

## Method

- Score 20 matched surface/insight pairs on gemma-2-2b
- Extract mean [early_ratio, mid_ratio, late_ratio] per condition
- Compare against neuroscience reference signatures (from published fMRI studies)
- Compute cross-system variance and universality coefficient

## Signatures

| Source | Type | Condition | Early | Mid | Late | n |
|--------|------|-----------|-------|-----|------|---|
{sig_rows}

## Results

| Metric | Value |
|--------|-------|
| Sources | {result.n_sources} |
| Transformer sources | {result.n_transformer_sources} |
| Neuro references | {result.n_neuro_sources} |
| Surface signature variance | {result.surface_variance:.6f} |
| Insight signature variance | {result.insight_variance:.6f} |
| **Universality coefficient** | **{result.universality_coefficient:.4f}** |
| **Is universal (> 0.8)?** | **{'YES' if result.is_universal else 'NO'}** |

## Interpretation

{'UNIVERSAL SIGNATURE DETECTED. The reasoning depth topology holds across both transformer architectures and biological neural reference data. Universality coefficient = ' + f'{result.universality_coefficient:.3f}' + '.' if result.is_universal else f'Universality coefficient = {result.universality_coefficient:.3f}. Below threshold (0.8) for claiming universality with current data. More diverse sources needed (additional model architectures, scales).'}

{'**Discovery:** ' + result.discovery if result.discovery else ''}

## Raw Data

```json
{json.dumps(result.to_dict(), indent=2, default=str)}
```
"""
    write_experiment_md("EXPERIMENT_UNIVERSAL.md", "Universal Cognition Mapper", body)
    return result


# ══════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up dual logging
    log_path = PROJECT_ROOT / f"experiment_run_{timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print("=" * 70)
    print("FATHOM EXPERIMENT RUNNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Log: {log_path}")
    print("=" * 70)

    # GPU check
    gpu_name = gpu_check()

    # ── Load model once ───────────────────────────────────────────────
    print("\n[INIT] Loading DepthScorer (gemma-2-2b)...")
    t_load = time.time()

    # Aggressive memory clearing — double pass to catch ref cycles
    gc.collect()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # TF32 uses 10-bit mantissa — halves VRAM for matmuls with negligible precision loss
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"[INIT] VRAM free before load: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
    except Exception:
        pass
    gc.collect()
    gc.collect()

    from depth_scorer import DepthScorer

    try:
        scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=150)
    except OSError as e:
        print(f"[INIT] First load attempt failed: {e}")
        print("[INIT] Clearing CUDA cache and retrying with flash SDP disabled...")
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.backends.cuda.enable_flash_sdp(False)
        except Exception:
            pass
        scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=150)

    print(f"[INIT] Model loaded in {time.time() - t_load:.1f}s")

    # ── Run experiments ───────────────────────────────────────────────
    results = {}
    discoveries = []
    experiments = [
        ("hallucination", "Hallucination Correlator V2", run_hallucination_experiment),
        ("constant", "Fathom Constant", run_constant_experiment),
        ("phi", "Phi Approximator", run_phi_experiment),
        ("universal", "Universal Cognition Mapper", run_universal_experiment),
    ]

    for key, name, fn in experiments:
        print(f"\n{'─' * 70}")
        print(f"[RUN] Starting: {name}")
        print(f"{'─' * 70}")
        t_exp = time.time()

        try:
            result = fn(scorer)
            elapsed = time.time() - t_exp
            results[key] = {"status": "SUCCESS", "elapsed_s": elapsed}
            print(f"\n[DONE] {name} completed in {elapsed:.1f}s")

            # Check for discoveries
            if hasattr(result, "is_significant") and result.is_significant:
                discoveries.append(f"HALLUCINATION: {result.discovery}")
            if hasattr(result, "is_constant") and result.is_constant:
                discoveries.append(f"CONSTANT: {result.discovery}")
            if hasattr(result, "is_universal") and result.is_universal:
                discoveries.append(f"UNIVERSAL: {result.discovery}")
            # PhiDistribution doesn't have a discovery flag, check manually
            if hasattr(result, "pct_high_integration") and result.pct_high_integration > 0.5:
                discoveries.append(
                    f"PHI: >{result.pct_high_integration:.0%} of prompts show high information integration"
                )

        except Exception as e:
            elapsed = time.time() - t_exp
            results[key] = {"status": "FAILED", "error": str(e), "elapsed_s": elapsed}
            print(f"\n[FAIL] {name} failed after {elapsed:.1f}s: {e}")
            traceback.print_exc()

        # Flush GPU between experiments
        gpu_flush()

    # ── Summary ───────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("EXPERIMENT RUN COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"GPU: {gpu_name}")
    print()

    for key, name, _ in experiments:
        r = results.get(key, {"status": "SKIPPED"})
        status = r["status"]
        elapsed = r.get("elapsed_s", 0)
        marker = "OK" if status == "SUCCESS" else "FAIL" if status == "FAILED" else "SKIP"
        print(f"  [{marker}] {name:30s} {elapsed:7.1f}s  {r.get('error', '')}")

    if discoveries:
        print(f"\n{'!' * 70}")
        print("DISCOVERIES DETECTED:")
        print(f"{'!' * 70}")
        for d in discoveries:
            print(f"  >>> {d}")
    else:
        print("\n  No discovery thresholds triggered at current sample sizes.")

    # List generated files
    print("\nGenerated files:")
    for pattern in ["EXPERIMENT_*.md", "DISCOVERY_*.md", "PHI_ANALYSIS.md",
                     "FATHOM_CONSTANT_REPORT.md"]:
        for p in sorted(PROJECT_ROOT.glob(pattern)):
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name:40s} {size_kb:.1f} KB")

    print(f"\nFull log: {log_path}")
    print("Nothing was pushed. Everything stays local.")

    # Write summary JSON for programmatic access
    summary = {
        "timestamp": timestamp,
        "gpu": gpu_name,
        "total_elapsed_s": total_elapsed,
        "results": results,
        "discoveries": discoveries,
    }
    summary_path = PROJECT_ROOT / f"experiment_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")

    # Restore stdout
    sys.stdout = logger._stdout
    sys.stderr = logger._stdout
    logger.close()

    return 0 if all(r.get("status") == "SUCCESS" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
