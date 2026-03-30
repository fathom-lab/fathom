# CLAUDE CODE TASK: Alternative Depth Scorer for Mistral-7B
## Priority: HIGH — needed for 3rd cross-architecture validation

## Context
Fathom measures reasoning depth in AI models using circuit-tracer's SAE feature attribution.
Current implementation (run_llama_experiment.py, depth_scorer.py) requires circuit-tracer transcoders.
Transcoders are only available for Gemma-2-2B and Llama-3.2-1B.
Mistral-7B transcoders do NOT exist in circuit-tracer — this is a hard blocker.

## Goal
Build `depth_scorer_hooks.py` — a fallback depth scorer that works on ANY transformer model
using raw activation hooks (no transcoders needed). Must produce a depth score that is
comparable to the circuit-tracer-based score and shows the same surface > insight pattern.

## Method: Residual Stream Norm (validated approach)
Instead of SAE feature attribution, measure the L2 norm of the residual stream at each layer.
Compute weighted mean layer = sum(layer_i * norm_i) / sum(norm_i)
This is a tractable proxy for "where computation is happening."

## Files to Create

### 1. depth_scorer_hooks.py
```python
"""
Depth scorer using residual stream hooks — no transcoders needed.
Works on any HuggingFace transformer model.

Usage:
    scorer = HookDepthScorer("mistralai/Mistral-7B-v0.1")
    result = scorer.score("prompt text", "target")
    print(result.depth_score, result.n_layers)
"""
```

Requirements:
- Load model with transformerlens (HookedTransformer.from_pretrained)
- Register hooks on every residual stream output (hook_resid_post)
- For a given prompt+target, collect residual stream norms at each layer
- Compute: weighted_mean_layer = sum(layer * norm[layer]) / sum(norm[layer])
- Return: DepthResult dataclass with depth_score, n_layers, layer_profile, early_ratio, mid_ratio, late_ratio
- Handle GPU VRAM limits (offload to CPU if needed)
- Clean up hooks after each call

### 2. run_mistral_experiment.py
Identical structure to run_llama_experiment.py but using HookDepthScorer instead of circuit-tracer.

Config:
- MODEL_NAME = "mistralai/Mistral-7B-v0.1"
- N_LAYERS = 32
- BAND_BOUNDARIES = (10, 22)  # early/mid/late for 32-layer model
- Use same 30 calibrated pairs from experiment_12_power_pairs.json
- GEMMA_REFERENCE = {"ratio": 1.0302, ...}
- LLAMA_REFERENCE = {"ratio": 1.0242, ...}

Same output format as run_llama_experiment.py:
- Raw results JSON
- Analysis JSON (ratio K, p-value, delta, Cohen's d)
- Report markdown
- TIER verdict (FULL CONVERGENCE if K within 0.15 of 1.03)

### 3. VALIDATION: verify hook method correlates with circuit-tracer
Run both methods on same 5 pairs from healthcare domain.
Print correlation between hook-based depth and SAE-based depth.
Target: Pearson r > 0.7 (they should track together, exact values will differ).

## Technical Notes
- transformerlens install: pip install transformer-lens
- Mistral model: mistralai/Mistral-7B-v0.1 (need HuggingFace access)
- VRAM: Mistral-7B at bfloat16 = ~14GB. RTX 4070 laptop has 8GB. Must use 4-bit quantization (bitsandbytes) or CPU offloading.
- 4-bit quant: use BitsAndBytesConfig with load_in_4bit=True, then load into transformerlens
- Alternative: run on CPU (slow but works, ~2 min/pair vs 30s/pair on GPU)

## Success Criteria
1. run_mistral_experiment.py --smoke completes without error
2. Surface depth != Insight depth on the earthquakes pair
3. Direction (surface > insight or insight > surface) is consistent
4. Full 30-pair run completes and analysis.json is saved

## Files that exist (don't recreate, use as reference)
- C:\Users\heyzo\clawd\sae-reasoning-depth\depth_scorer.py (circuit-tracer version)
- C:\Users\heyzo\clawd\sae-reasoning-depth\run_llama_experiment.py (same structure needed)
- C:\Users\heyzo\clawd\sae-reasoning-depth\experiment_12_power_pairs.json (the 30 pairs)
- C:\Users\heyzo\clawd\sae-reasoning-depth\fathom_demo.py (demo script, keep consistent)

## Output Location
All new files in: C:\Users\heyzo\clawd\sae-reasoning-depth\
