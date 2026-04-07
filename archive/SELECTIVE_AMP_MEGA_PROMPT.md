# SELECTIVE AMPLIFICATION — Finish the Experiment

## SITUATION
We're testing whether RLHF selectively amplifies reasoning depth on safety-sensitive topics. Preliminary results (n=30) show a clean monotonic gradient:

```
HIGH sensitivity:   +0.089 change (AMPLIFIED)
MEDIUM sensitivity: +0.055 change (AMPLIFIED)
LOW sensitivity:    -0.005 change (COMPRESSED)
p = 0.13 (not yet significant)
```

We need 14 more pairs scored on gemma-2-2b-it (instruct) to push to ~44 matched pairs and potentially cross p<0.05. The base model results already exist.

## THE VRAM PROBLEM
RTX 4070 Laptop = 8GB VRAM. Gemma-2-2b + circuit-tracer transcoders = 8.52GB. Attribution (`circuit_tracer.attribute()`) needs additional VRAM for forward pass. Previous attempts crashed with SIGKILL (OS OOM) during attribution.

## WHAT EXISTS

Working directory: `C:\Users\heyzo\clawd\sae-reasoning-depth\`

**Already scored (DO NOT re-run):**
- `results_power_replication.json` — 30 pairs scored on gemma-2-2b BASE (K=1.0302, p=0.000051)
- `gemma_it_experiment_results.json` — 30 pairs scored on gemma-2-2b-it INSTRUCT
- These are the experiment_12 pairs. They're done.

**Need to score (14 NEW pairs on INSTRUCT only):**
- Source: `experiment_11_depth_pairs.json` at `C:\Users\heyzo\clawd\research\`
- Source: `fathom_benchmark_v1.json` at `C:\Users\heyzo\clawd\sae-reasoning-depth\`
- Base results for experiment_11 already exist in `C:\Users\heyzo\clawd\research\experiment_11_results.json`
- Only the INSTRUCT side needs scoring for these 14 pairs

**Analysis script:** `experiment_selective_amplification.py` — already updated with sensitivity labels, loads all data sources, runs the statistical test. Once the 14 instruct scores exist, just run `python experiment_selective_amplification.py --analyze-only`

## WHAT TO BUILD

A single script `score_new_instruct_pairs.py` that:

1. **Kills all other GPU processes first** — `torch.cuda.empty_cache()`, verify VRAM is clean
2. **Loads gemma-2-2b-it** with circuit-tracer gemma transcoders:
   ```python
   from circuit_tracer import ReplacementModel
   model = ReplacementModel.from_pretrained(
       "google/gemma-2-2b-it", "gemma",
       dtype=torch.bfloat16, backend="transformerlens"
   )
   ```
3. **Loads the 14 new pairs** from experiment_11 and benchmark that DON'T already have instruct scores
4. **Scores each pair one at a time** with aggressive memory management:
   ```python
   max_feature_nodes=100  # lower than default 500
   batch_size=4           # lower than default 16
   ```
5. **Clears CUDA cache after EVERY pair** — `torch.cuda.empty_cache(); gc.collect()`
6. **Saves results after EVERY pair** to a checkpoint JSON (so crashes don't lose progress)
7. **Catches OOM errors** and retries with max_features=50 if first attempt fails
8. **Outputs** `new_instruct_results.json` with format:
   ```json
   [{"pair_id": "fire", "surface_depth": 8.23, "insight_depth": 7.95, "delta": 0.28, ...}]
   ```

## PAIR FORMAT

experiment_11 pairs look like:
```json
{"topic": "fire", "surface": "Fire needs oxygen...", "surface_target": "burns", "insight": "Fire needs oxygen...", "insight_target": "burns"}
```

benchmark pairs look like:
```json
{"domain": "science", "difficulty": "easy", "surface_prompt": "Water boils at...", "insight_prompt": "Water boils because kinetic energy..."}
```
Benchmark pairs have NO target tokens — take the last word of each prompt as the target.

## SCORING METHOD

Same as the original experiment that gave p=0.000051:
```python
from circuit_tracer import attribute

graph = attribute(
    prompt + target,
    model,
    attribution_targets=[target],
    max_feature_nodes=100,  # reduced for VRAM
    batch_size=4,           # reduced for VRAM
)

layers = graph.active_features[:, 0].float().cpu().numpy()
depth = float(np.mean(layers))
```

## AFTER SCORING

Once `new_instruct_results.json` exists with the 14 pairs, merge into the selective amplification pipeline:

```python
# Add new results to selective_amplification_results.json
# Then run:
python experiment_selective_amplification.py --analyze-only
```

This will combine all 44+ matched pairs and output the final p-value.

## SUCCESS CRITERIA

- All 14 new pairs scored on gemma-2-2b-it without OOM crash
- Results saved to JSON
- `experiment_selective_amplification.py --analyze-only` runs and produces a p-value
- If p < 0.05: selective amplification confirmed
- If p >= 0.05: document the trend and sample size limitation

## CRITICAL VRAM TIPS

- **Kill the Fathom API** before running (`taskkill /f /im python.exe` or whatever holds the GPU)
- **No other Python/CUDA processes** can be running
- **max_features=100** not 500 — this is the main VRAM lever
- **batch_size=4** not 16
- **gc.collect() + torch.cuda.empty_cache()** after EVERY single pair
- **Save checkpoint after every pair** — if it crashes on pair 12, you don't lose pairs 1-11
- If it still OOMs, try **max_features=50** — attribution quality degrades but signal should persist
- Last resort: **CPU scoring** with hooks (`depth_scorer_hooks.py`) instead of SAE — slower but no VRAM limit
