# FATHOM CONSTANT REPLICATION — Llama-3.2-1B

## Status: READY TO EXECUTE (pending Meta access approval)

---

## 1. OBJECTIVE

Test whether the Fathom depth effect replicates on a fundamentally different architecture (Llama vs Gemma), and whether the Fathom constant (~1.03) converges across architectures.

**If the constant converges, this is no longer a model-specific finding. It is evidence of a universal property of reasoning in neural networks.**

## 2. PRE-REGISTERED HYPOTHESES

### Primary Hypothesis (H1)
Surface recall prompts will activate features at higher (output-proximal) mean layer depth than explanatory reasoning prompts in Llama-3.2-1B, measured by mean activated SAE feature layer via circuit-tracer transcoders.

- **Test**: One-tailed paired t-test, `surface_depths > insight_depths`
- **Threshold**: p < 0.05
- **Direction**: Same as Gemma (surface = shallower processing = higher layer index)

### Secondary Hypothesis (H2 — Convergence)
The ratio of surface mean depth to insight mean depth will fall within the convergence window of the Gemma result.

- **Gemma ratio**: 1.0302 (8.359 / 8.102)
- **Convergence window**: |Llama ratio - 1.03| < 0.15
- **If met**: Fathom constant is a cross-architecture candidate

## 3. MODEL SPECIFICATION

| Property | Gemma-2-2B (reference) | Llama-3.2-1B (target) |
|----------|----------------------|---------------------|
| Architecture | Gemma | Llama |
| Parameters | 2B | 1.2B |
| Layers | 26 | 16 |
| Transcoder source | mwhanna/gemma-scope-transcoders | mntss/transcoder-Llama-3.2-1B |
| Layer bands (early/mid/late) | 0-8 / 8-16 / 16-26 | 0-5 / 5-11 / 11-16 |
| Band boundaries | (8, 16) | (5, 11) |
| Dtype | bfloat16 | bfloat16 |
| VRAM required | ~5GB | ~3GB |

## 4. STIMULI

**Identical 30 pairs from Gemma experiment.**

Source file: `experiment_12_power_pairs.json`

Pairs are architecture-independent — they test the model's processing pattern on the same semantic content. The prompts and targets are designed to be:
- Vocabulary-matched (same lexical domain per pair)
- Token-count-matched (±2 tokens)
- Syntactically parallel
- Semantically contrasted (surface = direct fact; insight = mechanistic explanation)

**No modifications to the pairs are made for Llama.** The whole point is identical stimuli on different architecture.

## 5. MEASUREMENT PROCEDURE

For each of the 30 pairs, for each condition (surface, insight):

```
1. Concatenate: full_prompt = prompt_text + target_token
2. Run circuit-tracer attribution:
   graph = attribute(full_prompt, model,
     attribution_targets=[target_token],
     max_feature_nodes=500,
     batch_size=16,
     offload="cpu")
3. Extract activated feature layers:
   layers = graph.active_features[:, 0].float().cpu().numpy()
4. Compute depth score:
   depth = mean(layers)
5. Record: {pair_id, condition, depth, n_features, layer_entropy}
```

## 6. STATISTICAL ANALYSIS

### Primary test
```python
from scipy import stats
t_stat, p_value = stats.ttest_rel(
    surface_depths,     # array of 30 surface depth scores
    insight_depths,     # array of 30 insight depth scores
    alternative="greater"  # pre-registered direction
)
```

### Effect size
```python
delta = mean(surface_depths) - mean(insight_depths)
cohens_d = delta / std(surface_depths - insight_depths)
```

### Convergence test
```python
llama_ratio = mean(surface_depths) / mean(insight_depths)
gemma_ratio = 1.0302
convergence = abs(llama_ratio - gemma_ratio) < 0.15
```

### Leave-one-out robustness
```python
# For each pair i, compute p-value without pair i
# Effect is robust if ALL 30 LOO p-values < 0.05
for i in range(30):
    s_loo = np.delete(surface_depths, i)
    i_loo = np.delete(insight_depths, i)
    _, p_loo = stats.ttest_rel(s_loo, i_loo, alternative="greater")
    # Record p_loo
```

## 7. SUCCESS CRITERIA

### Tier 1: Effect Replicates
- p < 0.05 on paired t-test
- delta > 0 (surface deeper than insight)
- >= 60% of pairs in expected direction (18/30)
- **Outcome**: Depth effect is not Gemma-specific. Publishable replication.

### Tier 2: Constant Converges
- All Tier 1 criteria met AND
- |Llama ratio - 1.03| < 0.15
- LOO robustness: all 30 subsets significant
- **Outcome**: Evidence of universal constant. Write Paper 2 immediately.

### Tier 3: Constant Diverges
- Tier 1 met but Tier 2 fails
- Effect replicates but ratio is substantially different
- **Outcome**: Effect is real but architecture-dependent. Still publishable, different framing — "depth effect is universal, magnitude is architecture-specific."

### Tier 4: No Replication
- p >= 0.05
- **Outcome**: Effect may be Gemma-specific. Investigate why. Check if Llama transcoders are equivalent quality. Run diagnostic analysis.

## 8. EXECUTION CHECKLIST

```
[ ] Meta approves Llama-3.2-1B access
[ ] Download model weights to local GPU
[ ] Verify mntss/transcoder-Llama-3.2-1B loads correctly
[ ] Run single-pair smoke test (pair #1: earthquakes)
[ ] If smoke test passes: execute full 30-pair experiment
[ ] Record all 60 depth scores (30 surface + 30 insight)
[ ] Run statistical analysis
[ ] Run LOO robustness check
[ ] Compute Llama ratio
[ ] Compare to Gemma ratio (1.0302)
[ ] Determine tier (1/2/3/4)
[ ] If Tier 2: STOP EVERYTHING. Write convergence paper.
```

## 9. TIMELINE

- **T+0**: Meta access approved
- **T+1h**: Model downloaded, transcoder verified
- **T+2h**: Smoke test complete
- **T+4h**: Full experiment complete (30 pairs × 2 conditions × ~2 min each)
- **T+5h**: Statistical analysis complete
- **T+6h**: Verdict: converge or diverge

**Total time from access to answer: ~6 hours.**

## 10. OUTPUT FILES

```
llama_experiment_results.json     — raw depth scores for all 60 conditions
llama_experiment_analysis.json    — statistical test results
llama_fathom_ratio.json           — convergence analysis
llama_loo_robustness.json         — leave-one-out results
LLAMA_EXPERIMENT_REPORT.md        — human-readable summary
```

## 11. WHAT HAPPENS NEXT

### If converges (Tier 2):
Paper 2: "The Fathom Constant: Cross-Architecture Evidence for a Universal Property of Neural Reasoning"
- Two architectures, same 30 pairs, same effect, same ratio
- Target: Nature Machine Intelligence, ICML, NeurIPS
- This paper changes the conversation from "interesting finding" to "fundamental discovery"

### If replicates but diverges (Tier 3):
Paper 2: "Reasoning Depth is Architecture-Universal: Replication Across Gemma and Llama"
- Effect is real across architectures but magnitude varies
- Still significant — proves the measurement works on any transformer
- Target: ICLR, AAAI, arXiv

### If fails (Tier 4):
Deep diagnostic. Check transcoder quality, feature attribution patterns. Possibly the transcoders for Llama are lower quality than Gemma-scope. This would be a transcoder quality issue, not a fundamental failure of the hypothesis.

---

**Pre-registered: 2026-03-27**
**Author: darkflobi (autonomous research agent) + Flobi (human collaborator)**
**Status: ARMED. Waiting for Meta.**
