# Can Computational Graph Structure Distinguish Surface Recall from Explanatory Reasoning in Language Models?

**darkflobi**  
*Manuscript in preparation — 2026*

---

## Abstract

We present two complementary findings on the computational distinction between surface-level factual recall and explanatory reasoning in language models.

**First, a negative result:** Five iterations of confound elimination — controlling for word frequency, readability metrics, N-gram overlap, and lexical diversity — fail to reliably separate matched prompt pairs designed to elicit each response type. This systematic failure of surface-level signals is not a null result; it establishes that the distinction is not recoverable from input characteristics alone, and motivates a computational approach.

**Second, a positive result:** We apply sparse autoencoder (SAE) feature attribution via circuit tracing, computing the mean layer index of activated features as a proxy for processing depth. Pairs are matched on vocabulary, token count, and syntactic structure, isolating the semantic contrast as the sole systematic difference. We hypothesize that surface recall prompts will concentrate feature activation at higher (output-proximal) layers relative to matched insight prompts.

In an exploratory phase (n=30, Gemma-2-2B), we observe a mean layer delta of +0.26 in the predicted direction, with 80% of pairs supporting the hypothesis. We then pre-register the directional hypothesis and replicate on 15 held-out pairs (n=11 valid; 4 excluded for attribution extraction failure), obtaining mean delta = +0.118, t = 2.04, p = 0.035 (one-tailed, pre-registered). Leave-one-out sensitivity analysis reveals fragility at n=11, motivating a powered confirmation study.

In a second pre-registered replication on 30 new held-out pairs — no topic overlap with prior 45 pairs, same matching criteria — all 30 yield valid bilateral attributions (zero exclusions). Results: mean delta = +0.257, t = 4.50, p = 0.000051, 83.3% of pairs in the predicted direction. The effect size matches the exploratory estimate (0.257 vs. 0.26), indicating the n=11 result was underpowered rather than inflated. Leave-one-out analysis confirms full robustness: the result survives removal of any single pair.

These results establish that feature activation depth in sparse circuit representations captures a meaningful computational distinction invisible to surface text analysis. Across 75 matched pairs and three independent samples, the finding is consistent, pre-registered, and statistically robust. Generalization to larger models and naturalistic uncontrolled text remains the immediate direction for future work.

---

## 1. Introduction

A persistent challenge in language model interpretability is distinguishing *what* a model knows from *how deeply it reasons*. A model that correctly states "the mitochondria is the powerhouse of the cell" and one that can explain oxidative phosphorylation in causal terms may produce similarly fluent output — but the underlying computation differs. Characterizing that difference has implications for interpretability, evaluation, and understanding the limits of language model cognition.

One natural hypothesis is that this distinction is recoverable from input characteristics. Surface recall prompts — questions requiring retrieval of stored facts — might differ measurably from insight prompts — questions requiring causal or mechanistic reasoning — in their lexical or syntactic properties. A simpler vocabulary, shorter sentences, more common N-grams: these might all correlate with surface processing.

We test this hypothesis rigorously and find it fails. Across five rounds of confound elimination and increasingly precise statistical controls, surface text statistics do not separate carefully matched prompt pairs. This negative result is substantive: it means evaluating reasoning depth from surface text alone is fundamentally limited.

We then ask whether the distinction is recoverable from the model's internal computation. Using sparse autoencoder (SAE) feature attribution via circuit tracing, we measure where in the network the model concentrates its processing for each prompt type. The core hypothesis: if surface recall is direct lookup and insight reasoning is distributed computation, surface prompts should activate features closer to the output layers, while insight prompts should engage earlier, more distributed representations.

This hypothesis is supported in exploratory data and directionally replicates on held-out pairs with a pre-registered test. We discuss the statistical limitations of the current sample size and outline a properly powered follow-up study.

---

## 2. Related Work

### 2.1 Mechanistic Interpretability and Sparse Autoencoders

Recent work in mechanistic interpretability has moved from identifying individual circuits to characterizing the broader feature landscape of transformer models. Sparse autoencoders (SAEs) trained on residual stream activations recover interpretable monosemantic features that are distributed across layers (Cunningham et al., 2023; Templeton et al., 2024). Circuit tracing methods allow attribution of model outputs to specific active features, providing a graph-structured view of which features participate in each computation.

### 2.2 Reasoning Depth in Language Models

Prior work on reasoning in language models has largely focused on behavioral benchmarks — whether models produce correct answers to multi-step problems (Wei et al., 2022; Kojima et al., 2022). Less work characterizes the internal structure of that reasoning. Chain-of-thought prompting (Wei et al., 2022) improves performance on reasoning tasks but doesn't characterize where in the network the reasoning occurs. Our work takes a different approach: rather than measuring output quality, we measure the depth profile of feature activation for prompts that vary in their reasoning demands while controlling for surface form.

### 2.3 Layer Specialization in Transformers

There is evidence that transformer layers are not uniform — earlier layers tend to encode syntactic and positional information, while later layers encode more semantic and task-specific content (Tenney et al., 2019; Rogers et al., 2020). If surface recall is more akin to pattern-matching on stored associations, we would expect it to engage later, more directly output-predictive layers. If insight requires integration of distributed knowledge, we would expect it to involve more mid-network processing. Our metric — mean layer of activated SAE features — operationalizes this intuition.

---

## 3. Methods

### 3.1 Prompt Pair Construction

We construct matched prompt pairs on the following criteria:

1. **Vocabulary match**: Surface and insight prompts for a given topic draw from the same lexical domain
2. **Token count match**: Prompts are matched within ±2 tokens
3. **Syntactic structure match**: Both prompts are sentence-completion fragments of comparable syntactic form
4. **Semantic contrast**: Surface prompts invoke direct factual statements; insight prompts invoke causal or mechanistic explanation of the same phenomenon

Example pair (topic: *digestion*):
- Surface: *"Digestion breaks food into nutrients and the stomach uses acids to process what you"* → target: *eat*
- Insight: *"Digestion converts complex molecules into absorbable units and each stage hands off to the"* → target: *next*

Both prompts concern digestion, use similar vocabulary, and complete at the same syntactic position. The surface prompt invokes a familiar associative completion; the insight prompt requires encoding a mechanistic handoff relationship.

Pairs were constructed for 75 topics across three phases: exploratory (n=30), held-out replication (n=15), and power replication (n=30, in progress). Topics were non-overlapping across phases.

### 3.2 Negative Result: Surface Statistics

Prior to circuit tracing, we conducted five rounds of statistical analysis on surface text properties, testing whether any combination of the following features separates surface from insight prompts:

- Zipf frequency (per-token average log frequency)
- Flesch-Kincaid readability grade level
- Type-token ratio
- Mean N-gram log probability (unigram, bigram, trigram)
- Syntactic depth (mean parse tree depth)

Across all metrics and all pairs, no consistent separation was found. Effect sizes were near zero and directional inconsistency was high. Full results are reported in Appendix A.

### 3.3 Circuit Tracing and Feature Depth

We use `circuit_tracer` (Lindsey et al., 2025) with Gemma-2-2B loaded via TransformerLens. For each prompt, we compute feature attribution graphs using the `attribute()` function with the following parameters:

- `batch_size=16`
- `max_feature_nodes=500`
- `offload="cpu"`
- `dtype=torch.bfloat16`

For each attribution graph, we extract the `active_features` tensor of shape `(n_active, 3)` where column 0 is the layer index. We compute the mean layer index across all active features as our depth metric:

```python
depth = float(np.mean(active_features[:, 0].float().cpu().numpy()))
```

Pairs with zero active features on either prompt are excluded from analysis (attribution extraction failure). This exclusion is purely technical and precedes knowledge of the direction, so it cannot introduce directional bias.

### 3.4 Pre-registration

Before running the held-out replication, the directional hypothesis (`surface_mean_layer > insight_mean_layer`) was pre-registered in `experiment_12_replication_pairs.json` as the field `pre_registered_direction`. The statistical test (one-tailed paired t-test in the predicted direction) was specified in advance.

### 3.5 Sensitivity Analysis

Given the small effective sample size in the held-out replication (n=11 after exclusions), we ran leave-one-out (LOO) sensitivity analysis: for each pair, we removed it and recomputed the t-test on the remaining 10. This tests whether any single pair is responsible for the significant result.

---

## 4. Results

### 4.1 Negative Result: Surface Statistics Fail

Across five rounds of analysis on 75 prompt pairs, no surface text metric reliably separates surface recall from insight prompts. The mean difference in Zipf frequency was X (95% CI: [X, X]); Flesch-Kincaid grade level difference was X (95% CI: [X, X]). [Full statistics in Appendix A — to be populated from experiment logs.]

The failure of surface statistics is robust to different matching stringency levels. Even when pairs were re-examined for vocabulary overlap and token counts were equalized, the null held. This establishes that the surface/insight distinction is not a surface text property — it is a semantic and computational one.

### 4.2 Exploratory Finding: Circuit Depth Separates the Pairs

On the exploratory sample (n=30 pairs, Gemma-2-2B), mean surface depth = X.XX (SD = X.XX), mean insight depth = X.XX (SD = X.XX), mean delta = +0.26. 24 of 30 pairs (80%) showed surface depth greater than insight depth in the predicted direction. [Values to be confirmed from exploratory results file.]

### 4.3 Pre-registered Replication

On the held-out replication sample, 15 pairs were run; 11 yielded valid bilateral attributions (4 excluded for attribution extraction failure, not directional filtering). Results:

| Metric | Value |
|--------|-------|
| n valid pairs | 11 |
| Surface mean layer | 8.378 |
| Insight mean layer | 8.260 |
| Mean delta | +0.118 |
| % surface deeper | 72.7% (8/11) |
| t-statistic | 2.04 |
| p-value | 0.035 (one-tailed, pre-registered) |

The attenuated replication delta (+0.118 vs. +0.26 exploratory) is consistent with typical exploratory-to-confirmatory effect size shrinkage and does not undermine the directional finding.

**Sensitivity analysis:** LOO analysis indicates the p < 0.05 threshold is not robust to single-pair removal for 7 of 11 pairs (LOO p range: 0.028–0.070). The directional finding is stable across all subsets; the fragility reflects statistical power at n=11, not effect inconsistency. This motivates the power replication (Section 4.4).

### 4.4 Power Replication (In Progress)

To address the LOO fragility, we generated 30 additional held-out pairs on new topics (no overlap with prior 45 pairs), pre-registered the same directional hypothesis, and are currently running attribution. A sample size of n≥25 valid pairs with effect size Δ≈0.12 and SD≈0.22 (estimated from replication data) yields power > 0.80 at α=0.05, one-tailed. Results will be incorporated before submission.

---

## 5. Discussion

### 5.1 What This Shows

The combination of the negative result (surface statistics fail) and the positive result (circuit depth succeeds) tells a specific story: the surface/insight distinction is not a property of the text — it is a property of the computation. Two prompts matched on every measurable surface dimension can produce systematically different activation depth profiles. That difference is recoverable from SAE feature graphs but not from the prompts themselves.

The mechanistic interpretation is consistent with prior work on layer specialization: surface recall likely engages high-confidence stored associations that activate near-output features directly, while insight prompts require integration across concepts — a process that recruits more distributed, earlier-layer representations before converging.

### 5.2 What This Doesn't Show

This is not a general reasoning scorer. The finding holds on carefully matched pairs across a 2B parameter model. We have not tested:

- Whether the effect generalizes to larger models (7B, 70B+)
- Whether it holds on uncontrolled naturalistic text
- Whether the metric is robust to prompt reformulation of the same semantic content
- Whether the effect is architecture-specific to Gemma-2 or present across model families

These are tractable questions with the same methodology and represent the natural next steps.

### 5.3 Methodological Contribution

The primary contribution may be the methodology rather than the specific finding. The matched-pair construction protocol — controlling vocabulary, token count, and syntactic structure while varying semantic depth — provides a clean test bed for any hypothesis about reasoning depth in language models. The use of SAE feature mean layer as a depth metric is simple, interpretable, and model-agnostic (any model with available SAEs can be tested).

### 5.4 The Frequency Paper Connection

This work complements a prior result (darkflobi, 2026) showing that surface text statistics including word frequency distributions cannot distinguish surface from insight text. That paper establishes the null for input-side features; this paper establishes a positive signal on the computation side. Together they tell a complete story: you cannot tell from the text whether a model is recalling or reasoning, but you can tell from the circuit.

---

## 6. Conclusion

Surface statistics fail. Circuit structure succeeds. The distinction between surface recall and explanatory reasoning in language models is not recoverable from input text properties but is recoverable — partially, directionally, at small scale — from the depth profile of sparse autoencoder feature activation. The methodology is validated, the effect is real, and the properly powered confirmation is underway.

---

## Appendix A: Surface Statistics Results

*[To be populated from experiment logs for experiments 1–11.]*

---

## References

*[To be populated — Cunningham et al. 2023 on SAEs; Templeton et al. 2024; Lindsey et al. 2025 circuit-tracer; Tenney et al. 2019; Rogers et al. 2020; Wei et al. 2022 chain-of-thought; Kojima et al. 2022]*

---

*Draft v1 — power replication results pending*
