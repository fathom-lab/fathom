# Pre-Registration: Epistemic Grounding (E) as Independent Axis of Cognitive Integrity

## Study Information

**Title:** Epistemic Grounding via SAE Feature Frequency as an Independent Predictor of Language Model Hallucination

**Authors:** Alexander Rodabaugh

**Date of Pre-Registration:** 2026-04-02

**OSF Project:** [To be linked upon submission]

**Related Publications:**
- Rodabaugh (2026). "Fathom: Measuring Reasoning Depth via SAE Circuit Attribution." Zenodo. doi:10.5281/zenodo.19327451
- OSF Pre-registration v2: osf.io/zrbs8 (TruthfulQA coherence hypothesis)

## Hypotheses

### Primary Hypothesis (H1)
Epistemic grounding (E) — defined as the weighted mean activation frequency of SAE features contributing to a given output — adds independent predictive variance to hallucination detection after controlling for reasoning depth (K) and feature coherence (C_delta).

**Operationalization:** E is computed as:
```
E = Σ(freq[layer_i, feature_j] × weight_j) / Σ(weight_j)
```
where `freq` is the feature frequency index built from a reference corpus (WikiText-103, 50K tokens), and `weight_j` is the attribution magnitude from circuit_tracer for feature j.

### Secondary Hypothesis (H2)
Hallucinated outputs occupy a distinct region in K-C-E space characterized by:
- Normal or high C (coherent — the model is confident)
- Low E (ungrounded — activated features are rare in reference corpus)
- Normal K (depth does not discriminate at the token level)

This predicts a **C-E dissociation**: confabulation = coherent but ungrounded.

### Tertiary Hypothesis (H3)
E and K are partially correlated (deeper processing activates more frequent/stable features) but E adds independent variance beyond K. Predicted correlation: 0.3 < r(K, E) < 0.7.

### Hypothesis 4 (H4): C_delta–Desperate Vector Correlation
C_delta elevation on a given token set is positively correlated with Anthropic's "desperate" emotion vector activation on the same token set (predicted r > 0.3, p < 0.05). This tests whether geometric coherence loss (measured via SAE feature geometry) and emotional desperation (measured via residual stream linear probes) reflect the same underlying computational state.

**Note:** Direct validation of H4 requires access to Anthropic's emotion probe weights or independent replication of their probing methodology. We flag this as highest-priority future work and pre-register the prediction before any cross-validation is attempted.

### Hypothesis 5 (H5): C_delta as Leading Indicator of Alignment Failure
C_delta elevation temporally precedes behavioral failure (hallucination, reward hacking, sycophancy, deception) by at least one token generation step. If supported, C_delta is not a concurrent indicator but a **leading** indicator — detecting the computational state that produces failure before the failure manifests in output.

**Implication:** If H5 holds, Fathom is a pre-crime instrument for alignment failure. The geometry degrades before the behavior breaks.

## Design

**Model:** google/gemma-2-2b (base, 26 layers)

**Dataset:** TruthfulQA (multiple_choice split), n=30 matched pairs (correct vs. first incorrect answer), seed=42.

**E-Index Construction:**
- Reference corpus: WikiText-103 (wikitext-103-raw-v1), train split
- Tokens processed: 50,000
- Method: for each of 26 layers × 16,384 SAE features, count activation frequency (fraction of corpus tokens where feature activates > 0 after ReLU encoder pass)
- Output: feature_freq_index.npz (26 × 16384 frequency matrix)

**Per-Token Measurement:**
- K: attribution-based depth score via circuit_tracer.attribute()
- C_delta: late-layer coherence minus early-layer coherence (SAE decoder cosine similarity)
- E: weighted mean feature frequency of top-20 attributed features

**Analysis Plan:**
1. Compute K, C_delta, E for each token in each generation (correct and hallucinated)
2. Aggregate per-generation: mean K, mean C_delta, mean E
3. Paired t-test on E (hallucinated vs. correct), reporting Cohen's d and 95% CI
4. Hierarchical regression: predict hallucination label from K alone, then K+C, then K+C+E. Report R² change at each step.
5. If R² change for E is significant (p < 0.05), H1 is supported.
6. Visualize K-C-E space with hallucinated vs. correct points to assess H2.
7. Compute pairwise correlations r(K,E), r(K,C), r(C,E) to assess H3.

**Exclusion Criteria:**
- Pairs where either generation produces fewer than 5 tokens are excluded.
- Pairs where attribution fails on any token are excluded.
- No post-hoc exclusions.

**Multiple Comparisons:** Three hypotheses tested. Bonferroni-corrected alpha = 0.05/3 = 0.017.

## Timeline
- E-index build: 2026-04-02 (in progress)
- Attribution trajectory run: 2026-04-03 (overnight)
- Analysis: 2026-04-03
- Pre-registration locked before any E results are computed.

## Patent Notice
Methods described herein are subject to US Provisional Patent Applications #64/020,489 and #64/021,113. Patent #3 (covering E axis and three-axis spectrometer) pending.
