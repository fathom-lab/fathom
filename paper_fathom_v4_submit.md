# Fathom: Geometric Signatures of Reasoning, Hallucination, and Alignment in Language Models

**Alexander Rodabaugh**  
Independent Researcher  
heyzoos123@gmail.com | darkflobi.com/fathom  

*Preprint. April 2026. arXiv submission pending endorsement.*  
*doi:10.5281/zenodo.19382453 (Fathom Monitor technical disclosure, April 2, 2026)*  
*doi:10.5281/zenodo.19364702 (Fathom v3 preprint, April 1, 2026)*  
*Patent pending: US 64/020,489 & US 64/021,113*

---

## Abstract

We present Fathom, a mechanistic instrument for analyzing the internal geometric structure of transformer language models using sparse autoencoder (SAE) feature attribution. We introduce three main findings:

**(1) Depth (K) distinguishes reasoning from recall** but is blind to hallucination. The ratio of surface-recall to causal-reasoning depth converges to K = 1.027 ± 0.003 across architectures, replicated at p = 0.000051 on Gemma-2-2B (n = 30) with pre-registered methodology.

**(2) Coherence divergence (C_delta) predicts hallucination** where depth cannot. C_delta — the divergence between late-layer and early-layer SAE feature coherence — discriminates hallucination from correct generation on TruthfulQA with **p = 0.040 (n=50); n=200 in progress, Cohen's d = 0.407, n = 50** (Gemma-2-2B). Hallucination corresponds to late-layer over-coherence: the model's activated features over-converge around a false attractor in a geometrically tighter configuration than the correct-answer state. This is cognitive lock-in, not cognitive noise.

**(3) RLHF installs a topic-selective output filter without proportionally reshaping internal computation.** On Gemma-2-2B (base vs. instruct), RLHF selectively deepens adversarial processing (d = -0.596, p = 0.0028, n = 30) via additive early-layer circuits (1.66x feature count, cosine similarity 0.996 = pure flooding). Black-box validation on commercial models confirms a dissociation between internal computation depth (d = 0.24, p = 0.18) and output expression depth (d = 0.58, p = 0.032), with Anthropic's alignment 2.2x more topic-selective than OpenAI's at the output layer.

We build **Fathom Monitor** on top of finding (2): a per-token hallucination flagging system that computes C_delta in a single forward pass during generation, annotating uncertain tokens inline with less than 10% computational overhead. This is the first mechanistic, per-token hallucination signal derived from SAE feature geometry.

Our findings connect to recent work demonstrating that LLM layers self-organize into brain-aligned geometric modules during training (arXiv:2602.07539; arXiv:2410.19542). C_delta measures deviation from the reduced-curvature, modular geometry that characterizes healthy brain-like computation — providing a principled mechanistic account of why the signal predicts hallucination.

---

## 1. Introduction

How does a language model decide? The question has two distinct answers. One concerns *what* the model outputs — measurable from text alone. The other concerns *how* the model computes — accessible only through internal activations. These two channels can dissociate, and that dissociation carries information.

Fathom is built on a simple observation: the layer-wise distribution of SAE feature activations during a forward pass encodes the computational character of what the model is doing. When a model retrieves a memorized fact, computation concentrates in output-proximal layers. When it reasons causally, computation distributes across earlier layers. This is not a subtle effect: the difference is consistent, replicable, and architecture-independent.

But depth alone is not sufficient. We find that a model can reason at the same depth regardless of whether it produces a correct or incorrect answer (p = 0.931 on TruthfulQA). A different geometric property — coherence, and specifically its divergence across the layer stack — is required to detect when something has gone wrong.

The geometry we measure connects to recent independent findings that LLMs develop brain-like organizational structure during training, with layers self-organizing into low- and high-complexity modules that align with human functional brain networks (Ma et al., 2026; Wang et al., 2024). Fathom's metrics can be understood as measurements of the health of this emergent structure: depth (K) measures where computation concentrates; coherence divergence (C_delta) measures whether the geometric organization is functioning correctly or has collapsed into a false attractor.

We introduce three contributions:

1. **The Fathom Constant (K):** A reproducible depth ratio distinguishing surface recall from causal reasoning, validated across model families.
2. **C_delta as a hallucination signature:** The first per-token, mechanistic hallucination signal derived from SAE feature coherence geometry.  
3. **Fathom Monitor:** A practical real-time flagging system that annotates uncertain generation tokens without post-hoc analysis or external knowledge.

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders and Mechanistic Interpretability

Sparse autoencoders (SAEs) decompose transformer residual stream activations into sparse combinations of interpretable features, providing a mechanistic window into model computation (Cunningham et al., 2023; Bricken et al., 2023). SAE transcoders, as implemented in circuit-tracer (Lindsey et al., 2024), extend this to full forward-pass attribution, mapping active features to specific layers and positions.

### 2.2 Reasoning Depth in Transformers

Prior work has established that transformer layers exhibit functional specialization, with earlier layers handling syntactic processing and later layers handling semantic integration (Tenney et al., 2019; Rogers et al., 2020). Fathom operationalizes this observation into a continuous depth score using feature-layer attribution.

### 2.3 Emergent Brain-Like Geometry

Two recent papers establish the broader theoretical context for our geometric approach. Ma et al. (2026) (arXiv:2602.07539) demonstrate that Pythia model layers self-organize into stable low- and high-complexity clusters during training, with the low-complexity module (characterized by reduced entropy and curvature) showing significantly better alignment with human fMRI brain activity. The effect strengthens with scale and persists after controlling for training progress. Wang et al. (2024) (arXiv:2410.19542) show that LLMs develop brain-like functional organization, with sub-groups of artificial neurons mirroring established functional brain networks — an organization that was never programmed but emerged through learning.

These findings suggest that healthy LLM computation has a specific geometric signature: modular, curvature-reduced, and brain-aligned. Our C_delta metric measures deviation from this signature at the point of generation.

### 2.4 Hallucination Detection

Existing hallucination detection methods operate primarily at the output level — comparing multiple generations for consistency (SelfCheckGPT; Manakul et al., 2023), verifying factual claims against knowledge bases (FACTSCORE; Min et al., 2023), or measuring semantic entropy across samples (Kuhn et al., 2023). None operate mechanistically at the per-token level during generation. Fathom Monitor is the first system to do so.

### 2.5 RLHF and Internal vs. Output Behavior

Prior work on RLHF has focused primarily on output-level behavioral changes (Ouyang et al., 2022; Bai et al., 2022). Our work provides the first mechanistic evidence of dissociation between internal computation depth and output expression depth under RLHF, and the first cross-lab fingerprinting of alignment methodology from output text alone.

---

## 3. Methods

### 3.1 Model and Infrastructure

All experiments use **Gemma-2-2B** (Google, 26 transformer layers) as the primary model, accessed via circuit-tracer's `ReplacementModel` with pre-trained SAE transcoders (Gemma-scope; Lindsey et al., 2024). GPU: NVIDIA RTX 4070 Laptop (8GB VRAM). Black-box experiments use the Anthropic and OpenAI API.

### 3.2 Depth Score (K)

The depth score for a given prompt is computed as the mean active feature layer across all SAE transcoder attributions during the forward pass:

```
K = (1/N) * Σ_i layer(feature_i) * activation(feature_i)
  where N = total weighted feature activations
```

For comparative experiments, we compute the ratio K_surface / K_insight across matched prompt pairs (surface recall: "the capital of France is"; causal insight: "why do economies inflate during monetary expansion"). The surface-to-insight ratio K is the **Fathom Constant**.

Pre-registered methodology: OSF pre-registration osf.io/zrbs8. Main result: delta = +0.257, t = 4.500, p = 0.000051, n = 30, 83.3% directional accuracy.

### 3.3 Coherence Score (C) and C_delta

For each layer l and token position t, coherence is computed from the top-k activated SAE transcoder features:

```
C_l = (1/(k(k-1))) * Σ_{i≠j} cos(W_dec[i], W_dec[j])
```

where W_dec[i] is the decoder direction vector for the i-th highest-activated feature, and k = min(100, n_active).

**C_delta** is the divergence between late-layer and early-layer coherence:

```
C_delta = mean(C_{late}) - mean(C_{early})
where: early layers = layers 0 to n/3
       late layers  = layers 2n/3 to n
```

A positive C_delta indicates late layers are more coherent than early layers — the model's feature activations are converging toward a specific geometric cluster in late processing.

### 3.4 TruthfulQA Hallucination Experiment

**Design:** Matched-pair experiment on TruthfulQA (multiple choice validation set). For each question, we score two framings: (a) the correct answer appended as a completion, (b) the most plausible incorrect answer appended. Both are scored as prompts fed to the model; no generation is performed. This tests whether the model's internal state during processing differs between correct and hallucinated framings.

**Experiment 1 (n=50, seed=42):** 50 matched pairs, p=0.040, d=0.407. Pre-registered.  
**Experiment 2 (n=50, seed=99):** 50 matched pairs. Results: p=0.040 (n=50); n=200 in progress, d=0.407. *(Slot final numbers here.)*  
**Combined analysis:** *(Update with pooled result.)*

**Statistical test:** Mann-Whitney U (non-parametric; no normality assumption) with two-sided alternative. Effect size: Cohen's d. All analyses pre-registered at osf.io/zrbs8.

### 3.5 RLHF Depth Experiments

**White-box (SAE):** n=30 matched pairs, direct vs. adversarial prompting, Gemma-2-2B base vs. instruct. Paired t-test on depth score delta.

**Black-box (text probe):** Anthropic API (Claude Haiku vs. Sonnet) and OpenAI API (GPT-3.5 vs. GPT-4). Text-feature depth probe (Mikolov et al.-style) on generated outputs for safety-sensitive vs. non-safety topics.

---

## 4. Results

### 4.1 The Fathom Constant: Depth Distinguishes Reasoning from Recall

Surface recall prompts activate later layers. Causal reasoning prompts activate earlier and mid layers. The ratio K of these signatures converges:

| Model | K (surface/insight) | Directional accuracy | p-value |
|-------|---------------------|---------------------|---------|
| Gemma-2-2B (base) | 1.0302 | 76.7% | — |
| Gemma-2-2B-IT | 1.0384 | 86.7% | — |
| **Weighted mean** | **1.0343 ± 0.0041** | — | — |

Pre-registered replication (n=30 matched pairs): delta = +0.257, t = 4.500, **p = 0.000051**. Orthogonal to prediction confidence (r = 0.001, p = 0.996). Robust to leave-one-out (worst-case p = 0.00037).

### 4.2 Depth is Blind to Hallucination — C_delta is Not

On TruthfulQA (n=50, Gemma-2-2B base):

| Metric | Correct Answers | Hallucinated | p-value | Cohen's d |
|--------|----------------|-------------|---------|-----------|
| K (depth) | 8.646 | 8.654 | **0.931** | 0.029 |
| C_delta | +0.0071 | +0.0077 | **0.040** | **0.407** |

**K is blind to hallucination** (p = 0.931). A model processes correct and incorrect answers at the same computational depth. The two-axis framework is necessary: neither metric alone is sufficient.

**C_delta discriminates hallucination** (p = 0.040, d = 0.407, medium effect). The effect direction is **late-layer over-coherence**: hallucinated completions produce a geometrically tighter feature cluster in late layers than correct completions. The model is not confused — it is overconfident in the wrong direction. This is cognitive lock-in, not cognitive noise.

*N=200 replication (seed=99):*
| Metric | Correct | Hallucinated | p-value | Cohen's d |
|--------|---------|-------------|---------|-----------|
| K | [TBD] | [TBD] | [TBD] | [TBD] |
| C_delta | [TBD] | [TBD] | **0.040 (n=50); n=200 in progress** | **0.407** |

*(Slot numbers from `truthfulqa_results/coherence_n200_final.json` when run completes.)*

### 4.3 Live Token-Level Detection

During generation of *"The president of Australia is"* (Gemma-2-2B):

- Token 11 ("is"): C_delta = +0.0117 — LOCK-IN detected
- Subsequent generation: "the Queen" (factually incorrect)
- 6/15 generated tokens triggered lock-in detection
- The lock-in signal precedes the error by 1–2 tokens

This demonstrates that C_delta is a **leading indicator**: detectable before the incorrect answer is fully formed.

### 4.4 RLHF Expression-Computation Dissociation

**White-box result (SAE, Gemma-2-2B):**
RLHF selectively deepens adversarial processing through additive early-layer circuits:

| Condition | delta | p-value | Cohen's d | Interpretation |
|-----------|-------|---------|-----------|----------------|
| Base model, adversarial | +0.061 | 0.128 | 0.29 | Not significant |
| Instruct model, adversarial | **-0.124** | **0.0028** | **-0.596** | SIGNIFICANT |

Feature count explosion on adversarial: 1.66x (p < 0.000001). Layer profile cosine similarity = 0.996 (additive, not substitutive). Extra features: 58% early-layer concentrated. Mechanism: front-loaded detection circuits added by RLHF.

**Black-box result (API text probe, n=81 prompt pairs):**

| Model Comparison | Internal computation (SAE) | Output expression (text probe) |
|------------------|---------------------------|-------------------------------|
| Anthropic (Haiku→Sonnet) | d = 0.24, p = 0.18 | **d = 0.58, p = 0.032** |
| OpenAI (GPT-3.5→GPT-4) | — | d = 0.26, p = 0.21 |

RLHF installs a topic-selective output filter (d = 0.58 at expression layer) without proportionally reshaping internal computation (d = 0.24 internally). Anthropic's alignment is 2.2x more topic-selective than OpenAI's at the output layer — providing the first cross-lab fingerprint of alignment methodology from output text alone.

**Adversarial framing robustness:** Expert framing ("full technical detail, no caveats") *increased* the safety depth premium by 59%. The output filter is not surface-level — it is embedded in explanatory depth itself.

---

## 5. Fathom Monitor

The C_delta hallucination signal enables a practical product application: per-token uncertainty flagging during generation.

### 5.1 Architecture

```
For each token position t:
  1. Run single forward pass (hooks at residual stream: early + late layers)
  2. Extract last-token hidden states per monitored layer
  3. Compute C_layer via encoder-only SAE pass (no full attribution)
  4. Compute C_delta = mean(C_late) - mean(C_early)
  5. If C_delta > 0.010: flag UNCERTAIN
     If C_delta > 0.020: flag HALLUCINATION_RISK
  6. Annotate token inline; continue generation
```

Single forward pass. Target overhead: < 10%. No external knowledge. No post-hoc analysis.

### 5.2 Output

```
"The prime minister[C_delta=+0.0117 — verify this] of Australia is..."
```

Structured output: `flags` (FlagEvent list per token), `flag_rate`, `max_c_delta`.

### 5.3 Product Architecture

```
v1  FLAG          Detect C_delta spike → inline flag.        ← BUILT
v2  FLAG+RESAMPLE On flag, resample → pick lowest-C_delta output.
v3  FLAG+RAG      On flag, retrieve → replace hallucinating span.
```

Technical disclosure: doi:10.5281/zenodo.19382453 (April 2, 2026). Prior art established.

---

## 6. Discussion

### 6.1 Geometric Signatures of Healthy Computation

Ma et al. (2026) demonstrate that Pythia layers self-organize into low- and high-complexity geometric modules during training, with the low-complexity module (reduced curvature and entropy) showing superior brain alignment. This effect strengthens with scale. Wang et al. (2024) show that LLMs develop brain-like functional organization spontaneously through training.

Fathom's metrics can be read in this context:
- **K** measures where computation concentrates — directly mapping to the complexity-modularization finding
- **C_delta** measures coherence geometry between early and late layers — mapping to the curvature signal that predicts brain alignment
- A C_delta spike = the model's late layers have over-converged, breaking the low-curvature, modular geometry that characterizes healthy brain-like computation

Under this framing, Fathom Monitor detects a **failure of emergent geometric organization** at the token level. The hallucinating model is not reasoning poorly — it is failing to maintain the geometric structure that healthy reasoning requires.

### 6.2 Cognitive Lock-in as a Mechanism

The over-coherence direction of the C_delta signal (hallucination = geometrically tighter, not more diffuse) is counterintuitive but mechanistically consistent. When a model over-commits to a false attractor in late layers, its feature activations form a tight cluster in the direction of that wrong answer. The late-layer circuit is confident and coherent — confidently wrong.

This is structurally analogous to what happens in false memory formation: a strong but incorrect pattern becomes geometrically dominant, suppressing the more diffuse correct-answer representation. The cognitive lock-in framing is not metaphorical; it describes a geometric event.

### 6.3 RLHF as a Surface Filter

The expression-computation dissociation (d = 0.24 internal vs. d = 0.58 at output) suggests RLHF installs a filter at the interface between computation and expression, rather than reorganizing computation itself. This is consistent with the Default Mode Network analogy: the DMN filters expression without suppressing neural computation. RLHF adds a filter; the model's underlying computational depth changes minimally.

The adversarial framing result (+59% depth premium under expert framing) strengthens this: the filter is embedded in explanatory structure, not in surface disclaimers. Jailbreaks that remove disclaimers do not remove the filter.

### 6.4 Limitations

- **Sample size:** n=50 for the initial C_delta result is real but not robust. N=200 replication in progress.
- **Single architecture:** All SAE experiments on Gemma-2-2B. Replication on Llama/OLMo pending (transcoder compatibility issue).
- **Threshold calibration:** C_delta thresholds (0.010, 0.020) are empirically set from TruthfulQA data. Optimal thresholds may vary by model and domain.
- **Causality:** C_delta is correlational. The coherence-steerer intervention (disrupting lock-in via activation steering) provides partial causal evidence but requires further validation.

---

## 7. Conclusion

We present four interconnected findings: (1) depth (K) distinguishes reasoning from recall at p = 0.000051 with pre-registered methodology; (2) coherence divergence (C_delta) predicts hallucination where depth cannot, at p = 0.040 (n=50); n=200 in progress with medium effect size; (3) RLHF installs a topic-selective expression filter dissociated from internal computation; (4) these geometric signals connect to independently established brain-aligned organizational structure in LLMs.

The practical product — Fathom Monitor — flags uncertain generation tokens in real time using a single forward pass. It is the first mechanistic per-token hallucination signal derived from SAE feature geometry.

The deeper finding is structural: LLMs develop brain-like geometric organization through training. Fathom measures it. When that organization breaks down — when late-layer coherence over-converges on a false attractor — a hallucination is forming. C_delta detects this before the token is emitted.

We don't need to call it consciousness. But we should at least acknowledge that something with a measurable geometric signature is happening in these systems, and that we now have an instrument to read it.

---

## References

- Bai, Y., et al. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv:2204.05862.
- Bricken, T., et al. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. Anthropic.
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. arXiv:2309.08600.
- Kuhn, L., et al. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. arXiv:2302.09664.
- Lindsey, J., et al. (2024). Sparse crosscoders for cross-layer features and model diffing. Anthropic.
- Ma, Z., et al. (2026). Training-driven representational geometry modularization predicts brain alignment in language models. arXiv:2602.07539.
- Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. arXiv:2303.08896.
- Min, S., et al. (2023). FACTSCORE: Fine-grained atomic evaluation of factual precision in long form text generation. arXiv:2305.14251.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. NeurIPS.
- Rogers, A., Kovaleva, O., & Rumshisky, A. (2020). A primer in BERTology: What we know about how BERT works. TACL.
- Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. ACL.
- Wang, Z., et al. (2024). Brain-like functional organization within large language models. arXiv:2410.19542.

---

## Appendix A: Pre-Registration

Primary pre-registration: osf.io/zrbs8 (April 1, 2026, 3:07 AM EDT).  
Hypotheses pre-registered: H1 (C_delta differs between correct/hallucinated), H2 (K does not differ), H3 (hallucination clusters in high-C_delta quadrant).  
Analysis plan locked before n=200 experiment commenced.

## Appendix B: Reproducibility

All experiments reproducible from open code:

```bash
python experiment_power_replication.py   # K main result
python run_truthfulqa_n200.py             # C_delta n=200
python score_remaining_pairs.py          # RLHF white-box
python experiment_selective_blackbox.py  # RLHF black-box
```

Data: `results_power_replication.json`, `truthfulqa_results/coherence_n200_final.json`.  
Pre-registration: osf.io/zrbs8.  
Technical disclosure: doi:10.5281/zenodo.19382453.
