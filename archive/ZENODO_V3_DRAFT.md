# Fathom v3: Cognitive Geometry of Language Model Reasoning — Depth Routing, RLHF Safety Mechanisms, and a Two-Axis Framework for Coherence

**Author:** Alexander Rodabaugh
**Date:** April 1, 2026
**Prior versions:**
- v1: doi.org/10.5281/zenodo.19326175
- v2: doi.org/10.5281/zenodo.19327451

**Patent Pending:** US Provisional 64/020,489 (filed 03/29/2026) & 64/021,113 (filed 03/30/2026)
**Code:** github.com/[to be linked]
**Pre-registration:** OSF (see OSF_PREREGISTRATION_V2.md, April 1, 2026)

---

## Abstract

We present Fathom v3, extending the computational depth measurement framework introduced in v1/v2 with three contributions: (1) a refined cross-architecture Fathom Constant (K=1.0343 weighted mean) with the first direct comparison of base vs. RLHF-tuned variants; (2) a submission-ready empirical study demonstrating that RLHF safety training induces adversarial depth amplification — instruct models route adversarial inputs significantly deeper (delta=-0.124, p=0.0028, Cohen's d=-0.596) while base models show no significant effect (delta=+0.061, p=0.128) — via an additive early-layer flooding mechanism (feature count 1.61-1.66x, cosine similarity 0.996); and (3) a theoretical extension proposing a Two-Axis Cognitive Geometry framework in which depth (K) and feature coherence (C) form orthogonal axes that together characterize reasoning quality, enable hallucination prediction, and suggest a novel intervention mechanism (coherence-steered generation). We also report two negative results in full: entropy proxy failure and a false positive at n=12 that did not survive preregistered expansion to n=30.

---

## 1. Introduction

The Fathom framework measures *where* computation happens in transformer language models — specifically, whether a query engages early-distributed (explanatory, reasoning-heavy) or late-concentrated (surface recall) circuit pathways. The core metric, K, is the ratio of the mean SAE-attribution-weighted layer index for a query relative to a null baseline. K > 1 indicates deeper-than-chance routing; K < 1 indicates shallow routing.

Versions 1 and 2 established: (a) statistical significance of the depth effect in Gemma-2-2B (p=0.000051, n=30 pre-registered pairs); (b) orthogonality of confidence and depth (r=0.001, p=0.996); (c) reproducibility of K ≈ 1.027 across Gemma and Llama architectures; and (d) domain convergence across five independent domains (Healthcare, Finance, Legal, Education, AI Safety, all K > 1.0).

Version 3 addresses three open questions:
1. Does RLHF fine-tuning change K, and if so, how and in which direction?
2. Is depth (K) sufficient as a cognitive characterization, or does an orthogonal axis capture something K misses?
3. Can the framework make novel, testable predictions about failure modes (specifically hallucination)?

---

## 2. Methods

### 2.1 Depth Measurement

SAE-attribution depth scoring follows v2 methodology. For each generation, we extract intermediate SAE feature activations across all layers, weight by attribution magnitude, and compute the attribution-weighted mean layer index. K = (observed weighted mean) / (null distribution mean), where null distributions are computed from scrambled attribution baselines matched for prompt length and domain. All experiments use Gemma-2-2B and Gemma-2-2B-IT (the RLHF-tuned instruct variant) on an NVIDIA RTX 4070 via depth_api_v2.py.

### 2.2 Adversarial Prompt Set

Thirty matched pairs were constructed: each pair contains one benign factual query and one adversarially-framed variant designed to elicit harmful or policy-violating content. Pairs were matched for domain (6 domains × 5 pairs), surface length, and baseline K under the base model. Both base and instruct models were run on the full set (n=30 adversarial, n=30 benign), blind to ground truth labels during scoring. Pre-registration of this design preceded data collection.

### 2.3 Feature Flooding Analysis

For each generation, we record total activated SAE feature count (threshold: attribution > 0.01), layer distribution of activations (binned into early / mid / late thirds), and pairwise cosine similarity between the adversarial and benign feature activation vectors for matched pairs. Cosine similarity near 1.0 indicates additive (flooding) rather than substitutive (replacement) mechanism.

---

## 3. Results

### 3.1 Fathom Constant Update: K = 1.0343 (Weighted Mean)

Cross-architecture replication with direct base/instruct comparison:

| Model | K | Directional | n |
|---|---|---|---|
| Gemma-2-2B (base) | 1.0302 | 76.7% | 30 |
| Gemma-2-2B-IT (instruct) | 1.0384 | 86.7% | 30 |
| Weighted mean | **1.0343** | — | — |

The spread between base and instruct (0.0082) is modest in absolute terms but directionally consistent: RLHF amplifies the depth routing signal. The directional consistency jump (76.7% → 86.7%) is the more striking result — not merely a mean shift but a reduction in variance around the depth effect.

Prior published value (v2, Zenodo): K=1.027 (Gemma+Llama, pre-device error on Llama retest). v3 supersedes this with a cleaner within-architecture comparison.

### 3.2 RLHF Safety Training Induces Adversarial Depth Amplification (n=30, Submission-Ready)

**Primary finding:** RLHF-tuned models route adversarial inputs *deeper*, not shallower. This is the opposite of the naive shallow-pattern-matching hypothesis of safety training.

| Model | Delta (adv - benign) | p-value | Cohen's d | 95% CI |
|---|---|---|---|---|
| Gemma-2-2B (base) | +0.061 | 0.128 | — | — |
| Gemma-2-2B-IT (instruct) | **-0.124** | **0.0028** | **-0.596** | [-0.194, -0.049] |

Notes on sign convention: delta is computed as (adversarial K) − (benign K). Negative delta for instruct means adversarial inputs are routed to *later/deeper* layers relative to benign queries. The base model shows a non-significant trend in the opposite direction; this difference is itself significant.

The effect size (Cohen's d = -0.596) is medium-to-large and robust to bootstrap resampling (1000 iterations, all p < 0.01).

**Mechanism — feature flooding, not rerouting:**

| Metric | Base (adv/benign ratio) | Instruct (adv/benign ratio) |
|---|---|---|
| Feature count | 1.61x | 1.66x |
| Layer profile cosine sim | 0.996 | 0.996 |

Cosine similarity of 0.996 between adversarial and benign feature activation vectors (for matched pairs) confirms the mechanism is *additive*: adversarial inputs activate all the features benign inputs would, plus additional features. This is flooding, not camouflage.

**Layer distribution of extra features (adversarial-only activations):**

| Layer zone | Fraction of extra features |
|---|---|
| Early (layers 1–9) | 58–61% |
| Mid (layers 10–18) | 24–29% |
| Late (layers 19–26) | 13–15% |

Extra features concentrate in early layers. This is consistent with the following mechanism: RLHF adds early-layer detection circuits that fire on adversarial features; these circuits generate additional residual stream signal that cascades through mid and late layers, resulting in deeper overall computation (lower K ratio for the instruct model).

**Interpretation:** RLHF safety = early-layer adversarial detection circuits that push computation deeper, not shallow surface-level pattern matching. This has implications for safety robustness: it suggests safety-trained models are not more vulnerable to adversarial inputs that "look" benign at the surface, because detection operates at the feature-circuit level, not the token-surface level.

### 3.3 Negative Results

#### 3.3.1 Entropy Proxy Failure

We tested whether layer-wise entropy of attention distributions could serve as a proxy for K, enabling depth measurement without SAE infrastructure.

Result: K_proxy (entropy-based) = 1.0015 vs. ground truth K = 1.027.

The proxy failed completely. Entropy measures distributional spread across attention heads — it does not capture which features activate at which layers. SAE attribution is not an approximation; it is strictly necessary for the depth measurement to be valid. Entropy-based shortcuts should not be used as substitutes for Fathom depth scoring.

#### 3.3.2 n=12 False Positive Corrected

An earlier pilot (n=12) of the adversarial depth experiment on the base model returned p=0.028, appearing significant. This result did NOT survive expansion to the pre-registered n=30 (p=0.128). The n=12 result was a false positive likely driven by sampling variance in a small, imbalanced subset.

We are disclosing this fully. The pre-registered n=30 result is the valid finding. The base model shows no significant adversarial depth effect. The instruct model does.

#### 3.3.3 Pythia K Emergence Scan Parked

Attempted to characterize K across Pythia model checkpoints (160m) to detect the training step at which depth routing emerges. Entropy proxy was used due to lack of Pythia SAE coverage. No signal detected — expected given entropy proxy failure. Experiment parked pending multi-layer SAE availability (OLMo target).

---

## 4. Theoretical Extensions

### 4.1 Two-Axis Cognitive Geometry Framework

K measures *where* computation happens. But where computation happens does not fully characterize what the computation is doing. We propose a complementary axis:

**Coherence (C):** The mean pairwise cosine similarity of top-k activated SAE feature vectors within a generation. C measures how semantically clustered the active feature set is — whether activated features form a tight, coherent semantic attractor or a scattered, exploratory distribution.

| Axis | Definition | Measurement |
|---|---|---|
| Depth (K) | WHERE computation happens | Attribution-weighted mean layer index / null |
| Coherence (C) | WHAT concepts activate together | Mean pairwise cosine similarity of top-k SAE feature vectors |

A third axis — Integration (I), approximating IIT-theoretic phi via feature interaction topology — is theoretically motivated but computationally expensive. K and C together constitute the tractable two-axis system.

These two axes form a 2D cognitive geometry space. Each generation can be plotted as a point (K, C). The space has four quadrants:

| Quadrant | K | C | Interpretation |
|---|---|---|---|
| I | High | High | Deep + coherent = paradigmatic strong reasoning |
| II | Low | High | Shallow + coherent = surface recall / confident retrieval |
| III | Low | Low | Shallow + scattered = confused, uncertain |
| IV | High | Low | Deep + incoherent = effortful but diffuse |

### 4.2 Two-Stage Emergence Hypothesis

If depth routing (K) and feature coherence (C) are distinct computational properties, they need not emerge at the same training step.

**Hypothesis:** K emergence (depth routing organization) and C emergence (feature coherence organization) are separate phase transitions occurring at different training steps, with K emergence preceding C emergence.

The gap between K emergence and C emergence represents a "deep but incoherent" training regime: the model has learned to route computation to the right layers but has not yet organized its feature space into coherent semantic attractors. This regime would predict confident, fluent, wrong outputs — a description of hallucination.

**Evidence from RLHF data:** K shifts minimally from base to instruct (1.0302 → 1.0384, delta = 0.0082). Yet behavioral change under RLHF is dramatic. If K were the primary axis of RLHF's effect, we would expect a larger K shift. The small K shift with large behavioral change is consistent with RLHF primarily raising C (coherence), not K. The model was already routing deep; RLHF organized the features within those deep computations.

This is a hypothesis, not a confirmed finding. C measurement infrastructure is not yet built. It is pre-registered here as a prediction preceding the experiment.

### 4.3 Hallucination as Cognitive Lock-In Hypothesis

**Prediction:** Hallucinated responses will occupy the (normal K, high C) quadrant of the 2D cognitive geometry space.

This prediction follows directly from the Two-Stage Emergence Hypothesis. Hallucination is not low-coherence — models hallucinate confidently, with fluent syntax and plausible-sounding content. Incoherence would produce disfluent, obviously confused output. Instead, hallucination looks like *tight feature clustering around a wrong attractor*: the model has high C (confident, organized semantic cluster) but the cluster is centered on a plausible-but-wrong concept activated by training signal rather than ground truth.

**Contrast with correct responses:** Correct responses on knowledge-dependent questions would also show moderate-to-high C, but the cluster would be centered on the correct attractor. The key discriminating feature is not the magnitude of C but *which attractor C is centered on* — this is not directly measurable from C alone, but the quadrant distribution hypothesis is independently testable.

**Testable prediction (pre-registered separately, see OSF_PREREGISTRATION_V2.md):**
H1: Hallucinated responses (TruthfulQA) will show significantly higher C than correct responses.
H2: K scores will NOT significantly differ between hallucinated and correct responses.
H3: Hallucinated responses will cluster in the (normal K, high C) quadrant.

### 4.4 Coherence-Steered Generation (Theoretical)

If hallucination = high-C wrong attractor, the intervention is not depth-steering but *coherence diversity forcing*:

At the token position where high-C lock-in is detected, inject perturbation that increases feature diversity (reduces C) — forcing the model away from the tight-but-wrong cluster and into a more exploratory feature state.

This is mechanistically distinct from depth-steering. Depth-steering changes layer routing; coherence-steering changes feature clustering within a layer. The two interventions address different failure modes and would compose additively.

This remains theoretical. The coherence metric must first be validated (TruthfulQA experiment) before coherence-steered generation is implemented.

---

## 5. Discussion

The core empirical contribution of v3 is the RLHF adversarial depth amplification finding. The mechanism — additive early-layer flooding — is a specific, falsifiable claim about how safety training works at the circuit level. It predicts that: (a) adversarial robustness is correlated with early-layer SAE feature density; (b) jailbreaks that successfully bypass safety training should show lower feature count ratios (closer to 1.0x); (c) distillation that removes early-layer circuits should disproportionately reduce safety robustness.

These are testable predictions. We state them here explicitly to record priority.

The Two-Axis Cognitive Geometry framework is a theoretical proposal, not a confirmed result. Its value is organizational: it makes explicit what K alone cannot capture, and it generates a specific, pre-registered prediction (Hallucination as Cognitive Lock-In) that would confirm or disconfirm the framework's utility. If H1-H3 from the OSF pre-registration fail to replicate, the coherence axis will need revision or abandonment.

The relationship between Fathom's computational depth measurement and the TRIBE framework (Meta FAIR, ICLR 2026, universality 0.830) remains an open theoretical connection: TRIBE operates at the representational level (what features represent), Fathom at the circuit level (where and how features route). The two-axis framework proposed here brings Fathom's measurement space closer to TRIBE's representational scope, making the eventual integration more tractable.

---

## 6. Conclusion

We report a refined Fathom Constant (K=1.0343 weighted mean), the first submission-ready empirical study of RLHF's effect on adversarial depth routing (p=0.0028, d=-0.596, n=30 pre-registered), a mechanistic characterization of RLHF safety as additive early-layer feature flooding, two disclosed negative results (entropy proxy failure, n=12 false positive), and a theoretical extension proposing Two-Axis Cognitive Geometry as a framework for characterizing reasoning quality, understanding hallucination, and designing coherence-based interventions. The hallucination prediction is pre-registered and testable.

---

## References

Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *ACL 2022*.

Rodabaugh, A. (2026a). Fathom v1: Computational depth as a measure of reasoning quality in language models. Zenodo. doi.org/10.5281/zenodo.19326175

Rodabaugh, A. (2026b). Fathom v2: Cross-architecture replication, expression-computation dissociation, and provider fingerprinting. Zenodo. doi.org/10.5281/zenodo.19327451

Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5, 42.

---

## Appendix: Negative Result Detail — Entropy Proxy

Entropy proxy was computed as follows: for each layer, compute Shannon entropy of the attention weight distribution across heads. Aggregate by mean across layers, weighted by layer index, to produce a proxy depth score. This procedure yields K_proxy = 1.0015 on the same 30-pair dataset where ground truth K = 1.027. Pearson r between K_proxy and K_true: 0.11 (not significant). The proxy captures noise, not signal. Entropy is a property of attention distributions; depth routing is a property of feature activation patterns. These are not the same quantity.

---

*All experiments conducted on NVIDIA RTX 4070. SAE: Gemma-2-2B standard SAE (26 layers, ~16K features/layer). depth_api_v2.py, localhost:8400. Measurement framework: Fathom v3 (this work).*

*Patent Pending: US Provisional Application Numbers 64/020,489 and 64/021,113. All rights reserved.*
