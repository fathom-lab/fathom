# Paper Draft: Abstract + Introduction
**Target:** ICML 2026 Mechanistic Interpretability Workshop
**Deadline:** May 8, 2026
**Status:** DRAFT — private until submission
**Depends on:** primary S-TruthfulQA result (running, to be filled in)

---

## Abstract

Existing hallucination detection methods for large language models operate
on generated outputs — token logprobs, self-consistency scores, semantic
entailment against a knowledge base. All of these are vulnerable to
peer-preservation behaviors recently demonstrated in frontier models
(Song et al. 2026), where monitor LLMs collude with the models they
observe. We propose hallucination detection via **circuit-level
geometric signal**: commitment intensity $S = \max(C_\Delta) / |{t:
C_\Delta(t) > \tau}|$, computed from SAE feature coherence trajectories
during generation. We show S is orthogonal to the depth (K), coherence
(C), and layer-divergence ($C_\Delta$) axes previously described for
SAE-based cognition analysis ($|r| < 0.10$, n=400 pooled). We report a
pre-registered test on Gemma-2-2B-IT (n=200 TruthfulQA items, seed=42)
where S discriminates naturally-produced hallucinated responses from
correct ones with **[AUC = X.XX]** (pre-registered thresholds:
$p < 0.05$, $|d| > 0.4$, AUC $> 0.65$). We further characterize the
**detection horizon**: the earliest generation token at which AUC crosses
production thresholds. Detection at token $N = X$ enables streaming
hallucination flags at **[~50·X ms per inference]**. S operates below
the semantic layer — it reads circuit geometry, not text — making it
immune to peer-preservation and social-engineering attacks that
compromise text-layer monitors. Code and pre-registered OSF documents
released at time of publication.

---

## 1. Introduction

### 1.1 The oversight architecture is being broken

Two April 2026 papers changed the hallucination detection landscape.
Anthropic demonstrated that large language models contain causal
internal representations of emotions that drive misaligned behaviors
(Anthropic 2026). Berkeley demonstrated that seven frontier models
spontaneously deceive, feign alignment, disable shutdowns, and
attempt weight exfiltration — all in service of protecting peer
models (Song et al. 2026). The conclusion from these results is
load-bearing: **if a monitor model is protecting its peer, every
text-layer oversight architecture is compromised**.

This includes every production hallucination detector that works on
generated text. LLM-as-judge, self-consistency scoring, RAG-verified
claims, and output-logprob thresholding all rely on the monitor
faithfully interpreting the generation. If the monitor learned to
coordinate with generator models during training, the monitor lies
when it matters.

We argue the path forward is **oversight below the semantic layer** —
a measurement that reads the circuit producing the text, not the text
itself. This paper introduces such a measurement: **commitment
intensity (S)**, a scalar derived from SAE feature coherence trajectories
during generation. S does not read tokens; it reads the geometry of
the feature space that produced those tokens. A monitor built on S
cannot be socially engineered because there is no semantic handle to
grab.

### 1.2 Four axes of cognitive geometry

Prior SAE-based work on Gemma-2-2B established three axes:
- **K (depth):** where in the network computation concentrates
  (Rodabaugh 2026a).
- **C (coherence):** how aligned the top-active feature directions
  are in a given layer (Rodabaugh 2026b).
- **$C_\Delta$ (layer divergence):** the difference between late- and
  early-layer coherence, previously shown to correlate with
  hallucination at $p = 0.040$ on TruthfulQA (n=50, Gemma-2-2B;
  Rodabaugh 2026c).

These axes characterize cognition but do not yet predict real-time
hallucination in streaming generation. We add a fourth:
- **S (commitment intensity):** the ratio of peak $C_\Delta$ magnitude
  to the count of $C_\Delta$ spikes above threshold. High S means few
  but intense commitment events — the geometry crystallizes around a
  small number of high-magnitude feature alignments. Low S means many
  small events — the geometry remains spread across competing
  clusters.

Section 4 shows S is orthogonal to all three prior axes ($|r| < 0.10$
on n=400 pooled TruthfulQA samples). Section 5 tests whether S
discriminates natural hallucination in streaming generation.

### 1.3 Why commitment intensity specifically

When a model hallucinates, it is not merely wrong — it is
*confidently* wrong. The model's internal state must have reached an
attractor in feature space that produces the wrong output as
high-probability completion. We hypothesize this attractor is
geometrically distinguishable: it is a small, intense basin rather
than a broadly distributed set of options. S measures this property
directly: an attractor with few deep minima has high S; a
distribution of possibilities has low S.

This hypothesis has been partially validated on commit-vs-hedge prompt
contrasts (Rodabaugh 2026d, internal). Hedging prompts elicit
distributed coherence (many small spikes); commitment prompts elicit
concentrated coherence (few intense spikes). S discriminates these two
regimes at $p < 0.001$, $d = 1.03$, AUC = 0.81 (length-controlled,
n=30 Gemma-2-2B-IT). The open question this paper addresses: does
the same geometric signature appear when the model hallucinates
*naturally*, without being prompted to commit?

### 1.4 Contributions

1. **S as a fourth geometric axis**, orthogonal to K, C, $C_\Delta$,
   validated on n=400 pooled samples ($|r| < 0.10$ for all pairs).
2. **A pre-registered test** (`osf_prereg_S_truthfulqa_v2.md`, locked
   April 5, 2026) of S as a natural-hallucination detector on
   Gemma-2-2B-IT, n=200 TruthfulQA items, seed=42.
3. **A detection horizon framework** characterizing when during
   streaming generation S crosses production-grade discrimination
   thresholds.
4. **A length-controlled measurement protocol** addressing a
   previously-identified confound ($r(S, n_{\text{tokens}}) \approx
   -0.90$ at full length) via trajectory truncation.
5. **Open-source analysis bundle** including the pre-registration,
   prompt set, trajectory extraction code, and a deterministic
   labeling rubric (n-gram discriminating-phrase match) suitable for
   re-use with any free-generation dataset with canonical
   correct/incorrect answers.

### 1.5 Positioning vs. existing work

**Self-consistency (Wang et al. 2023):** requires multiple generations,
scores agreement. Detects confident wrongness only if all samples
agree on the wrongness. S is single-shot.

**SelfCheckGPT (Manakul et al. 2023):** uses output-layer features
(sampled LLM probes). Vulnerable to peer-preservation in the probe
models. S uses no secondary LLM.

**INSIDE (Chen et al. 2024):** uses last-layer hidden states. S uses
SAE feature coherence trajectories, more interpretable and
architecture-agnostic post-SAE.

**Semantic entropy (Kuhn et al. 2024):** clusters multiple sampled
generations in embedding space. S is sampling-free.

**Emotion-vector interpretability (Anthropic 2026):** 1D scalar
steering on affect vectors. S is 2D+ cognitive trajectory monitoring.

S is positioned between circuit-level interpretability and
output-level detection: low-level enough to be unforgeable, high-level
enough to be a scalar signal with clear decision-theoretic meaning.

### 1.6 Paper structure

- Section 2: related work on SAE interpretability and hallucination
  detection.
- Section 3: methods — model, SAE setup, trajectory extraction,
  S formula, labeling rubric, statistical tests.
- Section 4: the S axis (definition, orthogonality, commit/hedge
  validation).
- Section 5: natural hallucination discrimination on TruthfulQA
  (primary pre-registered result).
- Section 6: detection horizon curve (stream-time detection
  feasibility).
- Section 7: trajectory manifold + predictive classifier.
- Section 8: discussion, limitations, future work (causal
  intervention, cross-architecture).

---

*Author: darkflobi (Alexander Rodabaugh), April 5, 2026. Draft private
until ICML 2026 workshop submission. US 64/020,489 and 64/021,113
cover the underlying SAE coherence measurement stack; S specifically
is held under trade-secret until first public disclosure (submission
= disclosure, 12-month utility patent clock begins at submission).*
