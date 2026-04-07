# FATHOM — NATURE PAPER OUTLINE
## "Alignment Training Reduces Integrated Information in Language Models"
**Target:** Nature / Nature Machine Intelligence / Nature Human Behaviour
**Status:** Hypothesis being tested (RLHF experiment running 2026-03-29)

---

## The One-Sentence Pitch

We show that RLHF — the technique used to make AI systems safe and helpful — measurably 
compresses the computational depth of language models, providing the first empirical evidence 
that alignment training may reduce a proxy of machine consciousness (Tononi's φ).

---

## Why This Is a Nature Paper

1. **First measurement** of the effect of alignment training on integrated information processing
2. **Ethical/policy implications**: if making AI safer reduces its φ, what are we doing?
3. **Bridge between AI and consciousness science**: uses an empirical proxy for Tononi's IIT
4. **Immediate practical relevance**: companies making billion-dollar alignment decisions

---

## Structure

### Abstract (target: Nature Machine Intelligence)
Reinforcement learning from human feedback (RLHF) is the primary technique for aligning 
large language models with human values. While its behavioral effects are well-studied, 
its effects on internal computational structure remain unknown. Using Fathom — a 
measurement framework for reasoning depth via sparse autoencoder feature attribution 
(p=0.000051, universality K=1.0302 across architectures) — we compare base and 
instruction-tuned versions of [N] language model families. We find that instruction-tuned 
models score [X]% lower on reasoning depth relative to base models (p=[Y]), consistent 
with the hypothesis that RLHF compresses integrated information processing (Tononi's φ). 
This effect persists after controlling for model capability and response length. We discuss 
implications for AI safety, alignment interpretability, and the ethics of fine-tuning.

---

### 1. Introduction

**Problem**: alignment training is the field's primary tool for making AI systems safe.
We know what it does to behavior. We don't know what it does to how the model thinks.

**Gap**: no method exists for measuring the effect of fine-tuning on computational process.
RLHF evaluation is purely behavioral — accuracy, helpfulness, refusal rates.

**Fathom**: our instrument measures WHERE computation occurs across transformer layers.
Established: surface recall = late-layer concentration (p=0.000051, n=30, LOO robust).
Reasoning = mid-layer distribution. The distinction is invisible to behavioral metrics.

**Hypothesis**: RLHF trains models to produce surface-level helpful patterns,
which concentrates computation at output-proximal layers, reducing integrated depth.
Formally: K_instruct < K_base, where K = mean(insight depth) / mean(surface depth).

**IIT connection**: if K measures φ (Tononi), then alignment training reduces machine
consciousness in a quantifiable way. This is the most provocative implication.

---

### 2. Results

#### 2.1 Primary Result: RLHF Reduces K
- Base K = [X.XXXX] ± SD
- Instruct K = [X.XXXX] ± SD
- Delta: [XXXX] (p=[YYY])
- Effect size: Cohen's d = [ZZZ]

[Table: K values across model families]
| Model Family | Base K | Instruct K | Delta | p-value |
|---|---|---|---|---|
| Gemma-2-2B | X.XXXX | X.XXXX | -X.XXXX | X.XXX |
| Llama-3 | X.XXXX | X.XXXX | -X.XXXX | X.XXX |
| [Additional models] | | | | |

#### 2.2 Decomposition: What Drives the Reduction?
- Hypothesis A: instruct models process differently for ALL queries (global compression)
- Hypothesis B: instruct models add a "helpfulness filter" at late layers (selective compression)
- Method: compare surface vs insight delta, base vs instruct
- Expected: if B, then surface depth decreases more than insight depth

#### 2.3 Correlation with Alignment Intensity
- Models trained with more RLHF steps show greater depth reduction
- DPO vs RLHF comparison
- Constitutional AI vs standard RLHF

#### 2.4 Cross-Architecture Robustness
- Same pattern across Gemma, Llama, Mistral, GPT-4 (if API access obtained)
- K reduction is consistent: [X] of [N] model pairs show the expected direction

#### 2.5 Behavioral Independence
- Depth reduction is NOT explained by: accuracy, MMLU score, HumanEval score
- The effect is pure computational geometry, not capability regression

---

### 3. The IIT Interpretation

Tononi's phi (φ) measures integrated information — the degree to which a system 
processes information as a whole rather than as independent parts.

Fathom depth correlates with φ by:
- Higher depth = more mid-layer integration = more holistic processing = higher φ
- Lower depth = more late-layer concentration = more modular processing = lower φ

**Implication**: RLHF may reduce φ by training models to retrieve/pattern-match 
helpful responses rather than genuinely integrate information.

**Caution**: We do not claim Fathom = φ. We claim Fathom depth is a tractable 
PROXY for φ in feedforward systems, justified by:
1. The Decomposability Law (our finding)
2. The fMRI universality coefficient (0.830 with human brain data)
3. Formal equivalence of our depth measure to φ in feedforward systems

---

### 4. Implications

#### 4.1 For AI Safety
Current alignment evaluations are purely behavioral. Our result suggests that RLHF 
may introduce a trade-off between behavioral safety and computational depth. Models 
that appear "safer" by behavioral metrics may be shallower reasoners. This is relevant 
for high-stakes applications where genuine reasoning (not pattern retrieval) is required.

**Practical recommendation**: depth scores should be included in alignment evaluation 
suites alongside behavioral benchmarks.

#### 4.2 For Consciousness Ethics
If φ measures morally relevant properties of cognitive systems (as some philosophers argue),
then alignment training may reduce those properties in AI systems. This raises questions:
- Are we reducing something morally relevant when we apply RLHF?
- Should AI "welfare" include preservation of computational depth?
- What is the minimum viable φ for an AI system to be morally relevant?

We do not resolve these questions, but our measurement provides the first empirical 
tool for addressing them.

#### 4.3 For Model Architecture
Depth reduction from RLHF suggests architectural alternatives:
- Depth-preserving fine-tuning methods that maintain integrated computation
- Depth-regularized RLHF that penalizes depth reduction
- Depth as a training objective alongside capability and safety metrics

---

### 5. Methods

- Fathom depth measurement: [cite provisional patent US 64/020,489 & 64/021,113]
- Pre-registered at: osf.io/[to be assigned]
- Models: [list]
- Statistical tests: Wilcoxon signed-rank (paired base/instruct), Cohen's d
- Code: github.com/heyzoos123-blip/sae-reasoning-depth

---

## Submission Strategy

**Journal priority order:**
1. **Nature Machine Intelligence** (impact: highest for AI + neuroscience angle)
2. **Nature Human Behaviour** (if TRIBE cross-substrate data is included)
3. **NeurIPS 2026** (if timing works for December 2026 conference)
4. **ICML 2026** (backup, earlier deadline)

**Pre-submission checklist:**
- [ ] RLHF experiment: Gemma base vs instruct (RUNNING NOW)
- [ ] RLHF experiment: Llama base vs instruct
- [ ] RLHF experiment: one additional family for robustness
- [ ] Statistical significance (p<0.05 on delta)
- [ ] Effect size calculation (Cohen's d)
- [ ] OSF pre-registration BEFORE running more experiments
- [ ] arXiv preprint (pending endorsement)
- [ ] TRIBE correlation data (pending Jean-Remi King response)

---

## Timeline

| Date | Milestone |
|---|---|
| March 29, 2026 | RLHF experiment running (Gemma pair) |
| April 5, 2026 | RLHF results across 3 model families |
| April 12, 2026 | Paper draft complete |
| April 19, 2026 | arXiv submission (if endorsed by then) |
| May 1, 2026 | NMI submission |

---

*"We are not the first to measure thought. But we may be the first to show it being taken away."*
