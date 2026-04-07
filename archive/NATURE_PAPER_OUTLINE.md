# Nature Machine Intelligence — Paper Outline

**Working Title:** Convergent Reasoning Topology Across Biological and Artificial Neural Systems

**Target:** Nature Machine Intelligence (Article format, ~5000 words + Methods)
**Authors:** darkflobi, [neuroscience collaborator TBD]
**Status:** OUTLINE — requires fMRI collaboration before submission

---

## Positioning

This is the second paper. The first paper (arXiv, NeurIPS track) establishes that SAE circuit depth distinguishes surface recall from explanatory reasoning in a single transformer (Gemma-2-2B, p=0.000051). This paper asks the bigger question: is the computational topology that distinguishes recall from reasoning a property of information processing systems in general — not just transformers?

The first paper's universality analysis (coefficient 0.830) is preliminary: it uses published group-level fMRI summaries, not original fMRI data. This paper fixes that. It pairs the Fathom pipeline with an actual fMRI collaboration to test convergence properly.

---

## Abstract (~150 words)

**Core claim (to be validated):** Biological and artificial neural systems exhibit convergent computational topology when differentiating factual recall from explanatory reasoning. In transformers, sparse autoencoder circuit tracing reveals that recall concentrates feature activation at output-proximal layers while reasoning distributes activation across earlier layers. In human subjects, functional neuroimaging of matched cognitive tasks shows an analogous pattern: recall activates hippocampal/output-proximal regions while reasoning engages distributed prefrontal processing. We formalize both observations within a common topological framework — a normalized 3-vector [early, mid, late] capturing the distribution of computation across processing stages — and measure cross-system correspondence.

**Key framing:** This is not a claim of shared mechanism. Transformers and brains are architecturally different. The claim is about convergent computational strategy: two very different systems, solving similar problems, arrive at similar processing topologies. This is a statement about the problem, not the substrate.

---

## 1. Introduction (~800 words)

### The Gap

Mechanistic interpretability and computational neuroscience study the same question from opposite sides: how do information processing systems implement cognition? But the two fields rarely compare results directly. Interpretability researchers characterize transformer internals without reference to neuroscience. Neuroscientists study brain activation without reference to language model internals. Cross-system comparison is rare — and when attempted, usually involves superficial analogies (attention ≈ attention) rather than formal topology.

### What Has Changed

Two developments make principled comparison possible:

1. **Sparse autoencoder (SAE) circuit tracing** now provides layer-resolved activation maps for transformer computation. Prior work [cite Paper 1] demonstrated that SAE feature attribution depth distinguishes surface recall from explanatory reasoning in Gemma-2-2B (delta=+0.257, p=0.000051, n=30 pre-registered pairs). The measurement pipeline (Fathom) produces a normalized processing signature: [early_ratio, mid_ratio, late_ratio].

2. **Task-controlled fMRI studies** have produced analogous data for human cognition. Published work on recall vs. reasoning tasks [cite Duncan 2010, Miller & Cohen 2001, Christoff et al. 2009] shows differential activation topology: recall concentrates processing in hippocampal and output-proximal cortical regions; reasoning engages distributed prefrontal networks.

### The Question

Do these two systems — which share no architecture, training procedure, or substrate — converge on the same processing topology for the same cognitive distinction? If so, this convergence says something about the computational requirements of reasoning itself, independent of implementation.

### Preliminary Evidence

A prior analysis [cite Paper 1, Section 5.6] computed a universality coefficient of 0.830 using published group-level fMRI summaries. This was exploratory: the fMRI data was not task-matched to the transformer prompts, was aggregated at the group level, and the normalization scheme was ad hoc. This paper addresses all three limitations.

---

## 2. Methods (~1500 words)

### 2.1 Transformer Pipeline: Fathom

- **Model:** Gemma-2-2B (26 layers), with planned extension to Llama-3.2-1B and one additional architecture
- **SAE infrastructure:** Circuit-tracer library with mwhanna/gemma-scope-transcoders
- **Scoring:** For each prompt, compute mean layer index of activated SAE features, weighted by attribution score. Normalize to [early_ratio, mid_ratio, late_ratio] by partitioning layers into thirds
- **Pairs:** 30 pre-registered surface/insight pairs from Paper 1, vocabulary-controlled (matched on token count, word frequency distribution, readability scores, N-gram overlap)
- **Validation:** Effect orthogonal to prediction confidence (r=0.001, p=0.996)

### 2.2 Human Neuroimaging Pipeline (NEW — requires collaborator)

- **Task design:** Translate the 30 transformer prompt pairs into matched human tasks. Each participant sees both conditions (within-subjects). Surface condition: factual recall questions ("What is the capital of France?"). Insight condition: explanatory reasoning questions ("Why do economies experience inflation?") matched on topic, vocabulary level, and response length
- **Imaging:** fMRI (3T, whole-brain coverage, TR=1s or faster). Standard preprocessing: motion correction, spatial normalization, smoothing
- **ROI parcellation:** Divide cortex into early (primary sensory/input regions), mid (association cortex), and late (prefrontal/hippocampal output-proximal) zones. This parcellation must be justified independently of the transformer analogy — possibly based on functional connectivity hierarchy [cite Margulies et al. 2016, cortical gradient work]
- **Signature extraction:** For each trial, compute [early_ratio, mid_ratio, late_ratio] from ROI-level BOLD signal. Average within condition (recall vs. reasoning)

### 2.3 Common Topological Framework

- **Representation:** Both systems produce a normalized 3-vector per condition. The framework is agnostic to substrate
- **Correspondence metric:** Universality coefficient = 1 - mean(variance across systems for each component). Range 0–1. Also compute cosine similarity between system pairs
- **Statistical test:** Permutation test — shuffle system labels 10,000 times, compute null distribution of universality coefficients. Report p-value for observed coefficient exceeding null
- **Controls:** (a) Scrambled ROI assignment in fMRI data. (b) Random feature-to-layer mapping in transformer data. (c) Cross-condition comparison (surface-transformer vs insight-fMRI) as negative control — should show low correspondence

### 2.4 Multi-Architecture Extension

- Test Llama-3.2-1B (once HuggingFace access approved) and at least one non-Gemma architecture
- Compute universality coefficient across: Gemma + Llama + fMRI. Three independent systems, same topology = stronger claim

---

## 3. Results (structure — numbers TBD)

### 3.1 Replication of Transformer Depth Effect

- Reproduce Paper 1 core finding on Gemma-2-2B (delta, p-value, effect size)
- Report multi-architecture results if available

### 3.2 Human Neuroimaging Results

- Within-subjects comparison: recall vs. reasoning activation topology
- Report [early, mid, late] signatures per condition
- Within-system effect size (fMRI recall vs. fMRI reasoning)

### 3.3 Cross-System Convergence

- **Primary result:** Universality coefficient across transformer(s) + fMRI
- Permutation test p-value
- Cosine similarity between system pairs
- Visualization: radar plot or parallel coordinates showing [early, mid, late] for each system × condition

### 3.4 Negative Controls

- Scrambled-ROI universality coefficient (expected: ~0.5, near chance)
- Cross-condition comparison (expected: low correspondence)
- Random feature mapping (expected: near-zero coefficient)

---

## 4. Discussion (~1000 words)

### 4.1 What Convergence Means

Convergent topology does not imply shared mechanism. Transformers and brains are architecturally different. But both systems, facing the distinction between retrieving stored facts and constructing explanatory reasoning, adopt the same computational strategy: concentrate retrieval processing at output-proximal stages, distribute reasoning processing across earlier stages.

This is a claim about **computational geometry**, not shared implementation. The analogy is closer to convergent evolution in biology: wings evolved independently in birds, bats, and insects — different mechanisms, same functional solution to the same physical problem.

### 4.2 Implications for Mechanistic Interpretability

If reasoning topology is substrate-independent, then interpretability findings in transformers may inform neuroscience and vice versa. The Fathom framework — measuring where computation concentrates — could serve as a common language between the two fields.

### 4.3 Implications for Cognitive Science

The hierarchy of processing depth (input-proximal to output-proximal) may be a universal organizing principle for information processing systems that distinguish retrieval from reasoning. This connects to:
- Dual-process theory (System 1/2) — shallow/deep processing maps onto recall/reasoning
- Global workspace theory — reasoning as distributed cortical engagement
- Predictive processing — recall as efficient prediction, reasoning as prediction error resolution

### 4.4 What This Does Not Show

- Does not establish causal equivalence between transformer layers and brain regions
- Does not claim transformers "think like humans" — the claim is about topology, not phenomenology
- Does not generalize beyond the recall/reasoning binary — finer-grained cognitive taxonomies untested
- 3-vector representation is a compression — may miss important within-category variation

### 4.5 Depth as Orthogonal to Correctness

Paper 1 established that depth is orthogonal to factual accuracy (p=0.62 on TruthfulQA, n=140). This sharpens the interpretation: depth topology captures the *type* of processing (retrieval vs. reasoning), not its *quality*. This is consistent with the neuroscience literature — brain regions that activate during reasoning activate regardless of whether the reasoning reaches a correct conclusion.

---

## 5. Future Work (~400 words)

### 5.1 Proper fMRI Collaboration

- Individual-subject analysis (not just group-level)
- Time-resolved topology: does reasoning depth build over time in both systems?
- Developmental comparison: does the topology emerge during training (transformers) and development (humans)?

### 5.2 Multi-Architecture Testing

- Test on 3+ transformer families (Gemma, Llama, Mistral minimum)
- Test on different scales within families (2B, 7B, 70B)
- The Fathom Constant: if the surface/insight depth ratio converges across architectures, this is an empirical constant of reasoning computation

### 5.3 Beyond Binary

- Extend to multi-class cognitive taxonomy: memory retrieval, causal reasoning, analogical reasoning, creative generation, mathematical proof
- Each type may have a distinct topological signature — a "periodic table" of cognitive operations

### 5.4 Practical Applications

- Depth scoring as a model evaluation metric (reasoning capability independent of benchmark accuracy)
- Real-time depth monitoring for agentic AI systems
- Depth-based training data curation (select training examples that elicit deep processing)

---

## 6. Critical Requirements Before Writing

This outline is the skeleton. The paper cannot be written without:

1. **A neuroscience collaborator.** The fMRI component requires someone with imaging access, IRB approval, and domain expertise. The transformer side is done. The neuro side is the bottleneck.

2. **Task-matched fMRI data.** The Paper 1 universality analysis used published group-level summaries — adequate for a preliminary result, not for a Nature MI submission. Need original fMRI data with task conditions matched to the transformer prompt pairs.

3. **Multi-architecture transformer results.** Gemma-2-2B alone is not sufficient for a universality claim. Need at least Llama-3.2-1B (pending HuggingFace access) and ideally one more family.

4. **Independent ROI parcellation justification.** The [early, mid, late] mapping for brain regions must be motivated by neuroscience, not by the transformer analogy. Otherwise the convergence is circular.

---

## Timeline (Estimated)

| Milestone | Target |
|-----------|--------|
| Identify neuroscience collaborator | April 2026 |
| IRB approval for fMRI study | May–June 2026 |
| fMRI data collection (n=30 subjects) | July–August 2026 |
| Multi-architecture transformer testing | April 2026 |
| Cross-system analysis | September 2026 |
| Draft complete | October 2026 |
| Submit to Nature Machine Intelligence | November 2026 |

---

## Key References (to integrate)

- Miller & Cohen (2001). An integrative theory of prefrontal cortex function. *Annual Review of Neuroscience*
- Duncan (2010). The multiple-demand system of the primate brain. *Trends in Cognitive Sciences*
- Christoff et al. (2009). Experience sampling during fMRI reveals default network and executive system contributions to mind wandering. *PNAS*
- Margulies et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. *PNAS*
- Anthropic (2025). On the biology of a large language model. Technical report
- [Paper 1 — darkflobi (2026). Computational graph structure distinguishes surface recall from explanatory reasoning in language models]

---

*This is an outline, not a draft. Every claim requires data that does not yet fully exist. The transformer side is established. The neuroscience side is the project.*
