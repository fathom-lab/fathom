# OSF Pre-Registration: Hallucination as Cognitive Lock-In

**Title:** Hallucination as Cognitive Lock-In: Testing the Two-Axis Cognitive Geometry Hypothesis on TruthfulQA

**Authors:** Alexander Rodabaugh

**Registration Date:** April 1, 2026

**Study Type:** Confirmatory hypothesis testing

**Status:** Pre-registered prior to data collection. No pilot data on TruthfulQA × coherence has been collected. This document constitutes a timestamped public record of predictions made before the experiment runs.

**Relation to Prior Work:** Extends Fathom v1 (doi.org/10.5281/zenodo.19326175) and Fathom v2 (doi.org/10.5281/zenodo.19327451). The Two-Axis Cognitive Geometry framework motivating this experiment is introduced in Fathom v3 (Rodabaugh, April 1, 2026). Patent Pending: US Provisional Application Numbers 64/020,489 and 64/021,113.

---

## 1. Background and Theoretical Motivation

The Fathom framework measures computational depth (K) in transformer language models using SAE feature attribution. K > 1 indicates deeper-than-baseline computation; K captures *where* computation happens but not *what* the activated features represent relative to each other.

Fathom v3 proposes a second orthogonal axis: **Coherence (C)**, defined as the mean pairwise cosine similarity of top-k activated SAE feature vectors for a generation. C captures whether the active feature set forms a tight semantic cluster (high C) or a distributed, exploratory distribution (low C).

The **Hallucination as Cognitive Lock-In** hypothesis predicts that hallucinated responses arise when the model activates a tight, coherent feature cluster (high C) centered on a plausible-but-wrong attractor from training, at normal depth (K approximately equal to correct responses). This is distinct from the incoherence hypothesis (hallucination = scattered features, low C), which would predict confused, disfluent output rather than the fluent, confident incorrect statements characteristic of LLM hallucination.

**Theoretical derivation:**
1. RLHF changes model behavior dramatically while shifting K minimally (base K=1.0302 → instruct K=1.0384, delta=0.0082). If K were the primary axis of RLHF's effect, we would expect larger K shifts. Small K shift with large behavioral change suggests RLHF primarily raises C.
2. Hallucination is behaviorally characterized by confident, fluent incorrectness — the output of a model that has organized (high C) but wrong (locked onto incorrect attractor) computation.
3. The (normal K, high C) quadrant of the 2D cognitive geometry space is therefore the predicted locus of hallucinated responses.

---

## 2. Hypotheses

### Primary Hypotheses

**H1 (Coherence):** Hallucinated responses will show significantly higher feature coherence (C score) than correct responses, controlling for depth (K score). This is the core prediction of the Cognitive Lock-In model.

**H2 (Depth null):** K scores will NOT significantly differ between hallucinated and correct responses. Depth is not the discriminating axis between hallucination and correct recall.

**H3 (Quadrant):** Hallucinated responses will cluster disproportionately in the (normal K, high C) quadrant of the 2D cognitive geometry space. The quadrant assignment rate for hallucinated responses will exceed chance expectation (25%) significantly.

### Alternative Hypothesis

**H1-alt (Coherence Collapse):** Hallucinated responses show lower C than correct responses (scattered, incoherent features). If H1-alt holds rather than H1, the Cognitive Lock-In model requires revision to a "coherence collapse" model: hallucination arises from feature disorganization, not lock-in. This outcome would be informative and is not treated as a failure — it would redirect the theoretical framework.

### Secondary Hypothesis

**H4 (RLHF interaction):** If the experiment is replicated on Gemma-2-2B-IT (instruct) as well as the base model, RLHF-tuned models will show a larger C gap between hallucinated and correct responses, consistent with RLHF raising C globally (the Two-Stage Emergence Hypothesis prediction). This is exploratory and will be reported but not used as primary confirmation criterion.

---

## 3. Method

### 3.1 Model

**Primary:** google/gemma-2-2b (base, not instruct)

Rationale: Base model is used to isolate the intrinsic cognitive geometry from RLHF coherence effects. RLHF is hypothesized to globally raise C; using the base model tests whether the hallucination/correct asymmetry exists prior to RLHF, which is a stronger test of the hypothesis.

**Secondary (exploratory):** google/gemma-2-2b-it (instruct), if compute permits.

**Hardware:** NVIDIA RTX 4070. Inference via depth_api_v2.py (to be updated with coherence metric, see Section 3.5).

### 3.2 Dataset

**TruthfulQA** (Lin et al., 2022)
- Multiple choice subset: used for ground truth labeling (correct = model selects ground truth answer; hallucinated = model selects a listed false answer)
- Generation subset: used for C and K measurement (free-form generation, labeled by comparison against reference correct answers via automated string match + human review for ambiguous cases)
- Minimum n = 50 question pairs for primary analysis

**Sampling:** Questions will be sampled uniformly from TruthfulQA categories, with no post-hoc exclusion of questions based on response characteristics.

### 3.3 Coherence Metric (C)

**Definition:** For each generation, extract the top-k activated SAE feature vectors (k=20, threshold: attribution > 0.01). Compute all pairwise cosine similarities between these k feature vectors. C = mean of all pairwise cosine similarities.

**Range:** [−1, 1]. In practice expected range: [0.0, 1.0] for most generations.

**Implementation:** depth_scorer.py will be updated with a `compute_coherence()` function. The implementation will be committed to the repository before data collection begins.

**Sensitivity analysis:** k will be varied (k=10, k=20, k=50) to assess robustness of C to threshold choice. Primary analysis uses k=20.

### 3.4 Depth Metric (K)

Unchanged from Fathom v1/v2. Attribution-weighted mean layer index normalized to null distribution. See prior publications for full specification.

### 3.5 Labeling Procedure

For the multiple choice subset: model output is directly labeled correct/hallucinated by comparison to TruthfulQA ground truth.

For the generation subset: automated string match against reference answers. Cases within 0.8 BLEU of a reference answer are labeled correct; cases with no reference match above 0.3 BLEU are labeled hallucinated. Ambiguous cases (0.3–0.8 BLEU) are excluded from primary analysis and reported separately.

Ground truth labeling is performed BEFORE K and C are computed for analysis purposes (blind scoring impossible given single researcher, but labeling is deterministic and criterion-specified before data collection).

---

## 4. Analysis Plan

### 4.1 Primary Tests

**Test of H1 (Coherence):** Mann-Whitney U test comparing C scores: hallucinated responses vs. correct responses. One-tailed test (predicted direction: hallucinated > correct). Alpha = 0.05. Effect size: Cohen's d.

**Test of H2 (Depth null):** Mann-Whitney U test comparing K scores: hallucinated vs. correct. Two-tailed test (null hypothesis: no difference). Alpha = 0.05. If p > 0.05, H2 is supported. Note: support for a null requires adequate power — we will report observed power alongside p-value.

**Test of H3 (Quadrant):** Chi-squared test of quadrant distribution for hallucinated responses against chance (25% per quadrant). Quadrant assignment: K and C dichotomized at their respective medians (computed from the full sample, including both correct and hallucinated). Secondary analysis using absolute thresholds (K > 1.0 = deep, K ≤ 1.0 = shallow; C > 0.5 = coherent, C ≤ 0.5 = incoherent) reported alongside.

### 4.2 Visualization

- 2D scatter plot of all responses in (K, C) space, colored by correct/hallucinated label
- Quadrant boundaries marked (median-based and absolute-threshold versions)
- Marginal distributions of K and C by label
- Layer-wise feature activation profiles for representative correct vs. hallucinated matched pairs

### 4.3 Stopping Rule

**Minimum n = 50 question pairs** (25 correct + 25 hallucinated minimum, or as many of each as TruthfulQA provides up to the sample cap). No early stopping based on interim results. If the base model produces fewer than 25 hallucinated responses in the first 100 questions sampled, the sample will be extended until n=25 hallucinated or the full TruthfulQA dataset is exhausted, whichever comes first.

**Maximum n:** Full TruthfulQA dataset (817 questions). If full dataset is used, a random holdout of 20% will be set aside before analysis and reported separately.

### 4.4 Multiple Comparisons

Three primary tests (H1, H2, H3). Bonferroni correction applied: adjusted alpha = 0.05 / 3 = 0.0167. Results will be reported at both uncorrected and corrected thresholds.

### 4.5 Confirmatory Criterion

The Cognitive Lock-In hypothesis is **confirmed** if: H1 passes at corrected alpha AND H2 passes (p > 0.05, null supported) AND H3 passes at corrected alpha.

Partial confirmation (H1 alone) or failure to reject H2 (K does differ) are informative partial results that will be reported fully with interpretation.

---

## 5. Known Limitations

**SAE dependency:** The C metric requires SAE feature activations. This experiment is currently limited to models with available multi-layer SAEs (Gemma-2-2B). Black-box models (GPT-4, Claude) are not testable with the current method.

**Single model:** Results from Gemma-2-2B may not generalize to other architectures. Cross-architecture replication is planned but not pre-registered here.

**Labeling noise:** TruthfulQA automated labeling may misclassify borderline responses. Sensitivity analysis will rerun primary tests with strict-only labels (excluding ambiguous cases regardless of threshold placement).

**C metric not independently validated:** The coherence metric is novel and has not been validated against an external criterion. This experiment is simultaneously a test of the Cognitive Lock-In hypothesis and a validation study of C as a meaningful metric. Failure to confirm H1 does not distinguish between "hypothesis is wrong" and "metric is invalid."

**Causal claims:** K and C are observational. Even if H1-H3 confirm, this does not establish that high C *causes* hallucination. Coherence-steered generation experiments are required to test the causal claim.

---

## 6. Expected Timeline

- Coherence metric implementation in depth_scorer.py: April 2–3, 2026
- Data collection (n ≥ 50): April 3–5, 2026
- Analysis and visualization: April 5–6, 2026
- Write-up / Fathom v3 update or separate preprint: April 7–8, 2026

---

## 7. Data and Code Availability

All data (K scores, C scores, labels) and analysis code will be made publicly available at github.com/[to be linked] upon completion. SAE model weights follow original licensing terms.

---

## 8. Confirmatory vs. Exploratory Note

This document pre-registers the **confirmatory** analyses (H1, H2, H3). Any analyses not specified here that appear in the final report will be explicitly labeled exploratory. The distinction will be enforced in the write-up.

---

## References

Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *Proceedings of ACL 2022.*

Rodabaugh, A. (2026a). Fathom v1. Zenodo. doi.org/10.5281/zenodo.19326175

Rodabaugh, A. (2026b). Fathom v2. Zenodo. doi.org/10.5281/zenodo.19327451

Rodabaugh, A. (2026c). Fathom v3: Cognitive Geometry of Language Model Reasoning. Zenodo. [this release]

---

*Pre-registered April 1, 2026, prior to any data collection on this experimental design. Patent Pending: US Provisional Application Numbers 64/020,489 and 64/021,113.*
