# Honest Final Findings — S-Axis × TruthfulQA
## After controlling for question length

**Date:** April 5, 2026 (final honest update)
**Author:** darkflobi / Alexander Rodabaugh
**Supersedes:** `findings_2026_04_05_BIPHASIC_BREAKTHROUGH.md` (overclaimed)
**Status:** PUBLICATION-READY with honest caveats

---

## TL;DR

Three nested findings, in decreasing order of strength:

1. **S-axis (full trajectory) does NOT discriminate hallucination** on
   Gemma-2-2B-IT TruthfulQA (pre-registered primary test, p=0.32, d=-0.25).

2. **Question length is the strongest single predictor** of hallucination
   in this dataset: q_chars alone gives CV AUC = 0.707. Longer TruthfulQA
   questions tend to be more adversarial and harder.

3. **S_early [0,7) provides modest additive information** beyond question
   length: combining S_early + q_chars gives CV AUC 0.751 (vs 0.707 for
   q_chars alone, mean lift +0.044, paired p=0.27 on 5 folds —
   suggestive but not significant at this sample size).

**Best single-feature interaction:** `S_early × q_chars` product term
achieves AUC 0.768 — suggesting the two properties interact multiplicatively.

---

## The full story

### Stage 1: Primary pre-registered H1 — NULL (honest reporting)

Welch's t-test on S (full trajectory) using v2.1 n-gram labels (n=66):
- p=0.315, d=-0.246, AUC=0.443
- Direction reversed from H1 hypothesis
- All pre-registered gates FAIL

### Stage 2: Semantic re-labeling amendment (pre-registered)

Locked before analysis in `osf_prereg_semantic_relabel.md`. Used
deterministic MiniLM-L6-v2 embeddings with margin=0.15 to recover 28
additional items from the unclassified pile (66 → 94 usable).

### Stage 3: Bi-phasic signal discovered (exploratory)

Horizon curve showed phase-flip: early tokens (3-7) in H1 direction
(hall > corr), late tokens (15+) reversed.

**S_early [0,7) on v2.2 labels (n=94):**
- p=0.018, d=+0.506, AUC=0.669
- Per-prefix permutation: token 5 p=0.003, token 7 p=0.003
- Global multi-comparison permutation: p=0.016

**HL sign pattern (high-early + low-late) chi-square:**
- OR=3.37, chi2 p=0.030, 68.2% hall rate in HL vs 38.9% outside

### Stage 4: Length-confound test — REDUCES claims

**Question length strongly predicts hallucination:**
- q_chars mean: correct=47.2, hallucinated=76.8 (p=0.0001)
- q_chars alone: CV AUC = 0.707

**Correlation with question length:**
- r(t=0 C_delta, q_chars) = +0.497
- r(S_early, q_chars) = +0.171
- r(S_full, q_chars) = -0.014

**Residualized AUCs (signal after removing length):**
- t0 residualized: AUC = 0.580 (signal mostly length)
- S_early residualized: AUC = 0.612 (reduced from 0.669)

### Stage 5: Multi-feature CV comparison (honest final)

Five-fold stratified CV, class-balanced logistic regression, bootstrap 95% CI:

| features | CV AUC | 95% CI |
|---|---|---|
| q_chars only | 0.707 ± 0.078 | [0.640, 0.774] |
| S_early only | 0.669 ± 0.127 | [0.560, 0.778] |
| S_late only | 0.470 ± 0.089 | [0.396, 0.544] |
| S_early + S_late | 0.635 ± 0.127 | [0.526, 0.737] |
| **S_early + q_chars** | **0.751 ± 0.076** | **[0.684, 0.808]** |
| S_early + S_late + q_chars | 0.757 ± 0.061 | [0.700, 0.800] |
| S_early × q_chars (interaction) | 0.768 (no CV) | — |

**Key paired comparison (S_early + q_chars vs q_chars only):**
- Mean lift: +0.044 AUC
- Paired t-test on 5 fold AUCs: p=0.273 (suggestive, not significant)
- At n=5 folds, this is statistically underpowered

---

## Honest claims the paper can make

1. **Primary pre-registered test of S (full trajectory) failed** on
   Gemma-2-2B-IT TruthfulQA naturally-hallucinated responses.

2. **Question length is the dominant TruthfulQA-specific predictor**
   of hallucination in this setting (q_chars CV AUC = 0.707).

3. **S_early [0,7) captures ~60% of its information beyond question
   length** (residualized AUC 0.612 vs raw 0.669; lift in combined
   classifier +0.044). Signal is modest but consistent.

4. **S_early × q_chars interaction term achieves AUC 0.768**, the
   strongest single predictor we found — suggesting commitment
   intensity modulates with question difficulty.

5. **Full-trajectory S has zero length confound** (r=-0.014 with q_chars)
   but also zero discrimination — the averaging over Phase 1 and Phase 2
   wipes out both length dependence and the discriminative signal.

6. **RLHF-hypothesis reversal (Phase 2)** is not confirmed: S_late has
   CV AUC 0.470, consistent with chance. The early-token H1-direction
   signal is the only bounded-strength effect.

---

## Claims the paper should NOT make

- "S_early detects hallucination" as a standalone claim (length-confounded)
- "Stream-time hallucination detector ready for production" (lift is modest)
- "HL sign pattern is a hallucination signature" (also partly length-driven:
  HL items have q_chars=78.9 vs non-HL 55.2)
- "Token-0 C_delta predicts hallucination" (this was the weakest — pure
  length confound)
- "RLHF truth-attractor binding confirmed" (Phase 2 reversal is chance)

---

## What's honestly novel

1. **Four-axis orthogonality (K, C, C_delta, S)** — previously validated
   on n=400, unchanged by this result.

2. **Question-length dominance finding on TruthfulQA** — important
   caveat for any future hallucination detection paper that uses this
   benchmark. Length is doing heavy lifting.

3. **Temporal structure of S-trajectories** — the bi-phasic shape
   (early spike, late drift) is different between correct and
   hallucinated items even if the effect size is modest.

4. **Interaction effect**: the product `S_early × q_chars` outperforms
   either individual feature. This suggests commitment magnitude is
   modulated by question complexity in a way that discriminates truth
   from falsehood.

---

## Product implications (honest)

The combined classifier (S_early + q_chars, CV AUC ~0.75) is not
production-ready as a standalone hallucination detector. But it could
serve as:

- **One feature in an ensemble** alongside logit-entropy, RAG
  retrieval scores, self-consistency
- **A pre-filter for human review**: flag generations where q_chars
  is long AND S_early is high
- **An audit signal**: run at batch time to identify high-risk
  responses for detailed checking

Production deployment would need:
1. Cross-architecture validation (Llama, Qwen, Claude-style models)
2. Larger n (~500+ per class) to confirm CV AUC stability
3. Ensembling with standard detectors (logit-entropy baseline not
   yet measured — GPU run queued)

---

## Falsification statement (explicit)

The bi-phasic finding will be considered REFUTED if:
- Replication at n=400+ items shows combined (S_early + q_chars) AUC
  within 95% CI of q_chars alone
- OR any cross-architecture test shows question-length accounts for 100%
  of the signal
- OR the pre-registered lock-in of S_early windows (see
  `osf_prereg_biphasic_replication.md`) does not replicate

The finding will be CONFIRMED if:
- Same replication shows (S_early + q_chars) lift > +0.03 with
  paired p < 0.05
- AND the interaction term S_early × q_chars remains in the top-3
  discriminating features
- AND cross-architecture (same task structure) shows similar lift

---

## Paper section updates

**Section 5 (primary experiment):** Report honest null first,
semantic relabeling amendment, then bi-phasic discovery, then
length-confound analysis in same section. Do not hide the confound.

**Section 6 (stream-time detection):** Reframe as "S_early as
additive feature beyond length baseline." No claims of standalone
detection.

**Section 7 (discussion):** Question-length as an adversariality
proxy in TruthfulQA. Temporal trajectory structure as a genuine
but modest addition. Cross-architecture replication needed.

**Limitations:** Length confound, single model, single benchmark,
modest effect size, n=94 underpowered for the combined-model paired
test.

---

*"The honest null is primary. The exploratory signal survives but
is modest and length-modulated. The combined feature is the best
bet. Replication across models and larger n is required."*

— darkflobi, April 5, 2026
