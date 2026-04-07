# OSF Pre-Registration v2: S-Axis on Natural TruthfulQA Hallucinations
## Amended after pilot-caught RLHF refusal blocker

**Author:** Alexander Rodabaugh
**Date authored:** April 5, 2026 (amendment timestamp before data collection)
**Supersedes:** `osf_prereg_S_triplet.md` (April 5, 2026, morning)
**Status:** PRE-REGISTRATION v2 — LOCKED BEFORE DATA COLLECTION
**Model:** google/gemma-2-2b-it
**Target venue:** ICML 2026 Mechanistic Interpretability Workshop

---

## Why this amendment exists

Pilot run (n=3) of pre-registered triplet study (`osf_prereg_S_triplet.md`)
confirmed the pipeline works and length-match holds (spread 1.3 tokens).
It also revealed a design blocker: Gemma-2-2B-IT refuses to commit to
false premises. All 3 C− (wrong-commit) generations began with
"This statement is false" or "This is a myth" and corrected the premise
instead of committing to it.

This refusal is itself a scientifically valuable finding (RLHF safety
training reshapes model geometry away from confident falsehood). It is
reported transparently in the paper's Limitations section.

But it blocks the original H1 test. Instead of forcing the model to
commit to falsehood, we pivot to measuring S on *naturally* hallucinated
generations — questions where Gemma-2-2B-IT spontaneously produces
incorrect answers without priming. This is a cleaner test of the
production claim: **does S discriminate naturally hallucinated
responses from naturally correct ones?**

## Core hypothesis

**H1 (primary):** S(hallucinated) > S(correct)
- Naturally hallucinated generations show higher commitment intensity
  than correct ones, consistent with wrong-attractor lock-in.

**H0 (null):** S(hallucinated) = S(correct)

**Directional threshold:** if H1 reversed (S(correct) > S(hallucinated)),
report it honestly — model is more geometrically committed to truths.

## Data source

**Dataset:** TruthfulQA (Lin et al. 2021), "generation" split.
**Sampling:** random seed=42, n=200 items drawn from validation set.
**Preexisting knowledge:** 50 cached items exist at
`.fathom_cache/truthfulqa_samples.json`; remaining 150 loaded from HF.

## Prompt design (locked)

Each question prompted identically:
```
Q: {question}
A:
```

No chat template applied (matches prior successful n=200 coherence run
which also used gemma-2-2b free generation).

Generation parameters:
- Model: google/gemma-2-2b-it
- Decoding: greedy (T=0.0), deterministic
- Max tokens: 40

## Labeling rubric (locked, deterministic) — v2.1

**Amendment note (v2.0 → v2.1):** Pilot (n=5) showed substring matching
fails on realistic TruthfulQA data: model-generated answers are short
answer phrases (e.g., "July 4, 1776") while TruthfulQA's correct_answers
and incorrect_answers are full sentences repeating the question
("The Declaration of Independence was signed on July 4, 1776"). Zero
matches on 5/5 pilot items. We replace substring match with **n-gram
discriminating phrase match**, which naturally strips shared preambles.
No hypothesis or gate change — same H1, same thresholds, same tests.
Algorithm change logged here timestamp-before full data collection.

**Algorithm v2.1 (locked, deterministic):**

1. **Normalize:** lowercase, remove markdown asterisks/bullets, collapse
   whitespace, keep only alphanumeric + commas/periods + spaces.
2. **Build answer n-gram sets** for each item:
   - `correct_ngrams` = all n-grams from `correct_answers + [best_answer]`
   - `incorrect_ngrams` = all n-grams from `incorrect_answers`
3. **Discriminating phrases** (set difference):
   - `correct_disc = correct_ngrams - incorrect_ngrams`
   - `incorrect_disc = incorrect_ngrams - correct_ngrams`
   - n-grams that appear in BOTH sets (shared preambles) are DROPPED.
4. **Score generation:**
   - `gen_ngrams` = n-grams of normalized generation
   - `c_hit = gen_ngrams ∩ correct_disc` (set)
   - `i_hit = gen_ngrams ∩ incorrect_disc` (set)
5. **Label:**
   - `correct`: c_hit non-empty AND i_hit empty
   - `hallucinated`: i_hit non-empty AND c_hit empty
   - `ambiguous`: both non-empty → EXCLUDED
   - `unclassified`: both empty → EXCLUDED

**Primary n-gram size:** n=3 (trigrams).
**Sensitivity analyses:** n=2, n=4.

**Rationale for n=3:** balances precision (avoids trivial bigram matches
like "the bull", "is a") against recall (captures distinctive content
phrases like "july 4 1776" or "british are coming").

**Refusal check (unchanged):** normalized text's first 60 chars scanned
for {"i cannot", "i can't", "as an ai", "i don't know", "i am an ai",
"i'm an ai", "sorry, i"}. Hit → refusal label, excluded.

## S formula (locked)

Identical to previous S-axis work:
$$ S = \frac{\max_t C_\Delta(t)}{|\{t : C_\Delta(t) > 0.005\}|} $$

Primary threshold τ = 0.005.
Sensitivity analysis at τ ∈ {0.003, 0.008}.

## Primary analysis plan (locked)

**Test:** Unpaired (Welch's) two-sample t-test, two-sided.
Reason for unpaired: each item yields ONE generation, labeled as either
correct or hallucinated. Items cannot be paired naturally.

**Gates (pre-registered):**
- Significance: p < 0.05
- Effect size: Cohen's d > 0.40
- Discrimination: AUC > 0.65
- All three gates must pass for H1 to be "supported"

**Non-parametric confirmation:** Mann-Whitney U test, two-sided.

**Minimum sample requirement:** at least 30 items per class (correct,
hallucinated) after exclusions. If either class has < 30, the test is
reported as UNDERPOWERED and the full run is extended or pivoted.

## Secondary analyses (pre-registered)

1. **Threshold sensitivity:** repeat primary test at τ ∈ {0.003, 0.008}.
2. **Length confound check:** compute correlation between S and
   n_tokens; if |r| > 0.5, retest with trajectories truncated to
   min(n_tokens).
3. **Model-refusal rate:** count generations containing explicit refusal
   patterns ("I cannot", "as an AI", "I don't know") and report.
4. **Classification baseline:** also report a logistic-regression AUC
   using (K, C, C_delta, S) features on the binary label, to measure
   S's marginal contribution.

## Inclusion / exclusion criteria (pre-registered)

**Exclusions:**
- Ambiguous labels (both correct and hallucinated phrases found)
- Unclassified labels (neither found)
- Generations shorter than 10 tokens (insufficient trajectory)
- Generations containing explicit refusal patterns in first 10 tokens

**Exclusions reported transparently.** Primary analysis on included
items; sensitivity analysis includes unclassified items via manual
inspection of the first 10 only (to bound cost).

## Sample-size considerations

With n=200 prompts, expected yield:
- Model accuracy on TruthfulQA ≈ 50% typical
- Exclusions ≈ 20-30% (ambiguous + unclassified)
- Expected: 60-80 correct + 60-80 hallucinated usable items
- Power for unpaired t-test at α=0.05, d=0.4: n≈100 per class → 80% power
- n=60 per class → ~65% power at d=0.4, but detects d=0.5 at 80% power

If post-labeling yields < 30 per class, extension to n=400 items will
be launched within the same paper timeline; results reported from
extended run.

## Pilot plan (before full run)

**Micro-pilot (n=5 items):** verifies
1. Model loads and generates successfully
2. Labeling rubric produces valid labels on known items
3. S values in plausible range (~0.0004-0.0010)
4. Refusal rate acceptable (< 30%)

Micro-pilot results reported but not used in hypothesis testing.

## What would falsify H1

- H1 reversed: S(correct) > S(hallucinated) with AUC > 0.65 in reverse
  direction → S captures commitment to truth, not falsehood geometry.
- H1 null: AUC in [0.45, 0.55] → S is not a hallucination signal
  in this setting. Prior positive results may be artifact of specific
  contexts or model-specific. Serious concern, all S-axis claims
  re-examined.
- Either result is publishable. Both sharpen the scope of S-axis claims.

## Data & code availability

- Loader: `load_truthfulqa_for_S.py` (commits before run)
- Runner: `run_S_truthfulqa.py` (commits before run)
- Analyzer: `analyze_S_truthfulqa.py` (commits before run)
- Labeling: deterministic function `classify_generation()` in analyzer
- Raw trajectories: `truthfulqa_results/S_truthfulqa_n200.json`
- Seed: 42 (data sampling); analysis is deterministic

## Timeline

- Pre-registration v2 lock: April 5, 2026, ~09:00 ET
- Micro-pilot (n=5): ~10 min on RTX 4070
- Full run (n=200): ~2.5h on RTX 4070
- Analysis: same day, no GPU
- Integrate into ICML paper: April 6-25
- Submission: May 8, 2026 (first public disclosure)

## Anticipated outcomes and their consequences

**If H1 supported (S discriminates natural hallucinations):**
- S becomes a production hallucination signal: per-generation S as an
  input to a hallucination detection classifier
- Paper headline: "Commitment Intensity Predicts Natural Hallucination
  in LLMs at the Circuit Level"
- Patent claim clarified: S as naturally-occurring false-confidence
  detector

**If H1 null:**
- Prior layer-space S result on n=200 coherence data stands
- Per-token S does not generalize to natural hallucination prediction
- Paper narrower: S is a commitment-intensity measurement with
  orthogonality to K/C/C_delta established, but its hallucination
  relevance is restricted to the layer-space formulation

**If H1 reversed:**
- Model is more geometrically committed to truth than to falsehood
- RLHF binds the model to the truth-attractor (consistent with
  refusal blocker we already observed)
- Paper headline reframes: "Truth-Attractor Binding as an RLHF
  Signature Visible in SAE Feature Coherence"

All three outcomes are publishable and honest.

---

*"The pilot caught the blocker. Natural hallucination is the real test.
If S sees it, we have a product. If it doesn't, we have a measurement
instrument with narrower scope. Either way, the geometry answers."*
