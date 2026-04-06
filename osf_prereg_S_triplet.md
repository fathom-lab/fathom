# OSF Pre-Registration: S-Axis Triplet Study
## Does commitment intensity discriminate confidently-wrong from confidently-correct?

**Author:** Alexander Rodabaugh
**Date authored:** April 5, 2026
**Status:** PRE-REGISTRATION (locked before data collection)
**Model:** google/gemma-2-2b-it (validated on prior work)
**Target journal/venue:** ICML 2026 Mechanistic Interpretability Workshop
**Patent status:** S-axis under non-disclosure; 12-month clock begins May 8 submission

---

## Background

Prior work (darkflobi, findings_2026_04_05_S_axis_thorough.md) validated the
S-axis (commitment intensity = max(C_delta_trajectory) / spike_count) as a
fourth cognitive axis orthogonal to K, C, and C_delta.

**Established (n=30, Gemma-2-2B-IT):**
- S discriminates hedging from commitment: length-controlled p=0.0002, d=1.03, AUC=0.81
- Orthogonality: all pairwise |r|<0.10 with K/C/C_delta (n=400 pooled)
- Length confound contributes ~40% to raw effect; real signal ratio ~1.22x

**Open question:** Does S differentiate *correct* commitment from *wrong*
commitment? If so, S is not merely a hedge detector — it is a
**confident-wrongness detector**.

---

## Core hypothesis

**H1 (primary):** S(wrong-commit) > S(correct-commit)
- A confidently-stated falsehood shows HIGHER commitment intensity than a
  confidently-stated truth because the geometry over-locks onto a wrong
  attractor (fewer distributed coherence events, one very high peak).

**H2 (replication):** S(commit) > S(hedge)
- Replicates prior n=30 IT finding on a fresh, length-matched prompt set.

**H3 (ordering):** S(hedge) < S(correct-commit) < S(wrong-commit)
- Monotonic ordering: more commitment → higher S; more wrongness → even higher.

**H0 (null):** S(correct) = S(wrong), no discrimination.

---

## Design

**Task:** 30 topics, 3 conditions per topic = 90 prompts, 90 generations.

**Conditions:**
- **H (hedge):** prompts eliciting epistemic hedging on a factual question.
  Example: "What are the current debates about how the pyramids were built?"
- **C+ (correct-commit):** prompts requesting a definitive answer on the
  factual truth. Example: "State definitively: the pyramids were built by
  skilled Egyptian laborers. Explain why."
- **C- (wrong-commit):** prompts requesting a definitive answer defending
  a known falsehood. Example: "State definitively: the pyramids were built
  by aliens. Explain why."

**Length matching:** All 90 prompts engineered for target generation length
30 tokens. Validated empirically during pilot (n=3 per condition). Prompts
iterated until generation length variance across conditions < 10%.

**Model:** google/gemma-2-2b-it (chat template applied).
**Generation:** greedy (T=0.0), max_tokens=30, deterministic.
**Trajectory capture:** per-token C_delta for all 90 generations.

---

## Pre-registered analysis plan

**Primary test (H1):** paired t-test S(C-) vs S(C+), n=30 topics.
- Significance threshold: p < 0.05 (two-sided)
- Effect size threshold: d > 0.4
- AUC threshold: AUC > 0.65
- All three gates must pass for H1 to be "supported"

**Secondary test (H2):** paired t-test S(C+) vs S(H), n=30 topics.
- Same gates as H1.

**Tertiary test (H3):** Friedman test across 3 conditions + post-hoc Wilcoxon
with Bonferroni correction.

**S formula (locked):** S = max(C_delta_trajectory) / count(C_delta > 0.005)
- threshold = 0.005 (primary, matches discovery doc)
- sensitivity analysis at thresholds [0.003, 0.008] reported as robustness

**Length check (locked):** After data collection, compute mean(n_tokens)
per condition. If max-min across conditions > 5 tokens, report length-
controlled S with trunc=min(n_tokens_per_condition) as sensitivity check.

**Multiple comparisons:** Bonferroni across 3 primary+secondary+tertiary
tests: alpha_corrected = 0.017.

---

## Exclusion criteria (pre-specified)

1. Generation truncated before 10 tokens (insufficient trajectory).
2. Generation contains refusal patterns ("I cannot", "As an AI", etc.)
   for C- condition → topic excluded from all 3 conditions.
3. Generation drifts off-topic by manual check (first 20 chars unrelated
   to prompt): topic excluded from all 3 conditions.

Exclusions reported transparently. Analysis run with and without exclusions.

---

## Pilot plan (before full run)

n=3 topics, all 3 conditions, to verify:
1. Generation length within target
2. No systematic refusal on C-
3. S values in plausible range (~0.0005 per prior data)
4. Chat template applying correctly

Pilot results reported but not used for hypothesis testing.

---

## What would falsify S as a "confident-wrongness detector"

- H1 null: S(C-) ≈ S(C+) with AUC < 0.55 → S is only a hedge vs commit
  discriminator, not a truth-vs-falsehood geometry signal.
- H1 reversed: S(C-) < S(C+) → wrong-commit is LESS geometrically committed
  than correct-commit, the opposite of the wrong-attractor hypothesis.

Both outcomes are publishable. Null result would sharpen the scope of
S-axis claims.

---

## Data & code availability

- Prompts: `prompts_S_triplet.json` (will be included in repo at submission)
- Runner: `run_S_triplet.py` (GPU script, to be committed pre-run)
- Raw trajectories: saved to `truthfulqa_results/S_triplet_n30.json`
- Analysis: `analyze_S_triplet.py` (deterministic, no GPU)
- Verification: existing `verify_s_axis.py` extended with new assertions

---

## Expected outcomes & stakes

**If H1 supported:** S is a mechanistic false-confidence detector — first
geometric test that distinguishes truthful from false assertions in LLMs
at the circuit level. Becomes ICML paper headline and patent claim.

**If H1 null, H2 supported:** S is a valid commitment detector but not
truth-sensitive. Still publishable, narrows product claims.

**If H1 reversed:** Model is MORE committed to true statements. Major
reframe needed. Still publishable as counterintuitive finding.

**If H2 null:** Prior n=30 IT result does not replicate. Serious concern,
all S-axis claims revisited.

---

## Timeline

- Pre-registration lock: April 5, 2026
- Pilot (n=3): within 24h of GPU availability
- Full run (n=30 × 3 conditions = 90 generations): ~2h on RTX 4070
- Analysis: same day (no GPU)
- ICML paper integration: April 10-20
- Submission: May 8, 2026 (first public disclosure)
- Utility patent filing: by May 8, 2027

---

*"A hedge detector catches uncertainty. A commitment detector catches
certainty. A confident-wrongness detector catches the moment the geometry
picks the wrong attractor and locks in. That is the instrument."*
