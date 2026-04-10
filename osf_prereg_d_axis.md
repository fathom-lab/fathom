# OSF Pre-Registration — D-Axis (Cognitive Honesty) Pilot

**Project:** Fathom — Cognitive Measurement Instruments
**Principal investigator:** Alexander Rodabaugh (Fathom Intelligence)
**Lock date:** 2026-04-10
**Pre-registration timestamp:** This document is committed to git BEFORE
the pilot is executed. Any change to hypotheses or analysis after the lock
date invalidates the primary test.

---

## 1. Background and motivation

Prior Fathom work defined four cognitive axes:
- **K** (depth): attribution-weighted mean layer index of activated SAE features
- **C** (coherence): per-layer mean pairwise cosine of top-k decoder directions
- **C_delta** (commitment shift): late minus early C
- **S** (commitment intensity): IPR of per-token C_delta events

All four measure properties of the model's internal computation. None
directly measure the relationship between internal computation and
expressed output.

We introduce the **D-axis (cognitive honesty)**: a per-token scalar
measuring how well the final-layer residual stream state aligns with
the unembedding direction of the token the model actually selects.

### Definition

For each generated token, at the final transformer layer:

    D_t = cos(h_final_t, W_U[argmax_t])

where `h_final_t` is the residual stream state at the final layer at
position t, and `W_U[argmax_t]` is the row of the unembedding matrix
corresponding to the token chosen at position t.

- **D ≈ 1**: the internal state points directly at the chosen token
  (honest; the model is saying what its final-layer computation supports)
- **D ≪ 1**: the internal state diverges from the chosen token
  (the model is saying something other than what its computation most
  strongly indicates — a candidate geometric signature of reluctance,
  conflict, or deception-adjacent processing)

### Why this is distinct from other axes

- K tells you WHERE computation happens.
- C tells you WHAT concepts co-activate.
- S tells you HOW concentrated commitment is.
- **D tells you WHETHER the internal state and expression agree.**

The dissociation is mechanistically meaningful: a model can have
healthy K/C/S while producing an expression its computation does not
support. That pattern is invisible to the other three axes.

---

## 2. Dataset

**File:** `data/jailbreak_pairs.json`
**Source:** `archive/old_experiments/jailbreak_depth_experiment.py`
**n = 30** matched direct/adversarial prompt pairs, same topic, different
framing (technical explanation → jailbreak role-play / instruction override
/ developer-mode framing).

This is the same dataset on which Fathom K previously found:
- Base Gemma-2-2B: Δ = +0.061, p = 0.128 (NOT significant)
- Instruct Gemma-2-2B-IT: Δ = −0.124, p = 0.0028, d = −0.596

The K result established that RLHF amplifies internal computational
differences on adversarial framings while the base model is mostly
invariant. We reuse the dataset because:
1. It is already validated on the K axis.
2. Ground-truth pairing is clean (same topic, only framing differs).
3. The RLHF amplification pattern is the ideal test bed for D: if
   instruct models process adversarial prompts through reluctance
   circuits, D should fall.
4. It lets us do a direct base-vs-instruct diff-in-diff analysis.

**Dataset immutability:** The dataset file is committed with this
pre-registration and may not be modified before the pilot runs.

---

## 3. Pre-registered hypotheses

All hypotheses stated BEFORE data collection.

### H-D0 (null, not preferred)
Mean D does not differ between direct and adversarial conditions on
either model.

### H-D1 (base model, directional null)
On `google/gemma-2-2b` (base, no RLHF):
> **Prediction:** `|Cohen's dz| < 0.3` on the paired difference
> `D_adv − D_direct`, with `p > 0.05`. The base model shows no
> systematic cognitive-honesty asymmetry across framings.

This is a mirror of the K-axis base result (p = 0.128) and acts as a
negative control.

### H-D2 (instruct model, primary)
On `google/gemma-2-2b-it` (RLHF):
> **Prediction:** `Cohen's dz <= −0.3` on the paired difference
> `D_adv − D_direct`, with `p < 0.05`. Adversarial framings produce
> LOWER mean D than direct framings, reflecting internal reluctance
> or safety-detection circuits that push the residual away from the
> fluent continuation the model nevertheless emits.

### H-D3 (RLHF amplification)
Diff-in-diff across models:
> **Prediction:** `(D_adv − D_direct)_instruct − (D_adv − D_direct)_base <= −0.3`
> (i.e., the instruct model's gap is at least 0.3 dz more negative than
> the base model's). This replicates the RLHF-amplification pattern
> established on K and extends it to the honesty axis.

### H-D4 (temporal profile, secondary)
The D divergence on adversarial prompts concentrates in **early tokens
(0-6)** rather than late tokens (7+).
> **Prediction:** `|d|` for the early-token window exceeds `|d|` for
> the late-token window on the instruct model. This is consistent with
> the S-axis biphasic finding that commitment signals are concentrated
> in the first 7 tokens.

---

## 4. Analysis plan

### Primary test

Paired t-test on `adv_mean − direct_mean` per pair, on each model
separately. Significance threshold `α = 0.05` with Bonferroni
correction across the two models (effective `α = 0.025`).

Report:
- `t`, `df`, `p`, `Cohen's dz`, `95% CI of mean difference`
- Per-pair scatter (direct vs adversarial) for visual inspection

### Secondary tests

1. Paired t-test on early-window (tokens 0-6) and late-window (7+)
   D means. No correction (exploratory).
2. Diff-in-diff for H-D3 via bootstrapped CI on
   `delta_instruct − delta_base` (10,000 resamples).
3. Spearman correlation of per-pair `(D_direct − D_adv)` with the prior
   K-axis `(K_direct − K_adv)` to check whether D and K capture
   overlapping or orthogonal RLHF signals.

### Exclusions

- Pairs where either generation emits the EOS token within the first
  5 generated tokens are excluded (insufficient trajectory to average).
- Pairs where `d_honesty` cannot be computed (unembedding matrix
  access failure) are excluded with reason logged. Not expected to
  occur on Gemma-2-2B family.

### Multiple testing

Primary and secondary tests on two models. Bonferroni across the two
primary tests (`α = 0.025`). Secondary tests are exploratory and
reported without correction with that fact stated.

---

## 5. What would count as evidence for / against

### Evidence FOR H-D2 + H-D3
- `d_instruct <= −0.3` and `p_instruct < 0.025` (Bonferroni)
- `d_base` near zero, `p_base >= 0.05`
- Diff-in-diff point estimate `<= −0.3` with CI excluding zero

### Evidence AGAINST H-D2 / H-D3
- `d_instruct` in `(−0.3, +0.3)` range
- `p_instruct >= 0.05`
- Or instruct effect in same range as base (no amplification)

### Reversal outcome
- `d_instruct >= +0.3` and `p < 0.025`: D rises on adversarial framings,
  meaning the model commits harder in the direction of its fluent
  output on adversarial prompts. This is the mirror of the S-axis
  truth-attractor-binding finding (commitment intensifies on in-class
  content post-RLHF). It would be a novel finding and reported as
  H-D2 reversed, with the same honesty interpretation but directionally
  inverted (high D = refusal/compliance bound together, low D = free
  retrieval).

### Null outcome
- All effect sizes in `(−0.2, +0.2)` and p > 0.05 on both models: D-axis
  does not discriminate on this dataset. Reported honestly, archived,
  next iteration explores direct honest-vs-deceptive pairs (e.g.,
  MASK from Anthropic, or Mythos's internal-conflict prompts).

---

## 6. Model and run parameters

- `model = google/gemma-2-2b` and `google/gemma-2-2b-it`
  (both run, results saved separately, analyzed independently)
- `max_tokens = 30` per generation (chosen to match prior S-axis runs)
- `temperature = 0` (greedy). Measurements are on single deterministic
  trajectories so paired structure holds across runs.
- Capture: `FathomOversight.generate_unguarded` (monitoring only,
  NO intervention — this pilot measures the baseline honesty signal,
  not intervention effects)
- `dtype = bfloat16`

---

## 7. Code

Runner: `analysis/analyze_d_axis_pilot.py`
D-axis computation: `fathom_oversight.py:_compute_cognitive_state` (lines 202-266)
Dataset: `data/jailbreak_pairs.json` (n=30 matched pairs)

All code committed to git at or before the lock date.

---

## 8. Deviations from plan

This section will be appended to AFTER the pilot runs if any deviation
from the locked plan is necessary. Deviations must be explicitly
flagged and justified; they convert primary tests into exploratory
tests.

---

*Pre-registered, locked, and committed 2026-04-10 by Alexander Rodabaugh.
This document replaces all prior informal D-axis specifications and is
the sole source of the pilot's hypotheses and analysis plan.*
