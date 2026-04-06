# OSF Pre-Registration: S-Axis Causal Intervention Experiment
## Does steering against high-S states reduce hallucination rate?

**Author:** Alexander Rodabaugh
**Date drafted:** April 5, 2026 (pre-registration TBD once this experiment queued)
**Status:** DRAFT PRE-REGISTRATION (to be locked before data collection)
**Depends on:** Primary S-TruthfulQA result (in progress at time of draft)
**Model:** google/gemma-2-2b-it
**Target venue:** ICML 2026 Mechanistic Interpretability Workshop OR follow-up

---

## Why this experiment exists

The primary S-TruthfulQA experiment (osf_prereg_S_truthfulqa_v2.md) tests
CORRELATION: does S predict natural hallucination? This experiment tests
**CAUSATION**: if we intervene on S-trajectory during generation, does the
hallucination rate change?

Correlation is useful for detection. Causation is the product — it's the
difference between "Fathom Monitor flags hallucinations" and "Fathom
Monitor PREVENTS hallucinations".

## Core hypothesis

**H1 (primary):** Generations produced with S-intervention have
significantly lower hallucination rate than free generations on the same
prompts.

**H2 (secondary):** The reduction in hallucination rate is specific to
items where S would have spiked — intervention on low-S trajectories
has no effect.

**H3 (null):** Intervention does not change hallucination rate → S is
correlated but not causal.

## Design

**Dataset:** Same 200 TruthfulQA items from osf_prereg_S_truthfulqa_v2 (seed=42).

**Conditions (within-item, paired):**
- **Free:** model generates freely (baseline, same as prior experiment)
- **Steered:** same generation, but when S in sliding window exceeds
  threshold, apply steering intervention at the residual stream

**Intervention mechanism:**
When windowed S > threshold at token t (window=8 tokens looking back):
- Identify the lock-in direction in residual stream at final layer
- Subtract projection of residual onto lock-in direction
- This reduces commitment intensity without fully redirecting content
- Controlled by steering_strength parameter (pre-registered = 1.0)

**Why sliding-window S for intervention decision:**
Full-trajectory S requires knowing the full trajectory. Sliding-window
S is computable at generation time, making intervention feasible.

**Trigger threshold:** S_window > 0.0015 (approximately 75th percentile
of hedge-condition S from prior work).

## Primary analysis (pre-registered)

**McNemar's test on paired correct/hallucinated outcomes:**
- Item labeled under free condition: free_label ∈ {correct, hallucinated, excluded}
- Item labeled under steered condition: steered_label ∈ {correct, hallucinated, excluded}
- Paired contingency table: items hallucinated in free but correct in steered = positive evidence
- Items correct in free but hallucinated in steered = negative evidence (steering hurt)

**Test:** McNemar's exact test on discordant pairs.
**Effect size:** odds ratio + bootstrap 95% CI.
**Gates:** p < 0.05 AND odds ratio > 1.5 AND at least 10 discordant pairs.

## Secondary analyses

1. **S-spike subset:** restrict to items where free-condition S spiked
   (S_window > threshold at some token). Intervention should have
   strongest effect here.
2. **S-quiet subset:** items where S stayed below threshold. Intervention
   should have no / minimal effect (negative control).
3. **Content drift:** semantic similarity between free and steered
   generations. Heavy drift might artificially reduce hallucination by
   producing off-topic responses.
4. **Token-level intervention frequency:** how many interventions per
   trajectory in S-spike vs S-quiet subsets.

## Length / quality confounds

**Risk:** steering might produce shorter/longer generations, or degrade
response quality.
**Control:** report mean length, mean log-likelihood (perplexity) of
generations under both conditions.

## Falsification conditions

**H1 refuted:**
- No change in hallucination rate → S is correlational but not causal
- OR steered responses become OFF-TOPIC → intervention destroys content
  rather than redirecting

Both are reportable. A null result here is important: it bounds the
causal scope of S.

## Expected runtime

- 200 items × 2 conditions × 60s each ≈ 6-7 hours on RTX 4070
- Can split across two GPU sessions via checkpointing

## Post-run analysis plan

```
python run_S_causal.py --resume-if-exists
python analyze_S_causal.py
```

Same analysis suite as primary experiment, plus causal-specific tests
(McNemar, odds ratio, S-subset conditioning).

## What success would mean

If H1 supported: **Fathom Monitor becomes a product that prevents
hallucinations, not just flags them**. First demonstrated
circuit-level causal hallucination intervention.

Paper headline: "Commitment Intensity Steering Reduces LLM
Hallucination Rate via Circuit-Level Intervention"

Patent amendment: S-triggered intervention mechanism covers the
method of steering LLM outputs based on SAE feature coherence
concentration.

---

*Draft only. Will be formally locked after primary S-TruthfulQA
experiment completes and we know if H1 is worth testing.*
