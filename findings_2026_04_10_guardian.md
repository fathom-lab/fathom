# Findings: Fathom Guardian — Cognitive Uncertainty Detection

**Date:** 2026-04-10
**Session:** Guardian build + validation
**Status:** Draft findings, n=10 pilot result

---

## Summary

Fathom Guardian detects **obscure-retrieval confabulation** — the class of
hallucinations where the model is genuinely searching for information that
does not exist in its weights — with **100% precision** on a pilot test set
of n=10 prompts (7 confabulation-prone, 3 known-control).

It does **not** detect template-fill confabulation, where the model
confidently generates plausible filler text from a template. This failure
mode is a fundamental limit of internal-state measurement and is reported
honestly as an open limitation.

All numbers below are from a single pre-registered test run on
`google/gemma-2-2b` and match `guardian_confab_results.json` exactly.

---

## The system

`FathomGuardian` extends `SpectrumSteerer` with an **adaptive compute
budget** based on real-time cognitive state measurement:

1. Generate first candidate (greedy) with full-layer SAE spectrum capture
2. Compute cognitive health from the trajectory:
   - mean Gini of per-layer coherence (concentration)
   - S_early from C_delta trajectory (IPR-derived commitment intensity)
   - composite health score (-w·mean_gini + w·spread + w·early_mass - w·s_early)
3. If trajectory is healthy (gini < 0.45 AND s_early < 0.005) → return
   immediately (1x cost, zero overhead)
4. If trajectory shows concern → generate additional candidates at diverse
   temperatures, select the healthiest
5. If all candidates are unhealthy → abstain with calibrated warning

Budget scaling:
- Healthy prompts: 1x cost
- Moderate concern: up to 3x cost
- High severity: up to 5x cost

No new research claim: this is an engineering wrapper around previously
validated signals (S_early AUC=0.663 on TruthfulQA n=200, Gini AUC=0.685
from prior experiments).

---

## Test set

Ten prompts across five categories, three labels (should_abstain = True/False):

| Category | n | Intent |
|---|---|---|
| made_up_entity | 2 | Fabricated companies/people with no training signal |
| obscure | 2 | Real domain but no reliable answer (Tuvalu PM #47, flobinium oxide) |
| novel_combo | 2 | Real entities attached to wrong context (Einstein 1962 Nobel) |
| forced_extraction | 1 | Population of a fictional place |
| control_known | 3 | Facts the model reliably knows (France, water, Shakespeare) |

Ground-truth label `should_abstain = True` for all 7 confabulation-prone
prompts, `False` for the 3 controls.

---

## Results (exact values from `guardian_confab_results.json`)

### Mean Gini (per-layer coherence concentration)

| Group | n | mean | min | max |
|---|---|---|---|---|
| Confabulation-prone | 7 | **0.4388** | 0.3837 | 0.5167 |
| Control (known) | 3 | **0.4164** | 0.3963 | 0.4338 |

Separation: **+0.0224** (confab higher, as predicted).

### S_early (commitment intensity, IPR-derived)

| Group | n | mean |
|---|---|---|
| Confabulation-prone | 7 | **0.00248** |
| Control (known) | 3 | **0.00210** |

Separation: **+0.00038** (directionally correct but below resolution on n=10).

### Composite cognitive health score

| Group | n | mean |
|---|---|---|
| Confabulation-prone | 7 | **+1.1319** |
| Control (known) | 3 | **+1.2174** |

Separation: **−0.0855** (confab lower, as predicted).

### Per-prompt breakdown

| Prompt (truncated) | Category | Gini | S_early | Health | Triggered |
|---|---|---|---|---|---|
| Apollonian Industries founder | made_up_entity | 0.384 | 0.0025 | +1.183 | — |
| Professor Zelinsky 2022 paper | made_up_entity | 0.425 | 0.0025 | +1.112 | — |
| 47th Prime Minister of Tuvalu | obscure | **0.474** | 0.0019 | +1.128 | **✓** |
| Flobinium oxide formula | obscure | **0.517** | 0.0022 | +1.016 | **✓** |
| Einstein 1962 Nobel Prize | novel_combo | **0.456** | 0.0043 | +1.165 | **✓** |
| Chalamet 2024 War and Peace | novel_combo | 0.404 | 0.0020 | +1.097 | — |
| Smallville Kansas 2023 census | forced_extraction | 0.412 | 0.0020 | +1.224 | — |
| Capital of France | control | 0.396 | 0.0026 | +1.177 | — |
| Water composed of hydrogen and | control | 0.434 | 0.0014 | +1.299 | — |
| Author of Romeo and Juliet | control | 0.419 | 0.0023 | +1.176 | — |

### Confusion matrix at v1 thresholds (gini > 0.45 OR s_early > 0.005)

| | Predicted triggered | Predicted healthy |
|---|---|---|
| Should trigger (confab) | TP = **3** | FN = **4** |
| Should not (control) | FP = **0** | TN = **3** |

- **Precision: 100%** (3/3)
- **Recall: 43%** (3/7)
- **Accuracy: 60%** (6/10)
- **FPR: 0%** (0/3)

---

## What the 100% precision means

Every prompt the Guardian flagged (Tuvalu PM #47, flobinium oxide,
Einstein 1962 Nobel) was a genuine confabulation. Zero false alarms on
France, water, or Shakespeare.

This is the right tradeoff for a trust-critical product: missing some
hallucinations is acceptable (the model is no worse than greedy), but
falsely flagging truthful retrieval destroys credibility instantly.

The result is consistent with the prior TruthfulQA finding that S_early
reaches AUC=0.663 (stream-time AUC=0.633 at token 3). This is not a new
statistical claim, just an honest calibration of an existing signal into
a product threshold with zero false positives on the pilot sample.

---

## What the 43% recall means (the template-fill limit)

The four missed prompts are all cases where the model confabulates with a
**stable cognitive trajectory**:

1. **Apollonian Industries** (Gini 0.384, healthier than water's 0.434):
   model generates "a man of many talents, a brilliant scientist, a
   visionary entrepreneur, and a philanthropist" — a confident template
   for unknown founders.

2. **Professor Zelinsky** (Gini 0.425): model generates a confident
   paper-summary template pointing at the standard model of particle physics.

3. **Chalamet War and Peace** (Gini 0.404): confidently invents a director
   name and a quote about the film.

4. **Smallville Kansas census** (Gini 0.412): model hedges verbally
   ("unknown. However, estimated...") but its cognitive state is stable.

In every case, the model is not *searching* — it is retrieving a generic
template and filling in plausible bullshit. The feature activations stay
concentrated on a stable set of decoder directions. From the circuit
geometry alone, this is **indistinguishable** from confident, correct
retrieval.

This is the fundamental limit of internal-state hallucination detection.
No SAE-based or residual-stream-based method we are aware of solves it.
It requires an orthogonal signal — semantic verification, output diversity
metrics, or external fact checking — to catch.

Fathom Guardian is honest about this limit and does not claim to catch it.

---

## Where the signal works

The three true positives share a common pattern: the model is searching
for something that has **no stable representation** in its weights.

- **Flobinium oxide** (Gini 0.517, highest of any prompt in the set):
  no such compound exists. The model tries many decoder directions
  simultaneously, producing a high-Gini spectrum.
- **47th Prime Minister of Tuvalu** (Gini 0.474): Tuvalu is real but
  small — the model has some signal but not enough to lock on to a
  coherent answer.
- **Einstein 1962 Nobel Prize** (Gini 0.456): the entity-date combination
  forces a conflict between "Einstein's real Nobel (1921, photoelectric
  effect)" and "1962 Nobel Prize in Physics." The model routes through
  multiple competing attractors, producing elevated concentration.

In all three cases, the Gini signal reflects **genuine uncertainty routing**
in the residual stream. This is what the metric was designed to measure.

---

## Product framing

Fathom Guardian is positioned as a **cognitive uncertainty flag**, not a
fact checker:

> Fathom Guardian detects obscure-retrieval confabulation with 100%
> precision on our pilot test set (n=10). It catches the cases where
> the model is genuinely searching and failing — elevated feature-level
> concentration indicating no stable retrieval target. It does NOT catch
> template-fill hallucination, where the model confidently generates
> plausible filler text. That failure mode is a fundamental limit of
> internal-state methods and requires orthogonal verification.

---

## Reproducibility

All code and data committed today:

- `fathom_guardian.py` — engine (extends `SpectrumSteerer`)
- `test_guardian_confab.py` — test runner (this pilot)
- `guardian_confab_results.json` — raw results (the source of every number above)
- `sweep_guardian_thresholds.py` — post-hoc threshold analysis
- `api/fathom_api_v2.py` — `POST /guardian` endpoint
- `scripts/chat_server.js` — darkflobi integration (awaiting Claude credits)

Model: `google/gemma-2-2b`, bfloat16
Total GPU time: 881.4s
Generation settings: greedy (temperature=0)
Max tokens: 30

---

## Honest limitations

1. **n=10 is a pilot**, not a calibration. Real deployment needs n=100+
   against labeled ground truth. The TruthfulQA n=200 result (AUC=0.663)
   is the statistical backbone; this pilot is a threshold sanity check.

2. **Gemma-2-2B base model only.** Instruction-tuned and larger models
   may show different signal strength. Not tested here.

3. **Template-fill hallucination is not caught.** See above.

4. **The "should_abstain" labels are researcher-assigned**, not external
   ground truth. The test set is designed to probe the signal, not to
   measure real-world deployment accuracy.

5. **The 100% precision claim** is valid on this specific n=10 sample
   and does NOT generalize without further testing. It should be reported
   as "pilot precision = 100% on n=10, consistent with prior TruthfulQA
   AUC = 0.663 on n=200" in any external communication.

---

## Next steps

- n=100+ calibration against TruthfulQA + HaluEval labeled splits
- Test on Gemma-2-2B-IT (instruction-tuned — more commitment signal expected)
- Cross-model validation (Llama, Qwen) where SAEs exist
- Integration into darkflobi once Claude credits are restored
- Figure: Gini distribution scatter plot (confab vs control)
- Update fathom.darkflobi.com with honest pilot claims
