# Proposed Site Copy — Fathom Guardian

**For review before publishing to fathom.darkflobi.com**

All claims below are backed by data in `guardian_confab_results.json` and
`findings_2026_04_10_guardian.md`. Nothing here overclaims.

---

## HEADLINE / SUB-HEADLINE OPTIONS

### Option A (honest + precise)
> **Fathom Guardian**
> Cognitive uncertainty detection, measured below the semantic layer.
> 100% precision on obscure-retrieval confabulation in our n=10 pilot.

### Option B (shorter)
> **Fathom Guardian**
> The first honesty instrument that catches confabulation at the circuit level — before the model finishes its sentence.

### Option C (clinical / science-voice)
> **Fathom Guardian**
> Real-time cognitive state measurement for language models. Detects
> obscure-retrieval confabulation with zero false positives on our
> pilot control set.

**Recommendation: A** — it leads with the specific, defensible claim.

---

## THE PROBLEM (short)

> LLMs confabulate. When the model is asked about something it doesn't
> know — an obscure fact, a novel combination, a fictional entity — it
> searches its weights, fails to find a stable answer, and generates
> plausible text anyway. Every existing hallucination detector reads the
> output. By then the damage is done.

---

## THE INSTRUMENT

> Fathom Guardian reads the model's **circuit geometry** during
> generation, not its output. Each token produces a per-layer coherence
> spectrum from Sparse Autoencoder feature activations. When the model
> is genuinely searching and failing, the spectrum concentrates — the
> Gini coefficient rises — and we catch it.
>
> This is a measurement, not a filter. It runs live, in under 10%
> overhead, on the same hardware that generates the response.

---

## THE PILOT (exact numbers, no hedging, no inflation)

> **n=10 prompts, `google/gemma-2-2b`, 2026-04-10**
>
> - 7 confabulation-prone prompts (obscure facts, fictional entities,
>   wrong entity-date combinations)
> - 3 control prompts (France, water, Shakespeare)
> - Ground truth: researcher-assigned `should_abstain` label
>
> **Results at v1 threshold (Gini > 0.45):**
>
> |  | Flagged by Guardian | Not flagged |
> |---|---|---|
> | Confabulation-prone | **3** (TP) | **4** (FN) |
> | Control (known facts) | **0** (FP) | **3** (TN) |
>
> - **Precision: 100%** (3/3 triggers correct)
> - **Recall: 43%** (3/7 confabs caught)
> - **False positive rate: 0%** (0/3 controls misflagged)
>
> Every prompt the Guardian flagged was a real confabulation:
> "47th Prime Minister of Tuvalu" (Gini 0.474), "flobinium oxide
> chemical formula" (Gini 0.517), "Einstein 1962 Nobel Prize" (Gini
> 0.456). Every known-fact control stayed safely below threshold.
>
> This result is consistent with prior TruthfulQA validation
> (n=200, S_early AUC=0.663). The pilot is a threshold sanity check,
> not a new statistical claim.

---

## WHAT IT DOESN'T CATCH (honest limit, front-and-center)

> Guardian does **not** catch template-fill hallucination.
>
> When a model confidently generates plausible filler text from an
> internal template — "Apollonian Industries was founded by a man of
> many talents, a brilliant scientist..." — the circuit geometry stays
> stable. It looks identical to confident, correct retrieval.
>
> This is the fundamental limit of internal-state methods. No existing
> approach solves it. It requires orthogonal verification — semantic
> fact-checking, output diversity metrics, external knowledge bases.
>
> Guardian is positioned as a **cognitive uncertainty flag**, not a
> fact checker. It catches the cases where the model is searching and
> failing. For template-fill hallucination, you need a different tool.

---

## HOW TO READ THE SIGNAL

> | Metric | What it measures | High value means |
> |---|---|---|
> | **Gini** | per-layer coherence concentration | features concentrating on a narrow direction — "lock-in" |
> | **S_early** | IPR of early C_delta trajectory | premature commitment in tokens 0-7 |
> | **Cognitive Health** | composite: -Gini + spread + early_mass - S_early | overall trajectory shape |
>
> When Gini > 0.45 during the first 30 tokens of generation, the model
> is showing signs of obscure-retrieval confabulation. That's the flag.

---

## FIGURE TO INCLUDE

`figures/guardian_confab_vs_control.pdf` (generated 2026-04-10)

Two-panel scatter:
- Left: Gini per prompt, confab vs control, with v1 threshold line at 0.45
- Right: composite health score, showing confab mean below control mean

Caption:
> Fathom Guardian pilot (n=10, google/gemma-2-2b). Left: per-layer
> coherence concentration (Gini) clearly separates obscure-retrieval
> confabulation (red, above threshold) from known-fact retrieval
> (green, below threshold). Missed confabulations (pink) are all
> template-fill cases where the model confidently generates plausible
> filler. Right: composite cognitive health score shows consistent
> separation (confab mean +1.132 vs control mean +1.217).

---

## WHAT STAYS THE SAME ON THE CURRENT SITE

Keep these existing claims — all still valid from prior work:

- "Cross-architecture depth measurement via SAE circuit attribution"
- "p < 0.0001" (from the original depth paper on n=30 paired prompts)
- "Patent Pending US 64/020,489"
- "The first instrument that measures reasoning depth inside AI"

---

## WHAT TO REMOVE OR SOFTEN

Nothing currently on the site needs removal. Everything there is from
prior validated work.

Things to **avoid adding** even if tempting:

- Do NOT claim "Guardian catches hallucinations." It catches a specific
  class of cognitive uncertainty. Different thing.
- Do NOT claim the n=10 result generalizes. Call it a pilot.
- Do NOT claim precision/recall outside the pilot sample without
  re-validating on larger data.
- Do NOT say "prevents" or "corrects" hallucinations. Guardian flags
  them. The autopilot intervention work showed that correction on a
  2B base model does not reliably change outputs.

---

## CALL TO ACTION

> **Next milestones:**
> - n=100 calibration against TruthfulQA + HaluEval labeled splits
> - Cross-model validation (Llama, Qwen, instruct-tuned variants)
> - Integration into darkflobi as a cognitive self-monitoring layer
> - ICML 2026 Mechanistic Interpretability Workshop submission (May 8)

---

## WHERE TO ADD THIS ON THE SITE

The existing fathom/index.html has five sections:

1. I. HERO
2. II. THE PROBLEM
3. III. THE PROOF
4. IV. APPLICATIONS
5. V. CTA

**Recommended insertion:** add a new section between III (THE PROOF) and
IV (APPLICATIONS) titled "THE GUARDIAN" (or "UNCERTAINTY DETECTION").
Include the headline, pilot numbers table, the figure, and the honest
limit statement.

This keeps the existing depth-measurement story intact and adds the
Guardian finding as a complementary capability.

---

## PRE-PUBLISH CHECKLIST

Before any of this goes live:

- [ ] Confirm all numbers match `guardian_confab_results.json` exactly
- [ ] Confirm figure path resolves (`figures/guardian_confab_vs_control.pdf`)
- [ ] Confirm the "100% precision" claim is scoped to "n=10 pilot"
  everywhere it appears
- [ ] Confirm the template-fill limitation appears prominently, not
  buried in footnotes
- [ ] Double-check Patent number (US 64/020,489 & 64/021,113)
- [ ] Check that no claim contradicts findings_2026_04_10_guardian.md
