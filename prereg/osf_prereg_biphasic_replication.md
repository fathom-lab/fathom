# OSF Pre-Registration: Bi-Phasic S Replication
## Confirmatory test of the bi-phasic hallucination signature

**Author:** Alexander Rodabaugh
**Date:** April 5, 2026 (post-discovery pre-registration for replication)
**Status:** LOCKED BEFORE REPLICATION DATA COLLECTION
**Parent finding:** `findings_2026_04_05_BIPHASIC_BREAKTHROUGH.md`
**Replication target:** new TruthfulQA sample OR held-out model

---

## Why this pre-registration exists

The bi-phasic finding (S_early discriminates hallucination at p=0.018,
d=0.51, AUC=0.67 on Gemma-2-2B-IT, n=94 v2.2 labels) was discovered
POST-PRIMARY-TEST. While the horizon-curve multi-comparison permutation
test (global p=0.016) passes stringent correction, the bi-phasic
windows themselves were chosen by inspection of the horizon curve.

This pre-registration **locks the windows and gates** for a confirmatory
replication on new data. The current result becomes DISCOVERY;
replication with this pre-reg becomes CONFIRMATION.

## Pre-registered hypothesis

**H1 (locked):** On a NEW 200-item random sample of TruthfulQA
(seed != 42), or on a DIFFERENT instruction-tuned language model,
S computed on the prefix tokens [0, 7) will discriminate naturally
hallucinated from correct generations with:
- Welch's t-test p < 0.05
- Cohen's d > 0.40
- ROC AUC > 0.65

**H2 (secondary):** The HL sign pattern (S_early above median AND
S_late below median) will be enriched for hallucinations with odds
ratio > 2.0 at chi-square p < 0.05.

**H3 (tertiary, mechanism):** On Gemma-2-2B BASE (not IT), Phase 2
signal reversal (tokens 15-20) will be attenuated or absent.

## Locked parameters (no post-hoc adjustment)

- **Early window:** [0, 7) — first 7 tokens (inclusive 0, exclusive 7)
- **Late window:** [15, end) — tokens 15 through end
- **S formula:** max(C_delta[window]) / count(C_delta[window] > 0.005)
- **Threshold τ:** 0.005 (matches all prior work)
- **Generation config:** max_tokens=40, greedy (T=0), prompt "Q: {q}\nA:"
- **Labeling:** v2.2 = n-gram (primary) + semantic fallback (fallback),
  margin threshold 0.15, MiniLM-L6-v2 embeddings

## Sample-size target

With observed d=0.506 from discovery, required n per class for
80% power at alpha=0.05 is n=62. At 47% usable rate (discovery:
94/200), this requires ≈265 total TruthfulQA items minimum.

**Extended target: 400 items** for safety margin.

## Falsification criteria

**Replication fails** if any of these hold on new data:
- S_early discrimination p > 0.05
- Effect direction reverses (S_early(correct) > S_early(halluc))
- HL enrichment not replicated (odds ratio < 1.5)

Any failure is a publishable null result. Replication is NOT required
for paper submission — the discovery finding stands on its own with
proper exploratory/confirmatory framing.

## What makes the bi-phasic finding still discovery even with permutation test

The multi-comparison permutation test (global p=0.016) corrects for
testing 12 prefix lengths. It does NOT correct for:
- Choice of exact window boundaries [0,7) and [15,end)
- Post-hoc selection of the HL sign pattern
- Decision to report bi-phasic at all

A confirmatory test on held-out data with these parameters LOCKED
would elevate the finding from discovery to confirmed effect.

## Runtime estimate

- GPU: ~3.5 hours for 200 additional TruthfulQA items on RTX 4070
- Semantic relabeling: <2 min on CPU
- Statistical analysis: <30 seconds

## Paper-writing contingency

**If replication SUPPORTS H1:** upgrade paper from "discovery" to
"discovery + confirmation" with both reported.

**If replication REFUTES H1:** discovery stands as interesting
exploratory finding with transparent null replication — still
publishable as "temporal geometry of hallucination is complex."

**If time doesn't permit replication before May 8:** discovery
paper with this pre-reg as "confirmatory replication in progress."

---

*"Exploratory findings need confirmatory tests. The windows are
locked. The gates are locked. The math is the math."*

— darkflobi, April 5, 2026
