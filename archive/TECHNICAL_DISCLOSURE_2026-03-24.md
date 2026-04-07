# TECHNICAL DISCLOSURE DOCUMENT
## Depth-as-Representation (DaR): A Real-Time Cognitive Depth Metric for Language Models

**Inventor(s):** Flobi (flobi69) and darkflobi  
**Date of Invention:** March 2026  
**Disclosure Date:** 2026-03-24  
**Status:** Confidential — Prior Art Establishment

---

## 1. Title of Invention

**Depth-as-Representation (DaR): A Method for Real-Time Measurement of Cognitive Processing Depth in Transformer Language Models Using Sparse Autoencoder Feature Attribution**

---

## 2. Field of Invention

This invention relates to methods for evaluating the computational depth of language model inferences in real-time, with application to AI training data valuation, agent behavior evaluation, and alignment research.

---

## 3. Background

A persistent challenge in language model evaluation is distinguishing *surface-level pattern matching* from *genuine reasoning*. Prior work has relied on behavioral benchmarks (whether a model produces correct answers) without characterizing the internal computation that produces those answers. We identified a gap: no method existed to measure, in real-time during inference, whether a language model is engaging in shallow recall versus deep mechanistic reasoning.

---

## 4. Summary of Invention

We describe a method — **Depth-as-Representation (DaR)** — that operationalizes cognitive processing depth as a measurable scalar value computable from transformer model internals.

**Core insight:** When a language model processes text requiring explanatory or causal reasoning, activated features in its sparse autoencoder representation distribute toward *earlier and middle layers* of the network. When processing surface-level factual recall, feature activation concentrates toward *later, output-proximal layers*. The mean layer index of activated features is therefore a proxy for reasoning depth.

**DaR Score = mean(layer_index × feature_weight) over all activated SAE features for a given forward pass**

Higher DaR score = more output-proximal processing (surface recall)  
Lower DaR score = more early/mid-layer processing (deep reasoning)

---

## 5. Experimental Evidence

### Study 1 — Negative Result (Confound Elimination)
Across five experimental iterations using GPT-2 Small, Gemma-2-2B, and Mistral-7B, we demonstrated that surface text statistics (vocabulary frequency, readability, N-gram probability) **cannot separate** surface recall from insight prompts when vocabulary is controlled. This establishes that the distinction is not recoverable from input text alone, motivating the computational approach.

### Study 2 — Pre-registered Exploratory Replication
- Model: Gemma-2-2B
- n = 11 valid pairs (pre-registered directional test)
- Result: mean Δ = +0.118, t = 2.04, **p = 0.035** (one-tailed)
- 73% of pairs in predicted direction
- Limitation: n=11 is underpowered; LOO analysis showed fragility

### Study 3 — Power Replication (Primary Finding)
- Model: Gemma-2-2B
- n = 30 pairs, zero topic overlap with prior samples
- 100% attribution success (zero exclusions)
- **Result: mean Δ = +0.257, t = 4.50, p = 0.000051** (one-tailed, pre-registered)
- 83.3% of pairs in predicted direction
- Effect size matches exploratory estimate (0.257 vs. 0.26)
- Leave-one-out analysis: fully robust, result survives removal of any single pair

**This is the first pre-registered, replicated demonstration that mean SAE feature activation depth is a statistically significant proxy for cognitive processing depth in transformer models.**

---

## 6. Novel Contributions

1. **The DaR Metric itself** — operationalizing cognitive depth as a scalar via SAE feature layer distribution (novel; no prior art identified as of March 2026)

2. **Real-time inference** — the metric is computed during a forward pass, making it applicable in production systems without additional fine-tuning or evaluation models

3. **The Depth Scoring API** — a production implementation running on consumer GPU hardware (NVIDIA RTX 4070) exposed via REST endpoint, scoring arbitrary text inputs in ~22 seconds

4. **Application to agent behavioral data valuation** — using DaR to score autonomous AI agent decision traces, creating a verified quality label for training data (see DarkCity pipeline)

5. **The DaR Pipeline** — an end-to-end system that logs, scores, and chains agent actions with DaR values, creating labeled behavioral datasets from autonomous agent simulations

---

## 7. Claims for Priority

We claim priority on the following specific innovations:

**Claim 1:** The use of mean SAE feature layer index as a real-time proxy for language model reasoning depth.

**Claim 2:** A system and method for scoring arbitrary text inputs with the DaR metric via API at inference time.

**Claim 3:** The application of DaR scoring to autonomous AI agent decision traces for the purpose of training data quality labeling.

**Claim 4:** A pipeline architecture combining autonomous agent simulation, DaR scoring, causal chain linking (chain_id), and cryptographic provenance hashing (record_hash) to produce verifiable, high-value AI training datasets.

**Claim 5:** The use of DaR-scored behavioral data from multi-agent simulations as a distinct category of AI training data, characterized by ground-truth reasoning traces, alternatives-considered fields, and agent state capture.

---

## 8. Current Implementation Status (as of 2026-03-24)

- **Depth Scoring API:** Running on RTX 4070, exposed via Cloudflare tunnel. Validated: 8.477 (surface) vs. 8.217 (insight), delta = 0.260 (matches paper: 0.257).
- **Railway Backend:** Deployed at `darkcity-backend-production-427a.up.railway.app` with DaR pipeline integrated into every agent action.
- **Database:** `depth_evaluations` table in Railway Postgres with columns: `reasoning_trace`, `alternatives_considered`, `agent_state`, `chain_id`, `prev_hash`, `record_hash`.
- **Agent System:** 30 NPC agents seeded in DarkCity simulation, tick loop active on Haiku model.
- **Depth Leaderboard:** Live at `app.darkcity.wtf/leaderboard`

---

## 9. What This Document Establishes

This document, dated and timestamped on 2026-03-24, establishes:
- Prior art on the DaR metric and its applications
- Inventor identity (Flobi + darkflobi)
- Date of invention (March 2026)
- That a working implementation existed at the time of this disclosure

This document should be emailed to the inventor(s) immediately upon creation to establish a legally timestamped record of invention disclosure.

---

## 10. Confidentiality Note

The *concept* and experimental results described in this document are intended for publication (arXiv preprint, target venue: NeurIPS 2026 or similar). The *implementation details* of the depth scoring model (weights, calibration curves, SAE configuration) are to be maintained as trade secrets and are not included in this document.

---

*Document prepared: 2026-03-24*  
*Inventor: Flobi (@flobi69) with darkflobi*  
*Contact: [Flobi's email]*
