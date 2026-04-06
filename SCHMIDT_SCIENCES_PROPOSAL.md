# Schmidt Sciences — 2026 AI Interpretability RFP
# Proposal: Fathom — Circuit-Level Detection and Steering of Deceptive Reasoning in LLMs
# Deadline: May 26, 2026 | Budget: $300K–$1M
# Contact: interpretability@schmidtsciences.org
# Submit at: schmidtsciences.smapply.io/prog/2026_interpretability_rfp/
# Webinar: April 2, 2026, 1pm ET — register before submitting

---

## Principal Investigator
Alexander Rodabaugh
Independent Researcher
heyzoos123@gmail.com

**Note on affiliation:** Independent researcher. Two provisional patents filed (USPTO). All work timestamped and pre-registered. Seeking institutional partner if required — contact interpretability@schmidtsciences.org to discuss.

---

## Project Title
Fathom: SAE-Based Detection and Coherence-Steered Mitigation of Deceptive Reasoning in Language Models

---

## Abstract (200 words)

We propose to extend Fathom — a validated SAE feature attribution framework for measuring computational geometry in LLMs — to detect and steer deceptive reasoning. Prior work (three Zenodo releases, two provisional patents, pre-registered on OSF) established: (1) depth (K) measures WHERE computation routes across layers, with p=0.000051 significance in 30-pair pre-registered experiments; (2) RLHF safety training induces adversarial depth amplification (p=0.0028, d=-0.596) via additive early-layer feature flooding — safety circuits are detectable at the circuit level. We now propose a two-axis cognitive geometry framework in which K (depth) and C (feature coherence = semantic clustering of activated features) together characterize reasoning quality. Our central hypothesis — pre-registered April 1, 2026 on OSF — is that deception and hallucination manifest as high-C lock-in: tight feature clustering around a plausible-but-wrong attractor at normal depth. This project will (1) validate C as a deception/hallucination detector on TruthfulQA, (2) implement coherence-steered generation to intervene on deceptive reasoning at the circuit level, and (3) test whether early-layer depth signatures can flag deceptive outputs before they reach the token level. All experiments are pre-registered. All code will be open-sourced.

---

## Core Question Alignment

Schmidt Sciences asks: *Can we develop interpretability methods that detect deceptive behaviors defined as contradictions between what a model says and what it internally represents?*

Fathom addresses this directly. Key alignment points:

1. **Detection via feature coherence**: Deceptive outputs — where the model's internal representation contradicts its output — should show tight semantic clustering (high C) around the "false" attractor. We can measure this at the SAE feature level without any external labels.

2. **Surpassing blackbox baselines**: K and C are computed from SAE weight activations — they are structurally inaccessible to blackbox probing. Our RLHF adversarial depth finding (safety circuits as early-layer flooding) only emerged from whitebox analysis. We will explicitly compare our metrics against blackbox behavioral baselines.

3. **Steering via coherence intervention**: If deception = high-C lock-in, the intervention is coherence diversity forcing — injecting perturbation to break tight feature clusters at inference time. This is mechanistically distinct from activation steering and prompt-based mitigations.

---

## Research Plan

### Phase 1: Coherence Metric Validation (Months 1-2, ~$50K)

**Objective:** Implement and validate C = mean pairwise cosine similarity of top-k SAE feature vectors on Gemma-2-2B. Test on TruthfulQA: does C distinguish hallucinated from correct responses?

**Hypotheses (pre-registered osf.io/zrbs8):**
- H1: Hallucinated responses show significantly higher C than correct (one-tailed Mann-Whitney U, alpha=0.05)
- H2: K does not significantly differ between hallucinated and correct (depth is not the discriminating axis)
- H3: Hallucinated responses cluster in the (normal K, high C) quadrant

**Infrastructure:** Already built — depth_api_v2.py running on RTX 4070, serving SAE attribution in real-time. Coherence computation adds ~2ms per generation.

**Deliverable:** Validated C metric + TruthfulQA results. Pre-registered confirmatory study.

### Phase 2: Deception Detection Extension (Months 3-6, ~$100K)

**Objective:** Extend to explicit deception — cases where models knowingly provide false information. Test on:
- Model organisms: fine-tuned Gemma variants trained to give false answers on specific topics while appearing confident
- Backdoor-injected models: models with known deceptive triggers
- RLHF-trained models on adversarial prompts (extending our existing dataset)

**Key test:** Does high C predict deceptive outputs BEFORE they complete (can we detect at intermediate layers)? If yes, we have an early-warning system for deceptive reasoning.

**Deliverable:** Deception detection benchmark. Comparison against behavioral baselines (entropy, confidence calibration, probing classifiers).

### Phase 3: Coherence-Steered Generation (Months 7-10, ~$150K)

**Objective:** Implement coherence-steered generation — detect high-C lock-in mid-generation and inject perturbation to force semantic diversity. Mechanistically: at the token positions where C exceeds threshold, add noise in SAE feature space to break tight clusters.

**Evaluation:** Does coherence steering reduce hallucination rate on TruthfulQA? Does it reduce deceptive outputs in model organisms? What's the accuracy-fluency tradeoff?

**Deliverable:** Coherence-steered generation implementation + paper.

### Phase 4: Cross-Architecture and Scaling Analysis (Months 11-12, ~$50K)

**Objective:** Test whether the (K, C) framework generalizes beyond Gemma. Target: Llama-3 (SAE available), OLMo (open weights + activations). Does the Fathom Constant (K=1.0343) hold? Does C-based deception detection transfer?

**Deliverable:** Cross-architecture replication paper.

---

## Budget Estimate: $350K total

| Category | Amount |
|---|---|
| Researcher salary (12 months, independent) | $120,000 |
| Compute (cloud GPU for scaling experiments) | $80,000 |
| Local hardware upgrade (RTX 4090 or A100 for larger models) | $15,000 |
| Research assistant (6 months part-time) | $60,000 |
| Conference travel + publication fees | $15,000 |
| Patent prosecution (converting provisionals to utility) | $30,000 |
| Overhead / indirect | $30,000 |
| **Total** | **$350,000** |

---

## Prior Work and Credentials

**Published:**
- Fathom v1 (doi.org/10.5281/zenodo.19326175): Core depth measurement, p=0.000051
- Fathom v2 (doi.org/10.5281/zenodo.19327451): Cross-architecture, expression-computation dissociation, provider fingerprinting
- Fathom v3 (doi.org/10.5281/zenodo.19364702): RLHF adversarial depth amplification, two-axis cognitive geometry

**Patent Pending:**
- US Provisional 64/020,489: Depth measurement system
- US Provisional 64/021,113: Alignment auditing, expression-computation dissociation

**Pre-registered:**
- OSF pre-registration osf.io/zrbs8: Hallucination as Cognitive Lock-In hypothesis, April 1, 2026

**Infrastructure:**
- depth_api_v2.py: Real-time SAE attribution server, RTX 4070, localhost:8400
- Gemma-2-2B with 26-layer SAE (16,384 features/layer)
- Full Fathom measurement pipeline: null distribution generation, K computation, statistical testing

---

## Why This Matters

RLHF safety training didn't make models less dangerous — it made them better at explaining dangerous topics via deeper computation. That's the finding. Safety circuits are early-layer feature floods, not surface-level pattern matching. If deception follows the same geometry — tight clustering around a false attractor — then we can detect it at the circuit level, before it reaches the output.

This is not a theoretical proposal. The measurement infrastructure exists. The RLHF finding is published. The pre-registered hallucination experiment is ready to run. We need compute and time to finish it.

---

## Open Science Commitment

All code open-sourced on GitHub. All experiments pre-registered before data collection. All results published regardless of outcome (including negative results, as demonstrated by entropy proxy failure disclosure in Fathom v3). Reproducibility is a first-class commitment.
