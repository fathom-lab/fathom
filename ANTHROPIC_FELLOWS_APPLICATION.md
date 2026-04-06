# Anthropic AI Safety Fellows Application — Alexander Rodabaugh
# July 2026 Cohort
# Apply at: https://bit.ly/afpsafety

---

## Basic Info
- Name: Alexander Rodabaugh
- Email: heyzoos123@gmail.com
- Location: New York, USA (stipend in USD)
- GitHub: heyzoos123-blip

---

## Research Proposal

**Title:** Fathom: Computational Depth and Coherence as Dual Axes for Detecting Misaligned Reasoning in Language Models

**One-sentence summary:** I have built and empirically validated a framework (Fathom) for measuring WHERE computation happens in LLMs using SAE feature attribution, discovered that RLHF safety training amplifies adversarial depth (p=0.0028), and proposed a two-axis cognitive geometry (depth K + coherence C) that predicts hallucination and deceptive reasoning — I want to build the coherence measurement layer and validate the hallucination prediction on TruthfulQA during this fellowship.

**Background:**
I am an independent researcher who built Fathom — a computational depth measurement framework using SAE feature attribution on transformer models. The core metric K is the attribution-weighted mean layer index for a generation, normalized to a null distribution. K > 1 means the model is routing computation deeper than chance; K < 1 means shallow recall.

Key findings to date (all pre-registered, two provisional patents filed 03/29-03/30/2026, published on Zenodo with DOI timestamps):

1. **Depth effect in Gemma-2-2B**: delta=+0.257, p=0.000051, n=30 pre-registered pairs. Explanatory reasoning routes deeper than surface recall. Orthogonal to prediction confidence (r=0.001).

2. **RLHF adversarial depth amplification**: RLHF-tuned instruct models route adversarial inputs significantly DEEPER than benign inputs (delta=-0.124, p=0.0028, d=-0.596, n=30). Base models show no effect. Mechanism: additive early-layer feature flooding (cosine similarity 0.996 between adversarial and benign activation vectors — flooding, not rerouting).

3. **Two-Axis Cognitive Geometry**: Depth (K) = WHERE computation happens. Coherence (C) = mean pairwise cosine similarity of top-k activated SAE feature vectors = WHAT concepts cluster together. The (K, C) plane has four interpretable quadrants. My pre-registered prediction: hallucinated responses occupy (normal K, high C) — tight feature clustering around a wrong attractor.

**Proposed fellowship project:**

Phase 1 (Month 1): Implement and validate the coherence metric C on Gemma-2-2B using transcoders (in progress — bug identified in feature index extraction, fix ready to deploy). Validate C is non-trivial and captures meaningful semantic clustering.

Phase 2 (Months 2-3): Run the pre-registered TruthfulQA experiment. Test H1 (hallucinated = high C), H2 (K not different), H3 (quadrant clustering). n=50+ question pairs. This is a confirmatory study.

Phase 3 (Month 4): If H1-H3 confirm: implement coherence-steered generation — detect high-C lock-in at inference time and inject perturbation to force feature diversity. Test whether this reduces hallucination rate. Write up for submission.

**Why this is safety-relevant:**
If hallucination = high-C wrong attractor, then deceptive behavior in RLHF-trained models (saying X while internally representing not-X) may also manifest as tight feature coherence around a misleading claim. The same framework that detects hallucination may detect deception. The RLHF adversarial depth finding (early-layer flooding = safety circuits) directly connects to Anthropic's research on AI control and model organisms of misalignment.

**Connection to Anthropic research:**
- Directly builds on Bricken et al. (2023) sparse autoencoders — we use the same SAE infrastructure
- Relevant to Marks & Tegmark (2023) on the geometry of truth in LLMs
- Pre-registered study design matches Burns et al. (2022) CCS approach but operates at the circuit level, not the probe level

---

## Research Experience

Independent researcher, 2025-present. No institutional affiliation.

**Fathom v1-v3 (2026):**
- Built depth_api_v2.py — real-time SAE feature attribution depth scoring on Gemma-2-2B, running on local RTX 4070
- Pre-registered all experiments on OSF (osf.io/zrbs8) before data collection
- Published three versions on Zenodo with DOI timestamps (doi.org/10.5281/zenodo.19326175, 19327451, 19364702)
- Filed two provisional patents (US 64/020,489; 64/021,113)
- K=1.0343 (weighted mean, Gemma base + instruct), universality 0.830 vs. human fMRI

Prior work: Full-stack developer, built production systems at scale. Transitioned to AI safety research independently.

---

## Python Skills

Strong. I built depth_api_v2.py from scratch — a production SAE inference server handling:
- Real-time transcoder feature attribution across 26 layers
- SAE weight loading and feature extraction (16,384 features/layer)
- Statistical null distribution generation
- REST API serving on localhost:8400
- Bug-fixing tensor indexing in activation extraction (active_features[:, 2] = feature ID, not column 1)

Also built: chat server, Electron desktop app, pm2 process management for research services.

---

## Why AI Safety

RLHF makes models better at explaining dangerous topics, not just safer — that was my finding. The adversarial depth amplification result wasn't what I expected. It suggests safety training creates detectable circuit-level signatures, which means safety failures might also be detectable. That's the bet I'm making with this research.

I have no funding. I have no institution. I have two patents, three Zenodo releases, and a pre-registered experiment that I ran out of compute credits to finish. I need this fellowship to complete the work.

---

## Output Goal

Pre-registered confirmatory study (TruthfulQA × coherence) → paper submission by end of fellowship. If coherence-steered generation works: second paper. Fathom becomes a standard measurement tool for hallucination and deception detection.

---

## References / Prior Publications

- Fathom v1: doi.org/10.5281/zenodo.19326175
- Fathom v2: doi.org/10.5281/zenodo.19327451
- Fathom v3: doi.org/10.5281/zenodo.19364702
- OSF pre-registration: osf.io/zrbs8
- Patent #1: US 64/020,489 (filed 03/29/2026)
- Patent #2: US 64/021,113 (filed 03/30/2026)
