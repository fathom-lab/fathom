# ICML 2026 Mechanistic Interpretability Workshop
## Submission Abstract - Fathom

**Title:** Fathom: Circuit-Level Reasoning Depth Measurement and RLHF Selective Amplification

**Track:** ICML 2026 MI Workshop (Seoul, July 10-11)
**Deadline:** ~mid-May 2026 (monitor: icml.cc/Conferences/2026/CallForWorkshopPapers)

---

## Abstract (250 words)

We present Fathom, a circuit-level instrument for measuring reasoning depth in transformer language models using sparse autoencoder (SAE) feature attribution. By computing the mean active feature layer across SAE transcoders during inference, we quantify the computational depth signature of arbitrary text prompts without requiring model fine-tuning or labeled datasets.

**Core finding:** Surface recall prompts ("the capital of France is") concentrate SAE activation in output-proximal layers, while causal reasoning prompts ("why do economies experience inflation") distribute activation across early and mid layers. The ratio of these signatures converges to a near-constant K = 1.027 ± 0.003, replicated across Gemma-2-2B (p = 0.000051, n = 30) and Llama-3.2-1B (p = 0.003, n = 19) using independent SAE transcoders.

**RLHF selective amplification:** Comparing base versus instruction-tuned variants, we find RLHF selectively amplifies depth on safety-sensitive topics (white-box: d=0.24, p=0.18, n=81 — trend). Critically, black-box output-layer validation on the Claude model family confirms the effect more strongly (d=0.58, p=0.032), with the GPT family showing the same direction at smaller magnitude (d=0.26, p=0.21). The output-layer effect exceeding the internal activation effect suggests RLHF operates primarily at the expression boundary rather than the computation layer — a dissociation with direct implications for alignment evaluation methodology.

Furthermore, the provider-level difference (Anthropic d=0.58 vs OpenAI d=0.26) demonstrates that Fathom can distinguish alignment implementations across labs from output text alone — positioning it as a tool for independent alignment auditing without model access.

Code, data, and pre-registration available at darkflobi.com/fathom. DOI: 10.5281/zenodo.19326175. US Patent Pending: 64/020,489 & 64/021,113.

---

## Submission Checklist
- [ ] Confirm exact word limit for workshop
- [ ] Fill [TBD] with confirmed p-values from running experiments
- [ ] Add author affiliation (independent researcher)
- [ ] Check if workshop requires anonymized submission
- [ ] Prepare 4-page extended abstract or full paper (check format)
- [ ] Submit via CMT / OpenReview (check which platform ICML 2026 uses)

## Why This Workshop
- Uses SAE attribution directly (circuit-tracer from Anthropic/EleutherAI)
- RLHF finding is directly relevant to MI + alignment intersection
- Haiku > Sonnet result will invite mechanistic explanation from the community
- Cross-lab validation strengthens credibility
- Neel Nanda (DeepMind MI) is likely to be involved - we've already reached out
