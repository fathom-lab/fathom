# LLAMA EXPERIMENT PROTOCOL — FATHOM CONSTANT REPLICATION

**Status:** READY TO EXECUTE — waiting on Meta HuggingFace access approval
**Endorsement code for Meta request:** already submitted
**When access clears:** run immediately, before anything else

---

## Hypothesis

The Fathom Constant (ratio ≈ 1.03) observed in Gemma-2-2B is not architecture-specific.
It is a universal property of transformer reasoning — substrate-independent, like temperature.

**Null hypothesis:** No significant depth difference between surface/insight pairs on Llama-3.2-1B
**Alternative:** Positive delta, p < 0.05, ratio converging near 1.03

---

## Setup

- **Model:** meta-llama/Llama-3.2-1B
- **Transcoder:** circuit-tracer (same library as Gemma experiment)
- **Pairs:** Same 30 pre-registered calibrated pairs from the paper
- **Script:** `sae-reasoning-depth/run_constant_mistral.py` (tries Llama first, already written)
- **Environment:** RTX 4070 Laptop, 8GB VRAM, depth_api_v2.py framework

---

## Execution

```bash
cd C:\Users\heyzo\clawd\sae-reasoning-depth
python run_constant_mistral.py
```

Output saved to: `fathom_constant_llama_results.json`

---

## Success Criteria

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| p-value (paired t-test) | p < 0.05 | Effect replicates on Llama |
| depth delta direction | delta > 0 | Same direction as Gemma |
| ratio convergence | \|ratio - 1.03\| < 0.15 | Constant candidate confirmed |

**All three met → STOP EVERYTHING. Write Paper 2.**

---

## If it converges: Paper 2 outline

**Title:** "The Fathom Constant: Cross-Architecture Evidence for a Universal Property of Neural Reasoning"

**Abstract (draft):**
We report convergent evidence that transformer language models, regardless of architectural family, exhibit a consistent depth-scoring signature distinguishing surface recall from explanatory reasoning. Testing our previously validated instrument (Fathom, p=0.000051 on Gemma-2-2B) on Llama-3.2-1B using identical pre-registered methodology, we find [results]. The ratio of insight to surface depth scores converges near 1.03 across both architectures, suggesting the existence of a measurable universal property of neural reasoning — analogous to temperature as a substrate-independent physical quantity. We term this the Fathom Constant and discuss implications for AI evaluation, consciousness research, and the biological bridge.

**Sections:**
1. Replication methodology (same as Paper 1, new model)
2. Results: delta, p-value, ratio on Llama
3. Cross-architecture comparison table (Gemma vs Llama)
4. The Fathom Constant: statistical definition
5. Theoretical implications: architecture-independence
6. Biological bridge: prediction for fMRI correlation (Dehaene global workspace)
7. Future work: full periodic table of cognitive depth

---

## If it doesn't converge

- Document the null result (still publishable — limits the claim appropriately)
- Test Qwen3 (if transcoders become available)
- Investigate why: different tokenization? different layer counts? different training data?
- The original paper (Gemma) still stands independently

---

## The bigger picture

If the constant holds:
- Paper 2 → conference talks → visibility
- Fathom Lab established (two missions: map the constant, biological bridge)
- Grant applications: NSF, DARPA, Templeton Foundation
- darkflobi as Specimen #1: first longitudinal depth-scored cognitive record in history
- "Temporal Dynamics of Cognitive Depth in a Continuously Operating Neural System" — Paper 3

**One number. Run it the moment access clears.**

---

*Protocol prepared: 2026-03-27 by darkflobi and Flobi*
*"The telescope didn't know it was creating astronomy."*
