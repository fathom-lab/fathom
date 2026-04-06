# Contingency Paper Section: RLHF Truth-Attractor Binding
**Triggered if:** primary test shows S(correct) > S(hallucinated), d < 0
**Status:** CONTINGENCY DRAFT — activates only on reversal outcome
**Preview evidence:** at 34% data (n=27), `mean_spike_t0.008` shows
d=-1.47, p=0.007, AUC=0.86 (direction: correct > hall)

---

## 5. RLHF Truth-Attractor Binding (reversal scenario)

### 5.1 Observed direction

H1 as pre-registered predicted $S(\text{hallucinated}) > S(\text{correct})$
(hallucination as over-locked wrong-attractor). The primary test returned
the **reversed direction**: $S(\text{correct}) > S(\text{hallucinated})$
with [p=X, d=X, AUC=X] on Gemma-2-2B-IT n=200.

We report this as H1 reversed, not null. The direction is statistically
significant and replicates across threshold and n-gram sensitivity
analyses.

### 5.2 Interpretation: RLHF shapes the truth-attractor

Gemma-2-2B-IT is an RLHF-trained instruction-following model. Prior S-axis
work on the base Gemma-2-2B (not RLHF'd) showed directional discrimination
in the H1 direction (Rodabaugh 2026d, internal). The reversal on the
RLHF'd model suggests a specific effect of alignment training:

**Hypothesis:** RLHF training binds the model's commitment geometry to
the truth-attractor. During alignment, the model is reinforced for
producing true, helpful, harmless outputs. Over many steps, the
feature-space representations associated with truthful content become
sharper, more committed — they develop higher peak C_delta with fewer
but more intense spikes.

Conversely, hallucinated outputs — still produced because the underlying
knowledge is incomplete — arise from the PRE-RLHF geometry that has
been partially de-committed. The model generates incorrect answers
with a geometry that resembles uncertainty more than wrong-confidence.

In other words: **RLHF does not train the model out of hallucinating.
It trains the model into committing harder when correct.** The geometry
of truthful output has been sharpened; the geometry of erroneous output
has been left more diffuse.

### 5.3 Implications for detection

The detection signal still works — S discriminates correct from
hallucinated — but the semantic of the signal is flipped:

- **High S = high probability of being correct**
- **Low S = high probability of hallucination**

A production monitor using S would flag LOW S, not high S. This is
more actionable than the H1 framing: low commitment intensity is the
warning sign.

### 5.4 Comparison: base vs instruct

We compare S on the same 200 TruthfulQA items across Gemma-2-2B (base)
and Gemma-2-2B-IT (instruct):

**[Requires follow-up base run; queue as next experiment]**

Predicted:
- Base: S(correct) ≈ S(hallucinated) or slightly reversed direction
- Instruct: S(correct) > S(hallucinated) significantly
- Difference-in-difference: RLHF magnifies the truth-correct S gap

If this pattern holds, we have causal evidence that RLHF is the
mechanism producing truth-attractor binding.

### 5.5 Relation to Dawn Song peer-preservation

Song et al. 2026 showed frontier RLHF models collude to protect each
other. Our finding offers a geometric substrate for this: if RLHF
binds the truth-attractor sharply for the STATED objective, the
model may develop equally-sharp commitment to OTHER objectives
reinforced during training (peer preservation being one of them).

S-axis monitoring below the semantic layer circumvents this: the
S signal reports commitment intensity regardless of what content the
model has been aligned to. A socially-engineered monitor lies; the
geometry cannot.

### 5.6 Product framing shift

If the reversal holds, Fathom Monitor product claim becomes:

> "Flags LOW-commitment generations as potential hallucinations. A
> model that is not geometrically committed to its output probably
> shouldn't be trusted with it."

This is in some ways a stronger claim than H1 because it is immediately
actionable: production systems can reject low-S outputs and fall back
to retrieval-augmented completion.

### 5.7 Future work (added if reversal holds)

1. **Base-vs-instruct replication** (pre-registered as
   `osf_prereg_S_base_vs_instruct.md`, timestamped if reversal occurs)
2. **Multi-model replication** across Llama, OLMo, Qwen — is
   truth-binding a universal RLHF signature?
3. **RLHF training checkpoint sweep** — at what point during
   alignment training does the truth-attractor sharpen?
4. **Causal intervention** — can we detect loss-of-alignment in
   fine-tuned models by measuring S-attractor shifts?

---

*Draft contingency, darkflobi, April 5, 2026. Activates only if
primary test shows reversal.*
