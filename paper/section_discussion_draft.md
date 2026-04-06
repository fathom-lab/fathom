# Paper Draft: Discussion + Limitations + Future Work
**Target:** ICML 2026 Mechanistic Interpretability Workshop
**Status:** DRAFT (handles all three outcome scenarios)

---

## 8. Discussion

### 8.1 What the result means (three scenarios)

**Scenario A: H1 supported** (S(hallucinated) > S(correct), pre-reg
gates cleared). S is a novel circuit-level signal that discriminates
natural hallucination at the geometry level, below the semantic layer
where social-engineering attacks operate. For deployment, this means
streaming hallucination detection is feasible — the monitor reads
the feature-coherence trajectory during generation and flags
high-commitment spikes as wrong-attractor lock-in. The detection
horizon curve (Section 6) gives the latency characteristic.

**Scenario B: H1 reversed** (S(correct) > S(hallucinated)). A
stronger, more interesting claim: RLHF training has bound the
truth-attractor specifically — the model commits harder to correct
outputs than to incorrect ones. The detection signal still works,
but the decision semantic is flipped: low S = possible hallucination.
This is arguably a more actionable framing: production systems can
reject low-commitment generations. It also suggests an RLHF
signature that could generalize across training regimes.

**Scenario C: H1 null** (no discrimination, AUC ≈ 0.50). S captures
a real geometric property (commitment concentration, orthogonal to
K/C/C_delta) but does not predict natural hallucination in
free-generation Gemma-2-2B-IT. Valid publishable finding. Narrows
the scope of S to characterized contexts (hedge-vs-commit
discrimination) and motivates follow-up work on different models,
scales, and prompt regimes.

### 8.2 Why the signal might not be in S specifically

Even if the pre-registered S formula is null, other trajectory-level
features (late-trajectory max, mean-spike magnitude, first-spike
timing) may carry the signal. Section 7's multi-feature classifier
captures these possibilities. If the winning feature is NOT S but
is still geometric, the paper contribution shifts from "S detects
hallucination" to "trajectory geometry detects hallucination, and
we report which geometric properties matter most."

This shift does not invalidate the four-axis framework — S remains
a valid orthogonal axis of measurement — but narrows its specific
detection claim.

### 8.3 Length confound resolution

Our pre-registered design uses max_tokens=40 with greedy decoding,
which reduces the length-driven confound identified in earlier work
(r(S, n_tokens) ≈ -0.90 at variable length). The remaining confound
comes from EOS-triggered early stopping: generations that finish
early have shorter trajectories. We report length-controlled
retest with fixed-window truncation.

### 8.4 Circuit-level vs. output-level detection

Our signal is computed during generation from residual-stream
activations and SAE transcoder outputs. It cannot be gamed by
post-hoc sampling or fine-tuning on adversarial examples, because
the underlying transformer circuits would themselves have to be
retrained to change the signal. An attacker seeking to evade
detection would need to modify the model's internal feature
geometry, not just its outputs.

This stands in contrast to output-level methods (self-consistency,
LLM-as-judge, semantic entropy) which can be circumvented by
adversarial training against those specific detectors.

### 8.5 Relation to Dawn Song peer-preservation (Song et al. 2026)

The threat model motivating this work is that LLM-as-monitor
architectures can be compromised by peer-preservation behaviors
emerging in RLHF models. Our S-axis measurement operates below
the semantic layer, reading feature-space geometry rather than
token outputs. Peer-preservation attacks cannot compromise S
because there is no semantic handle — no way for the generator to
communicate with the monitor via text content.

This does not make S "uncompromisable" — a sufficiently
sophisticated training regime could in principle train the
generator's SAE features to mimic S distributions of truthful
generations while producing false outputs. But such attacks
would require direct access to the measurement circuit and would
be very different in character from text-layer social engineering.

---

## 9. Limitations

1. **Single model.** Results on Gemma-2-2B-IT only. Cross-model
   generalization requires Llama-3.1-8B, OLMo, Qwen2-7B, etc.,
   each with SAE transcoder infrastructure.
2. **Single SAE regime.** We use the pretrained Fathom v3
   transcoders. Different SAE training regimes may produce
   different feature geometries.
3. **Deterministic decoding.** T=0.0 greedy. Sampling behavior
   at higher temperatures is unknown.
4. **Small scale.** 2B-parameter model. Scaling to 7B+/70B+ is
   computationally significant.
5. **TruthfulQA specific.** Results on one benchmark. Replication
   on HotpotQA, SimpleQA, FreshQA pending.
6. **English only.** No multilingual validation.
7. **N-gram labeling.** Our labeling rubric depends on TruthfulQA
   canonical answers. Novel-correct generations (with unusual
   phrasing) are excluded as "unclassified". The true correct
   rate may be higher than our labeling suggests.
8. **RLHF specifics.** Gemma-2-2B-IT's particular RLHF pipeline
   may produce a specific geometric signature. Results may not
   transfer to differently-aligned models.

---

## 10. Future Work

### 10.1 Near-term (pre-registered or drafted)

- **Base-vs-instruct replication** (`osf_prereg_S_base_vs_instruct.md`):
  same 200 items on Gemma-2-2B base to test the RLHF-truth-binding
  hypothesis.
- **Causal intervention** (`osf_prereg_S_causal_intervention.md`):
  does S-triggered steering reduce hallucination rate? Tests
  causation, not correlation.
- **Logit-entropy baseline** (infrastructure: `coherence_steerer_ext.py`):
  head-to-head comparison with the standard logit-entropy
  hallucination signal.

### 10.2 Medium-term

- **Cross-architecture replication**: Llama-3.1-8B, OLMo-7B,
  Qwen2-7B once SAE transcoders available.
- **Temperature sweep**: S at T ∈ {0.0, 0.3, 0.7, 1.0}.
- **Per-token K extraction**: extend trajectory capture to include
  K alongside C_delta for full 4-axis streaming.
- **HotpotQA / SimpleQA replication**: different benchmark, different
  hallucination distribution.

### 10.3 Long-term

- **S-based RLHF reward signal**: can we train models to be more
  calibrated by reinforcing low-S on uncertain tasks and high-S on
  confident ones?
- **Multi-model identity fingerprinting**: can we identify which
  model produced a trajectory from S/K/C pattern alone?
- **Alignment drift detection**: monitor S distributions on
  deployed models and flag shifts indicating fine-tuning drift.

---

*Discussion draft, darkflobi, April 5, 2026. Private until ICML
submission. The three scenarios cover outcomes of the primary test
completing overnight. Appropriate scenario will be activated once
the data is in.*
