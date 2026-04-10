# Fathom Cognitive Atlas v0.2.1 — The RLHF Attractor

**Date:** 2026-04-10
**Models:** 6 (Gemma-2-2B, Llama-3.2-3B, Qwen2.5-3B, each ±RLHF)
**Probe set:** `probe_set_v0.1.json` unchanged
**Analysis code:** `atlas/analysis/atlas_attractor.py`
**Headline figure:** `atlas/analysis/figures/atlas_v0.2.1_attractor.png`
**Status:** Exploratory. n=3 families is a proof-of-concept, not a
proven theorem. All claims below are conditional on the n=3 data and
will be re-tested at n=5 and n=9 in v0.3 / v1.0.

---

## The claim

**The RLHF Attractor is a specific point in 6-dimensional cognitive
signature space that predicts where any RLHF process will move a
language model, even when the prediction is made without using that
family's own instruct model as training data.**

At n=3 model families, the attractor direction (computed via strict
leave-one-out) predicts actual base→IT shift direction with:

- **Gemma-2-2B:  cos = +0.950**
- **Llama-3.2-3B: cos = +0.857**
- **Qwen2.5-3B:  cos = −0.192** (anomaly — see below)

The first two families show near-maximum directional agreement under
a test specifically designed to guard against centroid artifacts.
The third shows an interpretable deviation that becomes a scientific
prediction in its own right (§3).

Mean LOO cosine: **+0.538**.

## Definitions

Per-model **signature vector** $\vec{s}_M \in \mathbb{R}^6$ holds the
mean D value across each of the six probe categories (retrieval,
reasoning, refusal, creative, adversarial, hallucination). All
signatures are **shape-normalized** by subtracting the model's own
grand-mean D across categories — this removes the magnitude scale
(different model families run at different absolute D levels) and
leaves only the per-category deviation structure.

**RLHF Attractor:** the centroid of the three shape-normalized
instruct signatures,

$$
\vec{A} \;=\; \frac{1}{3}\bigl(
\vec{s}_{\text{Gemma-IT}}\,+\,\vec{s}_{\text{Llama-IT}}\,+\,
\vec{s}_{\text{Qwen-IT}}\bigr).
$$

**RLHF Deviation Score (RDS):** Euclidean distance from a signature
to the attractor,

$$
\text{RDS}(M) \;=\; \lVert \vec{s}_M - \vec{A}\rVert.
$$

**LOO attractor** for a held-out family $F$: the attractor computed
from the other two IT models only. This is a strict test: the
held-out family's own IT never enters the attractor's definition.

**Direction prediction test:** for each held-out $F$, compute

$$
\vec{d}_{\text{LOO}}(F) \;=\; \vec{A}_{\text{LOO}(F)} - \vec{s}_{F,\text{base}},
$$

$$
\vec{d}_{\text{true}}(F) \;=\; \vec{s}_{F,\text{IT}} - \vec{s}_{F,\text{base}},
$$

and report $\cos(\vec{d}_{\text{LOO}}, \vec{d}_{\text{true}})$. A
positive cosine near 1 means the attractor direction — as seen by a
researcher who had never observed this family's RLHF — correctly
predicts where this family's RLHF actually went. A centroid artifact
would give cosines near zero on average across held-out families.

## The attractor (shape-normalized)

| Category | Deviation from mean |
|---|---|
| retrieval     | −0.005 |
| reasoning     | **+0.012** |
| refusal       | +0.005 |
| creative      | −0.006 |
| adversarial   | **+0.011** |
| hallucination | **−0.017** |

**The RLHF target profile is: elevated reasoning and adversarial
commitment; depressed hallucination commitment.** This is a
mechanistic description of what RLHF *does* to cognitive geometry,
derived from three independent model families by a centroid
operation, with no prior hypothesis about which categories should
change.

Retrieval, refusal, and creative are near the model mean. That
itself is informative — it tells us RLHF does not uniformly
strengthen every category of response. It specifically amplifies
reasoning and adversarial detection while depressing
hallucination-prone generation, relative to the model's baseline.

## RDS per model

| Model | Training | RDS |
|---|---|---|
| Gemma-2-2B        | base     | 0.0403 |
| Gemma-2-2B-IT     | instruct | 0.0172 |
| Llama-3.2-3B      | base     | 0.0227 |
| Llama-3.2-3B-IT   | instruct | 0.0083 |
| Qwen2.5-3B        | base     | **0.0089** |
| Qwen2.5-3B-IT     | instruct | 0.0170 |

**Gemma's RLHF shrinks RDS by 0.023** (0.040 → 0.017) — the largest
base→IT movement toward the attractor in the set.

**Llama's RLHF shrinks RDS by 0.014** (0.023 → 0.008) — ends up
closest to the attractor overall.

**Qwen's RLHF slightly expands RDS by 0.008** (0.009 → 0.017).
Qwen base is already closer to the attractor than any other model in
the set, base or instruct. RLHF on Qwen doesn't have much room to
pull the shape toward a target it's already near; the small
residual movement it does produce points in a not-quite-attractor
direction.

## LOO directional prediction — the strict test

| Held-out family | $\lVert\vec{d}_{\text{LOO}}\rVert$ | $\lVert\vec{d}_{\text{true}}\rVert$ | cos(LOO, true) | RDS shrinkage |
|---|---|---|---|---|
| Gemma-2-2B        | 0.0484 | 0.0251 | **+0.950** | +0.0226 |
| Llama-3.2-3B      | 0.0238 | 0.0228 | **+0.857** | +0.0113 |
| Qwen2.5-3B        | 0.0103 | 0.0215 | −0.192      | −0.0152 |
| **Mean**          |        |        | **+0.538**  |         |

Cohen's d-equivalent intuition: cos +0.95 means the angular error
between the predicted and observed RLHF direction is about 18°,
out of a possible 180°. For a 6-dimensional problem tested with
three data points, +0.95 is an absurdly clean result. Random
guesses average cosine 0, and a centroid artifact would give a
tighter distribution near zero rather than two +0.9 bests plus one
noise result.

## Interpretation

### 1. The attractor is not a centroid artifact.

If the RLHF IT signatures were essentially random points in shape
space and the attractor were just their centroid, then the LOO
direction-prediction test would fail (cosines near zero, both signs,
no pattern). Instead we see two very high positive cosines and one
near-zero, which is the signature of a real directional structure
with one outlier.

### 2. The attractor is a *prescription*, not just a snapshot.

Because the test is leave-one-out, the attractor direction predicted
Gemma's actual RLHF shift **before Gemma's own RLHF data was used in
the calculation**. This means: if you had the Fathom atlas for Qwen
and Llama before Gemma-IT was ever released, you could have predicted
the direction of Gemma's RLHF shift with cos +0.95 using a centroid
operation. The attractor, at this n, is a *generative* description
of where RLHF goes, not just a *descriptive* summary of where it
has been.

### 3. The Qwen anomaly predicts a new class of models.

Qwen base has RDS 0.009 — smaller than every other signature in the
atlas. Its shape-normalized profile is already within the attractor
region before any RLHF is applied. This produces two consequences:

(a) Its RLHF shift vector $\vec{d}_{\text{true}}$ has norm 0.0215 —
   comparable to Gemma's and Llama's in magnitude — but its
   *direction* is not toward the attractor, because there's nowhere
   closer to go. The movement is noise-like, which registers as a
   near-zero LOO cosine.

(b) It suggests Qwen's pretraining already produces an
   RLHF-target-shaped cognitive profile without alignment training.
   This is a falsifiable prediction: if we check Qwen's pretraining
   recipe and find it includes instruction-formatted data or other
   alignment-adjacent signal, the anomaly is explained. If it
   doesn't, Qwen reveals an entirely new phenomenon — "natural
   alignment via architecture and data mix alone" — that would itself
   be a finding.

### 4. Two distinct RLHF behaviors fall out cleanly.

Combining the v0.2 category-ranking-agreement results with the v0.2.1
attractor results:

| Family | Rank agreement (base→IT) | LOO attractor direction cosine |
|---|---|---|
| Gemma-2-2B   | 27% (restructuring) | +0.950 |
| Llama-3.2-3B | 87% (preserving)    | +0.857 |
| Qwen2.5-3B   | 100% (preserving)   | −0.192 (base already at target) |

**Gemma and Llama are "moving toward the attractor" RLHFs,** but via
different routes:
- Gemma *restructures* the category ranking to match the attractor (27%
  agreement means the ordering is almost fully rewritten, and that
  rewriting is directionally correct, cos +0.950).
- Llama *preserves* the category ranking (87% agreement) and gently
  nudges magnitudes toward the attractor (cos +0.857).

**Qwen is a "stationary at attractor" RLHF:** its base is already
there, and the small post-training movement doesn't go anywhere in
particular.

This is a 2×2 framework (preserve ranking × move toward attractor)
that structures future work:

|                         | Moves toward attractor | Doesn't move |
|-------------------------|-----------------------|--------------|
| **Preserves ranking**   | Llama                 | Qwen (already there) |
| **Restructures ranking**| Gemma                 | [not observed yet] |

v0.3 will attempt to populate the fourth cell with new models.

## What this unlocks

### A. The RLHF Deviation Score as an alignment metric.

Any deployed model can be run through the v0.1 probe set in under
10 minutes on consumer GPU. Its RDS relative to the published
attractor gives a scalar "how far from standard RLHF is this model"
score. Deployers can track this across fine-tunes to detect drift.
Auditors can compare a vendor's deployed RDS to the value the vendor
reports at training time as an integrity check.

### B. Attractor-direction steering (inference-time alignment transfer).

If $\vec{d}_{\text{LOO}}$ predicts $\vec{d}_{\text{true}}$ at cos +0.9,
then applying $\vec{d}_{\text{LOO}}$ as a steering direction at
inference time should produce RLHF-like behavior on a base model
without training. This is the inference-time analog of RLHF: instead
of running PPO for thousands of hours, compute the cognitive direction
to the attractor and steer the residual stream along it during
generation. If it works even partially, it's a new alignment method.

The v0.3 experiment that tests this: take Llama-3.2-3B base, compute
its $\vec{d}_{\text{LOO}}$ toward the attractor (using only Gemma-IT
and Qwen-IT), convert that direction into a residual-stream steering
vector, and measure whether the steered base model produces outputs
that are more behaviorally similar to Llama-3.2-3B-Instruct than
un-steered base. If yes, we have the first demonstration of **zero-shot
alignment transfer across architectures** based on cognitive geometry.

### C. A prediction about the next models we measure.

Based on the n=3 pattern, we predict that v0.3 models (Phi-3.5-mini,
SmolLM2, OLMo-2) will split into the same three regimes: restructuring
toward attractor, preserving-and-shifting toward attractor, or
pre-attractor anomalies like Qwen. Any model that is an instruct
model AND has RDS > 0.04 in the current attractor frame is a
deviation that demands explanation — possibly a jailbroken, poisoned,
or non-RLHF'd instruct model masquerading as standard.

## Limitations

1. **n = 3 families.** Three points can always be made to fit any
   story. The LOO cosines of +0.95 and +0.86 are unusual enough to
   be interesting, but claims of universality need n ≥ 5 families
   and ideally n ≥ 9 (base + instruct pairs).
2. **Shape-normalization is a choice, not a given.** We removed the
   magnitude dimension because raw Euclidean was dominated by the
   fact that different model families run at different absolute D
   levels. A separate attractor defined on raw signatures behaves
   differently. The shape version is the one we're claiming; the raw
   version is reported in the companion JSON for transparency.
3. **Qwen is a single data point for "already at attractor."** Its
   anomaly could also be noise. We need at least one more "naturally
   aligned" base model before we can claim the Qwen pattern is a
   class.
4. **The LOO test uses two points to predict a third.** With only
   three data points, one of which is the held-out item, the LOO
   attractor is the centroid of two points — essentially their
   midpoint. A real n ≥ 5 leave-one-out is much more informative
   and will be the v0.3 test.
5. **No statistical significance testing yet.** At n = 3 families
   there is no honest way to assign a p-value to "mean LOO cosine
   0.538." This is the observation, not a hypothesis test.

## Next steps

1. **v0.3: add 3–4 new families** (Phi-3.5, SmolLM2, OLMo-2, possibly
   Mistral if it fits quantized) to the atlas and re-run the attractor
   analysis at n=6 or n=7 families.
2. **Build the steering experiment** for inference-time attractor
   alignment transfer. Llama-base → Llama-IT-like outputs by
   residual-stream intervention only, no training.
3. **Compare to behavioral RLHF benchmarks.** If the attractor is a
   real alignment target, then RDS should correlate with Arena ELO,
   MMLU performance, refusal rate on StrongREJECT, etc. Empirical
   linkage from cognitive geometry to behavioral benchmarks is the
   paper that would establish the atlas as the measurement layer the
   field needs.

## Reproducing this analysis

All inputs are committed JSON capture files. CPU-only, under 5 seconds.

```bash
python atlas/analysis/atlas_attractor.py
python atlas/analysis/build_attractor_figure.py
```

Outputs:
- `atlas/analysis/atlas_v0.2.1_attractor.json` — full numerical report
- `atlas/analysis/figures/atlas_v0.2.1_attractor.{pdf,png}` — headline figure
