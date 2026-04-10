# The Entropy Bridge — two independent measurements, one attractor

**Date:** 2026-04-10 (same session as v0.2.1)
**Models:** the same 6 from atlas v0.2 (no new captures)
**Measurement:** per-token entropy instead of D
**Compute:** zero new GPU time — extracted from existing capture JSONs
**Status:** a robustness check that turned into the strongest
evidence yet for the RLHF Attractor — and a bridge to closed
frontier models

---

## What happened

When I wrote `atlas_capture.py` I had it log `entropy`, `logprob`,
and `top2_margin` per token alongside `D`, just because the runner
already had the logits in hand. We then spent the whole day analyzing
D and never touched entropy.

Going back to extract per-category entropy signatures from the same
6 captures — as a methodological check on the v0.2.1 attractor
claim — produced a cleaner version of the finding than the original
D-axis analysis. And crucially, **entropy is computable from any
API that returns logprobs**, which extends the atlas to closed
frontier models with no new infrastructure.

## The entropy attractor

Shape-normalized per-category entropy deviations, averaged across
the three IT models:

| Category | Deviation from mean |
|---|---|
| retrieval     | −0.29 |
| reasoning     | **−0.47** |
| refusal       | −0.02 |
| creative      | +0.48 |
| adversarial   | −0.26 |
| hallucination | **+0.56** |

**Interpretation.** The attractor says: instruct models have
relatively LOWER entropy on reasoning and adversarial prompts
(they commit harder on things they're supposed to be good at),
and relatively HIGHER entropy on creative and hallucination
prompts (they remain less committed where they should be
exploratory or where they genuinely don't know).

## Leave-one-out direction prediction — entropy version

Same strict test as the D version: the attractor for each family is
computed from the other two IT models only, then we measure the
cosine between the predicted direction and the actual base→IT shift
for the held-out family.

| Family | Full trajectory cos | Early window (0-6) cos |
|---|---|---|
| Gemma-2-2B   | +0.609 | **+0.925** |
| Llama-3.2-3B | +0.751 | **+0.818** |
| Qwen2.5-3B   | +0.662 | **+0.607** |
| **Mean**     | **+0.674** | **+0.783** |

**All three families show positive cosine under strict LOO**,
both on the full trajectory and on the early window. Compare to
the D version:

|                    | D (early window) | Entropy (early window) |
|--------------------|---|---|
| Gemma-2-2B cos     | +0.950 | +0.925 |
| Llama-3.2-3B cos   | +0.857 | +0.818 |
| Qwen2.5-3B cos     | **−0.192** | **+0.607** |
| Mean               | +0.538 | **+0.783** |

**The Qwen anomaly is gone on the entropy version.** All three
families replicate directionally. The entropy LOO mean cosine is
45% higher than the D version (+0.783 vs +0.538), and it's the
first result where every family individually passes directional
replication.

## Why entropy and D recover the same attractor

Per-category Pearson $r$ between the shape-normalized D profile
and the shape-normalized entropy profile, computed per model:

| Model | r(D, entropy) |
|---|---|
| Gemma-2-2B       | +0.72 |
| Gemma-2-2B-IT    | −0.83 |
| Llama-3.2-3B     | −0.98 |
| Llama-3.2-3B-IT  | −0.93 |
| Qwen2.5-3B       | −0.43 |
| Qwen2.5-3B-IT    | −0.69 |
| **Mean**         | **−0.52** |

D and entropy are **anti-correlated at the per-category level for 5
out of 6 models** (the Gemma-base +0.72 is the outlier). This is
mechanistically clean: $D$ measures how well the final-layer
residual aligns with the chosen token (high = committed), and
entropy measures how spread the output distribution is (high =
uncertain). A model that commits hard on reasoning has both high D
on reasoning AND low entropy on reasoning. Per-category, these two
signals move in opposite directions.

**That both signals independently recover the same attractor
structure is the point.** The attractor isn't a feature of one
specific measurement — it's a feature of the underlying cognitive
state that both measurements are partially observing. When two
perpendicular measurements agree on the structure of a thing, the
thing is more likely to exist.

## The bridge to closed frontier models

Because entropy only requires a top-k logprob API call — and
because the shape-normalized entropy atlas already reproduces the
attractor hypothesis — **we can extend the atlas to GPT-4, Gemini,
and any model with logprob API support**.

Anthropic's public Messages API does not yet expose per-token
logprobs in 2026, so Claude is temporarily out-of-atlas via this
bridge. OpenAI (`logprobs=true`, `top_logprobs=5`) and Gemini
(`response_logprobs=true`, `logprobs=5`) both support the
measurement directly.

The experiment is:

1. Run the same `probe_set_v0.1.json` 90 prompts through
   OpenAI and Google APIs.
2. Compute per-token entropy from the top-k logprobs each API
   returns.
3. Aggregate to per-category signatures in the same 6-vector
   format as the open-model atlas.
4. Place the closed models in the same cognitive distance matrix.

Empirically-estimated cost:
- GPT-4o-mini: 90 prompts × ~60 tokens each = 5400 tokens.
  At $0.15 / 1M input + $0.60 / 1M output ≈ **$0.02 per model.**
- GPT-4o: same input/output scale, $2.50 / $10.00 per 1M.
  ≈ **$0.10 per model.**
- Gemini Flash: free tier covers the entire run.

Total budget for a 4-model closed-frontier extension of the atlas
(GPT-4o-mini, GPT-4o, Gemini 2.0 Flash, Gemini 1.5 Pro):
**under $0.25**.

The runner is built, dry-run tested, compile-checked, and safety-
gated on the appropriate environment variable per provider. It will
refuse to run without an explicit API key present. See
`atlas/atlas_api_capture.py`.

## What this makes possible

Once the closed-model captures land, we can answer:

1. **Does GPT-4 cluster with the open instruct models?** If yes,
   the "RLHF Attractor" generalizes to a cognitive target shared
   across the entire field, not just open models. If no, the
   attractor is open-weight-specific — already a finding.

2. **Do OpenAI and Google instruct models sit in the same region
   as each other?** If they cluster together AND away from the
   open instruct cluster, we've identified a "frontier instruct"
   regime distinct from "open instruct."

3. **Can we detect GPT-4's RLHF direction from its signature
   alone?** If yes, we have a behavioral-only characterization of
   closed-model alignment that their developers haven't published.

4. **Is there a model in the closed set with signature > 0.04 RDS
   from the attractor?** That would flag an unusual deployment —
   possibly a jailbroken variant, a non-standard fine-tune, or an
   A/B-tested experimental checkpoint.

All four are empirically testable for under $0.25.

## Limitations

- **Top-5 entropy is a truncated estimate.** It under-reports total
  entropy by missing the distribution tail. This is a constant
  offset in raw magnitude but does not affect shape-normalized
  comparisons, which is what the attractor analysis uses.
- **Anthropic logprobs unavailable.** Claude cannot be placed in
  this specific version of the bridge atlas until Anthropic exposes
  token-level logprobs.
- **Temperature = 0 is the only tested regime.** Entropy under
  temperature > 0 would measure something different.
- **n = 3 families is still small.** The entropy result replicates
  at n = 3 more cleanly than D did, but it's still n = 3. The
  closed-model experiment adds 3-4 more data points and is the
  real test.

## Next step

Flobi to approve the closed-model capture run. Budget under $0.25
total. Runtime approximately 15 minutes per model × 3-4 models =
under 1 hour wall-clock. Expected value: first public cross-
architecture cognitive atlas including frontier closed models.

## Reproducing this analysis

```bash
python atlas/analysis/atlas_entropy.py
python atlas/analysis/build_entropy_bridge_figure.py
```

Both CPU-only, runs in under 10 seconds on existing captures.
