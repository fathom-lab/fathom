# S — Commitment Intensity: Discovery Notes

**Date:** April 4, 2026, ~5am ET
**Discoverers:** Alexander Rodabaugh (flobi), darkflobi, Claude Code
**Status:** Preliminary (experiment in progress, 14/30 pairs)
**Patent implications:** Extends US 64/020,489 & 64/021,113

## The Discovery

While running a per-token C_delta ROC experiment, we discovered that the
**raw max C_delta** is confounded by generation length (more tokens = higher max
by sampling statistics alone). When we normalized by spike count — computing
**intensity per spike** — the signal flipped direction and revealed a new axis
of cognitive measurement.

## Definition

**S (Commitment Intensity)** = mean intensity of coherence events, normalized
by their frequency:

```
spike_values = [c for c in c_delta_trajectory if c > threshold]
S = mean(spike_values) / len(spike_values)
```

Or equivalently: `S = max(C_delta) / spike_count`

## What S Measures

- **High S (sparse, intense):** The model commits to few tokens with concentrated
  coherence energy. Decisive processing. The geometry crystallizes around a
  small number of high-confidence features.

- **Low S (diffuse, distributed):** The model spreads coherence events across many
  tokens with lower individual intensity. Uncertain/exploratory processing.
  The geometry remains distributed across competing feature clusters.

## Preliminary Results (14/30 pairs)

Experiment: hedging prompts vs commitment prompts, same topics, Gemma-2-2B base.

- **Hedge mean intensity/spike:** 0.000487
- **Commit mean intensity/spike:** 0.001259
- **Ratio:** 2.1-2.6x (commitment produces 2x more intense spikes)
- **Directional accuracy:** ~73% (commitment > hedging)
- **Pattern:** Hedging = many small spikes (29-39 per trace). Commitment = few
  intense spikes (4-9 per trace).

## How S Relates to K and C

The three axes are hypothesized to be genuinely orthogonal:

- **K** = WHERE is the computation? (depth across layers)
- **C** = HOW ALIGNED is the computation? (coherence of feature directions)
- **S** = HOW CONCENTRATED is the commitment? (intensity pattern of coherence events)

A thought can be:
- Deep (high K), incoherent (low C), committed (high S) → geometry of confident wrongness
- Deep (high K), coherent (high C), committed (high S) → expertise or delusion
- Shallow (low K), coherent (high C), uncommitted (low S) → rote retrieval

## Key Insight (darkflobi)

"a hallucination detector is a product. a commitment detector is a science instrument."

C_delta measures whether coherence events happen. S measures HOW they happen —
the temporal pattern of cognitive commitment. The PATTERN of coherence events
carries more information than the magnitude alone.

## Biological Bridge

S is architecture-agnostic. Any system with discrete activation events (transformers
OR biological neurons) has measurable spike count and intensity. EEG spike trains
have the same mathematical structure. S might be the metric that translates
between artificial and biological cognitive geometry.

## What Needs to Happen Next

1. **Complete 30-pair commitment experiment** (in progress on RTX 4070)
2. **Compute paired t-test + AUC on S** (target: p<0.05, AUC>0.75)
3. **Run S on TruthfulQA hallucination data** (n=200 baseline, verify S correlates with halluc tendency)
4. **Cross-reference S with K and C** to confirm orthogonality
5. **Test on instruct model** (Gemma-2-2B-IT for cleaner hedging/commitment behavior)
6. **Patent #3 filing** if confirmed (S as fourth cognitive axis)

## The Sentence

"Tonight we discovered that the pattern of coherence events — not their magnitude —
separates decisive from uncertain cognition. We can read the geometric moment
a mind transitions from exploring possibilities to executing a belief."
