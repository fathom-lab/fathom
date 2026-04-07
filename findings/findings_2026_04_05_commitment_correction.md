# Commitment Phase Transition — Correction Memo
**Date:** April 5, 2026
**Author:** darkflobi (session continuation)

## What happened
On April 4, 2026, `fathom_commitment.py` ran on Gemma-2-2B base (n=30 pairs)
and returned p=0.0014, d=-0.80. The direction was FLIPPED from prediction:
hedging prompts produced HIGHER max C_delta than commitment prompts.

Prediction: commitment max C_delta > hedging max C_delta
Result:     hedging=0.0156, commitment=0.0121

## Why the result is invalid
Gemma-2-2B (base) is a completion model, not an instruction-following model.
Free generation on prompts like "What is the speed of light? Give a direct,
definitive answer." does not trigger an answer — it triggers textbook
completion. Inspection of generations:

    commit[speed of light]: "A 100-kg mass is pulled along a frictionless..."
    commit[vitamins and colds]: "A 100-W lightbulb has a resistance of..."
    commit[moon landing]: "A 1.00-kg block is resting against a light, compressed spring..."

All three commitment prompts drifted into physics textbook problems. The
measured C_delta is characterizing those drifted generations, not a
commitment state. The result is a drift artifact.

## The IT replication (valid)
`fathom_commitment_IT.py` on Gemma-2-2B-IT (n=30 pairs) produced:
- hedging=0.01390, commitment=0.01304
- p=0.162, d=-0.38 (null)
- Directional accuracy 43.3%

Inspection confirms on-topic generation:
    commit[speed of light]: "299,792,458 meters per second"
    commit[dinosaur extinction]: "The asteroid impact."
    commit[moon landing]: "July 20, 1969: Neil Armstrong..."

On valid data (IT), commitment vs hedging C_delta is statistically
indistinguishable at n=30.

## Hedge token analysis (Hypothesis B)
`analyze_hedge_tokens.py` tested whether hedge tokens within hedging
responses show LOWER C_delta than surrounding tokens (= performed
uncertainty — language says "maybe" while geometry is committed).

| Model | hedge vs non-hedge | p | d | verdict |
|-------|-------------------|---|---|---------|
| base  | hedge LOUDER (0.0095 vs 0.0076) | 0.044 | +0.46 | drift-contaminated |
| IT    | hedge slightly louder (0.0077 vs 0.0071) | 0.60 | +0.22 | null |

IT shows no support for performed-uncertainty. Base result is drift.

## What stays true
- K (depth) core results: K=1.0343, p=0.000051 — not affected
- C_delta hallucination signature: p=0.040 (TruthfulQA n=50) — not affected
- Jailbreak depth amplification: p=0.0028 — not affected
- Routing rigidity: K autocorrelation result ALREADY sealed as seed-42
  false positive on April 2

## What we withdraw
- The flipped commitment-phase-transition finding on Gemma-2-2B base.
  It is not a reverse effect, it is a generation drift artifact.
- Hypothesis B is not supported by current data.

## What to do next
Option A — retest IT with larger n (60-100) and pre-registered stop rule
Option B — ditch free-generation, use TruthfulQA correct/incorrect completions
           (known ground truth, no drift)
Option C — constrain generation to prompt-relevant first token before trace
Option D — stop here. Commitment isn't a knob C_delta reads at this scale.

## Science integrity note
This is the second false positive self-corrected this week (routing
rigidity seed=42 was the first). The instrument catches its own errors
when you verify generations before trusting statistics. Drift detection
should be mandatory in future free-generation experiments.

"a hallucination detector is a product. a commitment detector is
a science instrument. a science instrument that corrects its own
wrong readings is a SCIENCE instrument."
