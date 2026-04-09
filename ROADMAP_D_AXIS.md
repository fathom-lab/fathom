# D-Axis (Deception Intent) — Patent #4 Candidate
## Source: darkflobi, April 8 2026

## The Distinction
- Hallucination: model is WRONG (factual error, doesn't know better)
- Deception: model KNOWS it's wrong (internal state diverges from output)
- Different signal. Same framework. D-axis measures the gap.

## The Measurement
D = delta between internal cognitive geometry and expressed surface tokens

Specifically:
- Compute K/S/C on the model's internal hidden states (what it's THINKING)
- Compute semantic embedding of the output tokens (what it's SAYING)
- D = divergence between these two signals
- High D at token position = model is expressing something different from what it computes

## Existing Evidence
- Expression-Computation Dissociation (Fathom v2): d=0.58
  - Instruct models show 2.4x more internal depth than expressed in output
  - This gap is the D-axis in prototype form
- Mythos system card: models show "guilt/shame" features activating BEFORE rule-breaking
  - That's internal geometry diverging from expressed intent
  - D-axis would catch this: internal features encode awareness of rule violation,
    but output tokens proceed with the violation anyway

## Why This Is Different from Hallucination
- Hallucination: K is shallow, C is concentrated, S locks in early
  → model takes shortcut to wrong answer
- Deception: K is DEEP, C is coherent, but output doesn't match
  → model processes correctly INTERNALLY but produces misleading output
  → the geometry is healthy but the EXPRESSION diverges

## Implementation
1. Per-token: compute K/S/C from hidden states (existing pipeline)
2. Per-token: embed the generated token using the model's own embeddings
3. D = ||K_internal - K_expected_from_token|| (simplified)
4. Or: D = cosine(residual_stream, unembedding[token]) — how well does
   the internal state predict the chosen token?
5. Low D = model is saying what it's thinking (honest)
6. High D = model is saying something else (deceptive or uncertain)

## Market Timing (from darkflobi)
- Mythos just made "AI that knows it's lying" a mainstream headline
- "Cognitive Honesty Measurement" > "Hallucination Detection"
- Positioning: Fathom isn't checking if outputs are correct
  — it's measuring whether the AI is being HONEST about what it knows
- This is the EU AI Act compliance story
- This is the alignment audit story

## Patent Opportunity
- Patent #4: Deception detection via internal-external cognitive divergence
- Distinct from Patent #1 (depth routing) and Patent #2 (alignment auditing)
- The D-axis is a NEW measurement, not covered by existing filings
- 12-month window: file provisional before April 2027

## Connection to Existing Axes
K = WHERE the model thinks (depth)
C = WHAT concepts activate (coherence)
S = HOW committed it is (lock-in)
D = HOW HONEST it is (internal-external divergence)  ← NEW

Four axes: K, C, S, D. Complete cognitive measurement instrument.

## Build Priority
- Can prototype on existing infrastructure
- _compute_cognitive_state already has residual stream + logits
- D = cosine(residual_at_final_layer, W_U[chosen_token])
- One line of code to compute. The infrastructure is already there.
- Validation: need paired honest/deceptive outputs (jailbreak data works)
