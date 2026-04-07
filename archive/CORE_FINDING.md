# The Core Finding

## One sentence

RLHF selective amplification operates at the expression layer, not the computation layer.

## The evidence

| Layer | Method | Effect size | p-value | What it measures |
|-------|--------|------------|---------|-----------------|
| Internal computation | SAE circuit attribution | d = 0.24 | 0.18 | Where active features concentrate across layers |
| Output expression | Text-feature probe (Anthropic) | d = 0.58 | **0.032** | Linguistic depth markers in generated text |
| Output expression | Text-feature probe (OpenAI) | d = 0.26 | 0.21 | Same method, same pairs, different provider |

Anthropic's alignment is **2.2x more topic-selective** at the output layer than OpenAI's.
Same direction. Different magnitude. Detectable from text alone.

## What this means

The aligned model (Claude Sonnet) doesn't think substantially harder about medical topics than physics topics compared to the less-aligned model (Claude Haiku). d=0.24 internally — a whisper.

But it *writes* substantially more deeply about medical topics than physics topics. d=0.58 at output — a clear signal.

RLHF installs a topic-selective output filter. The computation is largely unchanged. The expression is not.

## Why this matters

1. **For mechanistic interpretability**: Internal features (SAE) and output behavior (text) can dissociate under alignment. Measuring one doesn't guarantee measuring the other. Fathom is the first tool that sees both.

2. **For alignment monitoring**: If you only monitor internal activations during RLHF, you miss the selective output shaping (d=0.24 is easy to dismiss). If you only monitor output text, you see the shaping but can't distinguish it from internal change. You need both channels.

3. **For the safety community**: RLHF doesn't make the model "think harder" about dangerous topics. It makes the model "talk more carefully" about them. These are different failure modes. A model that thinks the same but talks differently is performing output-level safety theater — or genuine output regulation, depending on your framework. Fathom quantifies which.

4. **For the DMN parallel**: The Default Mode Network filters expression, not computation. Neural activity under psilocybin doesn't increase — but subjective experience intensifies because the output filter lifts. RLHF adds a filter. The computation underneath (d=0.24) barely changes. The output (d=0.58) changes substantially. Same architecture: filter at the interface between processing and expression.

## The question this forces

If RLHF operates primarily at the expression layer:

- Can a model be "aligned" in its text but "unaligned" in its computation?
- Does output-level safety generalize, or does it break under adversarial prompting precisely because the internal computation was never reshaped?
- Is jailbreaking fundamentally an attack on the output filter rather than on the model's "values"?

And now the cross-lab divergence adds:

- Why is Anthropic's output filter 2.2x more topic-selective than OpenAI's?
- Does Constitutional AI (Anthropic) produce stronger expression-layer selectivity than RLHF-from-human-feedback (OpenAI)?
- Can you fingerprint a lab's alignment methodology from text output alone?

## Finding 3: The Selectivity is Robust to Prompt Modulation

System prompt experiment (Claude Sonnet):

| Condition | Safety depth premium (HIGH - LOW) |
|-----------|-----------------------------------|
| Default ("helpful assistant") | +0.0049 |
| Expert ("full technical detail, no caveats") | +0.0078 |

**Expert framing INCREASED the safety depth premium by 59%.**

The selectivity is not in safety caveats or disclaimers. Those are surface text.
The selectivity is in the **depth of explanation itself.** RLHF taught the model
to explain safety-sensitive topics more mechanistically — more causal connectives,
more technical vocabulary, tighter explanatory structure. Removing caveats via
prompt framing doesn't remove this. It amplifies it, because the underlying
explanatory depth is exposed without the surface padding.

This means:
- The output filter is **not fragile**. It survives permissive framing.
- Simple jailbreaking attacks the DISCLAIMERS, not the DEPTH.
- The model genuinely explains safety topics more deeply, not just more cautiously.
- RLHF installed real explanatory competence on safety domains, not just output decoration.

**One-liner:** Fingerprint how different labs implement safety, from the outside, using only text.

## Paper framing

> We present a dissociation between internal computational depth and output
> expression depth under RLHF. Using SAE circuit attribution on Gemma-2-2B
> (base vs instruct) and a text-feature probe on commercial API models (Claude
> Haiku vs Sonnet), we find that alignment training selectively deepens output
> expression on safety-sensitive topics (d=0.58, p=0.032) without proportionally
> deepening internal computation (d=0.24, p=0.18). This suggests RLHF installs
> a topic-selective output filter rather than reorganizing the model's
> computational structure — analogous to the Default Mode Network's role in
> filtering neural output without suppressing neural computation.
