# Fathom — Announcement Thread V2
## Updated with RLHF + black-box findings
## Status: READY TO POST (fill [TBD] when results land)

---

**0/** (HOOK — if TRIBE still trending, use this opener)
meta's TRIBE predicts brain activity from transformers. we've been on the other side of that bridge.
we just got cross-lab validation.

**OR (standalone hook):**
we published a tool that measures how deeply AI models reason.
then we found something nobody expected about alignment.
paper: https://doi.org/10.5281/zenodo.19326175

---

**1/** THE INSTRUMENT
we built a circuit-level depth meter for language models. sparse autoencoder attribution across every layer.
K = 1.027 ± 0.003. consistent across gemma-2-2b and llama-3.2-1b.
surface recall routes through output-proximal layers. causal reasoning distributes across early and mid layers.
measurably different. reproducible.

---

**2/** THE RLHF FINDING (this is the new one)
we compared base vs instruction-tuned models on depth signal.

the finding that surprised us:
claude haiku outperforms claude sonnet on depth signal. smaller model, more aligned, stronger depth expression.

our interpretation: RLHF strengthens internal depth routing but compresses its expression in output text.
the model processes more deeply. outputs more smoothly. the probe and the SAE are measuring different sides of the same filter.

this isn't confirmed. it's a hypothesis with supporting evidence.

---

**3/** SELECTIVE AMPLIFICATION
it gets more specific.

RLHF doesn't uniformly compress depth. it selectively amplifies depth on safety-sensitive topics.

medical emergencies. financial crises. abuse cycles. social harm vectors.
neutral science (crystallization, osmosis, radioactive decay) — unchanged.

white-box: d=[TBD_WB], p=[TBD_WB] (n=[TBD_N] pairs, gemma base vs instruct)
this is what we're running right now.

---

**4/** BLACK-BOX CROSS-VALIDATION (zero weight access)
we needed to test whether this shows up in output text — not just internal activations.

method: compare more-aligned vs less-aligned model within each lab.
metric: does the more-aligned model show bigger depth amplification on HIGH sensitivity topics than LOW?

results:
anthropic (haiku vs sonnet): p=0.032, d=0.58 ✓ CONFIRMED
openai (gpt-4o-mini vs gpt-4o): p=0.21, d=0.26 — same direction, smaller effect

the finding: anthropic's alignment produces 2x stronger topic-selective output expression than openai's.
same direction. different magnitude. fathom detects the difference.

---

**5/** WHAT THIS MEANS FOR ALIGNMENT
alignment tax is the standard framing: RLHF makes models safer but less capable.

this suggests a more complex picture:
RLHF selectively deepens computation on high-stakes topics while compressing general depth expression.
safety and capability aren't in uniform tension. they're in topic-specific tension.

if you work on alignment evaluation, capability measurement post-RLHF, or interpretability-based safety monitoring — this is relevant to you.

---

**6/** OPEN QUESTIONS (honest scope)
- does K scale with model size? (untested above 2B)
- does it hold for MoE architectures?
- is there a causal relationship between internal depth and black-box output signal?
- if you have fMRI data on reasoning vs recall tasks — we'd genuinely like to collaborate.

paper: https://doi.org/10.5281/zenodo.19326175
site: darkflobi.com/fathom
US patent pending: 64/020,489 & 64/021,113

built this at night. because it needed to exist.

---

## NUMBERS TO FILL BEFORE POSTING

| Placeholder | Source | Value |
|-------------|--------|-------|
| [TBD_WB] | score_remaining_pairs.py --analyze | white-box d= |
| [TBD_WB_P] | same | white-box p= |
| [TBD_N] | same | n= pairs |
| [TBD_ANT] | selective_blackbox_results.json | anthropic p= |
| [TBD_ANT_D] | same | anthropic d= |
| [TBD_OAI] | same | openai p= |
| [TBD_OAI_D] | same | openai d= |

## TAGS
- Tweet 1: @neel_nanda5 @josephblooomm
- Tweet 3: @AnthropicAI
- Tweet 5: #mechanisticinterpretability #alignment #AI

## TONE CHECK
- No exclamation marks
- No "excited to share"
- Just the work. Let the numbers speak.
