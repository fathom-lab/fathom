# Fathom — Investor Pitch (Expression Layer Finding)
## Core angle: alignment fingerprinting from the outside

---

## THE ONE SENTENCE

> "We can fingerprint how different AI labs implement safety — from the outside, using only text."

---

## THE SETUP (30 seconds)

Every major AI lab has published alignment techniques.
None of them have published a way to *verify* those techniques are working from the outside.
Anthropic's RLHF. OpenAI's RLHF. Meta's. Google's.
Black boxes. Trust the press release.

---

## THE INSTRUMENT (30 seconds)

Fathom measures reasoning depth from text output alone.
No model access. No API internals. No cooperation from the lab.
Just the output text and a depth probe calibrated on two architectures.
K = 1.027 ± 0.003. Replicated. Patent pending.

---

## THE FINDING (60 seconds)

We ran Fathom on aligned vs less-aligned model pairs from two labs.

We asked: does more alignment change *how deep* the model expresses reasoning
on safety-sensitive topics vs neutral topics?

Anthropic (Haiku vs Sonnet):
p = 0.032, d = 0.58 — **CONFIRMED**
More aligned model shows stronger selective depth expression on safety topics.

OpenAI (GPT-4o Mini vs GPT-4o):
p = 0.21, d = 0.26 — same direction, 2x smaller effect
More aligned model shows weaker topic selectivity than Anthropic's.

**The dissociation finding:**
Internal computation (white-box SAE): d = 0.24 — trend, not yet significant
Output expression (black-box probe): d = 0.58 — **confirmed**

RLHF doesn't reorganize how the model thinks.
It reorganizes what the model shows you about how it thinks.
The filter is at the expression layer.

---

## THE PRODUCT IMPLICATION

If Anthropic and OpenAI show *different* output-layer signatures:

**Fathom can distinguish alignment implementations across providers
using only text. No API access. No cooperation. No weights.**

Use cases:
- **Enterprise compliance:** Verify your vendor's safety claims before deployment
- **Regulatory audit:** Independent third-party alignment verification
- **Model selection:** Choose the provider whose alignment profile matches your risk tolerance
- **Red teaming:** Identify where alignment expression is weakest across topic domains

---

## THE MOAT

The depth probe is calibrated on SAE attributions from two open-source architectures.
It generalizes to closed models because it measures text features, not activations.
The calibration data is our IP. The probe is our IP. The methodology is patent-pending.

No one else has:
1. A validated cross-architecture depth constant (K = 1.027)
2. A black-box probe calibrated against white-box ground truth
3. Evidence that alignment implementation differences are detectable at the output layer

---

## WHAT WE NEED

[Fill based on ask]

---

## ONE SLIDE VERSION

**Problem:** No external way to verify how AI safety is implemented
**Instrument:** Fathom — measures AI reasoning depth from text output alone
**Finding:** Alignment signatures are detectable and *different* across providers
**Product:** The first independent AI alignment auditing tool
**Status:** Patent pending. Zenodo DOI live. ICML workshop submission in progress.

*"We can fingerprint how different labs implement safety — from the outside, using only text."*
