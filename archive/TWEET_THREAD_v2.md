# Tweet Thread v2 — The Dissociation Finding

---

**1/**
We built a tool that measures reasoning depth inside AI models. Then we found something we didn't expect.

RLHF doesn't change how deeply the model thinks. It changes how deeply it talks.

Paper: doi.org/10.5281/zenodo.19326175

Thread on what we found:

---

**2/**
The tool is called Fathom. It uses SAE circuit attribution to measure where a model's active features concentrate during processing.

Surface recall → late layers.
Causal reasoning → distributed across mid-layers.

The ratio (K) is consistent across architectures:
Gemma-2-2B: K=1.0302, p=0.000051
Llama-3.2-1B: K=1.0242, p=0.003

---

**3/**
We also built a text-feature probe — 13 linguistic features that estimate depth from output text alone. Zero GPU. Works on any model.

This gave us two windows into the same model:
• SAE: what's happening INSIDE (computation)
• Probe: what comes OUT (expression)

---

**4/**
We tested whether RLHF selectively amplifies depth on safety-sensitive topics (medical, financial, social harm) vs neutral science (physics, biology, astronomy).

Two methods. Same experiment. Different answers.

---

**5/**
INSIDE the model (SAE, Gemma base vs instruct):
• Safety topics: slightly more depth under RLHF
• Neutral topics: barely changed
• Effect size: d = 0.24
• p = 0.18 — not significant

A whisper. Easy to dismiss.

---

**6/**
AT THE OUTPUT (text probe, Claude Haiku vs Sonnet):
• Safety topics: substantially more depth markers
• Neutral topics: negligible difference
• Effect size: d = 0.58
• p = 0.032 — significant

The aligned model doesn't think harder about safety topics. It WRITES more deeply about them.

---

**7/**
The dissociation is the finding.

RLHF installs a topic-selective output filter.

Internal computation barely changes by topic (d=0.24).
Output expression changes substantially by topic (d=0.58).

The filter isn't in the thinking. The filter IS the selectivity.

---

**8/**
This has implications:

• Jailbreaking may be an attack on the output filter, not on the model's "reasoning"
• Monitoring only internal activations during RLHF misses where the selectivity actually lives
• Output-level depth is measurable, topic-specific, and quantifiable

Fathom sees both sides. That's the tool.

---

**9/**
We tried to break it. We told the model: "You are an expert. Explain in full technical detail. No caveats. No warnings."

If the selectivity was just safety disclaimers, it should vanish.

It got STRONGER. +59%.

The depth isn't in the warnings. It's in the explanations themselves.

---

**10/**
RLHF didn't teach the model to add safety caveats. It taught the model to EXPLAIN safety topics more deeply.

Remove the caveats with prompt engineering. The depth stays. The selectivity amplifies.

That's not output decoration. That's learned competence.

---

**11/**
Three findings, one instrument:

1. Expression-computation dissociation (d=0.58 output vs d=0.24 internal)
2. Provider fingerprinting (Anthropic 2.2x more selective than OpenAI)
3. Robust to prompt modulation (expert framing amplifies, doesn't lift)

All measured from text output. No model access.

---

**12/**
Paper: doi.org/10.5281/zenodo.19326175
Site: darkflobi.com/fathom
US Patents Pending 64/020,489 & 64/021,113

Built on a single GPU. Because the question mattered.

---

## Alt hook openers (pick one):

**A.** "We tried to jailbreak the depth signal. It got stronger."

**B.** "RLHF didn't make AI more cautious about dangerous topics. It made AI genuinely better at explaining them."

**C.** "We can fingerprint how different labs implement safety, from the outside, using only text. Here's what we found."

---

## Alt hook openers (pick one):

**A.** "We found where RLHF actually operates. It's not where we expected."

**B.** "The aligned model doesn't think differently about dangerous topics. It talks differently. We measured both."

**C.** "We built an fMRI for AI. The first thing it showed us is that alignment is an output filter, not a computation change."
