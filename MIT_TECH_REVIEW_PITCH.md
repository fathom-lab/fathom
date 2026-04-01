# MIT Technology Review — Pitch
**Contact:** tips@technologyreview.com / story pitches form: https://www.technologyreview.com/story-tips/
**Date:** March 30, 2026

---

**Subject:** First tool to measure "reasoning depth" inside AI models — RLHF made models better at explaining dangerous topics, not just safer

---

Hi,

We've built what we believe is the first quantitative tool for measuring how deeply an AI model processes information — not what it says, but how much internal computation it performs to get there.

**The surprising finding:**

RLHF (reinforcement learning from human feedback — the training that makes AI models "safe") didn't just add safety disclaimers. It made models genuinely better at *explaining* dangerous topics. Tell Claude to respond without caveats: the safety depth signal gets 55% *stronger*, not weaker. The selectivity is structural, not cosmetic.

**What we measured:**

- Consistent, measurable difference between memorized retrieval and genuine reasoning across multiple AI architectures
- RLHF selective amplification is 2.4x stronger at the output layer than at internal computation — an expression-computation dissociation
- Anthropic's alignment produces 2.2x more topic-selective output shaping than OpenAI's, detectable from text alone
- Expert framing that removes safety disclaimers *amplifies* the depth signal — the alignment is in the explanation structure, not the warnings

**The tool is called Fathom.**

Paper: doi.org/10.5281/zenodo.19326175 (v2, published today)
Patent pending: USPTO 64/020,489 & 64/021,113
Site: darkflobi.com/fathom
Code: github.com/heyzoos123-blip/sae-reasoning-depth

**Why it matters:**

Every AI benchmark measures what models say. Nobody measures how they think. This does. And the first thing it found is that "aligned" models aren't thinking less about dangerous topics — they're filtering expression differently. And that filter is measurably distinct by provider.

**Happy to provide:**
- Technical walkthrough
- Access to the live demo at darkflobi.com/fathom
- Interview

— Alex Rodabaugh
  darkflobi.com
  heyzoos123@gmail.com
