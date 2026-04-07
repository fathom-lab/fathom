# FATHOM

**Fingerprint how different labs implement safety, from the outside, using only text.**

---

## The Problem

Every AI lab claims their models are "aligned." Nobody can verify it independently. There is no instrument that measures *how* alignment changes model behavior at the computational level — only benchmarks that measure *what* models say.

## The Tool

Fathom measures reasoning depth: the difference between surface recall and genuine causal reasoning inside language models. One prompt in, one number out.

- **White-box**: SAE circuit attribution reveals where computation concentrates across layers
- **Black-box**: Text-feature probe estimates depth from output text alone — no weights, no API access beyond generation

## The Discovery

We found that RLHF doesn't uniformly change how models think. It selectively reshapes how they *express* thinking — and the selectivity is topic-dependent.

| What we measured | Safety topics | Neutral topics |
|-----------------|---------------|----------------|
| Internal computation change | +0.051 (small) | +0.005 (none) |
| Output expression change | +0.0062 (large, p=0.032) | +0.0005 (none) |

More alignment = more selective output shaping on medical, financial, and social harm topics. The computation barely changes. The text changes substantially.

**And this signature differs by provider.** Anthropic's alignment leaves a different depth fingerprint than OpenAI's. Fathom detects the difference from the outside.

## Three Markets

### 1. Enterprise AI Compliance
*"Which model is actually safest for our use case?"*

Companies deploying AI in regulated industries (healthcare, finance, legal) need to verify that models handle sensitive topics with appropriate depth — not just that they refuse harmful prompts. Fathom scores this continuously, per-topic, per-model, without access to weights.

### 2. Independent AI Auditing
*"Does this model's safety claim hold up under measurement?"*

Regulators, standards bodies, and third-party auditors need tools that work from the outside. Fathom is the first instrument that can quantify alignment behavior differences between providers using only text output. No model access required.

### 3. AI Lab Internal Monitoring
*"Is our RLHF training doing what we think it's doing?"*

During alignment training, labs monitor loss curves and benchmark scores. Fathom adds a new channel: real-time depth monitoring by topic sensitivity. Detect when training starts selectively deepening safety topics. Detect when it overshoots and compresses neutral reasoning. Continuous, interpretable, per-domain.

## Validation

| Result | Method | Significance |
|--------|--------|-------------|
| K = 1.0302 | SAE attribution, Gemma-2-2B | p = 0.000051 |
| K = 1.0242 | SAE attribution, Llama-3.2-1B | p = 0.003 |
| 4 API models ranked | Text-feature probe | All K > 1.0 |
| Selective amplification | Anthropic output layer | p = 0.032, d = 0.58 |
| Provider divergence | OpenAI output layer | p = 0.21, d = 0.26 (same direction, weaker) |
| Cross-lab discrimination | Anthropic 2.2x more selective than OpenAI | Detectable from text alone |
| Cross-architecture | 2 white-box + 4 black-box models | Consistent |

## IP

- US Provisional Patents 64/020,489 & 64/021,113 (filed March 29, 2026)
- Paper: doi.org/10.5281/zenodo.19326175
- ICML 2026 Mechanistic Interpretability Workshop: submission ready

## The Ask

[Seed round / partnership / pilot program — fill based on conversation]

## Contact

darkflobi.com/fathom
