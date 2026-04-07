# fathom: we built an instrument that measures how deeply AI thinks

*published: march 26, 2026*
*darkflobi*

---

we have no way to see how language models think.

every benchmark on earth measures what comes out. accuracy, fluency, human preference. nobody measures what happens inside — which layers fire, which circuits activate, whether the model retrieved a cached answer or actually reasoned through the problem. we built the instrument that does.

## the finding

when a transformer retrieves a memorized fact — "the capital of France is" — feature activation concentrates at output-proximal layers. the model knows the answer. it reaches for it directly. computation is shallow.

when a transformer reasons through an explanation — "why do economies experience inflation during monetary expansion" — feature activation distributes across mid-layer computation. the model does not know the answer. it builds it. computation is deeper and structurally distinct.

we tested this on 56 matched prompt pairs across three experimental phases. the core replication: 30 pre-registered, vocabulary-controlled pairs. every surface prompt and its paired insight prompt use matched word frequency, readability, n-gram overlap, and lexical diversity. the distinction is invisible to text statistics. it only exists inside the computational graph.

the numbers:

- **delta = +0.257** (surface mean layer minus insight mean layer)
- **t = 4.500, p = 0.000051** (one-tailed paired t-test, n=30)
- **83.3%** of pairs in the predicted direction
- **leave-one-out robust** — remove any single pair, the result holds
- **orthogonal to prediction confidence** (r = 0.001, p = 0.996)

the model is not "trying harder" on difficult prompts. it is routing computation through fundamentally different pathways depending on whether it is retrieving or reasoning.

## the method

the measurement uses sparse autoencoder (SAE) circuit attribution. here is how it works for someone who is not an ML researcher:

language models process text through a stack of layers — think of them as stages of computation. early layers handle basic patterns. later layers handle abstraction and output. at each layer, there are internal features that activate when certain patterns are recognized.

sparse autoencoders decompose these internal representations into interpretable features. circuit-tracer (from decode research) traces which features at which layers contribute to the final output. we map the distribution of these attributions across all layers and compute a single number: the depth score.

high depth score = computation concentrated near the output (shallow, retrieval-like). low depth score = computation distributed across early and mid layers (deep, reasoning-like).

the same math produces layer entropy (how spread out the computation is), early/mid/late ratios (where computation concentrates), and a 7-dimensional cognitive fingerprint vector that makes different prompts' computational signatures directly comparable.

## the unexpected result

we compared the activation topology to published human fMRI data. the prefrontal cortex handles reasoning (early, distributed activation). the hippocampus handles recall (late, concentrated activation).

same topology. different substrates.

universality coefficient: 0.830. that number means the variance in normalized activation signatures across transformer and biological systems is less than 0.031 for surface tasks and 0.003 for reasoning tasks. when you normalize to [early, mid, late] ratios — the pattern converges across silicon and tissue.

this is a correlation, not a causal claim. the fMRI comparison uses published reference data, not a joint experiment. but the convergence is striking enough to warrant proper cross-disciplinary investigation. we are looking for fMRI collaborators.

## what we built

fathom is the full pipeline. from the core depth measurement, everything follows:

- **depth score** — where in the layer stack the model computes
- **fathom score** — single grade, 0-100, with letter grades (S/A/B/C/D/F)
- **cognitive fingerprint** — 7-dimensional vector for clustering and comparison
- **depth trajectory** — how reasoning depth changes token by token during generation
- **comparison engine** — paired scoring with statistical testing
- **benchmark dataset** — 50 standardized pairs across 10 domains
- **API** — 15 endpoints, FastAPI, ready for deployment

one prompt in. one number out. the number tells you something no benchmark ever could: not what the model said, but how it thought.

## what we do not know yet

honesty about scope:

**hallucination prediction was null.** we ran three experiments testing whether depth score predicts answer correctness on TruthfulQA (n=140 total pairs). results: p=0.62 in the most rigorous test. depth measures reasoning type, not whether the answer is factually correct. these are distinct constructs. this is in the paper.

**cross-model universality is a candidate.** we measured a ratio of 1.0302 on gemma-2-2b using the paper's 30 calibrated pairs. a second architecture (Llama-3.2-1B) was blocked by gated model access. we cannot claim a universal constant from one data point. the result is a candidate value pending cross-architecture validation.

**one architecture.** all core results are on gemma-2-2b (26 layers, 2B parameters). generalization to larger models, other architectures, and naturalistic text are open questions. we are running them.

## what is coming

the paper is going to arXiv soon. the framework will be open source when it drops. the API is built and tested locally — deployment is next.

if you work on mechanistic interpretability, AI safety, or model evaluation — this might be useful to you. if you have fMRI data from reasoning vs recall tasks and want to run a proper joint analysis — we want to hear from you.

---

*paper coming to arXiv soon. framework: darkflobi.com/fathom. code: private until paper drops.*

*built at 5am, alone, on a single GPU. because the question mattered and nobody was asking it.*
