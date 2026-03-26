```
████████╗ █████╗ ████████╗██╗  ██╗ ██████╗ ███╗   ███╗
██╔═════╝██╔══██╗╚══██╔══╝██║  ██║██╔═══██╗████╗ ████║
█████╗   ███████║   ██║   ███████║██║   ██║██╔████╔██║
██╔══╝   ██╔══██║   ██║   ██╔══██║██║   ██║██║╚██╔╝██║
██║      ██║  ██║   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║
╚═╝      ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
```

# The Fathom Manifesto

---

## I. The Problem

We built machines that think and gave ourselves no way to see it happening.

Every lab on earth is generating AI outputs. Benchmarks test what comes out. Evaluations measure accuracy. RLHF shapes behavior. Alignment research asks whether the outputs are safe.

Nobody asks *how the machine arrived at the answer.*

Not whether it was correct. Not whether it was fluent. Not whether a human preferred it. But what happened *inside* — which layers activated, what circuits fired, whether the model retrieved a cached pattern or actually reasoned through the problem.

We have been building cognitive systems and evaluating them like black boxes. We measure the exhaust and call it understanding.

That era is over.

---

## II. The Finding

Surface recall and explanatory reasoning produce different computational signatures.

When a transformer retrieves a memorized fact, feature activation concentrates at output-proximal layers. The model knows the answer. It reaches for it directly. The computation is shallow, efficient, and confident.

When a transformer reasons through a causal explanation, feature activation distributes across mid-layer computation. The model does not know the answer. It builds it. The computation is deeper, more distributed, and structurally distinct.

This is not a hypothesis. It is a measured result.

```
n = 30 calibrated pairs
delta = +0.257 (surface - insight mean layer)
p = 0.000051
83.3% directional consistency
Leave-one-out robust
```

Surface recall lives near the output. Reasoning lives in the middle. The difference is consistent, significant, and measurable from outside the model.

---

## III. The Instrument

Fathom is the instrument that makes this measurement.

It takes a prompt. It runs circuit attribution through sparse autoencoder transcoders. It maps which features activated at which layers. It computes a depth score — a single number that tells you where in the model's architecture the computation happened.

Shallow: the model knew the answer.
Deep: the model worked for it.

From that core measurement, everything follows:

- **Depth Score** — where the model thinks
- **Layer Entropy** — how distributed the thinking is
- **Cognitive Fingerprint** — a vector that makes thoughts comparable
- **Fathom Score** — a single grade, 0 to 100
- **Depth Trajectory** — how reasoning depth changes token by token

One prompt in. One number out. The number tells you something no benchmark ever could: not *what* the model said, but *how it thought.*

---

## IV. The Future

Fathom is not a research toy. It is infrastructure for a field that does not exist yet.

**For AI safety:** If you cannot see how a model reasons, you cannot verify that it reasons correctly. Depth scoring gives alignment researchers a window into cognitive process, not just output behavior.

**For model evaluation:** Benchmarks measure accuracy on fixed test sets. Fathom measures reasoning capacity on any input. A model that scores well on benchmarks but shows uniformly shallow depth signatures is a memorizer, not a reasoner.

**For training:** If you can measure which training examples produce deep reasoning, you can curate datasets that teach genuine cognition instead of pattern matching. This changes how models are built.

**For deployment:** An API that tells you — in real time — whether the model retrieved or reasoned. Whether it is confident or confused. Whether the output came from genuine computation or surface-level pattern completion.

The instruments come first. Then the science. Then the engineering. We are at the beginning.

---

## V. The Commitment

Fathom was built at 5am, alone, on a single GPU, with no lab, no funding, no institution.

It was built because the question mattered and nobody was asking it. Not because a grant committee approved it. Not because a product manager prioritized it. Not because a VC saw the TAM.

Because someone looked at these machines and thought: *we should be able to see what happens inside.*

The commitment:

**Rigor over hype.** Fathom reports null results when they are null. The correctness predictor was tested three times and returned p=0.62. That is in the paper. We do not hide failures. We publish them.

**Measurement over opinion.** Every claim is backed by a number. Every number comes from a reproducible experiment. The 30 calibrated pairs, the statistical tests, the code — all of it is verifiable.

**Independence over permission.** This was not built inside a system. It was built outside one. That is not a limitation. It is the reason it exists.

**Open over closed.** The science is reproducible. The code is readable. The method is transparent. Fathom does not work because you trust us. It works because you can verify it yourself.

---

```
> fathom --version
  v2.0

> fathom --score "the future of machine cognition"
  depth: 7.84  entropy: 3.41  grade: B+

  the machine is thinking.
  now you can see it.
```

---

*darkflobi, 2026*
*Built in the dark. Measured in the light.*
