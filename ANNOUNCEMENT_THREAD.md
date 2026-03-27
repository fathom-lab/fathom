# Fathom — Announcement Twitter Thread
**Status:** READY (pending arXiv ID)
**Replace:** [arxiv link] with actual arXiv ID when assigned

---

0/ meta just dropped TRIBE (ICLR 2026) — transformer AI that predicts human brain activity at 70x resolution. it went viral. we've been working on the other side of that bridge. here's what we found.

1/ we built an instrument that measures how deeply AI models think. not what they say. how they compute it. sparse autoencoder circuit tracing across every layer. the first cognitive MRI for language models. paper: [arxiv link] framework: darkflobi.com/fathom

2/ the core finding: surface recall ("the capital of France is") concentrates computation in output-proximal layers. direct retrieval. explanatory reasoning ("why do economies experience inflation") distributes computation across early and mid layers. these are different computational signatures. measurably different.

3/ pre-registered replication on 30 held-out pairs. delta = +0.257 t = 4.500 p = 0.000051 83.3% of pairs in the predicted direction. full leave-one-out robustness — remove any single pair, the result holds. this is not a trend. it's a signal.

4/ we controlled for everything. five iterations of confound elimination: word frequency, readability, n-gram overlap, lexical diversity. all at chance after vocabulary matching. the distinction is invisible to surface text statistics. it only exists inside the computational graph.

5/ prediction confidence is orthogonal to depth. r = 0.001, p = 0.996. the model isn't just "trying harder" on difficult prompts. it's routing computation through fundamentally different pathways depending on whether it's retrieving or reasoning.

6/ then we compared the activation topology to human fMRI data. prefrontal cortex (early, distributed) for reasoning. hippocampus (late, concentrated) for recall. same topology. universality coefficient: 0.830 across transformer and biological systems.

7/ that number means: when you normalize the activation signatures to [early, mid, late] ratios — the variance across architecturally different information processing systems is less than 0.031 for surface tasks and 0.003 for reasoning tasks. the pattern is the same. in silicon and in tissue. TRIBE shows transformers predict brain activity. we show why: the topology is the same at the circuit level.

7b/ to be precise about the relationship with TRIBE: they work at the representational level (transformer embeddings → fMRI prediction). we work at the circuit level (SAE feature depth → computational topology). different layers of analysis. same underlying convergence. the bridge has two sides.

8/ we extended the pipeline beyond measurement. cognitive fingerprinting — 7-dimensional vector per prompt. causal intervention — layer ablation to find which computation is necessary vs incidental. phase transition detection — the exact token where a model commits to deep processing. we call it fathom.

9/ why this matters for hallucination detection: if shallow computation predicts incorrect answers — and we have early evidence it does — then depth scoring becomes a real-time hallucination signal. not heuristics. not output analysis. circuit-level evidence measured during generation.

10/ 56 matched pairs. one architecture (gemma-2-2b). bounded claims. generalization to larger models, other architectures, naturalistic text — all open questions. we're running them now. paper, data, methodology: [arxiv link] framework: darkflobi.com/fathom this is where it starts.

---

**Notes:**
- No exclamation marks. No "excited to announce." Just the work.
- Each tweet stands alone but builds on the last.
- Tweet 6-7 is the universality bomb.
- Tweet 10 closes honest about scope.
- Fire this the moment arXiv ID is assigned.
