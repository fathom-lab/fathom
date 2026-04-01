# Fathom Outreach — Ready to Send

## 1. Circuit-Tracer Authors (Joseph Bloom / Logan Riggs)

**DM / Email:**

Hey — we built something on top of circuit-tracer that found a clean result.

We used your SAE transcoders on Gemma-2-2B to measure reasoning depth: the mean active feature layer when a model processes "surface recall" prompts vs "explanatory insight" prompts.

The ratio is consistent: K=1.0302, p=0.000051 (n=30 calibrated pairs). Replicated on Llama-3.2-1B (K=1.0242, p=0.003). Generalizes to black-box API models via a text-feature probe.

We're also finding preliminary evidence that RLHF selectively amplifies depth on safety-sensitive topics (d=0.44, scoring more pairs now).

Paper: https://doi.org/10.5281/zenodo.19326175

Thought you'd want to see this since it's built directly on your tool. Happy to answer any methodology questions.

— Alex

---

## 2. Neel Nanda (Mechanistic Interpretability Lead, Google DeepMind)

**DM:**

Built a depth metric using SAE circuit attribution that consistently distinguishes surface recall from causal reasoning in transformers.

K=1.0302 on Gemma-2-2B (p=0.000051), replicates on Llama. Claude Haiku outperforms Claude Sonnet on depth signal — less alignment → stronger depth expression.

Paper live: https://doi.org/10.5281/zenodo.19326175

Would love your take on the methodology — especially whether the active feature layer distribution is capturing something real about computation depth vs just prompt length artifacts (we controlled for this, but a second opinion from someone who lives in SAE space would be valuable).

---

## 3. AI Safety / Alignment Researchers (general template)

**Subject:** Paper — RLHF selectively amplifies reasoning depth on safety-sensitive topics

We've published initial evidence that RLHF doesn't uniformly compress reasoning depth — it selectively amplifies depth on topics with high safety/alignment training pressure (medical, financial, social harm) while leaving neutral science topics unchanged.

White-box measurement via SAE circuit attribution on Gemma-2-2B base vs instruct.
Black-box cross-validation on Claude and GPT model families.

Core finding (Fathom Constant): https://doi.org/10.5281/zenodo.19326175
Selective amplification: scoring in progress, preliminary d=0.44

Relevant if you work on: alignment tax measurement, capability evaluation post-RLHF, interpretability-based safety monitoring.

— Alex Rodabaugh
darkflobi.com/fathom

---

## 4. MIT Tech Review / The Information Pitch

**Subject:** First tool to measure "reasoning depth" inside AI models — smaller Claude outperforms flagship

We've built what we believe is the first quantitative tool for measuring how deeply an AI model processes information — not what it says, but how much internal computation it performs to get there.

Key findings:
- Consistent measurable difference between memorized retrieval and genuine reasoning across multiple AI architectures
- Claude Haiku (Anthropic's smaller model) produces deeper reasoning signals than Claude Sonnet (the flagship) — suggesting alignment training smooths output expression
- Preliminary evidence that safety training selectively deepens processing on sensitive topics (medical, financial harm)

The tool is called Fathom. Paper: doi.org/10.5281/zenodo.19326175. Patent pending (64/020,489 & 64/021,113).

We think this is relevant because: every AI benchmark measures what models say. Nobody measures how they think. This does. And the first thing it found is that the "aligned" model isn't thinking less — it's filtering more.

Happy to do a technical walkthrough or provide access to the demo.

— Alex Rodabaugh

---

## 5. Workshop Submissions (check deadlines)

- ICML 2026 Mechanistic Interpretability Workshop
- NeurIPS 2026 SafeAI Workshop
- ICLR 2026 Workshops (may have passed)
- ACL 2026 Workshop on Interpretability

---

## 6. Quick DM to Hoagy / Aidan (endorsement update)

Paper's live on Zenodo: https://doi.org/10.5281/zenodo.19326175

Still need arXiv endorsement (code XOZS9V, cs.LG or cs.AI). If you can endorse, everything else is ready to go — just need the gate opened. The paper's real and public now, so you can read and evaluate before deciding.

Thanks either way.
