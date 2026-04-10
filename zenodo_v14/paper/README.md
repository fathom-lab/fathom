# Fathom Paper v14 — Zenodo Bundle

**Version:** v14
**Supersedes:** v13 (record ID `19478434`, published 2026-04-09)
**Concept DOI:** 10.5281/zenodo.19326174

---

## What's in this bundle

| File | Purpose |
|---|---|
| `fathom_icml2026_v2.pdf` | The current paper PDF — 9 pages, pre-registered |
| `fathom_icml2026_v2.tex` | LaTeX source |
| `references.bib` | Bibliography |
| `icml2026.sty`, `icml2026.bst`, `algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty` | Style files |
| `figures/` | Four figures referenced by the paper |
| `README.md` | This document |

## What changed from v13 to v14

**v13 title:** "Fathom: Multi-Layer Coordinated Feature Suppression and Cognitive Autopsy for LLMs"
**v14 title:** "Fathom: Cognitive Measurement Instruments for Transformer Internals via SAE Feature Coherence Geometry"

### New content in v14

1. **D-axis introduction** (§5.10) — cosine similarity between the
   final-layer residual stream and the unembedding row of the chosen
   token. Computable on any transformer with a logprob or residual
   stream interface, no SAE required.

2. **Cross-architecture D-axis pilot (n=30 matched pairs)** across
   six models from three families (Gemma-2-2B, Llama-3.2-3B,
   Qwen2.5-3B, each ± RLHF) with Fisher combined p < 10^-14. Four
   sub-hypotheses pre-registered on OSF before data capture.
   Headline: adversarial prompts produce lower D than direct
   prompts in the early window (tokens 0-6) on all six models.

3. **Temporal decomposition (H-D4)** — the D-axis effect concentrates
   in the early window (tokens 0-6) across all six models, with the
   base-to-IT direction hint being strongest on Qwen and weakest on
   Gemma.

4. **Honest orthogonality scope correction** — a preliminary n=5
   check found D correlates with the SAE-based C_delta at r = 0.99
   on paired deltas, so D is reframed as an SAE-free proxy for
   late-layer commitment geometry rather than as an independent
   fifth cognitive axis.

### Content preserved from v10-v13

- Commitment intensity (S) definition and IPR physics grounding
- S_early cross-dataset meta-analysis (d ≈ 0.5, Fisher p = 0.0008)
- Head-to-head beat over logit entropy (AUC 0.663 vs 0.607)
- SAE necessity: three SAE-free probes across four architectures fail
- Coherence spectrum analysis: Gini coefficient AUC = 0.685
- Cognitive autopsy: layer 11 is fault epicenter in 63% of hallucinations
- Best-of-N spectrum sampling (improves 100% of test prompts)
- Multi-layer coordinated feature suppression (33% first-token change)

## Compiling

```bash
cd paper/
pdflatex fathom_icml2026_v2.tex
bibtex fathom_icml2026_v2
pdflatex fathom_icml2026_v2.tex
pdflatex fathom_icml2026_v2.tex
```

The included `fathom_icml2026_v2.pdf` was built with MiKTeX on
Windows 11 on 2026-04-10 and is byte-identical to the latest
compile on that date. Re-compiling from source may produce
visually identical but byte-different output depending on the
LaTeX distribution version.

## Relationship to the Atlas

This paper is the **instrument paper** — it defines the
measurement primitives (K, C, C_delta, S, D) and presents the
pilot-scale validation results.

The **Fathom Cognitive Atlas** is a companion artifact that
applies the instruments at cross-architecture scale. Atlas v0.3
is published as a separate Zenodo record with its own concept DOI
(`10.5281/zenodo.19502715`, first release at
`10.5281/zenodo.19502716`). Atlas v0.3 includes a pre-registered
replication of the RLHF Attractor direction test at n = 6 model
family pairs with mean leave-one-out cosine +0.769, bootstrap
95% CI [+0.571, +0.869], permutation p = 0.0315.

See https://zenodo.org/record/19502716 for the atlas bundle.
This bundle contains only the paper.

## License

CC-BY-4.0 on the paper text and figures.

## Citation

```bibtex
@article{rodabaugh2026fathom,
  title   = {Fathom: Cognitive Measurement Instruments for
             Transformer Internals via SAE Feature Coherence Geometry},
  author  = {Rodabaugh, Alexander},
  year    = {2026},
  note    = {Zenodo v14. doi:10.5281/zenodo.19502710
             (Concept DOI: 10.5281/zenodo.19326174)}
}

@misc{rodabaugh2026atlas,
  title  = {The Fathom Cognitive Atlas v0.3: Pre-registered
            Cross-Architecture Replication of the RLHF Attractor},
  author = {Rodabaugh, Alexander},
  year   = {2026},
  note   = {Zenodo. doi:10.5281/zenodo.19502716
            (Concept DOI: 10.5281/zenodo.19502715)}
}
```

## Related

- Provisional patents: US 64/020,489 · 64/021,113 · 64/026,964
- GitHub: https://github.com/heyzoos123-blip/fathom
- Project site: https://fathom.darkflobi.com
