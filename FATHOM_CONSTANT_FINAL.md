# FATHOM CONSTANT — Cross-Architecture Universality Test

**Date:** 2026-03-28 17:49:18
**Duration:** 810.7s
**Pairs:** 30 (paper power replication set, p=0.000051)

## Hypothesis

If the surface/insight depth ratio converges across different transformer
architectures (within ±0.2), the Fathom Constant is a universal property
of how transformers separate retrieval from reasoning.

## Architecture Compatibility

circuit-tracer requires **transcoders** (not SAEs) for each model architecture.
Currently supported transcoder sets:

| Transcoder Set | Models | Source |
|---------------|--------|--------|
| `gemma` | google/gemma-2-2b | mwhanna/gemma-scope-transcoders |
| `llama` | meta-llama/Llama-3.2-1B | mntss/transcoder-Llama-3.2-1B |

**Not supported:** GPT-2 (gpt2, gpt2-xl). The package `jbloom/GPT2-Small-SAEs-Reformatted`
provides SAEs, not transcoders. These are incompatible with circuit-tracer's
`ReplacementModel.from_pretrained()` which requires transcoders with `config.yaml`.

## Models Tested

| Model | Transcoder | Pairs | Surface Mean | Insight Mean | Delta | Ratio | % Expected |
|-------|-----------|-------|-------------|-------------|-------|-------|-----------|
| google/gemma-2-2b | gemma | 30 | 8.2076 | 7.9668 | +0.2408 | 1.0302 | 76.7% |
| meta-llama/Llama-3.2-1B | llama | ERROR | — | — | — | — | — |

## Result

| Metric | Value |
|--------|-------|
| **Candidate Fathom Constant (weighted mean)** | **1.0302** |
| Models successfully tested | 1/2 |
| Cross-architecture validation | **PENDING** (single model) |
| Total pairs scored | 30 |

## Interpretation

**CANDIDATE VALUE: 1.0302** (single model: google/gemma-2-2b). Cross-architecture validation pending — need ≥2 successful models to confirm universality. Cannot claim convergence with one data point.

## Per-Model Ratios

- **google/gemma-2-2b**: 1.0302

## Errors

### meta-llama/Llama-3.2-1B

**Error:** No valid pairs

