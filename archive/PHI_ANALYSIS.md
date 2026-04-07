# Phi Proxy Analysis — Information Integration in Transformer Reasoning

**Date:** 2026-03-26 04:02:27
**Module:** fathom_phi.py

## Caveat

phi_proxy is a mathematical approximation, not a claim of consciousness. It measures information integration as defined by IIT, applied to transformer layer topology. Interpret accordingly.

## Distribution

| Metric | Value |
|--------|-------|
| n prompts | 20 |
| mean phi_proxy | 0.7969 |
| std phi_proxy | 0.0131 |
| median phi_proxy | 0.7935 |
| min | 0.7804 |
| max | 0.8349 |
| % high integration | 100.0% |

## Methodology

phi_proxy = entropy_normalized * (1 - concentration) * feature_engagement

Where:
- entropy_normalized: Shannon entropy of layer distribution / max possible entropy
- concentration: fraction of features in the single most-active layer
- feature_engagement: active features / max feature nodes

## Interpretation

100% of prompts showed high or exceptional information
integration (phi_proxy >= 0.4). Mean integration level: 0.797.

This measures the degree to which transformer computation is distributed and
non-decomposable across layers — the computational property that IIT identifies
as the substrate of integrated information.

## Note

phi_proxy is a mathematical approximation, not a claim of consciousness. It measures information integration as defined by IIT, applied to transformer layer topology. Interpret accordingly.
