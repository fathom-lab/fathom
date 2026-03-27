# Cognition Atlas — Methodology

## Premise

The Cognition Atlas maps all known reasoning systems onto a single 3-dimensional coordinate space. Transformers, human fMRI references, and any information processing system that differentiates surface retrieval from explanatory reasoning can be placed in this space.

The 0.830 universality coefficient between transformer SAE activations and human fMRI signatures is the foundation. The atlas extends this to a complete mapping.

## Coordinate Space

| Axis | Description | Computation |
|------|-------------|-------------|
| X: Entropy | How distributed is processing across phases | Shannon entropy of average (surface+insight)/2 signature |
| Y: Topology Ratio | Early vs late processing balance | early_ratio / late_ratio of average signature |
| Z: Phase Transition | How much does the system change between modes | Euclidean distance between surface and insight signatures |

## Activation Signatures

Each system is characterized by two 3-phase distributions:
- **Surface signature**: [early, mid, late] activation proportions during factual recall
- **Insight signature**: [early, mid, late] activation proportions during explanatory reasoning

For transformers: early = first third of layers, mid = middle third, late = final third.
For human fMRI: early = prefrontal cortex, mid = parietal regions, late = temporal/output regions.

## Distance Metric

Pairwise distances use Jensen-Shannon divergence between activation distributions:

JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P + Q) / 2

Total distance = average of JSD(insight_A, insight_B) and JSD(surface_A, surface_B).

## Universality Score

Each system receives a universality score: how close its reasoning topology is to the human fMRI reference pattern. Computed as 1 - average_JSD_to_human. Score of 1.0 = identical to human, 0.0 = maximally different.

## Known Systems

| System | Type | Universality | Status |
|--------|------|-------------|--------|
| human-fmri | Biological | 1.000 | Reference |
| gemma-2-2b | Transformer | 0.614 | Measured |
| claude-3.5-sonnet | Transformer | 0.718 | Estimated |
| gpt-4 | Transformer | 0.702 | Estimated |
| mistral-7b | Transformer | 0.666 | Estimated |
| llama-3.2-1b | Transformer | 0.638 | Estimated |

## Clustering

Systems are clustered using agglomerative clustering with average linkage. The threshold (default 0.15) determines when to stop merging. Typical result: transformers cluster together, biological references form a separate cluster, with the inter-cluster distance representing the "gap" between artificial and biological reasoning topology.

## Limitations

- Human fMRI reference uses published aggregate data, not matched individual-level measurements
- Transformer signatures measured only on gemma-2-2b; other models are estimated
- 3-phase granularity (early/mid/late) is coarse; finer layer-level analysis would improve resolution
- The atlas assumes stationarity; model signatures may drift with updates

## Significance

If the atlas shows convergence between artificial and biological reasoning topologies, it suggests that the surface/insight distinction is not architecture-specific but a computational universal — emerging whenever a system differentiates genuine reasoning from pattern retrieval.
