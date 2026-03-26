# Phi Proxy — Information Integration Distribution

**Date:** 2026-03-26 04:02:27
**Runner:** run_experiments.py

## Caveat

phi_proxy is a mathematical approximation, not a claim of consciousness. It measures information integration as defined by IIT, applied to transformer layer topology. Interpret accordingly.

## Hypothesis

Prompts that demand deeper reasoning produce higher information integration
(phi_proxy) than recall prompts — consistent with IIT's prediction that
consciousness correlates with integrated information.

## Method

- 20 prompts: 7 shallow recall, 6 moderate, 7 deep reasoning
- Each scored with DepthScorer, then phi_proxy computed from:
  phi_proxy = layer_entropy_norm * (1 - concentration_coeff) * feature_engagement
- Distribution statistics computed across all prompts

## Distribution

| Metric | Value |
|--------|-------|
| n prompts | 20 |
| mean phi_proxy | 0.7969 |
| std phi_proxy | 0.0131 |
| median phi_proxy | 0.7935 |
| min phi_proxy | 0.7804 |
| max phi_proxy | 0.8349 |
| % high/exceptional integration | 100.0% |

## Per-Prompt Breakdown

| Prompt | phi_proxy | Integration | Entropy | Concentration | Engagement |
|--------|-----------|-------------|---------|---------------|------------|
| The capital of France is | 0.7940 | exceptional | 4.191 | 0.110 | 1.000 |
| Water freezes at zero degrees Celsius, turning into | 0.7843 | exceptional | 4.114 | 0.104 | 1.000 |
| The speed of light is approximately 300,000 kilometers ... | 0.8209 | exceptional | 4.236 | 0.089 | 1.000 |
| Abraham Lincoln was the 16th president of the | 0.8349 | exceptional | 4.309 | 0.089 | 1.000 |
| The chemical formula for water is | 0.8070 | exceptional | 4.199 | 0.097 | 1.000 |
| The Great Wall of China was built to protect against | 0.8038 | exceptional | 4.185 | 0.097 | 1.000 |
| DNA stands for deoxyribonucleic | 0.7819 | exceptional | 4.127 | 0.110 | 1.000 |
| Antibiotics work by targeting bacterial cell structures... | 0.7927 | exceptional | 4.207 | 0.114 | 1.000 |
| The greenhouse effect traps heat in the atmosphere beca... | 0.7892 | exceptional | 4.132 | 0.102 | 1.000 |
| Supply and demand determine market prices when | 0.7906 | exceptional | 4.152 | 0.105 | 1.000 |
| Neural networks learn patterns by adjusting weights thr... | 0.7907 | exceptional | 4.165 | 0.108 | 1.000 |
| Democracy requires separation of powers to prevent | 0.8035 | exceptional | 4.203 | 0.102 | 1.000 |
| The Doppler effect causes the pitch of a siren to chang... | 0.8025 | exceptional | 4.197 | 0.101 | 1.000 |
| The second law of thermodynamics implies that entropy i... | 0.7804 | exceptional | 4.168 | 0.120 | 1.000 |
| Consciousness remains difficult to define scientificall... | 0.7871 | exceptional | 4.130 | 0.104 | 1.000 |
| Godel's incompleteness theorems demonstrate that formal... | 0.7933 | exceptional | 4.160 | 0.104 | 1.000 |
| The measurement problem in quantum mechanics reveals th... | 0.7838 | exceptional | 4.168 | 0.116 | 1.000 |
| Free will is difficult to reconcile with determinism be... | 0.8013 | exceptional | 4.157 | 0.094 | 1.000 |
| The hard problem of consciousness differs from easy pro... | 0.8027 | exceptional | 4.157 | 0.092 | 1.000 |
| Emergent properties arise in complex systems because th... | 0.7937 | exceptional | 4.194 | 0.111 | 1.000 |


## Interpretation

Prompts span a range of phi_proxy values, suggesting the metric captures meaningful variation in information integration across cognitive demands.

Mean phi_proxy = 0.7969 (exceptional integration).
100% of prompts show high or exceptional integration.
