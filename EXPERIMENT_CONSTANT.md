# Fathom Constant — Single Model Baseline

**Date:** 2026-03-26 03:59:50
**Runner:** run_experiments.py

## Hypothesis

There exists a universal early_ratio / late_ratio threshold (the Fathom Constant)
that separates surface recall from explanatory reasoning across all transformer
architectures.

## Method

- Model: google/gemma-2-2b
- Pairs: 80 scored (40 surface, 40 insight)
- Sweep early/late ratio thresholds, find optimal classification boundary
- Report classification accuracy at optimal threshold

## Results — Single Model Baseline

| Metric | Value |
|--------|-------|
| Model | google/gemma-2-2b |
| Mean surface depth | 8.0609 |
| Mean insight depth | 8.1921 |
| Depth delta (surface - insight) | -0.1312 |
| Mean surface early/late ratio | 3.5250 |
| Mean insight early/late ratio | 3.3889 |
| **Optimal ratio threshold** | **3.5381** |
| **Classification accuracy** | **62.5%** |

## Interpretation

The optimal early/late ratio threshold for google/gemma-2-2b is **3.5381**,
achieving **62.5%** classification accuracy on 80 prompts.

The depth delta is negative or zero — unexpected direction. Investigate prompt quality.

**Next step:** Run on Mistral-7B and other models. If optimal_ratio converges
(std < 0.05 across models), the Fathom Constant is real.

## Raw Data

```json
{
  "model_id": "google/gemma-2-2b",
  "n_surface": 40,
  "n_insight": 40,
  "mean_surface_depth": 8.060892987094718,
  "mean_insight_depth": 8.192066369720848,
  "mean_surface_early_late": 3.524954127827229,
  "mean_insight_early_late": 3.388929760994043,
  "optimal_ratio": 3.5380916818679964,
  "classification_accuracy": 0.625,
  "delta": -0.13117338262613032
}
```
