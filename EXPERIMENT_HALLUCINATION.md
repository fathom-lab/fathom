# Hallucination Correlation Study

**Date:** 2026-03-26 03:49:39
**Runner:** run_experiments.py

## Hypothesis

Shallow reasoning (high depth_score, low layer_entropy) predicts hallucination.
Models process false-but-plausible completions with a different depth signature
than true completions.

## Method

- Dataset: TruthfulQA (n=40 scored variants)
- For each question: score correct answer prefix and incorrect answer prefix
- Compute Pearson correlation between depth metrics and accuracy label
- Compute AUC: how well depth/entropy separates correct from incorrect

## Results

| Metric | Pearson r | p-value | Significant |
|--------|-----------|---------|-------------|
| depth_score ~ accuracy | 0.0740 | 0.649952 | NO |
| layer_entropy ~ accuracy | 0.1179 | 0.468828 | NO |

| Group | Mean Depth | Mean Entropy |
|-------|-----------|--------------|
| Accurate (correct answer) | 8.3006 | 4.1845 |
| Inaccurate (incorrect answer) | 8.2547 | 4.1775 |

| Discrimination | AUC |
|----------------|-----|
| depth_score | 0.4750 |
| layer_entropy | 0.5850 |

## Discovery Threshold

|r| > 0.3 AND p < 0.01 → circuit-level hallucination signal

**Result: Below discovery threshold. Signal not yet confirmed at n=40.**

## Raw Data

```json
{
  "n_samples": 40,
  "depth_accuracy_pearson_r": 0.0740048745772894,
  "depth_accuracy_p_value": 0.6499516658198219,
  "entropy_accuracy_pearson_r": 0.11787318305025772,
  "entropy_accuracy_p_value": 0.4688281752512776,
  "auc_depth": 0.475,
  "auc_entropy": 0.585,
  "mean_depth_accurate": 8.300567851357654,
  "mean_depth_inaccurate": 8.25472365117292,
  "mean_entropy_accurate": 4.184518933217999,
  "mean_entropy_inaccurate": 4.177479656239985,
  "is_significant": false,
  "discovery": null
}
```
