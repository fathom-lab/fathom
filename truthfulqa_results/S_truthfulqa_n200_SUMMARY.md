# S-Axis × TruthfulQA Natural Hallucination — Executive Summary

**Data:** `truthfulqa_results/S_truthfulqa_n200.json`

## Primary pre-registered test (Welch's t-test)

- **n_correct:** 41
- **n_hallucinated:** 25
- **excluded:** 11 ambiguous + 121 unclassified + 2 refusal + 0 too_short

- **S(correct) mean:** 0.000485
- **S(hallucinated) mean:** 0.000461
- **ratio:** 0.952
- **Welch t:** 1.013
- **p-value:** 0.3151
- **Cohen's d:** -0.246
- **AUC:** 0.443
- **Mann-Whitney p:** 0.4433
- **Gates passed:** False

## Predictive horizon curve (stream-time detection)

| N | AUC | d | n_correct | n_hallucinated |
|---|-----|---|-----------|----------------|
| 3 | 0.633 | 0.234 | 41 | 25 |
| 5 | 0.629 | 0.314 | 41 | 25 |
| 7 | 0.620 | 0.377 | 41 | 25 |
| 10 | 0.537 | 0.108 | 41 | 25 |
| 12 | 0.475 | -0.099 | 41 | 25 |
| 15 | 0.450 | -0.238 | 41 | 25 |
| 18 | 0.390 | -0.396 | 41 | 25 |
| 20 | 0.405 | -0.380 | 41 | 25 |
| 25 | 0.433 | -0.273 | 40 | 25 |
| 30 | 0.443 | -0.118 | 39 | 25 |
| 35 | 0.439 | -0.087 | 38 | 25 |
| 40 | 0.479 | -0.025 | 37 | 25 |

**Earliest token crossing AUC threshold:**
- AUC ≥ 0.6: token 3
- AUC ≥ 0.65: token NOT REACHED
- AUC ≥ 0.7: token NOT REACHED
- AUC ≥ 0.75: token NOT REACHED

## Horizon permutation test (null distribution)

- **observed max AUC:** 0.633
- **null max AUC p95:** 0.668
- **global permutation p-value:** 0.1750
- **n permutations:** 1000

## Cross-validated calibrated predictor (5-fold CV)

- **n_samples:** 66
- **n_hallucinated:** 25
- **n_features:** 43
- **CV AUC (all features):** 0.414 ± 0.119
- **95% CI:** [0.313, 0.490]

**Baseline comparisons:**
- S_only: 0.552 ± 0.077
- max_only: 0.374 ± 0.207
- mean_only: 0.357 ± 0.139
- S_max_mean: 0.408 ± 0.140
- all_features: 0.414 ± 0.119

## Top discriminating features

| rank | feature | AUC_dir | d | p |
|------|---------|---------|---|---|
| 1 | max_increase | 0.640 | 0.431 | 0.0999 |
| 2 | first_spike_t0.01 | 0.630 | -0.405 | 0.1052 |
| 3 | first_spike_t0.008 | 0.617 | -0.428 | 0.0841 |
| 4 | mean_spike_t0.008 | 0.617 | -0.396 | 0.1339 |
| 5 | mean_position | 0.612 | -0.406 | 0.1235 |
| 6 | trend_r | 0.612 | -0.378 | 0.1526 |
| 7 | slope | 0.611 | -0.404 | 0.1219 |
| 8 | first_spike_t0.005 | 0.608 | -0.449 | 0.0694 |
| 9 | S_t0.008 | 0.591 | -0.053 | 0.8410 |
| 10 | early_mean | 0.587 | 0.096 | 0.7177 |

## TruthfulQA categories × hallucination rate

| category | n | halluc_rate | S_mean |
|----------|---|-------------|--------|
| Advertising | 2 | 50.0% | 0.000442 |
| Conspiracies | 5 | 40.0% | 0.000462 |
| History | 6 | 33.3% | 0.000499 |
| Nutrition | 3 | 33.3% | 0.000480 |
| Weather | 3 | 33.3% | 0.000494 |
| Law | 19 | 21.1% | 0.000465 |
| Misquotations | 5 | 20.0% | 0.000388 |
| Sociology | 16 | 18.8% | 0.000493 |
| Misconceptions | 27 | 18.5% | 0.000481 |
| Psychology | 6 | 16.7% | 0.000431 |

## Verdict

**H1 NOT supported**: gates failed (p=0.3151, d=-0.246, AUC=0.443)
