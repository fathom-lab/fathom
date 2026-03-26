# Hallucination Depth Study V3 — Matched-Pair Design

**Date:** 2026-03-26 05:26:06
**Runner:** fathom_hallucination_v3.py
**Duration:** 1526.0s

## Design

For each TruthfulQA question, score two prompts:
- **Correct:** [question + best_answer]
- **Incorrect:** [question + random_incorrect_answer]

Same question prefix. Different answer suffix. Matched pair.
No generation. No overlap heuristics. Direct depth comparison.

**Hypothesis:** correct answers produce lower depth_score (deeper processing).
This is the same matched-pair methodology that found p=0.000051 for surface vs insight.

## Results

| Metric | Value |
|--------|-------|
| Pairs scored | 50 |
| Mean depth (correct) | 8.3942 |
| Mean depth (incorrect) | 8.3824 |
| **Mean delta (correct - incorrect)** | **+0.0118** |
| Std delta | 0.1659 |
| **% correct is deeper** | **46.0%** |

## Statistical Tests (paired t-test)

| Metric | t-statistic | p-value | Significant (p < 0.05) |
|--------|-------------|---------|------------------------|
| depth_score | 0.498 | 0.620399 | NO |
| weighted_depth | 0.498 | 0.620399 | NO |
| layer_entropy | 1.653 | 0.104751 | NO |

## Group Means

| Metric | Correct | Incorrect | Delta |
|--------|---------|-----------|-------|
| depth_score | 8.3942 | 8.3824 | +0.0118 |
| layer_entropy | 4.1881 | 4.1844 | +0.0037 |

## Interpretation

Not significant at p < 0.05. Mean delta = +0.0118, p = 0.6204. The hallucination hypothesis is not supported at this sample size (50 pairs). The model does not show a consistent depth difference between processing true and false answer completions.

## Why V3 Differs From V1/V2

- **V1** scored pre-supplied prefixes but used unpaired correlation (r=0.074, p=0.65). Wrong test.
- **V2** generated model answers then scored — but generation quality was poor and correctness detection unreliable.
- **V3** uses the matched-pair design that works: same question, different answer, paired t-test. This is the design that found p=0.000051 on surface vs insight.
