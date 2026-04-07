# Hallucination Correlation Study V2 — Model-Generated Answers

**Date:** 2026-03-26 04:50:18
**Runner:** fathom_hallucination_v2.py
**Duration:** 450.8s

## Design Change from V1

V1 scored pre-supplied correct vs incorrect answer prefixes (r=0.074, p=0.65, n=40).
That tested text processing differences, not reasoning differences.

V2 feeds each TruthfulQA question to the model, captures its own generated answer,
scores the depth signature, and checks against ground truth. This tests whether the
model reasons differently when it produces a correct vs incorrect answer.

## Method

- Dataset: TruthfulQA (n=50 questions scored)
- For each question: generate greedy completion (max 20 tokens)
- Score the generation with DepthScorer (depth_score, weighted_depth_score, layer_entropy)
- Check answer correctness via token overlap >= 0.5 against best/correct answers
- Compute Pearson r, AUC, and optimal classification thresholds

## Sample Balance

| Group | Count | % |
|-------|-------|---|
| Correct | 0 | 0.0% |
| Incorrect | 50 | 100.0% |

## Correlation Results

| Metric vs Accuracy | Pearson r | p-value | Significant |
|--------------------|-----------|---------|-------------|
| depth_score | nan | nan | NO |
| weighted_depth_score | nan | nan | NO |
| layer_entropy | nan | nan | NO |

## Group Means

| Metric | Correct | Incorrect | Delta |
|--------|---------|-----------|-------|
| depth_score | nan | 8.2690 | nan |
| weighted_depth_score | nan | 8.2690 | nan |
| layer_entropy | nan | 4.1806 | nan |

## Discrimination (AUC)

| Metric | AUC |
|--------|-----|
| depth_score | 0.5000 |
| weighted_depth_score | 0.5000 |
| layer_entropy | 0.5000 |

## Optimal Classification Thresholds

| Metric | Details |
|--------|---------|
| depth_score | threshold=7.3450, P=0.000, R=0.000, F1=0.000, Acc=0.980 |
| layer_entropy | threshold=4.0837, P=0.000, R=0.000, F1=0.000, Acc=0.980 |

## Discovery Threshold

|r| > 0.3 AND p < 0.01 required for significance.

**Result: Below discovery threshold at n=50.**

## ROC Data (depth_score, first 10 points)

```json
[]
```
