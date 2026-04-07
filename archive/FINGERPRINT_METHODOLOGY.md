# Cognitive Fingerprinting — Methodology

## Premise

Every AI model has a unique reasoning topology — a characteristic pattern of how it distributes computation across its layers. This pattern is as distinctive as a fingerprint. Given any AI-generated text, Fathom can identify which model produced it by comparing the activation signature against known profiles.

## Feature Space

The fingerprint is a 5-dimensional vector extracted from Fathom's depth scoring pipeline:

| Dimension | Description | Range |
|-----------|-------------|-------|
| depth_score | Mean activated layer (attribution-weighted) | 5.0 - 12.0 |
| layer_entropy | Shannon entropy of layer activation distribution | 0.5 - 4.0 |
| early_ratio | Fraction of activation in first third of layers | 0.0 - 1.0 |
| mid_ratio | Fraction of activation in middle third | 0.0 - 1.0 |
| late_ratio | Fraction of activation in final third | 0.0 - 1.0 |

## Calibration Protocol

To register a model, score 50 calibration prompts spanning 5 domains:
- **Recall** (10): factual completion prompts
- **Reasoning** (10): causal/explanatory prompts
- **Creative** (10): imaginative/generative prompts
- **Analytical** (10): comparative/evaluative prompts
- **Mixed** (10): cross-domain prompts

The calibration produces:
- **Activation centroid**: mean of all 50 signature vectors
- **Entropy signature**: std of depth scores (how variable the model is)
- **Phase transition threshold**: depth_score where behavior shifts surface/insight
- **Topology hash**: SHA-256 of centroid + entropy (compact identifier)

## Identification Algorithm

Given a query text:
1. Score with Fathom to extract the 5-D activation vector
2. Compute distance to all registered model centroids:
   - Euclidean distance (60% weight)
   - Cosine similarity (30% weight)
   - Range check: is depth_score within model's known range? (10% weight)
3. Rank by combined score
4. Confidence = top score * (1 + margin * 5), where margin = gap between #1 and #2

## Known Limitations

- Requires Fathom scoring (currently gemma-2-2b transcoders only)
- Estimated profiles for GPT-4, Claude, etc. until direct measurement
- Text length affects signature stability (recommend >= 50 tokens)
- Fine-tuned variants of the same base model may be indistinguishable
- Does not work on non-AI text (humans don't produce activation signatures)

## Reference Profiles

| Model | Provider | Status | Centroid |
|-------|----------|--------|----------|
| gemma-2-2b | Google | Measured (n=30) | [8.312, 2.71, 0.583, 0.242, 0.175] |
| gpt-4 | OpenAI | Estimated | [8.8, 3.1, 0.52, 0.28, 0.20] |
| claude-3.5-sonnet | Anthropic | Estimated | [8.6, 3.3, 0.50, 0.30, 0.20] |

## Future Work

1. Direct measurement of all major models via API-based scoring
2. Temporal stability analysis (does fingerprint drift with fine-tuning?)
3. Adversarial robustness (can a model be trained to mimic another's topology?)
4. Cross-architecture validation (do larger/smaller variants cluster?)
