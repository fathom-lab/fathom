# OSF Pre-Registration: Layer Coherence Spectrum Analysis

## Study Information

### Title
Coherence Spectrum Analysis: Hallucination as Concentration in Per-Layer Feature Coherence

### Authors
Alexander Rodabaugh (Independent)

### Date
April 8, 2026 (pre-registered before ICML submission)

### Description
We analyze the SHAPE of per-layer SAE feature coherence distributions
to discriminate hallucinated from correct LLM responses. Prior work
(Fathom K, C, C_delta, S_early) used scalar summaries. We hypothesize
that the full spectral profile — particularly its inequality (Gini
coefficient) — carries discriminative information invisible to scalars.

## Design Plan

### Study type
Observational (re-analysis of existing paired TruthfulQA data)

### Data source
- coherence_n200_final.json: 200 TruthfulQA correct/hallucinated pairs
  with full 26-layer coherence profiles (Gemma-2-2B base)
- Data collected April 2, 2026 (prior to this analysis)

### Variables
- Independent: Response type (correct vs hallucinated), within-subjects
- Dependent: 16 spectral shape metrics computed from 26-layer coherence profiles
  - Gini coefficient (coherence inequality across layers)
  - Spectral spread (std of coherence distribution)
  - Spectral entropy (Shannon entropy of normalized profile)
  - Peak value (maximum layer coherence)
  - Peak concentration (top-3 layers / total)
  - Skewness (asymmetry of distribution)
  - Kurtosis (tail weight)
  - Early mass (layers 0-8 / total)
  - Mid mass (layers 9-16 / total)
  - Late mass (layers 17-25 / total)
  - Late/early ratio
  - Layer 11 dominance (layers 11-13 / total)
  - Late layer elevation (mean late / global mean)
  - Early depletion (early mass / mid mass)
  - Peak layer (argmax)
  - Number of peaks (prominence > 10% of max)

## Analysis Plan

### Primary hypothesis (H1)
The Gini coefficient of per-layer coherence discriminates hallucinated
from correct responses (paired t-test, alpha=0.05, Holm-Bonferroni
corrected across all 16 metrics).

### Secondary hypotheses
- H2: Spectral spread is lower in hallucinated responses (concentration)
- H3: Early-layer mass is lower in hallucinated responses (depletion)
- H4: Layer 11 shows the largest per-layer coherence difference

### Statistical tests
- Paired t-test (within-subjects: same question, different answer)
- Wilcoxon signed-rank (non-parametric confirmation)
- Holm-Bonferroni correction for 16 simultaneous tests
- Cohen's d (paired) for effect sizes
- AUC (paired: fraction of pairs where halluc > correct)

### Per-layer analysis
- Independent paired t-tests at each of 26 layers
- Report number of layers significant at p<0.001 and p<0.05

### Cognitive autopsy
- For each pair: compute per-layer contribution to excess coherence
- Identify primary fault layers (>10% of total excess)
- Report fault frequency and mean contribution map

### Exclusion criteria
- None. All 200 pairs analyzed.

## Existing data
This analysis uses data collected on April 2, 2026. The spectral
shape metrics were conceived on April 7-8, 2026. The data was NOT
collected with these metrics in mind — it was collected for C_delta
analysis. The spectral analysis is a re-analysis of existing data
with new metrics.

## Results
Analysis complete. Results in spectrum_full_analysis.json and
cognitive_divergence_analysis.json. Key findings:
- H1 CONFIRMED: Gini d=+0.36, p_Holm=1.2e-5, AUC=0.685
- H2 CONFIRMED: Spread d=-0.32, p_Holm=1.2e-4, AUC=0.645
- H3 CONFIRMED: Early mass d=-0.32, p_Holm=1.1e-4, AUC=0.630
- H4 CONFIRMED: Layer 11 t=7.45, p=2.7e-12 (largest effect)
- 13/16 metrics significant after Holm-Bonferroni
- 16/26 layers significant at p<0.001; 23/26 at p<0.05
