# Pre-Registration: Fathom Depth Measurement — Architecture Universality and Temperature Effects

**Registration Date:** March 29, 2026
**Principal Investigator:** Alexander Rodabaugh (darkflobi)
**OSF Project URL:** [to be assigned upon registration]

---

## Study Information

### Title
Architecture Universality of Reasoning Depth Topology in Transformer Language Models

### Description
We hypothesize that the structural distinction between surface recall (output-proximal activation concentration) and integrated reasoning (mid-layer activation distribution) is a universal property of transformer architectures, not an artifact of a single model. We will test this by measuring Fathom depth scores on a minimum of 5 architectures from different labs and comparing the surface/insight depth ratio (the "Fathom Constant") across all tested systems.

Additionally, we will complete the temperature sweep experiment (T in {0.0, 0.3, 0.7, 1.0}) to determine whether task-specific stochastic resonance is robust across the full temperature range.

### Hypotheses

**H1 (Architecture Universality):** The Fathom Constant (surface_mean_depth / insight_mean_depth) will fall within the range [0.98, 1.06] for all tested architectures, indicating a conserved structural property.

**H2 (Temperature Resonance):** Concept-level integration tasks will show peak depth at T=0.3 (not T=0.0), while formal-precision tasks will show peak depth at T=0.0. This pattern will hold across at least 3 architectures.

**H3 (Decomposability):** For any pair of responses to the same question where one organizes information as an irreducible causal web and the other as independent modular components, the causal web response will score higher on Fathom depth regardless of response length.

---

## Design Plan

### Study Type
Observational — computational measurement of existing model behavior

### Study Design
Within-subjects (same prompt pairs across all architectures). Each architecture receives the same 30 vocabulary-controlled prompt pairs from the validated calibration set.

### Randomization
Not applicable — deterministic scoring at T=0.0.

### Blinding
Not applicable — automated measurement, no subjective coding.

---

## Sampling Plan

### Existing Data
Completed: Gemma-2-2B (26 layers, Google), Qwen3-0.6B (28 layers, Alibaba)
These results are LOCKED and will not be re-run.

### Data Collection Procedures

**Architectures to test (minimum 5 total):**
1. Gemma-2-2B (COMPLETED) — Google, 26 layers
2. Qwen3-0.6B (COMPLETED) — Alibaba, 28 layers
3. GPT-2 (124M) — OpenAI, 12 layers — PENDING
4. Llama-3.2-1B — Meta, 16 layers — PENDING
5. Mistral-7B — Mistral AI, 32 layers — PENDING (requires >8GB GPU or quantization)

Additional architectures if accessible:
6. Phi-3-mini (3.8B) — Microsoft
7. RWKV-6 (non-transformer architecture, falsification test)

### Sample Size
30 prompt pairs per architecture (fixed calibration set).
Total measurements: 30 pairs x 5+ architectures x 4 temperatures = 600+ depth scores.

### Sample Size Rationale
Power analysis from Phase 3 replication: with delta=0.257 and SD=0.312, n=30 achieves power=0.97 at alpha=0.05 (one-tailed). This is sufficient per-architecture.

### Stopping Rule
Data collection stops when all planned architectures have been scored. No sequential testing or early stopping.

---

## Variables

### Measured Variables
- **depth_score**: Normalized depth from SAE circuit attribution (continuous, [0, 1])
- **layer_entropy**: Shannon entropy of activation distribution across layers
- **early_ratio, mid_ratio, late_ratio**: Proportion of activation in early/mid/late layer thirds
- **feature_count**: Number of SAE features exceeding activation threshold
- **weighted_depth_score**: Activation-magnitude-weighted mean layer index
- **fathom_constant**: Ratio of mean surface depth to mean insight depth per architecture

### Indices
- **Fathom Constant** = mean(depth_score_surface) / mean(depth_score_insight)
- Computed per architecture from the 30-pair calibration set at T=0.0

---

## Analysis Plan

### Statistical Models

**H1 test:** One-sample t-test for each architecture's Fathom Constant against the null value of 1.00 (no difference between surface and insight). Additionally, compute the cross-architecture variance of the Fathom Constant; if variance < 0.01 across 5+ architectures, we claim structural universality.

**H2 test:** Two-way repeated-measures ANOVA: Task Type (concept-integration vs formal-precision) x Temperature (0.0, 0.3, 0.7, 1.0), with depth_score as the dependent variable. We predict a significant interaction: concept tasks increase at T=0.3, formal tasks decrease.

**H3 test:** Paired t-test comparing depth scores of causal-web vs modular responses to the same questions, controlling for response length as a covariate.

### Transformations
Depth scores are already normalized to [0, 1]. No additional transformations planned.

### Inference Criteria
Alpha = 0.05, two-tailed for H1 and H3, with Bonferroni correction across architectures for H1.
Effect sizes reported as Cohen's d.

### Data Exclusion
Prompts that fail to generate coherent responses (e.g., degenerate repetition, empty output) will be excluded and documented. Maximum 3 exclusions per architecture; if more, the architecture is flagged as incompatible.

### Exploratory Analyses
- Cross-architecture activation profile clustering (are surface/insight profiles more similar across architectures than within?)
- Correlation between model size (parameter count) and Fathom Constant magnitude
- Non-transformer architectures (RWKV) as falsification test

---

## Scripts and Materials

### Analysis Scripts
All code is publicly available at: https://github.com/darkflobi/fathom

Key files:
- `depth_scorer.py` — Core SAE circuit attribution engine
- `experiment_power_replication.py` — 30-pair calibration protocol
- `experiment_temperature.py` — Temperature sweep protocol
- `fathom_constant_v2.py` — Cross-architecture ratio calculator

### Calibration Dataset
The 30 vocabulary-controlled prompt pairs are fixed and published in `results_power_replication.json`. Topics span: earthquakes, metabolism, ocean currents, immune response, dreams, migration, combustion, photosynthesis, gravity, black holes, and 20 others.

---

## Prior Results (Locked — Not Re-Analyzable)

### Phase 3 Pre-Registered Replication (n=30)
- delta = +0.257 (surface mean layer minus insight mean layer)
- t(29) = 4.500, p = 0.000051 (one-tailed)
- 83.3% directional support
- Leave-one-out robust (result holds when removing any single pair)
- Orthogonal to prediction confidence (r = 0.001, p = 0.996)

### Fathom Constant (2 architectures)
- Gemma-2-2B: 1.0302
- Qwen3-0.6B: 1.0020
- Mean: 1.0212, spread: 0.028

---

## Timeline

| Date | Milestone |
|------|-----------|
| March 29, 2026 | Pre-registration submitted |
| March 30, 2026 | GPT-2 depth scoring |
| March 31, 2026 | Llama-3.2-1B depth scoring |
| April 2, 2026 | Temperature sweep completion (T=0.7, T=1.0) |
| April 5, 2026 | Mistral-7B depth scoring (if GPU available) |
| April 7, 2026 | Analysis and write-up |
| April 10, 2026 | Results posted to OSF |

---

## Conflict of Interest
The principal investigator has filed a provisional patent (US 64/020,489 & 64/021,113) covering the Fathom measurement method. This pre-registration is filed to ensure scientific transparency despite commercial interest.

## Funding
Self-funded. No external grants or corporate sponsorship.
