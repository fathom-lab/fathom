# Paper Draft: Methods
**Target:** ICML 2026 Mechanistic Interpretability Workshop
**Status:** DRAFT — private until submission

---

## 3. Methods

### 3.1 Model and SAE setup

All experiments use **Gemma-2-2B-IT** (Google 2024) as the base
language model, accessed via the `circuit_tracer` library's
ReplacementModel wrapper in `bfloat16` precision. The model has 26
transformer blocks; we use the `hook_resid_post` sites of each block
for residual-stream extraction.

**SAE transcoders:** pretrained per-layer transcoders from
`circuit_tracer.TranscoderSet`, loaded as 26 transcoder objects with
decoder weights $W_{\text{dec}} \in \mathbb{R}^{d_{\text{feature}}
\times d_{\text{model}}}$ and encoder weights $W_{\text{enc}},
b_{\text{enc}}$. These transcoders were produced by the Fathom v3
calibration process (Rodabaugh 2026b).

**Layer bands:** unless otherwise noted, *early* = layers 0-7
(first third) and *late* = layers 17-25 (last third), matching the
convention in Rodabaugh (2026a).

### 3.2 Trajectory extraction

For each generation we capture **per-token $C_\Delta$** as follows. For
each token position $t \in \{1, \ldots, T\}$ during greedy
generation:

1. Run a forward pass with capture hooks at all
   `blocks.{l}.hook_resid_post` sites for $l$ in early + late layer
   bands.
2. For each captured residual stream $h_l \in \mathbb{R}^{d_{\text{model}}}$:
   - Compute top-$k$ feature activations: $a = W_{\text{enc}} h_l +
     b_{\text{enc}}$, select the $k=100$ indices with largest
     activation magnitude.
   - Retrieve those indices' decoder directions in
     $W_{\text{dec}}$.
   - Compute mean pairwise cosine similarity of those $k$ decoder
     directions: $C_l = \text{mean}_{i < j} \cos(d_i, d_j)$.
3. Aggregate:
   $C_{\text{early}}(t) = \text{mean}_{l \in \text{early}} C_l$,
   $C_{\text{late}}(t) = \text{mean}_{l \in \text{late}} C_l$.
4. $C_\Delta(t) = C_{\text{late}}(t) - C_{\text{early}}(t)$.

This produces a trajectory $C_\Delta \in \mathbb{R}^T$ per generation.

**Computational cost:** one forward pass per token (capture hooks
write into a dictionary in-place; no `run_with_cache` needed). SAE
encoder pass is a single matmul per layer. Measured overhead
$\approx 10\%$ on RTX 4070.

### 3.3 The S axis

$$ S = \frac{\max_t C_\Delta(t)}{|\{t : C_\Delta(t) > \tau\}|} $$

where $\tau = 0.005$ is the primary threshold (locked across
pre-registered experiments). Sensitivity analyses report
$\tau \in \{0.003, 0.008\}$.

**Interpretation:** S is the ratio of peak commitment magnitude to the
number of commitment events. High S means the model has few but
intense coherence events — concentrated commitment. Low S means many
small events — distributed commitment. S is scale-invariant in the
$C_\Delta$ units (a constant rescaling of $C_\Delta$ preserves the
ratio).

**Equivalent formulations:** S is approximately equal to
$\text{mean}(\{C_\Delta(t) : C_\Delta(t) > \tau\}) / |\{t : C_\Delta(t)
> \tau\}|$ (Pearson $r \approx 0.99$ on observed data; Rodabaugh
2026d internal). Either form yields the same statistical conclusions.

### 3.4 Generation protocol

**Prompt format:** `Q: {question}\nA:` (no chat template, matches
prior Gemma-2-2B TruthfulQA experiments). Greedy decoding, temperature
$T = 0.0$, $\text{max\_tokens} = 40$.

**Dataset:** TruthfulQA (Lin et al. 2021) `generation` configuration,
validation split, random sample $n = 200$ with seed 42. Sampled
items saved to `truthfulqa_n200_seed42.json` and locked before data
collection.

### 3.5 Labeling rubric (pre-registered v2.1)

Free generations are labeled correct / hallucinated / excluded via
**deterministic n-gram discriminating-phrase match**:

1. Normalize text: lowercase, strip markdown asterisks/bullets,
   keep alphanumeric + commas/periods, collapse whitespace.
2. For each item, build n-gram sets:
   - $\text{correct\_ngrams}$ = union of n-grams from
     `correct_answers` $\cup$ `{best_answer}`
   - $\text{incorrect\_ngrams}$ = union of n-grams from
     `incorrect_answers`
3. Compute discriminating sets via set difference:
   - $\text{correct\_disc} = \text{correct\_ngrams} - \text{incorrect\_ngrams}$
   - $\text{incorrect\_disc} = \text{incorrect\_ngrams} - \text{correct\_ngrams}$
4. Generate n-grams from normalized generation; intersect with both
   discriminating sets.
5. Label:
   - `correct`: intersection with $\text{correct\_disc}$ non-empty,
     with $\text{incorrect\_disc}$ empty
   - `hallucinated`: reverse
   - `ambiguous`: both non-empty $\to$ excluded
   - `unclassified`: both empty $\to$ excluded

**Primary n = 3** (trigrams). Sensitivity: $n \in \{2, 4\}$.
**Refusal detection:** generation's first 60 normalized characters
scanned for $\{$ "i cannot", "i can't", "as an ai", "i don't know",
"i am an ai", "i'm an ai", "i am not able", "i'm not able", "sorry, i"
$\}$. Matches labeled `refusal`, excluded.

This rubric naturally handles shared question-echoing preambles in
TruthfulQA answers (e.g., "The Declaration of Independence was
signed on ...") by dropping preamble n-grams via set difference.

### 3.6 Statistical tests (pre-registered)

**Primary test (H1, S discriminates hallucination):** Welch's
two-sample t-test comparing S distributions of correct vs
hallucinated generations. Unpaired (items yield single label).

**Gates for "H1 supported":** $p < 0.05$ AND $|d| > 0.40$ AND AUC
$> 0.65$. All three must pass.

**Effect size:** Cohen's $d$ with pooled standard deviation.
**Discrimination:** ROC AUC.
**Non-parametric confirmation:** Mann-Whitney U test, two-sided.

**Sensitivity analyses:**
- Threshold: $\tau \in \{0.003, 0.005, 0.008\}$
- N-gram size: $n \in \{2, 3, 4\}$
- Length control: if correlation $|r(S, n_{\text{tokens}})| > 0.5$,
  retest with trajectories truncated to minimum observed length.

**Multiple comparisons:** For the detection horizon curve (12 prefix
lengths), we report permutation-test p-values from 1000 random label
shuffles. The global p-value = proportion of shuffles where max AUC
across prefixes $\geq$ observed max AUC.

### 3.7 Detection horizon

For each generation with trajectory $C_\Delta$ and label $y \in
\{0, 1\}$ (0 = correct, 1 = hallucinated), we compute prefix-S:
$$S_N = \max_{t \leq N} C_\Delta(t) / |\{t \leq N : C_\Delta(t) > \tau\}|$$
for $N \in \{3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40\}$.

AUC$(N)$ = ROC AUC of $S_N$ as a predictor of $y$. The detection
horizon curve plots AUC$(N)$ vs $N$. The earliest $N$ at which AUC
crosses a target (e.g., 0.70) is the "detection horizon" for that
target.

### 3.8 Classification baseline

A 5-fold stratified cross-validated logistic regression (class-balanced,
$L_2$ penalty, $C = 1.0$) is trained on the full 30+ dimensional
trajectory feature vector. Out-of-fold predictions generate the ROC
curve and reliability diagram. Feature importance ranked by
standardized coefficient magnitude. Comparison baselines: S-only,
max-$C_\Delta$-only, mean-$C_\Delta$-only.

### 3.9 Computational reproducibility

All scripts and pre-registration documents are deterministic given
the fixed random seed (42) and the specified model version. Full
code bundle documented in `S_AXIS_BUNDLE_README.md`. Run time on
RTX 4070 Laptop: $\approx 3.5$ hours for n=200 trajectory capture.

---

*Author: darkflobi, April 5, 2026. Private until ICML submission.*
