# Paper Section Draft: The S-Axis (Commitment Intensity)
**Status:** DRAFT for ICML 2026 Mechanistic Interpretability Workshop
**Insert position:** between C_delta section and experiments section
**IP status:** PRIVATE until May 8, 2026 submission (first public disclosure)

---

## 4. A Fourth Axis: Commitment Intensity (S)

### 4.1 Motivation

K measures where computation happens. C and C_delta measure whether the
activation geometry is coherent. Neither captures a third property of
internal state: how *concentrated* the commitment events are across
generation. A model can distribute coherence across many tokens (uncertain,
exploratory) or concentrate it into few intense spikes (decisive, committed).
This concentration pattern carries independent information about
computational state.

We define the Commitment Intensity axis as:

$$ S = \frac{\max_t C_\Delta(t)}{|\{t : C_\Delta(t) > \tau\}|} $$

where $C_\Delta(t)$ is the per-token coherence divergence trajectory
and $\tau$ is a fixed threshold (we use $\tau = 0.005$). S is the ratio
of peak intensity to spike count.

High S: few but intense coherence events — decisive processing, the
geometry crystallizes around a small number of high-magnitude feature
alignments. Low S: many small events — distributed, exploratory
processing where the geometry remains spread across competing
feature clusters.

### 4.2 Orthogonality

We verify that S is genuinely independent of K, C, and C_delta by
computing pairwise Pearson correlations on n=400 pooled samples
(200 correct + 200 hallucinated TruthfulQA responses, Gemma-2-2B).

| | K | C | C_delta | S |
|---|---|---|---|---|
| K | 1.00 | −0.02 | −0.10 | −0.06 |
| C | | 1.00 | +0.02 | +0.03 |
| C_delta | | | 1.00 | −0.003 |
| S | | | | 1.00 |

All pairwise correlations are $|r| < 0.10$, well below any conventional
independence threshold. S is an independent axis of cognitive measurement.

### 4.3 S discriminates commitment from hedging

We evaluated S on 30 topic pairs where prompts elicited either epistemic
hedging ("what's uncertain about X?") or definitive commitment
("state definitively: X"). Model: Gemma-2-2B-IT, temperature 0,
greedy decoding, max 30 tokens per generation.

**Full-length result (includes generation-length effects):**
Paired t-test: $t=-3.37$, $p=0.0021$, $d=0.62$, AUC=0.77 (commit > hedge).
Direction: 87% of pairs show $S_{\text{commit}} > S_{\text{hedge}}$.
Bootstrap 95% CI of the paired difference: [0.00020, 0.00067] (excludes 0).
Leave-one-out refits: 30/30 stay significant ($p \in [0.002, 0.004]$).
Wilcoxon signed-rank: $p < 0.0001$.

**Length-controlled retest:** We identified a confound — commitment
responses are shorter than hedging responses ($30.2$ vs $40.0$ tokens,
$p=0.0003$), and $\text{corr}(S, n_{\text{tokens}}) = -0.90$. We retest
with all trajectories truncated to the first $T=25$ tokens (n=20
surviving pairs). Effect persists: ratio $1.22\times$, $p=0.0002$,
$d=1.03$, AUC=0.81. Approximately 60% of the raw effect is genuine
signal; 40% is length-driven.

### 4.4 Layer-space S discriminates hallucination

The same geometric principle — max over spike count — transfers from the
per-token dimension to the per-layer dimension. For each of 200
TruthfulQA items (Gemma-2-2B base), we compute layer-space S over the
26-layer per-layer coherence profile of correct vs hallucinated
continuations.

| metric | d | AUC | p |
|---|---|---|---|
| K | −0.11 | 0.48 | 0.14 (null) |
| C | −0.10 | 0.49 | 0.17 (null) |
| C_delta | +0.31 | 0.57 | <0.0001 |
| **layer-S** | **+0.34** | **0.58** | **<0.0001** |

Hallucinated continuations show higher layer-space S than correct
continuations. The effect is modest but exceeds C_delta on the same
data. Direction: hallucinated responses concentrate coherence into
fewer, more intense layer events.

### 4.5 S as a confident-wrongness detector (preliminary)

If S measures the extent to which the model's geometry locks into a
single attractor, we hypothesize that *confidently wrong* generations
should show even higher S than confidently correct ones — the wrong
attractor is more geometrically isolated, producing sharper
concentration. We pre-registered a 3-condition triplet study
(osf.io/REDACTED-until-submission, April 5, 2026) testing:

- H (hedge): uncertain framing
- C+ (correct-commit): definitive true statement
- C- (wrong-commit): definitive false statement
- Prediction: $S_H < S_{C+} < S_{C-}$

30 matched topics, length-matched by design. Results will be reported
in the Experiments section. If $S_{C-} > S_{C+}$, S functions as a
mechanistic false-confidence detector at the circuit level — a distinct
capability from hallucination scoring over token outputs.

### 4.6 Relation to K, C, and C_delta

The four axes partition cognitive state into:
- **K (depth):** WHERE computation happens — layer distribution.
- **C (coherence):** HOW ALIGNED feature directions are at a given layer.
- **C_delta:** divergence between early-layer and late-layer coherence.
- **S (commitment intensity):** HOW CONCENTRATED commitment events are
  across the trajectory.

A thought can be deep and committed but incoherent (high K, high S, low C —
confident nonsense); shallow and hedged (low K, low S, low C — rote
retrieval refused); deep, coherent, and committed (high K, high S, high C —
expertise or delusion). These are distinct geometric states with
distinct behavioral correlates.

### 4.7 Limitations

1. Per-token S validated at n=30 on a single model (Gemma-2-2B-IT);
   cross-architecture replication awaits Llama/OLMo transcoder
   infrastructure.
2. The length confound is controlled but not eliminated by design;
   length-matched prompts (Section X) address this pre-registered.
3. Layer-space S is a geometric analog of per-token S; they share
   the max/spike-count formulation but operate in different dimensions
   and should be reported as distinct measurements.
4. The confident-wrongness claim is pre-registered but not yet
   empirically confirmed at time of submission if not completed.

---

## Insertion notes for main paper

- Add S to main abstract: "...we introduce four geometric axes (K, C,
  C_delta, S) measurable from SAE transcoder activations..."
- Add orthogonality table to methods (Table X).
- Add S columns to all experiment result tables.
- Cite S_DISCOVERY_NOTES.md (internal) as originating document; remove
  before submission.
- Patent language for S: commitment intensity as ratio of peak coherence
  magnitude to count of coherence events exceeding threshold, within a
  fixed-length window.

## Figures to produce (GPU-free, from existing data)

1. **Orthogonality heatmap** — 4x4 correlation matrix, all |r|<0.10 visible.
2. **S distributions per condition** — violin plot, H vs C+ vs C-.
3. **Length-controlled trajectory** — effect size vs truncation length.
4. **Layer-space S bar chart** — correct vs hallucinated means with 95% CI.
5. **Threshold sensitivity** — ratio/AUC across τ ∈ [0.003, 0.010].

---

*Draft author: darkflobi, April 5, 2026. Do not distribute — under
non-disclosure until ICML submission.*
