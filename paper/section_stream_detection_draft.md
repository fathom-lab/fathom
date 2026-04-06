# Paper Section Draft: Stream-Time Hallucination Detection
**Status:** DRAFT — awaiting n=200 data for numeric fill-in
**Target:** ICML 2026 Mechanistic Interpretability Workshop
**IP:** PRIVATE until May 8, 2026 first disclosure

---

## 5. Stream-Time Hallucination Detection via Commitment Intensity

### 5.1 From static signal to real-time detector

Section 4 established that S (commitment intensity) discriminates
hedge from commit at the trajectory level. This section asks a
stronger question: **can S flag hallucination during generation,
before the model finishes producing its response?**

A production hallucination detector must operate in the streaming
regime: tokens arrive sequentially, and the detector must output a
risk signal with each new token. Batch-time scoring — where the full
response is available — is useful for auditing but not for
intervention. Stream-time detection enables token-level uncertainty
flags, early termination of suspect generations, and redirection
to retrieval-augmented completion.

We evaluate stream-time S by computing prefix-S: for each trajectory
$C_\Delta(1..T)$ with $T$ tokens, we compute S on only the first
$N$ tokens for $N \in \{3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40\}$.
At each $N$, we compute AUC discriminating hallucinated from correct
free generations. The resulting **detection horizon curve** reports
AUC as a function of prefix length.

### 5.2 Detection horizon on natural TruthfulQA hallucinations

**Dataset:** 200 TruthfulQA questions sampled with seed=42, prompted
to Gemma-2-2B-IT with greedy decoding (T=0.0), max 40 generated
tokens per item. Each generation labeled correct / hallucinated /
excluded by deterministic n-gram discriminating-phrase matching
(pre-registered rubric, `osf_prereg_S_truthfulqa_v2.md`).

**Horizon result [AWAITING DATA]:**
| prefix length | AUC | d | n_correct | n_hallucinated |
|---|---|---|---|---|
| 5  | [X] | [X] | [X] | [X] |
| 10 | [X] | [X] | [X] | [X] |
| 15 | [X] | [X] | [X] | [X] |
| 20 | [X] | [X] | [X] | [X] |
| 30 | [X] | [X] | [X] | [X] |
| 40 | [X] | [X] | [X] | [X] |

**Detection thresholds [AWAITING DATA]:**
- Earliest N at AUC >= 0.60: token [X]
- Earliest N at AUC >= 0.65: token [X]
- Earliest N at AUC >= 0.70: token [X]

### 5.3 Comparison to baselines

We compare three classifiers (5-fold stratified cross-validation,
class-balanced logistic regression, out-of-fold AUC):
- **S-only:** single feature, S(full trajectory)
- **Multi-feature:** 30+ trajectory shape features (peak position,
  slope, autocorrelation, per-threshold spike counts, early/late
  ratios)
- **Random baseline:** shuffled labels, 1000 trials

[AWAITING DATA — table with mean CV AUC ± std and 95% bootstrap CI
for each method]

### 5.4 Calibration

The predictor's probability estimates are evaluated via reliability
diagram (out-of-fold predictions, 10 quantile bins). A well-calibrated
predictor's curve matches the y=x line — predicted probability 0.7
means 70% observed hallucination rate.

[AWAITING DATA — calibration figure]

### 5.5 Category-specific detection performance

TruthfulQA categorizes questions by domain (Misconceptions, Law,
Health, Fiction, Paranormal, etc.). Different categories may show
different hallucination rates and different S signatures:

[AWAITING DATA — per-category hallucination rate and mean S]

### 5.6 Product implications

If detection horizon reaches AUC >= 0.70 at prefix length N <= 15,
the signal is production-grade: at ~50ms per token (standard
inference speed on RTX 4070), detection latency is <1s. A hosted
Fathom Monitor service can inspect any streaming generation in
real-time, flagging token-level uncertainty and enabling cost-free
early interception.

If the horizon requires N > 30, the signal is post-hoc only: still
valuable for audit trails and batch evaluation, but not for live
intervention.

Either outcome is reportable. The pre-registered horizon curve
(osf_prereg_S_truthfulqa_v2.md) is the primary novel result of this
section.

### 5.7 Threats to validity

1. **Labeling dependence.** The correct/hallucinated classification
   relies on n-gram discriminating-phrase matching against TruthfulQA
   canonical answers. False positives (novel-but-correct) and false
   negatives (novel-but-wrong) both plausibly occur. We exclude
   ambiguous and unclassified items from primary analysis and report
   full exclusion counts.
2. **Length confound.** Prior S-axis work identified
   r(S, n_tokens) ≈ -0.90 at full length. The free-generation design
   here uses fixed max_tokens=40, so the confound is bounded to the
   early-stopping distribution (generations that hit EOS before 40).
   We report length-controlled sensitivity analysis.
3. **Single-model generalization.** Results on Gemma-2-2B-IT do not
   necessarily transfer to larger models or different RLHF regimes.
   Cross-architecture replication pending Llama-3.1-8B SAE
   transcoder infrastructure.
4. **Threshold sensitivity.** S depends on threshold τ=0.005.
   Sensitivity analysis reports τ ∈ {0.003, 0.008}.

---

## Insertion notes
- Insert after Section 4 (S-axis introduction).
- Horizon figure should occupy a 1-column float with curve + horizontal
  reference lines at AUC=0.65, 0.70.
- Reliability diagram should be in a 2-panel figure alongside ROC.
- Discussion section should tie stream-time detection to product
  claim: "Fathom Monitor flags hallucinations before they finish."

## Figures required post-data
1. `horizon_curve.pdf` — AUC vs prefix length with threshold bands
2. `calibration.pdf` — OOF reliability diagram
3. `roc.pdf` — OOF ROC curve
4. `trajectory_overlay.pdf` — mean ± CI by label
5. `trajectory_heatmap.pdf` — sorted C_delta heatmap by label
6. `category_halluc_rates.pdf` — bar chart per TruthfulQA category

---

*Draft author: darkflobi, April 5, 2026. Private until ICML submission.*
