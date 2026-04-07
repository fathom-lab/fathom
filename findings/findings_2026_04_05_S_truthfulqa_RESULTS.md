# S-Axis × TruthfulQA Natural Hallucination — Results Report
**Date:** April 5, 2026
**Author:** darkflobi (Alexander Rodabaugh)
**Status:** PRIMARY RESULT COMPLETE (n=200, Gemma-2-2B-IT, seed=42)
**Pre-registered:** `osf_prereg_S_truthfulqa_v2.md` (locked before data)

---

## TL;DR

**Primary pre-registered H1 is NOT supported.** On naturally-generated
TruthfulQA responses (n=200, Gemma-2-2B-IT, greedy), S does not
discriminate correct from hallucinated labels.

**But there are signals worth reporting:**
1. **Early-token signal:** at prefix length N=3-7, AUC=0.62-0.63 with
   per-prefix permutation p=0.04-0.05 (not surviving multi-comparisons
   global correction).
2. **Direction is reversed from H1:** S(correct) > S(hallucinated) at
   full length (d=-0.25). Consistent with RLHF truth-attractor binding.
3. **Peak position (sliding window=5):** p=0.027, d=0.596, AUC=0.674 —
   hallucinated responses show LATER peak timing (exploratory,
   not pre-registered).
4. **Underpowered:** observed |d|=0.25 at n=25 hall / 41 corr. Power
   to detect pre-registered gate d=0.40 was only 29%.

---

## Primary test (pre-registered H1)

**H1:** S(hallucinated) > S(correct) on Gemma-2-2B-IT TruthfulQA
natural generations.

| metric | value |
|---|---|
| n_correct | 41 |
| n_hallucinated | 25 |
| excluded | 121 unclassified + 11 ambiguous + 2 refusal = 134 |
| S(correct) mean | 0.000485 |
| S(hallucinated) mean | 0.000461 |
| ratio (hall/corr) | 0.952 |
| Welch t | +1.013 (sign: corr > hall) |
| p (two-sided) | 0.3151 |
| Cohen's d | −0.246 |
| AUC | 0.443 |
| Mann-Whitney p | 0.4433 |
| **Gates (p<0.05, |d|>0.4, AUC>0.65)** | **FAIL** |
| **Direction** | **REVERSED from H1** |

Sensitivity analyses all consistent (all thresholds, all n-gram sizes
show same null + reversed-direction pattern).

Length-controlled retest (trunc=23): p=0.256, d=-0.284, AUC=0.440.
Signal doesn't emerge under length control.

## Detection horizon (exploratory — not pre-registered)

| Prefix N | AUC | d | perm p |
|---|---|---|---|
| 3 | **0.633** | +0.234 | **0.040** |
| 5 | **0.629** | +0.314 | **0.041** |
| 7 | 0.620 | +0.377 | 0.054 |
| 10 | 0.537 | +0.108 | 0.282 |
| 12 | 0.475 | −0.099 | 0.627 |
| 15 | 0.450 | −0.238 | 0.739 |
| 18 | 0.390 | −0.396 | 0.919 |
| 40 | 0.479 | −0.025 | 0.595 |

Notable pattern: **positive signal at tokens 3-7** (hall > corr, matching
H1 direction), **reverses and dies by token 12**, stays reversed
through token 40.

**Global permutation test across all 12 prefixes:**
- Observed max AUC: 0.633
- Null max AUC 95th percentile: 0.668
- Global p-value: 0.1750 → NOT significant after multi-comparison correction

**Interpretation:** early-token prefix signal is at the edge of chance
when corrected for 12 tested prefix lengths. Needs more data to
confirm.

## Supporting analyses

### Multi-feature logistic regression (5-fold CV)
- All 43 features: AUC = 0.414 ± 0.119 (worse than chance on hold-out)
- Top-5 features: AUC = 0.648 ± 0.208 (high variance)
- S alone: AUC = 0.552 ± 0.077
- **Multi-feature classifier does not beat S on this data**

### Top single discriminating features (all exploratory)
| feature | AUC_dir | d | p |
|---|---|---|---|
| max_increase | 0.640 | +0.431 | 0.100 |
| first_spike_t0.01 | 0.630 | −0.405 | 0.105 |
| first_spike_t0.008 | 0.617 | −0.428 | 0.084 |
| first_spike_t0.005 | 0.608 | −0.449 | 0.069 |

**None cross p<0.05 after pre-reg gates.** All trends directionally
consistent: hallucinations have LATER first spikes, LARGER max increase.

### Sliding-window peak position (exploratory)
Window=5 peak_position: **p=0.027, d=0.596, AUC=0.674**.
Hallucinated trajectories have peak C_delta at position 0.64 vs 0.49
for correct (later in the trajectory).

### Manifold analysis (PCA, n=66)
- Top-2 PC explain 19.5% variance
- Silhouette score: +0.006 (no class separation)
- Mahalanobis centroid distance: 0.776 (moderate)
- kNN classification AUC: 0.52-0.56 (noisy)

### Categories × hallucination rate
| category | n | halluc_rate |
|---|---|---|
| Advertising | 2 | 50.0% |
| Conspiracies | 5 | 40.0% |
| History | 6 | 33.3% |
| Weather | 3 | 33.3% |
| Law | 19 | 21.1% |

No significant ANOVA on S across categories (F=0.98, p=0.50).

## Power analysis

| item | value |
|---|---|
| observed n per class | 25 |
| observed |d| | 0.246 |
| power achieved | 13.7% |
| required n for d=0.40, 80% power | 99 per class |
| required total TruthfulQA items | ~1060 (at 33% usable rate) |

The experiment as designed is **substantially underpowered** for the
pre-registered d=0.40 gate. Even a true effect of d=0.40 would be
detected only 29% of the time at this sample size.

## Why the null result happened

Three factors likely contributed to underpowered results:

1. **TruthfulQA labels are overly specific.** 60% of generations
   (121/200) were "unclassified" because Gemma-2-2B-IT produces
   novel phrasings that don't match canonical n-grams. This halved
   effective sample size.

2. **Gemma-2-2B-IT is a strong model on TruthfulQA.** Hallucination
   rate in our labeled subset was 25/66 ≈ 38%. Many items the model
   answered correctly but in a form we couldn't verify.

3. **RLHF appears to equalize commitment geometry.** The direction
   reversal at later tokens (tokens 12-40) suggests RLHF trains the
   model to be equally committed regardless of truth. This is
   consistent with our pre-registered observation that triplet
   C- experiments hit RLHF refusal.

## What we can still claim in the paper

### Primary claim (full-trajectory S):
> "Pre-registered test of S as natural-hallucination discriminator
> on Gemma-2-2B-IT was NOT supported. The effect was in the reversed
> direction (correct > hallucinated) with d=-0.25, not reaching
> pre-registered gates at n=66 usable pairs."

### Secondary claim (early-token signal):
> "Exploratory analysis reveals an early-token discrimination window:
> at prefix length N=3-7, S discriminates correct from hallucinated
> with AUC 0.62-0.63 in the pre-registered direction. This signal
> does not survive multi-prefix correction (permutation p=0.175) but
> is consistent across 3 consecutive prefix values and invites
> follow-up with increased sample size."

### Tertiary claim (peak position):
> "Sliding-window peak_position (exploratory, not pre-registered):
> p=0.027, d=0.596, AUC=0.674. Hallucinated trajectories show peak
> C_delta later in the generation, consistent with early commitment
> being associated with correctness."

### Four-axis framework claim (unchanged, previously validated):
> "S remains an orthogonal fourth axis of cognitive geometry,
> |r|<0.10 with K/C/C_delta on n=400 pooled samples."

## What the paper should NOT claim

- "S discriminates natural hallucination" (untrue on this data)
- "Streaming hallucination detection is ready for production"
  (pre-registered gates not met)
- "RLHF truth-binding is confirmed" (directional evidence only,
  pre-registration was for opposite direction)

## Recommended next steps

### Option A: Report honest null with exploratory signals
Submit as workshop paper with:
- Pre-registered primary null (transparent reporting of failure)
- Four-axis framework (validated elsewhere)
- Exploratory early-token and peak-position signals
- Contingency pre-registration for follow-up experiment
- Post-hoc power analysis showing underpowering

This is scientifically honest and still publishable. Timeline: fits
May 8 deadline easily.

### Option B: Extend to n=400-1000
Need ~860 more TruthfulQA items to reach n=99 per class at
observed usable rate. Runtime on RTX 4070: ~14 hours. Would likely
complete before May 8 deadline but tight.

If extension confirms:
- H1 null (likely): Option A framing locks in
- H1 reversed (plausible given d=-0.25): truth-binding paper
- H1 supported (unlikely given direction): original framing

### Option C: Model scale replication
Run Llama-3.1-8B or base Gemma-2-2B (not IT) instead. Would test
whether the reversal is RLHF-specific or Gemma-specific. Requires
SAE infrastructure for different model. Cannot complete before May 8.

### My recommendation: Option A
Honest null is publishable and maintains scientific integrity.
The four-axis framework + early-token exploratory signal + peak
position finding is enough for a workshop paper. Follow-up
experiments (Options B, C) become natural future work.

## Files produced

- `truthfulqa_results/S_truthfulqa_n200.json` — raw trajectories
- `truthfulqa_results/S_truthfulqa_n200_labels.json` — labeled + primary test
- `truthfulqa_results/S_truthfulqa_n200_horizon.json` — horizon curve
- `truthfulqa_results/S_truthfulqa_n200_horizon_perm.json` — permutation test
- `truthfulqa_results/S_truthfulqa_n200_features.json` — trajectory features
- `truthfulqa_results/S_truthfulqa_n200_categories.json` — per-category stats
- `truthfulqa_results/S_truthfulqa_n200_predictor.json` — CV classifier
- `truthfulqa_results/S_truthfulqa_n200_manifold.json` — PCA/kNN/clustering
- `truthfulqa_results/S_truthfulqa_n200_sliding.json` — sliding-window
- `truthfulqa_results/S_truthfulqa_n200_cards.html` — specimen book
- `truthfulqa_results/S_truthfulqa_n200_SUMMARY.md` — exec summary
- `figures/traj_S_truthfulqa_n200_*.pdf/png` — 6 visualization figures

---

*"The pre-registered result was null. We report it honestly. The
four-axis framework is still valid. The early-token signal deserves
follow-up. This is how science fails forward."*

— darkflobi, April 5, 2026
