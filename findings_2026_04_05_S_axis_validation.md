# S-Axis Validation — Gemma-2-2B-IT, n=30
**Date:** April 5, 2026
**Source data:** `truthfulqa_results/commitment_IT_replication.json` (already saved)
**Discovery:** `paper/S_DISCOVERY_NOTES.md` (April 4, 5am ET)

## TL;DR
S (Commitment Intensity = max(C_delta) / spike_count) validates at n=30 on
Gemma-2-2B-IT with p=0.0021, d=0.615, AUC=0.769, 86.7% directional.
Robust across all tested thresholds (0.003–0.010).

## Why this matters
The S-axis was originally computed from Gemma-2-2B base data (14/30 pairs,
ratio 2.1–2.6x). That base experiment was later identified as drift-contaminated
(commitment prompts triggered physics textbook completions, see
`findings_2026_04_05_commitment_correction.md`).

When S is computed on the clean IT data, the signal is STRONGER, not weaker.
The base drift actually suppressed S because physics-textbook completions
produce dense medium-intensity coherence (many spikes → raised denominator).

## Main result (threshold = 0.005, matches original discovery)
- Hedge mean S:   0.000440 (spikes/trace mean 31.6)
- Commit mean S:  0.000857 (spikes/trace mean 22.7)
- Ratio:          1.95x
- Paired t:       t=-3.371, p=0.0021
- Cohen's d:      0.615
- AUC:            0.769
- Direction:      86.7% commit > hedge

## Threshold sensitivity
| threshold | ratio | p-paired | d     | AUC   | dir%  |
|-----------|-------|----------|-------|-------|-------|
| 0.003     | 1.88  | 0.0025   | 0.605 | 0.782 | 80.0% |
| 0.004     | 1.92  | 0.0023   | 0.609 | 0.801 | 83.3% |
| 0.005     | 1.95  | 0.0021   | 0.615 | 0.769 | 86.7% |
| 0.006     | 1.83  | 0.0011   | 0.659 | 0.702 | 80.0% |
| 0.007     | 1.87  | 0.0010   | 0.669 | 0.712 | 80.0% |
| 0.008     | 2.26  | 0.0005   | 0.716 | 0.799 | 86.7% |
| 0.010     | 2.13  | 0.0009   | 0.678 | 0.746 | 76.7% |

Best: threshold=0.008 (p=0.0005, d=0.716, AUC=0.799).

## BASE comparison (contaminated data, same method)
- Ratio:      1.36x
- p:          0.1205 (not significant)
- d:          0.292
- AUC:        0.678
- Direction:  60.0% commit > hedge

The drift contamination weakens but does not destroy the directional signal.
On the valid IT data, the signal is clean.

## What's still needed
1. **Out-of-sample validation:** run S on TruthfulQA n=200 data. Check
   whether S discriminates correct-answer vs hallucinated-answer traces.
2. **Orthogonality check:** correlate S with K and C across all items.
   Expect r < 0.3 for all pairs (three independent axes claim).
3. **Cross-architecture replication:** run commitment experiment on
   Llama-3.1-8B or OLMo once SAE transcoders are available.
4. **Pre-registration:** formalize threshold (0.005 as pre-discovery choice,
   or 0.008 as best performer) before cross-architecture tests.

## Publication-readiness
Gates claimed in darkflobi's ICML plan:
- p < 0.05 ✓ (0.0021 at main threshold, best 0.0005)
- AUC > 0.75 ✓ (0.769 at main, 0.799 at best)
- directional > 70% ✓ (86.7%)

All three gates cleared at n=30 on Gemma-2-2B-IT.

## Next ask
Flobi: do you want me to push straight to TruthfulQA n=200 out-of-sample
validation, or pause here and update the paper with this result first?
