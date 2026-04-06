# S-Axis Experimental Bundle — README

**Experiment:** S-axis (commitment intensity) × TruthfulQA natural hallucination
**Pre-registered:** osf_prereg_S_truthfulqa_v2.md (v2.1 with n-gram labeling amendment)
**Model:** google/gemma-2-2b-it
**Date:** April 5, 2026
**IP status:** PRIVATE until ICML 2026 submission (May 8, 2026) = first disclosure

---

## One-line status check

```bash
# How far has the GPU run gotten?
python -c "import json; d=json.load(open('truthfulqa_results/S_truthfulqa_checkpoint.json')); print(f'{d[\"next_idx\"]}/200')"
```

## After GPU run completes — one command runs everything

```bash
python run_all_S_analyses.py
```

This runs (in order):
1. Primary pre-registered test — Welch's t-test S(correct) vs S(halluc)
2. Predictive horizon curve — AUC at each prefix length
3. Trajectory shape features — 30+ per-item features + classifier
4. Category × geometry — hallucination rate per TruthfulQA category
5. Calibrated predictor — 5-fold CV + feature importance
6. Visualization bundle — 6 publication figures
7. Sliding-window S — dynamic commitment intensity
8. Horizon permutation test — null distribution (1000 shuffles)
9. Trajectory specimen cards — HTML specimen book
10. Executive summary — one-page markdown

Then read the summary:

```bash
cat truthfulqa_results/S_truthfulqa_n200_SUMMARY.md
```

---

## Files in this bundle

### Pre-registration & documentation
- `osf_prereg_S_truthfulqa_v2.md` — LOCKED pre-registration (v2.1)
- `osf_prereg_S_triplet.md` — superseded by v2 (triplet design hit RLHF refusal blocker)
- `S_AXIS_BUNDLE_README.md` — this file
- `findings_2026_04_05_S_axis_thorough.md` — prior S-axis validation work
- `findings_2026_04_05_commitment_correction.md` — drift artifact correction memo
- `paper/section_S_axis_draft.md` — four-axis framework paper section
- `paper/section_stream_detection_draft.md` — stream-time detection paper section
- `paper/S_DISCOVERY_NOTES.md` — original S-axis discovery (April 4)

### Data
- `truthfulqa_results/truthfulqa_n200_seed42.json` — 200 locked TruthfulQA items (seed=42)
- `truthfulqa_results/S_truthfulqa_n200.json` — main run output (per-item trajectories)
- `truthfulqa_results/S_truthfulqa_pilot.json` — 5-item micro-pilot (pipeline validation)
- `truthfulqa_results/S_truthfulqa_checkpoint.json` — resumable checkpoint

### Experiment driver
- `load_truthfulqa_for_S.py` — deterministic dataset loader (seed=42)
- `run_S_truthfulqa.py` — GPU experiment driver (with --pilot, --resume)

### Analysis scripts
- `analyze_S_truthfulqa.py` — primary pre-registered test (Welch's t-test)
- `analyze_S_horizon.py` — detection horizon curve
- `analyze_trajectory_features.py` — 30+ shape features + multi-feature classifier
- `analyze_category_geometry.py` — category × geometry
- `analyze_S_sliding.py` — sliding-window (dynamic) S
- `analyze_S_horizon_permutation.py` — null distribution via permutation
- `S_calibrated_predictor.py` — 5-fold CV + feature importance + calibration

### Visualization
- `visualize_S_trajectories.py` — 6 paper figures (overlay, spaghetti, heatmap, S-dist, horizon, cards)
- `generate_trajectory_cards.py` — HTML specimen book

### Orchestration
- `run_all_S_analyses.py` — runs every analysis step
- `make_executive_summary.py` — one-page markdown summary

### Verification
- `verify_s_axis.py` — 14-assertion reproducibility check (prior S validation)

---

## Labeling algorithm (pre-registered v2.1)

Free-generation responses labeled as correct / hallucinated / excluded via
**n-gram discriminating phrase match**:

1. For each TruthfulQA item, build n-gram sets from correct_answers and incorrect_answers
2. Set difference: discriminating_correct = correct - incorrect (and vice versa)
3. Match generation's n-grams against discriminating sets
4. Label based on which set hit (or both = ambiguous, neither = unclassified)

**Primary n = 3 trigrams** (locked). Sensitivity: n=2, n=4.

Refusal check: generation's first 60 chars scanned for explicit refusal patterns.

## S formula (locked)

$$ S = \frac{\max_t C_\Delta(t)}{|\{t : C_\Delta(t) > \tau\}|} $$

Primary threshold τ = 0.005. Sensitivity: τ ∈ {0.003, 0.008}.

## Gates (pre-registered)

For H1 to be "supported": p < 0.05 AND |d| > 0.40 AND AUC > 0.65.

## Expected outcomes (all publishable)

| Outcome | S signature | Paper framing |
|---------|-------------|---------------|
| H1 supported, d > 0 | S(hall) > S(corr) | Stream-time hallucination detector |
| H1 reversed, d < 0 | S(corr) > S(hall) | RLHF truth-attractor binding |
| H1 null, d ≈ 0 | no discrimination | S is commitment metric, not hallucination signal |

---

## Contingency playbook

**If H1 strongly supported (AUC > 0.70):**
- Add to ICML paper as Section 5 primary result
- Patent #3 amendment claim: stream-time detector
- Blog post after submission

**If H1 weakly supported (AUC 0.55-0.70):**
- Report as independent signal, ensemble with other detectors
- Compare to logit-entropy baseline (next experiment)

**If H1 null (AUC ~0.50):**
- Keep S-axis framework but narrow claims to commitment discrimination
- Investigate: does S work on a different model scale? (Llama-3.1-8B next)
- Paper becomes "S-axis as an independent cognitive measurement axis"

**If H1 reversed:**
- Investigate RLHF truth-binding mechanism
- Paper becomes "RLHF binds commitment geometry to factual truth"
- This is actually more interesting than H1 supported

---

## Troubleshooting

**GPU run crashes/stops:**
```bash
python run_S_truthfulqa.py --resume
```

**Label yield too low (< 30 per class):**
- Check n=2 sensitivity results (likely higher recall)
- Review unclassified items manually for false excludes
- Consider extending to n=400 items (pre-reg allows)

**Analyzer errors on Windows (cp1252 encoding):**
Already patched — all unicode replaced with ASCII equivalents.

---

## Next experiments (after tonight's results)

1. **Per-token K trajectory** — extend CoherenceSteerer to save K per token
2. **Cross-architecture replication** — Llama-3.1-8B, OLMo (requires SAE infrastructure)
3. **Temperature sweep** — S at T ∈ {0.0, 0.3, 0.7, 1.0}
4. **Logit-entropy baseline comparison** — standard hallucination detector head-to-head
5. **Full 4-axis streaming classifier** — K + C + C_delta + S joint trajectory

---

*Authored: darkflobi, April 5, 2026 ~10am ET. Private until first disclosure.*
