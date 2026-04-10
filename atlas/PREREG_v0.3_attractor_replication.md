# OSF-style Pre-Registration — Atlas v0.3 Attractor Replication

**Project:** Fathom Cognitive Atlas
**Lead:** Alexander Rodabaugh (Fathom Intelligence)
**Lock date:** 2026-04-10
**Commit hash before v0.3 data capture:** (this document is committed
to git BEFORE any v0.3 capture is run; the git commit timestamp
serves as the pre-registration seal)

---

## 1. Purpose

The v0.2.1 RLHF Attractor observation (Gemma + Llama + Qwen, n=3
families) produced a directional result that cleared bootstrap
prompt-resampling robustness on aggregate but failed to reach
conventional statistical significance under a permutation test.
The failure is structural: at n=3 families, the permutation space
has only $2^3 = 8$ configurations, so the minimum achievable
p-value is 1/8 = 0.125. A sample size of $n \geq 4$ is required
for any conventional significance threshold to be reachable.

This pre-registration locks the hypotheses, analysis plan, and
decision rule for the v0.3 atlas expansion BEFORE the new capture
data exists. The goal is to convert the v0.2.1 observation into a
statistical finding (or cleanly reject it) by adding 2-3 new
model families under a prespecified test.

## 2. Hypotheses

**Primary hypothesis (H1).** When the atlas is expanded to $n \geq 5$
independent (base, instruct) model family pairs, the leave-one-out
attractor direction prediction test (as implemented in
`atlas/analysis/attractor_bootstrap.py`) produces:

- **(a) Mean LOO cosine $\geq 0.4$** across all $n$ families on
  the shape-normalized entropy atlas, early window (tokens 0-6).
- **(b) Permutation-test one-sided $p < 0.05$** under family-level
  base/instruct label swaps with $\geq 2000$ permutations.
- **(c) Bootstrap 95% CI on the mean LOO cosine excludes zero**
  (lower bound $> 0.0$).

H1 is supported if all three conditions hold. H1 is rejected if any
of (a), (b), or (c) fail.

**Secondary hypothesis (H2).** The three existing families (Gemma,
Llama, Qwen) preserve their v0.2.1 directional pattern under
expansion: Gemma and Llama LOO cosines remain bootstrap-robust
(individual 95% CI excludes zero on entropy early-window); Qwen
remains the marginal outlier (individual CI marginally robust
but with lowest absolute magnitude).

H2 is a consistency check. It does not determine whether H1 is
supported; it tests whether the new data changes our interpretation
of the original three families.

**Tertiary hypothesis (H3).** The top-k entropy estimator validation
result from the bulletproof audit (mean shape-normalized r = 0.902
across 3 families) holds on the new families added in v0.3. Any
new family with estimator shape r < 0.70 is flagged as unreliable
for inclusion in the cross-vendor analysis.

H3 is a quality gate. It is evaluated per-family on new captures
before those captures enter the LOO analysis.

## 3. Dataset

**Probe set.** `atlas/probes/probe_set_v0.1.json`, SHA hash
`34a7254920bce654`. No modifications. This is the identical probe
set used for v0.1, v0.2, and v0.2.1 captures. Any analysis that
deviates from this probe set is NOT covered by this
pre-registration.

**New families to capture (target $n = 5$-$6$ total).** In order of
priority:

1. **Phi-3.5-mini (3.8B)** — `microsoft/Phi-3.5-mini-instruct`
   for instruct; `microsoft/phi-2` as closest base proxy if
   Phi-3.5 base not directly available. Different training data
   regime, different architecture family.
2. **SmolLM2-1.7B** — `HuggingFaceTB/SmolLM2-1.7B` (base) +
   `HuggingFaceTB/SmolLM2-1.7B-Instruct` (instruct). Smallest
   models in the atlas; tests whether attractor scales down.
3. **OLMo-2-1B** — `allenai/OLMo-2-0425-1B` (base) +
   `allenai/OLMo-2-0425-1B-Instruct` (instruct). Fully open
   training recipe; tests whether attractor holds on a model
   family where we actually know the training data.

Target minimum: 2 additional families (n=5 total). Stretch goal:
3 additional families (n=6 total).

**Existing families (locked).** Gemma-2-2B, Llama-3.2-3B,
Qwen2.5-3B captures from v0.2 / v0.2.1 are unchanged and enter the
expanded analysis as-is. They are not re-captured.

## 4. Capture protocol

Identical to v0.2:
- `max_tokens = 30`
- `temperature = 0` (greedy decoding)
- `dtype = bfloat16`
- Capture pipeline: `atlas/atlas_capture.py`
- Measurements: D (full residual access), entropy, logprob,
  top2_margin
- Device: single RTX 4070 Laptop, CUDA, no quantization except
  where model does not fit in 8 GB VRAM (in which case 4-bit
  quant is applied and documented per-capture)

Exclusions:
- Any family whose full-precision load exceeds the 8 GB VRAM
  budget AND does not function correctly under 4-bit quantization
  is excluded and documented as "out of scope for v0.3."
- Any family whose capture fails on more than 10% of prompts
  (error field in more than 9 of 90 records) is excluded.

## 5. Analysis plan

All analyses are pre-specified and will be run on the final
committed v0.3 captures without modification. Scripts already
exist in the repository:

- `atlas/analysis/atlas_analyze.py` — baseline per-category atlas
- `atlas/analysis/atlas_attractor.py` — D attractor + LOO
- `atlas/analysis/atlas_entropy.py` — entropy attractor + LOO
- `atlas/analysis/attractor_bootstrap.py` — D bootstrap + permutation
- `atlas/analysis/attractor_bootstrap_entropy.py` — entropy bootstrap + permutation
- `atlas/analysis/validate_entropy_estimator.py` — per-family estimator quality

**Primary test (for H1):** Entropy attractor LOO cosine on the early
window (tokens 0-6), shape-normalized signatures, bootstrap 2000x,
permutation 2000x, across all $n$ families included in v0.3.

**Decision rule:**

- If mean LOO cos $\geq 0.4$ AND $p_{\mathrm{perm}} < 0.05$ AND
  bootstrap mean CI excludes zero → H1 SUPPORTED. Proceed to
  Zenodo v14 publication with the result as the headline claim.

- If mean LOO cos $< 0.4$ OR $p_{\mathrm{perm}} \geq 0.05$ OR
  CI includes zero → H1 NOT SUPPORTED at v0.3. Report honestly
  as a preliminary directional observation that did not replicate
  under expansion. Hold the attractor claim. Publish v14 with the
  atlas infrastructure and the observation note, not the attractor
  claim.

- If mean LOO cos is in the $[0.3, 0.4)$ band → borderline.
  Report exact values, do not round, do not call the result
  supported. Flag as requiring v0.4 expansion.

**Consistency test (for H2):** Compute per-family bootstrap CIs on
the LOO cosine for Gemma, Llama, Qwen at the new $n$. Report whether
their P(>0) values and CI widths match the v0.2.1 expected
behaviors. If Gemma OR Llama drops below P(>0) = 0.90 under the
expansion, flag as a concern and re-examine before publication.

**Quality gate (for H3):** Run `validate_entropy_estimator.py` on
each new family BEFORE it enters the LOO analysis. Families with
shape r < 0.70 are excluded from the LOO test and reported as
"captured but estimator-unreliable."

## 6. What counts as a null result

A null result for H1 is not a publication failure. Either outcome
is publishable with appropriate framing:

- **H1 supported:** "The RLHF Attractor direction replicates at
  $n \geq 5$ families with permutation $p < 0.05$ and bootstrap
  95% CI excluding zero. The attractor is a reproducible feature
  of cross-architecture RLHF."
- **H1 not supported:** "The v0.2.1 preliminary directional
  observation did not replicate at $n \geq 5$. We report the
  expanded atlas without the attractor claim and document the
  observation for historical record."

Reporting a null result honestly is the point of pre-registration.

## 7. Deviations allowed

The following ARE allowed without invalidating the pre-registration:
- Swapping model choices if a planned model cannot be loaded
  (e.g., phi-2 substituted for Phi-3.5-mini-base). Exact model
  used will be documented per capture.
- Running OpenAI / Gemini closed-model captures alongside the
  open-model v0.3 captures. The closed-model data is an
  EXPLORATORY extension and does NOT enter the H1 primary test.
- Additional robustness checks beyond the pre-registered tests
  (e.g., leave-one-prompt-out, subset analyses). These are
  exploratory.

The following are NOT allowed:
- Changing the primary test specification after any v0.3 data
  has been inspected.
- Swapping from entropy early-window to D or any other measurement
  as the primary test.
- Excluding families based on their individual LOO values
  (which would be post-hoc cherry-picking).
- Moving the decision-rule thresholds after the fact.

## 8. Code commitment

At the time of pre-registration seal, the following code files
are committed to git and will NOT be modified before v0.3 data
is captured and analyzed:

- `atlas/atlas_capture.py` (capture runner)
- `atlas/analysis/atlas_attractor.py` (D attractor analysis)
- `atlas/analysis/atlas_entropy.py` (entropy attractor analysis)
- `atlas/analysis/attractor_bootstrap.py` (D rigor)
- `atlas/analysis/attractor_bootstrap_entropy.py` (entropy rigor)
- `atlas/analysis/validate_entropy_estimator.py` (estimator validation)

Any bug fix to these files before v0.3 analysis must be logged
with the bug description, the fix, and a re-run on the v0.2.1 data
to confirm that the bug fix does not change the v0.2.1 results.

## 9. Reporting

Upon completion of v0.3 captures and analyses, a
`FINDINGS_v0.3.md` document will be written that explicitly:

1. States the H1, H2, H3 outcomes.
2. Reports the exact mean LOO cosine, permutation p, and bootstrap
   CI for each test specified in §5.
3. Compares the v0.2.1 point estimates side-by-side with the
   v0.3 point estimates for each of the 3 original families.
4. Documents any deviations from this pre-registration with
   justification.
5. States the decision rule outcome and the publication action.

---

*Pre-registration sealed 2026-04-10 by Alexander Rodabaugh, via
git commit before any v0.3 capture has been run. Replaces any
informal hypothesis statement in the v0.2.1 findings memo for the
purpose of evaluating the v0.3 expansion.*
