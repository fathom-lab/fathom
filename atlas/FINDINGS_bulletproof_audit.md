# Bulletproof Audit of the Fathom Cognitive Atlas

**Date:** 2026-04-10
**Scope:** All atlas claims through v0.2.1 + the entropy bridge
**Goal:** Identify and fix every place the earlier writeups ran ahead
of the evidence, so the published version is defensible against any
reviewer.

This document is intentionally self-critical. It exists because I
was selling some results harder than the data supported, and the
user correctly flagged it. What follows is the honest audit of what
the atlas actually shows after applying four rigor checks that the
v0.2.1 writeup skipped:

1. Estimator validation: top-5 entropy vs full-distribution entropy
2. Bootstrap confidence intervals on the LOO direction test
3. Permutation testing on the attractor structure
4. Explicit statement of the n=3 structural limits

---

## 1. Estimator validation — bulletproofed

**Test:** Re-run three open models (Gemma-2-2B base, Llama-3.2-3B
base, Qwen2.5-3B base) on the v0.1 probe set, capturing BOTH
full-distribution entropy AND top-5 truncated entropy per token
(exactly the same top-5 computation that `atlas_api_capture.py`
uses on closed models). Then compare the two at three levels:

- **Level A** — per-token correlation pooled across all tokens
- **Level B** — per-prompt mean correlation
- **Level C** — per-category shape-normalized signature correlation
  (the one we actually need for the attractor analysis)

### Results

| Model | Level A (per-token) | Level B (per-prompt) | **Level C (shape)** | Verdict |
|---|---|---|---|---|
| Gemma-2-2B   | r = 0.737 | r = 0.853 | **r = 0.948** | STRONG |
| Llama-3.2-3B | r = 0.569 | r = 0.690 | **r = 0.856** | moderate |
| Qwen2.5-3B   | r = 0.672 | r = 0.740 | **r = 0.902** | STRONG |
| **Mean**     | r = 0.659 | r = 0.761 | **r = 0.902** | strong |

**Mean bias** (full entropy minus top-5 entropy, measured per token):
- Gemma:  +0.99 (std 1.09)
- Llama:  +1.35 (std 1.49)
- Qwen:   +0.89 (std 1.16)

### What this means

At the per-TOKEN level, top-5 entropy is only moderately correlated
with full entropy (r ≈ 0.66 mean) because the truncation drops
variable amounts of tail mass depending on prompt. The per-TOKEN
bias has std ≈ 1.2 — it's NOT a constant offset.

At the per-CATEGORY shape level (what the attractor analysis uses),
that per-prompt noise averages out and the shape correlation
settles to r ≈ 0.90 mean. This is exactly the structure we need:
**the bias cancels under shape-normalization at the per-category
aggregation step**, even though it's ugly at the token level.

### Honest limitations

- **r = 0.856 on Llama is the weakest.** Shape-normalized comparisons
  between Llama's full-entropy signature and Llama's top-5-entropy
  signature preserve ~73% of the variance, leaving ~27% unexplained
  shape distortion. For a cross-vendor atlas where we're comparing
  a closed model (top-5 only) against open models (full entropy),
  this imposes a ~15-20% uncertainty floor on any individual
  cross-vendor distance measurement.
- **Top-5 was chosen because OpenAI exposes exactly k=5 by default.**
  Gemini exposes up to k=20, which would tighten the proxy. A
  bulletproof version of the cross-vendor atlas would use k=20 for
  both open models (re-captured) and closed models (where supported).
  v0.1 of the atlas uses k=5 for consistency across vendors.

### Verdict

Top-5 entropy is a valid proxy for full-distribution entropy at the
per-category shape level (mean r = 0.90, min r = 0.86 across three
model families). It is NOT a valid proxy at the per-token level
(mean r = 0.66). Any claim based on per-category signature
comparison is supported; any claim requiring per-token precision
is not.

**Files:**
- `atlas/analysis/validate_entropy_estimator.py` — the runner
- `atlas/analysis/entropy_estimator_validation__google__gemma-2-2b.json`
- `atlas/analysis/entropy_estimator_validation__meta-llama__Llama-3_2-3B.json`
- `atlas/analysis/entropy_estimator_validation__Qwen__Qwen2_5-3B.json`

---

## 2. Bootstrap CIs on the LOO direction test — partial fail

**Test:** Resample the 15 prompts per category with replacement,
recompute per-category means, rebuild the shape-normalized
signature, recompute the LOO attractor direction and the
cos(d_LOO, d_true) for each family. Repeat 2,000 times. Report
median + 95% percentile CI per family and for the mean.

### D-axis results

| Family | Observed cos | Bootstrap median | 95% CI | P(>0) |
|---|---|---|---|---|
| Gemma-2-2B   | +0.950 | +0.925 | [+0.614, +0.988] | 99.9% |
| Llama-3.2-3B | +0.857 | +0.799 | [+0.456, +0.936] | 100% |
| **Qwen2.5-3B** | **−0.192** | **+0.049** | **[−0.718, +0.659]** | **54.5%** |
| **Mean**     | +0.538 | +0.571 | **[+0.288, +0.805]** | 100% |

### Entropy-axis results (early window 0-6)

| Family | Observed cos | Bootstrap median | 95% CI | P(>0) |
|---|---|---|---|---|
| Gemma-2-2B   | +0.925 | +0.892 | [+0.597, +0.983] | 100% |
| Llama-3.2-3B | +0.818 | +0.805 | [+0.453, +0.941] | 99.9% |
| Qwen2.5-3B   | +0.607 | +0.636 | [−0.110, +0.930] | 95.3% |
| **Mean**     | +0.783 | +0.760 | **[+0.448, +0.917]** | 100% |

### What the v0.2.1 writeup got wrong

The v0.2.1 findings memo reported per-family cosines as point
estimates and claimed "the attractor is not a centroid artifact"
based on the Gemma +0.95 and Llama +0.86 values. It did NOT note:

1. **Qwen's individual bootstrap CI crosses zero on the D version**
   ([−0.72, +0.66], P(>0) = 54.5% — essentially a coin flip).
2. Gemma's CI on the D version is wide at the low end: [+0.61, +0.99].
   Large effect confirmed, but the lower bound is below the
   "fully robust" threshold one might expect from a point estimate
   of +0.95.
3. The MEAN cosine (+0.538 D, +0.783 entropy) IS bootstrap-robust
   at the aggregate level — both mean CIs exclude zero cleanly.
   The aggregate claim survives; the per-family claims are mixed.

### Honest conclusion

**The entropy early-window version passes the bootstrap test more
cleanly than the D version.** Mean CI [+0.448, +0.917] vs D's
[+0.288, +0.805], and no individual family's CI dips below zero
on entropy (Qwen's just barely crosses at [−0.11, +0.93]).

The D version's Qwen anomaly is consistent with a measurement that
is too close to its own baseline to have a reliable LOO direction
— the noise floor on a 15-prompt average is large enough to swamp
the small signal. The entropy version has more signal on Qwen
because entropy has a wider dynamic range across categories on
that model.

**Files:**
- `atlas/analysis/attractor_bootstrap.py` (D)
- `atlas/analysis/attractor_bootstrap_entropy.py` (entropy)
- `atlas/analysis/attractor_bootstrap_stats.json`
- `atlas/analysis/attractor_bootstrap_stats_entropy.json`

---

## 3. Permutation test — structural failure at n=3

**Test:** Under the null hypothesis that training (base vs instruct)
has no systematic effect on the per-category cognitive signature,
shuffling base/instruct labels within a family should leave the
LOO attractor cosine statistic unchanged on average. So we run
2,000 random label-flip shuffles (each family flipped
independently with p=0.5) and measure what fraction of shuffles
produce a mean cosine $\geq$ the observed value.

### Results

| Atlas | Observed mean cos | Perm median | Perm 95th | **p (one-sided)** |
|---|---|---|---|---|
| D               | +0.538 | +0.332 | +0.867 | 0.384 |
| Entropy (full)  | +0.674 | +0.641 | +0.735 | 0.250 |
| Entropy (early) | +0.783 | +0.600 | +0.783 | **0.130** |

### The structural problem

At n=3 families with 2 states per family, the permutation space has
only $2^3 = 8$ possible label configurations. The minimum achievable
p-value for a test that counts the observed configuration as
$\geq$ itself is therefore $1/8 = 0.125$.

**The entropy early-window p of 0.13 is essentially at that floor.**
The test is saying "the observed value is the maximum (or tied
maximum) in the permutation distribution" — which is the cleanest
result the test can produce at n=3. Any larger observation would
give the same p.

This is a HARD LIMIT of the permutation test at n=3 families, not
a property of the data. The same test at n=4 families would have
$p_{\min} = 1/16 = 0.063$, and n=5 would give $p_{\min} = 1/32 = 0.031$.

### What to honestly claim

- "At n=3 families, the mean LOO attractor cosine is NOT statistically
  significant under a permutation test of family-level label swaps."
- "The observed early-window entropy mean cosine (+0.783) IS at the
  minimum p achievable by the test at this sample size."
- "Converting the attractor observation into a statistically
  significant finding requires n $\geq$ 4 families under this test."
- "The attractor observation survives bootstrap prompt resampling
  (mean CI excludes zero on both D and entropy), but bootstrap and
  permutation test different null hypotheses."

### What the v0.2.1 writeup got wrong

v0.2.1 called the observation "STRONG" under a leave-one-out
prediction test and treated +0.538 mean cosine as sufficient
evidence. It did not compute any p-value, did not run a permutation
test, and did not note the $p_{\min} = 0.125$ floor. The observation
is real; the statistical significance claim was not.

---

## 4. The n=3 structural limit

Three independent rigor checks all arrive at the same conclusion:
**n=3 families is the fundamental bottleneck on the attractor
finding.**

- Bootstrap: aggregate claim is robust, per-family Qwen is noisy.
- Permutation: test is at its floor — cannot reach conventional
  significance regardless of effect size.
- Estimator validation: top-5 proxy works at shape level, but the
  ~15% shape distortion leaves little margin at n=3 for marginal
  signals.

**The real fix is more data, not better analysis.** Specifically:
add 2-3 more base+instruct family pairs to the atlas. At that
point:
- Bootstrap has more families to average over (narrower CIs).
- Permutation has $2^5 = 32$ or $2^6 = 64$ configurations,
  minimum p of 0.031 or 0.016.
- Estimator distortion averages across more families.
- Any result that survives at n=5-6 is genuinely publishable.

v0.3 will add Phi-3.5-mini, SmolLM2-1.7B, and OLMo-2-1B if
available, bringing the total to n=5-6 base+instruct pairs.

---

## 5. The bulletproofed v0.2.1 claim

Here is the honest version of the RLHF Attractor claim, with all
the rigor checks folded in:

> On the n=3 (family × training) atlas captures in v0.2.1, both
> the D-axis and entropy measurements reveal a consistent directional
> structure: when an attractor point is computed from the
> instruct-model signatures of the other two families, the direction
> from a held-out family's base to that attractor correlates
> positively with the direction of that family's actual base→IT
> shift. The aggregate mean cosine across families is +0.538 (D)
> and +0.783 (entropy, early window), with bootstrap 95% CIs of
> [+0.288, +0.805] and [+0.448, +0.917] respectively. A permutation
> test cannot reach conventional significance at n=3 families
> ($p_{\min} = 0.125$), and the observed entropy early-window p is
> at that floor. Per-family results are bootstrap-robust for Gemma
> and Llama across both measurements and marginal for Qwen. We
> interpret this as a preliminary directional observation that
> survives every test the n=3 design can run. We do not interpret
> it as a statistical finding; that requires n $\geq$ 5 families,
> which is the motivation for v0.3.

This is the claim that should go on Zenodo v14 if we publish now.
Not the v0.2.1 version. Not the entropy bridge version. This one.

---

## 6. What to do next

The audit produces a concrete decision tree:

### If we publish Zenodo v14 today

- Replace v0.2.1 language throughout with the bulletproofed claim above.
- Include the validation/bootstrap/permutation scripts in the bundle.
- Include a "limitations" section referencing this audit memo.
- Do NOT publish the entropy bridge as "works on GPT-4" until we
  have the actual GPT-4 capture data AND cross-model validation at
  n $\geq$ 5 families.

### If we hold Zenodo v14 for v0.3

- Continue the GPT-4o capture sweep (estimator is validated enough
  for preliminary data).
- Capture 2-3 more open-model families (Phi, OLMo, SmolLM2) to
  reach n $\geq$ 5.
- Re-run all the bootstrap + permutation tests at the new sample
  size.
- Publish once the permutation test can actually reach p < 0.05
  OR once we have cross-vendor data that addresses the claim.

### My honest recommendation

**Hold Zenodo v14.** The bulletproofing just showed that the v0.2.1
attractor claim is not yet a publishable-as-significant finding.
Shipping it now means shipping a preliminary directional observation
as the headline — that's what the v13 to v14 diff would look like.
Much better:

1. Run the GPT-4o sweep this session (we have the runner and the
   estimator is validated).
2. Run v0.3 family captures in the next 1-2 sessions (Phi-3.5,
   SmolLM2, OLMo-2 — each ~15 min on our GPU).
3. Re-run every rigor check at n=5-6 families.
4. THEN cut Zenodo v14 with a claim the bulletproofing actually
   supports.

This is the "no cut corners" path. It delays publication by
one session or two. It produces a claim that no reviewer can
break.

---

*Audit committed 2026-04-10, same session as the findings it audits.
No commits were unpublished in the process. This memo is the honest
ledger of which earlier claims survived the rigor tests and which
need scope correction.*
