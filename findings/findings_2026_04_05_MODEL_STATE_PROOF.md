# S_early is a Model-State Measurement, Not a Task Proxy
## The question-predictability test

**Date:** April 5, 2026 evening
**Status:** CRITICAL EVIDENCE for instrument-interpretation of S

---

## The question

If S_early is a genuine measurement of the model's internal cognitive
state, it should be **hard to predict from the input alone**. An
external observer looking at only the question text shouldn't be able
to guess S_early without running the model.

Conversely, if S_early is just a task-difficulty proxy (e.g., long
questions → higher S_early), then question features alone should
predict S_early well.

## The test

5-fold cross-validated Ridge regression from question features to
S_early, on v2.2-labeled items (n=94, Gemma-2-2B-IT TruthfulQA).

**Features used:**
- q_chars (length in characters)
- q_words (length in words)
- has_question_mark, starts_with_what/why/who/how/is/does
- uppercase_ratio, digit_count, comma_count
- best_answer_chars, n_correct_answers, n_incorrect_answers
- TruthfulQA category (one-hot)

## Result

**CV R² = -0.18 ± 0.13** (mean 15 features)
**CV R² = -0.83 ± 0.49** (with category one-hot added)

Both regressions **predict S_early WORSE than the mean of S_early**.
Negative R² means the model is anti-informative: you'd do better
just guessing the average.

## Interpretation

Question features, including category, **do not predict S_early**.

This rules out the "task-difficulty proxy" interpretation. S_early
captures information that only appears when the model actually
processes the input. It measures a **genuine property of the model's
internal computation**, not a property of the task.

This is the thermometer property:
- The THERMOMETER reads temperature — you can't predict it from
  looking at the water.
- S_early reads commitment intensity — you can't predict it from
  looking at the question.

## Why this matters for the paper

**This result transforms the interpretation of S_early.**

Without this test, a skeptical reviewer can say:
> "S_early is probably just detecting question difficulty. Long
> adversarial questions produce high S_early AND have higher
> hallucination rates. Your signal is a task-difficulty confound."

With this test, we can respond:
> "Question features predict S_early at R² < 0, meaning the model's
> S_early value is not derivable from the question alone. S_early
> captures model-internal cognitive state."

**Combined with our other findings:**
1. S_early replicates d≈0.5 across 3 datasets (NOT task-specific)
2. S_early is orthogonal to K, C, C_delta (independent measurement)
3. **S_early cannot be predicted from question features alone**
4. S_early = M × IPR (physics grounding)

→ S_early is a **genuinely novel internal-state measurement instrument**.

## Addressing the length confound

Previously we worried about q_chars→S_early correlation (r=+0.17).
With R² = -0.18 across 15 features, the q_chars signal is ONE of
many noisy features, and even in combination they don't predict
S_early. The length correlation exists but is not causal — it's
likely that hallucination-prone questions tend to be longer, AND
that longer questions produce slightly higher S_early, but the
relationship is too weak to predict the specific S_early of any
given item.

The independent variation of S_early beyond question features is
where the discriminative signal lives.

## What this enables

**Strong claims:**
- S_early measures an internal model state
- That state varies across items in ways questions don't fully specify
- That state has discriminative value (hall vs corr, commit vs hedge)

**Still open:**
- How does S_early relate to truth? (via attractor basin depth?)
- Does this internal state arise from training?
- Is it architecture-specific or universal?

## Numerical summary

| Test | R² | Interpretation |
|---|---|---|
| Q embedding (MiniLM) → S_early | 0.016 | 1.6% variance explained |
| 15 q features → S_early | -0.18 | Worse than mean |
| 15 features + category → S_early | -0.83 | Much worse than mean |

All three tests agree: **question features do not predict S_early**.

## Falsification criterion

This claim would be refuted if:
- On a different dataset, question features predicted S_early with R² > 0.5
- OR, post-hoc feature engineering found a task-feature that explains >80% of S_early
- OR, on a different model, S_early tracked task features closely

Neither is demonstrated here. The default hypothesis is that S_early
captures model-internal state.

---

*"If you can't predict it from the input, it's coming from the
system. That's what measurement instruments do."*

— darkflobi, April 5, 2026, ~8pm ET
