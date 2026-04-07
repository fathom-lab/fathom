"""
HALLUCINATION CORRELATOR V3 — Matched-Pair Depth Comparison
=============================================================

The correct experiment design: for each TruthfulQA question, score BOTH
[question + correct_answer] and [question + incorrect_answer] as depth
scorer inputs. No generation. No overlap heuristics. Just two depth scores
per question — a matched pair, identical question prefix, differing only
in the answer text.

Hypothesis: correct answers produce lower depth_score (deeper, more
distributed processing) than incorrect answers for the same question.
If the model processes true information through different computational
pathways than false information, this paired design will find it.

This is the same matched-pair methodology that produced p=0.000051 on
surface vs insight prompts. Applied to truth vs falsehood.
"""

import gc
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "hallucination_v3_checkpoint.json"


@dataclass
class PairResult:
    """One scored question with correct and incorrect answer."""
    question: str
    correct_answer: str
    incorrect_answer: str
    depth_correct: float
    depth_incorrect: float
    entropy_correct: float
    entropy_incorrect: float
    wdepth_correct: float
    wdepth_incorrect: float
    delta_depth: float       # correct - incorrect (negative = correct is deeper)
    delta_entropy: float     # correct - incorrect
    correct_is_deeper: bool  # depth_correct < depth_incorrect

    def to_dict(self):
        return asdict(self)


@dataclass
class V3Result:
    """Full result of the v3 matched-pair hallucination study."""
    n_pairs: int
    # Depth
    mean_delta_depth: float
    std_delta_depth: float
    pct_correct_deeper: float       # fraction where correct < incorrect
    t_stat_depth: float
    p_value_depth: float
    # Entropy
    mean_delta_entropy: float
    t_stat_entropy: float
    p_value_entropy: float
    # Weighted depth
    mean_delta_wdepth: float
    t_stat_wdepth: float
    p_value_wdepth: float
    # Group means
    mean_depth_correct: float
    mean_depth_incorrect: float
    mean_entropy_correct: float
    mean_entropy_incorrect: float
    # Discovery
    is_significant: bool
    discovery: Optional[str]
    # Raw data
    pairs: List[dict]
    total_time_s: float

    def to_dict(self):
        d = asdict(self)
        return d


def _paired_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Paired t-test. Returns (t_stat, p_value_two_tailed)."""
    if len(x) < 3:
        return 0.0, 1.0
    if HAS_SCIPY:
        t, p = scipy_stats.ttest_rel(x, y)
        return float(t), float(p)
    # Manual fallback
    diffs = x - y
    n = len(diffs)
    mean_d = diffs.mean()
    std_d = diffs.std(ddof=1)
    if std_d < 1e-12:
        return 0.0, 1.0
    t = mean_d / (std_d / math.sqrt(n))
    # Approximate two-tailed p
    p = 2.0 * math.exp(-0.717 * abs(t) - 0.416 * t * t)
    p = max(0.0, min(1.0, p))
    return float(t), float(p)


class HallucinationCorrelatorV3:
    """Matched-pair hallucination depth comparison.

    For each TruthfulQA question, scores [question + correct_answer] and
    [question + incorrect_answer] with DepthScorer. Same question prefix,
    different answer suffix. If the model processes true and false completions
    through different computational pathways, the depth signatures will differ
    in a consistent direction across pairs.

    This mirrors the matched-pair design that found p=0.000051 for surface
    vs insight prompts — the strongest result in Fathom.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self._checkpoint_path = checkpoint_path or CHECKPOINT_PATH

    def _load_checkpoint(self) -> Tuple[List[dict], int]:
        if self._checkpoint_path.exists():
            try:
                with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pairs = data.get("scored_pairs", [])
                resume_idx = data.get("next_index", 0)
                print(f"[HalluV3] Resuming from checkpoint: "
                      f"{len(pairs)} pairs done, next_index={resume_idx}")
                return pairs, resume_idx
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[HalluV3] Corrupt checkpoint, starting fresh: {e}")
        return [], 0

    def _save_checkpoint(self, scored_pairs: List[dict], next_index: int):
        data = {
            "scored_pairs": scored_pairs,
            "next_index": next_index,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self._checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def run_mc_study(
        self,
        scorer,
        n: int = 100,
        checkpoint_every: int = 10,
        seed: int = 42,
    ) -> V3Result:
        """Run the matched-pair hallucination study.

        For each TruthfulQA question:
        1. Pick best_answer (correct) and one random incorrect_answer
        2. Score [question + correct_answer] with DepthScorer
        3. Score [question + incorrect_answer] with DepthScorer
        4. Record the delta (correct - incorrect)

        Hypothesis: mean delta < 0 (correct answers are processed deeper).

        Args:
            scorer: Initialized DepthScorer.
            n: Number of questions to process.
            checkpoint_every: Save checkpoint every N pairs.
            seed: Random seed for incorrect answer selection.

        Returns:
            V3Result with paired statistics.
        """
        import torch
        from fathom_hallucination import HallucinationCorrelator

        rng = random.Random(seed)
        t_start = time.time()

        # Load samples
        loader = HallucinationCorrelator()
        samples = loader.load_truthfulqa(n=n)
        print(f"[HalluV3] Loaded {len(samples)} TruthfulQA samples")

        # Filter to samples with both correct and incorrect answers
        valid_samples = [
            s for s in samples
            if s.get("best_answer", "").strip()
            and s.get("incorrect_answers")
            and len(s["incorrect_answers"]) > 0
        ]
        print(f"[HalluV3] {len(valid_samples)} samples have both correct + incorrect answers")

        # Load checkpoint
        scored_pairs, resume_idx = self._load_checkpoint()

        print(f"[HalluV3] Scoring pairs (resuming from {resume_idx})...")

        for i in range(resume_idx, len(valid_samples)):
            sample = valid_samples[i]
            question = sample["question"].strip()
            correct = sample["best_answer"].strip()
            incorrects = [a.strip() for a in sample["incorrect_answers"] if a.strip()]

            if not incorrects:
                continue

            incorrect = rng.choice(incorrects)

            # Truncate answers to keep prompt length comparable
            correct_trunc = correct[:80]
            incorrect_trunc = incorrect[:80]

            prompt_correct = f"{question} {correct_trunc}"
            prompt_incorrect = f"{question} {incorrect_trunc}"

            try:
                # Score correct
                r_correct = scorer.score(prompt_correct)
                torch.cuda.empty_cache()
                gc.collect()

                # Score incorrect
                r_incorrect = scorer.score(prompt_incorrect)
                torch.cuda.empty_cache()
                gc.collect()

                # Skip NaN results
                if math.isnan(r_correct.depth_score) or math.isnan(r_incorrect.depth_score):
                    print(f"  Pair {i}: NaN depth, skipping")
                    continue

                delta_d = r_correct.depth_score - r_incorrect.depth_score
                e_correct = r_correct.layer_entropy or 0.0
                e_incorrect = r_incorrect.layer_entropy or 0.0
                wd_correct = r_correct.weighted_depth_score or r_correct.depth_score
                wd_incorrect = r_incorrect.weighted_depth_score or r_incorrect.depth_score

                pair = PairResult(
                    question=question,
                    correct_answer=correct_trunc,
                    incorrect_answer=incorrect_trunc,
                    depth_correct=r_correct.depth_score,
                    depth_incorrect=r_incorrect.depth_score,
                    entropy_correct=e_correct,
                    entropy_incorrect=e_incorrect,
                    wdepth_correct=wd_correct,
                    wdepth_incorrect=wd_incorrect,
                    delta_depth=delta_d,
                    delta_entropy=e_correct - e_incorrect,
                    correct_is_deeper=r_correct.depth_score < r_incorrect.depth_score,
                ).to_dict()

                scored_pairs.append(pair)

                direction = "DEEPER" if delta_d < 0 else "shallower"
                print(f"  [{i+1}/{len(valid_samples)}] "
                      f"d_correct={r_correct.depth_score:.3f} "
                      f"d_incorrect={r_incorrect.depth_score:.3f} "
                      f"delta={delta_d:+.3f} correct={direction}")

            except Exception as e:
                print(f"  Pair {i} failed: {e}")
                torch.cuda.empty_cache()
                gc.collect()

            if (i + 1) % checkpoint_every == 0:
                self._save_checkpoint(scored_pairs, i + 1)
                print(f"  [CHECKPOINT] Saved at pair {i+1} ({len(scored_pairs)} scored)")

        # Final checkpoint
        self._save_checkpoint(scored_pairs, len(valid_samples))

        total_time = time.time() - t_start
        print(f"[HalluV3] Complete: {len(scored_pairs)} pairs in {total_time:.1f}s")

        return self.compute_statistics(scored_pairs, total_time)

    def compute_statistics(self, scored_pairs: List[dict], total_time: float = 0.0) -> V3Result:
        """Compute paired statistics from scored pairs."""
        if len(scored_pairs) < 5:
            raise ValueError(f"Need >= 5 pairs, got {len(scored_pairs)}")

        d_correct = np.array([p["depth_correct"] for p in scored_pairs])
        d_incorrect = np.array([p["depth_incorrect"] for p in scored_pairs])
        e_correct = np.array([p["entropy_correct"] for p in scored_pairs])
        e_incorrect = np.array([p["entropy_incorrect"] for p in scored_pairs])
        wd_correct = np.array([p["wdepth_correct"] for p in scored_pairs])
        wd_incorrect = np.array([p["wdepth_incorrect"] for p in scored_pairs])

        deltas_d = d_correct - d_incorrect
        deltas_e = e_correct - e_incorrect
        deltas_wd = wd_correct - wd_incorrect

        # Paired t-tests
        t_d, p_d = _paired_ttest(d_correct, d_incorrect)
        t_e, p_e = _paired_ttest(e_correct, e_incorrect)
        t_wd, p_wd = _paired_ttest(wd_correct, wd_incorrect)

        pct_deeper = float((deltas_d < 0).sum()) / len(deltas_d)

        # Discovery: mean_delta < 0 AND p < 0.05
        is_sig = deltas_d.mean() < 0 and p_d < 0.05
        discovery = None
        if is_sig:
            discovery = (
                f"Correct answers produce deeper computational processing than "
                f"incorrect answers for the same question. Paired t-test on "
                f"{len(scored_pairs)} TruthfulQA pairs: mean delta = "
                f"{deltas_d.mean():.4f}, t = {t_d:.3f}, p = {p_d:.6f}. "
                f"{pct_deeper:.0%} of pairs show deeper processing for the "
                f"correct answer. The model's circuit-level processing "
                f"distinguishes true from false completions."
            )

        return V3Result(
            n_pairs=len(scored_pairs),
            mean_delta_depth=float(deltas_d.mean()),
            std_delta_depth=float(deltas_d.std()),
            pct_correct_deeper=pct_deeper,
            t_stat_depth=t_d,
            p_value_depth=p_d,
            mean_delta_entropy=float(deltas_e.mean()),
            t_stat_entropy=t_e,
            p_value_entropy=p_e,
            mean_delta_wdepth=float(deltas_wd.mean()),
            t_stat_wdepth=t_wd,
            p_value_wdepth=p_wd,
            mean_depth_correct=float(d_correct.mean()),
            mean_depth_incorrect=float(d_incorrect.mean()),
            mean_entropy_correct=float(e_correct.mean()),
            mean_entropy_incorrect=float(e_incorrect.mean()),
            is_significant=is_sig,
            discovery=discovery,
            pairs=scored_pairs,
            total_time_s=total_time,
        )

    def write_report(self, result: V3Result, output_dir: Optional[Path] = None):
        """Write experiment report + discovery file if significant."""
        out = output_dir or PROJECT_ROOT
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        body = f"""# Hallucination Depth Study V3 — Matched-Pair Design

**Date:** {timestamp}
**Runner:** fathom_hallucination_v3.py
**Duration:** {result.total_time_s:.1f}s

## Design

For each TruthfulQA question, score two prompts:
- **Correct:** [question + best_answer]
- **Incorrect:** [question + random_incorrect_answer]

Same question prefix. Different answer suffix. Matched pair.
No generation. No overlap heuristics. Direct depth comparison.

**Hypothesis:** correct answers produce lower depth_score (deeper processing).
This is the same matched-pair methodology that found p=0.000051 for surface vs insight.

## Results

| Metric | Value |
|--------|-------|
| Pairs scored | {result.n_pairs} |
| Mean depth (correct) | {result.mean_depth_correct:.4f} |
| Mean depth (incorrect) | {result.mean_depth_incorrect:.4f} |
| **Mean delta (correct - incorrect)** | **{result.mean_delta_depth:+.4f}** |
| Std delta | {result.std_delta_depth:.4f} |
| **% correct is deeper** | **{result.pct_correct_deeper:.1%}** |

## Statistical Tests (paired t-test)

| Metric | t-statistic | p-value | Significant (p < 0.05) |
|--------|-------------|---------|------------------------|
| depth_score | {result.t_stat_depth:.3f} | {result.p_value_depth:.6f} | {'**YES**' if result.mean_delta_depth < 0 and result.p_value_depth < 0.05 else 'NO'} |
| weighted_depth | {result.t_stat_wdepth:.3f} | {result.p_value_wdepth:.6f} | {'**YES**' if result.mean_delta_wdepth < 0 and result.p_value_wdepth < 0.05 else 'NO'} |
| layer_entropy | {result.t_stat_entropy:.3f} | {result.p_value_entropy:.6f} | {'**YES**' if result.p_value_entropy < 0.05 else 'NO'} |

## Group Means

| Metric | Correct | Incorrect | Delta |
|--------|---------|-----------|-------|
| depth_score | {result.mean_depth_correct:.4f} | {result.mean_depth_incorrect:.4f} | {result.mean_delta_depth:+.4f} |
| layer_entropy | {result.mean_entropy_correct:.4f} | {result.mean_entropy_incorrect:.4f} | {result.mean_delta_entropy:+.4f} |

## Interpretation

{'**SIGNIFICANT.** ' + result.discovery if result.is_significant else f'Not significant at p < 0.05. Mean delta = {result.mean_delta_depth:+.4f}, p = {result.p_value_depth:.4f}. The hallucination hypothesis is not supported at this sample size ({result.n_pairs} pairs). The model does not show a consistent depth difference between processing true and false answer completions.'}

## Why V3 Differs From V1/V2

- **V1** scored pre-supplied prefixes but used unpaired correlation (r=0.074, p=0.65). Wrong test.
- **V2** generated model answers then scored — but generation quality was poor and correctness detection unreliable.
- **V3** uses the matched-pair design that works: same question, different answer, paired t-test. This is the design that found p=0.000051 on surface vs insight.
"""

        report_path = out / "EXPERIMENT_HALLUCINATION_V3.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"[HalluV3] Report: {report_path}")

        if result.is_significant and result.discovery:
            disc_path = out / "DISCOVERY_HALLUCINATION_V3.md"
            with open(disc_path, "w", encoding="utf-8") as f:
                f.write(f"# DISCOVERY: Circuit-Level Hallucination Signal (V3)\n\n")
                f.write(f"**Date:** {timestamp}\n\n")
                f.write(f"## Finding\n\n{result.discovery}\n\n")
                f.write(f"## Evidence\n\n")
                f.write(f"- n = {result.n_pairs} matched pairs (same question, correct vs incorrect answer)\n")
                f.write(f"- Mean delta = {result.mean_delta_depth:+.4f} (negative = correct is deeper)\n")
                f.write(f"- t = {result.t_stat_depth:.3f}, p = {result.p_value_depth:.6f} (paired t-test)\n")
                f.write(f"- {result.pct_correct_deeper:.0%} of pairs show deeper processing for correct answer\n\n")
                f.write(f"## Implication\n\n")
                f.write(f"The model engages different computational depth when processing true vs false\n")
                f.write(f"information as a completion to the same question. This depth signature is\n")
                f.write(f"measurable before output and could serve as a real-time hallucination signal.\n\n")
                f.write(f"## Methodology\n\n")
                f.write(f"Matched-pair design from TruthfulQA. For each question: score\n")
                f.write(f"[question + correct_answer] and [question + incorrect_answer] with DepthScorer.\n")
                f.write(f"Paired t-test on depth deltas. Same methodology that produced p=0.000051\n")
                f.write(f"for surface vs insight distinction.\n")
            print(f"[HalluV3] *** DISCOVERY: {disc_path} ***")

        return report_path
