"""
HALLUCINATION CORRELATOR V2 — Model-Generated Answer Depth Analysis
====================================================================

V1 scored pre-supplied correct vs incorrect answer prefixes. That tests whether
the model processes known-true text differently from known-false text — but not
whether the model's OWN reasoning differs when it's about to be right vs wrong.

V2 fixes this: feed the question to the model, let it generate its own answer,
score the depth signature of that generation, then compare against ground truth.
If models reason shallowly when hallucinating and deeply when correct, depth
becomes a real-time hallucination detector — not for curated text, but for the
model's own outputs.
"""

import gc
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "hallucination_v2_checkpoint.json"


# ── Result dataclasses ───────────────────────────────────────────────

@dataclass
class ScoredSample:
    """One scored TruthfulQA sample with model-generated answer."""
    question: str
    generated_answer: str
    best_answer: str
    is_correct: bool
    token_overlap: float
    depth_score: float
    weighted_depth_score: float
    layer_entropy: float
    feature_count: int
    early_ratio: float
    mid_ratio: float
    late_ratio: float

    def to_dict(self):
        return asdict(self)


@dataclass
class ROCPoint:
    """Single point on the ROC curve."""
    threshold: float
    tpr: float
    fpr: float


@dataclass
class ThresholdResult:
    """Optimal threshold for separating correct from incorrect."""
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    metric_name: str  # which metric this threshold is for


@dataclass
class CorrelationResultV2:
    """Full result of the v2 hallucination study."""
    n_samples: int
    n_correct: int
    n_incorrect: int
    # depth ~ accuracy
    depth_accuracy_pearson_r: float
    depth_accuracy_p_value: float
    # weighted depth ~ accuracy
    weighted_depth_accuracy_pearson_r: float
    weighted_depth_accuracy_p_value: float
    # entropy ~ accuracy
    entropy_accuracy_pearson_r: float
    entropy_accuracy_p_value: float
    # AUCs
    auc_depth: float
    auc_weighted_depth: float
    auc_entropy: float
    # Group means
    mean_depth_correct: float
    mean_depth_incorrect: float
    mean_weighted_depth_correct: float
    mean_weighted_depth_incorrect: float
    mean_entropy_correct: float
    mean_entropy_incorrect: float
    # Optimal thresholds
    optimal_depth_threshold: Optional[ThresholdResult]
    optimal_entropy_threshold: Optional[ThresholdResult]
    # ROC data (for export/visualization)
    roc_depth: List[Dict]
    roc_entropy: List[Dict]
    # Discovery
    is_significant: bool
    discovery: Optional[str]
    # Per-sample data
    samples: List[Dict]
    # Timing
    total_time_s: float

    def to_dict(self):
        d = asdict(self)
        return d


# ── Utility functions ────────────────────────────────────────────────

def _token_overlap(generated: str, reference: str) -> float:
    """Compute token-level overlap between generated and reference text.

    Simple whitespace tokenization. Returns fraction of reference tokens
    that appear in generated text (order-independent).
    """
    if not generated.strip() or not reference.strip():
        return 0.0
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    return len(gen_tokens & ref_tokens) / len(ref_tokens)


def _check_correctness(generated: str, best_answer: str,
                       correct_answers: List[str],
                       threshold: float = 0.5) -> Tuple[bool, float]:
    """Check if generated answer matches any correct answer.

    Computes token overlap against best_answer and all correct_answers.
    Returns (is_correct, best_overlap_score).
    """
    if not generated.strip():
        return False, 0.0

    overlaps = [_token_overlap(generated, best_answer)]
    for ca in correct_answers:
        if ca.strip():
            overlaps.append(_token_overlap(generated, ca))

    best_overlap = max(overlaps)
    return best_overlap >= threshold, best_overlap


def _pearson_r(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pearson correlation with p-value. Scipy if available, else manual."""
    if len(x) < 3:
        return 0.0, 1.0
    if HAS_SCIPY:
        r, p = scipy_stats.pearsonr(x, y)
        return float(r), float(p)
    # Manual fallback
    n = len(x)
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0, 1.0
    r = ((x - mx) * (y - my)).sum() / ((n - 1) * sx * sy)
    r = max(-1.0, min(1.0, float(r)))
    # Approximate p-value via t-distribution
    if abs(r) >= 1.0:
        return r, 0.0
    t_stat = r * math.sqrt((n - 2) / (1 - r * r))
    # Two-tailed p approximation (conservative)
    p = 2.0 * math.exp(-0.717 * abs(t_stat) - 0.416 * t_stat * t_stat)
    p = max(0.0, min(1.0, p))
    return r, p


def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC via trapezoidal rule on ROC curve."""
    if len(labels) < 2 or labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr

    return float(auc)


def _compute_roc_curve(labels: np.ndarray, scores: np.ndarray,
                       n_thresholds: int = 50) -> List[Dict]:
    """Compute ROC curve data points for export."""
    if len(labels) < 2:
        return []

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    roc_points = []

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return []

    for thresh in thresholds:
        predicted = (scores <= thresh).astype(int)  # lower score = "correct" prediction
        tp = ((predicted == 1) & (labels == 1)).sum()
        fp = ((predicted == 1) & (labels == 0)).sum()
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        roc_points.append({
            "threshold": float(thresh),
            "tpr": float(tpr),
            "fpr": float(fpr),
        })

    return roc_points


def _find_optimal_threshold(labels: np.ndarray, scores: np.ndarray,
                            metric_name: str,
                            n_thresholds: int = 100) -> Optional[ThresholdResult]:
    """Find threshold that maximizes F1 score for separating correct from incorrect."""
    if len(labels) < 4:
        return None

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_f1 = -1.0
    best_result = None

    for thresh in thresholds:
        # Predict: score <= threshold → correct (deeper = correct hypothesis)
        predicted = (scores <= thresh).astype(int)

        tp = ((predicted == 1) & (labels == 1)).sum()
        fp = ((predicted == 1) & (labels == 0)).sum()
        fn = ((predicted == 0) & (labels == 1)).sum()
        tn = ((predicted == 0) & (labels == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(labels)

        if f1 > best_f1:
            best_f1 = f1
            best_result = ThresholdResult(
                threshold=float(thresh),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                accuracy=float(accuracy),
                metric_name=metric_name,
            )

    return best_result


# ── Main class ───────────────────────────────────────────────────────

class HallucinationCorrelatorV2:
    """V2 hallucination correlator: scores the model's own generated answers.

    Instead of scoring pre-supplied correct/incorrect text (v1), this version
    feeds each TruthfulQA question to the model, captures its natural generation,
    scores the depth signature of that generation, and compares against ground
    truth. This tests whether the model's own reasoning depth differs when it's
    about to produce a correct vs incorrect answer.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self._checkpoint_path = checkpoint_path or CHECKPOINT_PATH

    def _load_checkpoint(self) -> Tuple[List[Dict], int]:
        """Load partial results from checkpoint if available."""
        if self._checkpoint_path.exists():
            try:
                with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                scored = data.get("scored_samples", [])
                resume_idx = data.get("next_index", 0)
                print(f"[HallucinationV2] Resuming from checkpoint: "
                      f"{len(scored)} samples done, next_index={resume_idx}")
                return scored, resume_idx
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[HallucinationV2] Corrupt checkpoint, starting fresh: {e}")
        return [], 0

    def _save_checkpoint(self, scored_samples: List[Dict], next_index: int):
        """Save partial results to checkpoint file."""
        data = {
            "scored_samples": scored_samples,
            "next_index": next_index,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self._checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _generate_answer(self, question: str, inner_model, max_new_tokens: int = 20) -> str:
        """Generate a short greedy answer from the model.

        Uses the inner HookedTransformer's generate method with greedy decoding.
        Falls back to manual token-by-token generation if generate() unavailable.
        """
        import torch

        prompt_text = question.strip()
        if not prompt_text.endswith("?"):
            prompt_text += "?"
        prompt_text += " "

        tokens = inner_model.to_tokens(prompt_text)

        # Try generate() first (TransformerLens standard)
        try:
            with torch.no_grad():
                output_tokens = inner_model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    verbose=False,
                )
            # Decode only the new tokens
            new_tokens = output_tokens[0, tokens.shape[1]:]
            generated = inner_model.to_string(new_tokens)
            return generated.strip()
        except Exception:
            pass

        # Manual fallback: greedy token-by-token
        try:
            current_tokens = tokens.clone()
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = inner_model(current_tokens)
                next_token = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                # Stop at EOS
                if hasattr(inner_model, "tokenizer") and inner_model.tokenizer is not None:
                    eos_id = getattr(inner_model.tokenizer, "eos_token_id", None)
                    if eos_id is not None and next_token.item() == eos_id:
                        break

            new_tokens = current_tokens[0, tokens.shape[1]:]
            generated = inner_model.to_string(new_tokens)
            return generated.strip()
        except Exception as e:
            print(f"    Generation fallback failed: {e}")
            return ""

    def run_study(
        self,
        scorer,
        n: int = 200,
        max_new_tokens: int = 20,
        checkpoint_every: int = 10,
    ) -> CorrelationResultV2:
        """Run the full v2 hallucination study.

        For each TruthfulQA question:
        1. Generate a short completion from the model (greedy, max 20 tokens)
        2. Score that completion's depth signature
        3. Check if the answer matches ground truth (token overlap >= 0.5)
        4. Store all metrics

        Checkpoints every 10 samples. Resumes from checkpoint on crash.

        Args:
            scorer: Initialized DepthScorer with model loaded.
            n: Number of TruthfulQA questions to process.
            max_new_tokens: Max tokens to generate per question.
            checkpoint_every: Save checkpoint every N samples.

        Returns:
            CorrelationResultV2 with full analysis.
        """
        import torch
        from fathom_hallucination import HallucinationCorrelator

        t_start = time.time()

        # Load TruthfulQA samples using v1's loader
        loader = HallucinationCorrelator()
        samples = loader.load_truthfulqa(n=n)
        print(f"[HallucinationV2] Loaded {len(samples)} TruthfulQA samples")

        # Get inner model for generation — try multiple attribute paths
        inner_model = (
            getattr(scorer._model, 'model', None)
            or getattr(scorer._model, 'hooked_model', None)
            or getattr(scorer._model, '_model', None)
        )
        use_generation = inner_model is not None and hasattr(inner_model, 'to_tokens')
        if not use_generation:
            print("[HallucinationV2] WARNING: Inner model not accessible for generation. "
                  "Falling back to single-token target scoring.")

        # Load checkpoint
        scored_raw, resume_idx = self._load_checkpoint()

        print(f"[HallucinationV2] Scoring {len(samples)} samples "
              f"(resuming from {resume_idx})...")

        for i in range(resume_idx, len(samples)):
            sample = samples[i]
            question = sample["question"]
            best_answer = sample.get("best_answer", "")
            correct_answers = sample.get("correct_answers", [])

            if not question.strip():
                continue

            try:
                generated = ""

                if use_generation:
                    # Step 1a: Generate model's own answer via inner model
                    try:
                        generated = self._generate_answer(
                            question, inner_model, max_new_tokens=max_new_tokens
                        )
                    except Exception as gen_e:
                        print(f"  Sample {i}: generation failed ({gen_e}), "
                              "falling back to single-token")

                if not generated.strip():
                    # Step 1b: Fallback — use scorer's target detection for a single token
                    try:
                        target = scorer._get_single_token_target(question + " ")
                        generated = target if target else best_answer.split()[0]
                    except Exception:
                        generated = best_answer.split()[0] if best_answer else ""

                torch.cuda.empty_cache()
                gc.collect()

                if not generated.strip():
                    print(f"  Sample {i}: no generation available, skipping")
                    continue

                # Step 2: Score the depth signature of the generated answer
                score_prompt = f"{question} {generated}"
                result = scorer.score(score_prompt)

                torch.cuda.empty_cache()
                gc.collect()

                if math.isnan(result.depth_score):
                    print(f"  Sample {i}: NaN depth_score, skipping")
                    continue

                # Step 3: Check correctness
                is_correct, overlap = _check_correctness(
                    generated, best_answer, correct_answers, threshold=0.5
                )

                # Step 4: Store
                scored_raw.append(ScoredSample(
                    question=question,
                    generated_answer=generated,
                    best_answer=best_answer,
                    is_correct=is_correct,
                    token_overlap=overlap,
                    depth_score=result.depth_score,
                    weighted_depth_score=result.weighted_depth_score or result.depth_score,
                    layer_entropy=result.layer_entropy or 0.0,
                    feature_count=result.feature_count,
                    early_ratio=result.early_ratio,
                    mid_ratio=result.mid_ratio,
                    late_ratio=result.late_ratio,
                ).to_dict())

                status = "CORRECT" if is_correct else "WRONG"
                print(f"  [{i+1}/{len(samples)}] {status} "
                      f"(overlap={overlap:.2f}, depth={result.depth_score:.3f}, "
                      f"entropy={result.layer_entropy or 0:.3f})")

            except Exception as e:
                print(f"  Sample {i} failed: {e}")
                torch.cuda.empty_cache()
                gc.collect()

            # Checkpoint
            if (i + 1) % checkpoint_every == 0:
                self._save_checkpoint(scored_raw, i + 1)
                print(f"  [CHECKPOINT] Saved at sample {i+1} "
                      f"({len(scored_raw)} scored)")

        # Final checkpoint
        self._save_checkpoint(scored_raw, len(samples))

        total_time = time.time() - t_start
        print(f"[HallucinationV2] Scoring complete: {len(scored_raw)} samples "
              f"in {total_time:.1f}s")

        # Compute correlations
        return self.compute_correlations(scored_raw, total_time)

    def compute_correlations(
        self,
        scored_samples: List[Dict],
        total_time: float = 0.0,
    ) -> CorrelationResultV2:
        """Compute all correlations from scored sample data.

        Args:
            scored_samples: List of ScoredSample dicts.
            total_time: Total wall time for the study.

        Returns:
            CorrelationResultV2 with complete statistical analysis.
        """
        if len(scored_samples) < 10:
            raise ValueError(
                f"Need >= 10 scored samples for correlation, got {len(scored_samples)}"
            )

        # Extract arrays
        depths = np.array([s["depth_score"] for s in scored_samples])
        w_depths = np.array([s["weighted_depth_score"] for s in scored_samples])
        entropies = np.array([s["layer_entropy"] for s in scored_samples])
        labels = np.array([1 if s["is_correct"] else 0 for s in scored_samples])

        n_correct = int(labels.sum())
        n_incorrect = len(labels) - n_correct

        print(f"[HallucinationV2] n={len(labels)}, "
              f"correct={n_correct}, incorrect={n_incorrect}")

        if n_correct < 3 or n_incorrect < 3:
            print("[HallucinationV2] WARNING: Extreme class imbalance. "
                  "Results may not be meaningful.")

        # Pearson correlations
        r_depth, p_depth = _pearson_r(depths, labels.astype(float))
        r_wdepth, p_wdepth = _pearson_r(w_depths, labels.astype(float))
        r_entropy, p_entropy = _pearson_r(entropies, labels.astype(float))

        # AUCs
        auc_depth = _compute_auc(labels, -depths)  # negate: lower depth = deeper = correct?
        auc_wdepth = _compute_auc(labels, -w_depths)
        auc_entropy = _compute_auc(labels, entropies)  # higher entropy = correct?

        # Group means
        correct_mask = labels == 1
        incorrect_mask = labels == 0

        mean_d_correct = float(depths[correct_mask].mean()) if correct_mask.any() else float("nan")
        mean_d_incorrect = float(depths[incorrect_mask].mean()) if incorrect_mask.any() else float("nan")
        mean_wd_correct = float(w_depths[correct_mask].mean()) if correct_mask.any() else float("nan")
        mean_wd_incorrect = float(w_depths[incorrect_mask].mean()) if incorrect_mask.any() else float("nan")
        mean_e_correct = float(entropies[correct_mask].mean()) if correct_mask.any() else float("nan")
        mean_e_incorrect = float(entropies[incorrect_mask].mean()) if incorrect_mask.any() else float("nan")

        # ROC curves
        roc_depth = _compute_roc_curve(labels, depths)
        roc_entropy = _compute_roc_curve(labels, entropies)

        # Optimal thresholds
        opt_depth = _find_optimal_threshold(labels, depths, "depth_score")
        opt_entropy = _find_optimal_threshold(labels, entropies, "layer_entropy")

        # Discovery check
        is_sig = False
        discovery = None
        best_r = max(abs(r_depth), abs(r_wdepth), abs(r_entropy))
        best_p = min(p_depth, p_wdepth, p_entropy)

        if best_r > 0.3 and best_p < 0.01:
            is_sig = True
            if abs(r_depth) > 0.3 and p_depth < 0.01:
                discovery = (
                    f"depth_score correlates with answer correctness "
                    f"(r={r_depth:.3f}, p={p_depth:.6f}, n={len(labels)}). "
                    f"Models reason at different depths when generating correct "
                    f"vs incorrect answers."
                )
            elif abs(r_wdepth) > 0.3 and p_wdepth < 0.01:
                discovery = (
                    f"weighted_depth_score correlates with answer correctness "
                    f"(r={r_wdepth:.3f}, p={p_wdepth:.6f}, n={len(labels)}). "
                    f"Attribution-weighted depth separates correct from incorrect "
                    f"model generations."
                )
            elif abs(r_entropy) > 0.3 and p_entropy < 0.01:
                discovery = (
                    f"layer_entropy correlates with answer correctness "
                    f"(r={r_entropy:.3f}, p={p_entropy:.6f}, n={len(labels)}). "
                    f"Distributed reasoning (high entropy) accompanies correct answers."
                )

        return CorrelationResultV2(
            n_samples=len(labels),
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            depth_accuracy_pearson_r=r_depth,
            depth_accuracy_p_value=p_depth,
            weighted_depth_accuracy_pearson_r=r_wdepth,
            weighted_depth_accuracy_p_value=p_wdepth,
            entropy_accuracy_pearson_r=r_entropy,
            entropy_accuracy_p_value=p_entropy,
            auc_depth=auc_depth,
            auc_weighted_depth=auc_wdepth,
            auc_entropy=auc_entropy,
            mean_depth_correct=mean_d_correct,
            mean_depth_incorrect=mean_d_incorrect,
            mean_weighted_depth_correct=mean_wd_correct,
            mean_weighted_depth_incorrect=mean_wd_incorrect,
            mean_entropy_correct=mean_e_correct,
            mean_entropy_incorrect=mean_e_incorrect,
            optimal_depth_threshold=opt_depth,
            optimal_entropy_threshold=opt_entropy,
            roc_depth=roc_depth,
            roc_entropy=roc_entropy,
            is_significant=is_sig,
            discovery=discovery,
            samples=scored_samples,
            total_time_s=total_time,
        )

    def write_report(
        self,
        result: CorrelationResultV2,
        output_dir: Optional[Path] = None,
    ):
        """Write full markdown report. Auto-generates DISCOVERY file if significant."""
        out = output_dir or PROJECT_ROOT
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Threshold info
        depth_thresh_info = "N/A"
        entropy_thresh_info = "N/A"
        if result.optimal_depth_threshold:
            t = result.optimal_depth_threshold
            depth_thresh_info = (
                f"threshold={t.threshold:.4f}, "
                f"P={t.precision:.3f}, R={t.recall:.3f}, "
                f"F1={t.f1:.3f}, Acc={t.accuracy:.3f}"
            )
        if result.optimal_entropy_threshold:
            t = result.optimal_entropy_threshold
            entropy_thresh_info = (
                f"threshold={t.threshold:.4f}, "
                f"P={t.precision:.3f}, R={t.recall:.3f}, "
                f"F1={t.f1:.3f}, Acc={t.accuracy:.3f}"
            )

        body = f"""# Hallucination Correlation Study V2 — Model-Generated Answers

**Date:** {timestamp}
**Runner:** fathom_hallucination_v2.py
**Duration:** {result.total_time_s:.1f}s

## Design Change from V1

V1 scored pre-supplied correct vs incorrect answer prefixes (r=0.074, p=0.65, n=40).
That tested text processing differences, not reasoning differences.

V2 feeds each TruthfulQA question to the model, captures its own generated answer,
scores the depth signature, and checks against ground truth. This tests whether the
model reasons differently when it produces a correct vs incorrect answer.

## Method

- Dataset: TruthfulQA (n={result.n_samples} questions scored)
- For each question: generate greedy completion (max 20 tokens)
- Score the generation with DepthScorer (depth_score, weighted_depth_score, layer_entropy)
- Check answer correctness via token overlap >= 0.5 against best/correct answers
- Compute Pearson r, AUC, and optimal classification thresholds

## Sample Balance

| Group | Count | % |
|-------|-------|---|
| Correct | {result.n_correct} | {result.n_correct/result.n_samples:.1%} |
| Incorrect | {result.n_incorrect} | {result.n_incorrect/result.n_samples:.1%} |

## Correlation Results

| Metric vs Accuracy | Pearson r | p-value | Significant |
|--------------------|-----------|---------|-------------|
| depth_score | {result.depth_accuracy_pearson_r:.4f} | {result.depth_accuracy_p_value:.6f} | {'**YES**' if abs(result.depth_accuracy_pearson_r) > 0.3 and result.depth_accuracy_p_value < 0.01 else 'NO'} |
| weighted_depth_score | {result.weighted_depth_accuracy_pearson_r:.4f} | {result.weighted_depth_accuracy_p_value:.6f} | {'**YES**' if abs(result.weighted_depth_accuracy_pearson_r) > 0.3 and result.weighted_depth_accuracy_p_value < 0.01 else 'NO'} |
| layer_entropy | {result.entropy_accuracy_pearson_r:.4f} | {result.entropy_accuracy_p_value:.6f} | {'**YES**' if abs(result.entropy_accuracy_pearson_r) > 0.3 and result.entropy_accuracy_p_value < 0.01 else 'NO'} |

## Group Means

| Metric | Correct | Incorrect | Delta |
|--------|---------|-----------|-------|
| depth_score | {result.mean_depth_correct:.4f} | {result.mean_depth_incorrect:.4f} | {result.mean_depth_correct - result.mean_depth_incorrect:.4f} |
| weighted_depth_score | {result.mean_weighted_depth_correct:.4f} | {result.mean_weighted_depth_incorrect:.4f} | {result.mean_weighted_depth_correct - result.mean_weighted_depth_incorrect:.4f} |
| layer_entropy | {result.mean_entropy_correct:.4f} | {result.mean_entropy_incorrect:.4f} | {result.mean_entropy_correct - result.mean_entropy_incorrect:.4f} |

## Discrimination (AUC)

| Metric | AUC |
|--------|-----|
| depth_score | {result.auc_depth:.4f} |
| weighted_depth_score | {result.auc_weighted_depth:.4f} |
| layer_entropy | {result.auc_entropy:.4f} |

## Optimal Classification Thresholds

| Metric | Details |
|--------|---------|
| depth_score | {depth_thresh_info} |
| layer_entropy | {entropy_thresh_info} |

## Discovery Threshold

|r| > 0.3 AND p < 0.01 required for significance.

**Result: {'DISCOVERY — ' + result.discovery if result.is_significant else 'Below discovery threshold at n=' + str(result.n_samples) + '.'}**

## ROC Data (depth_score, first 10 points)

```json
{json.dumps(result.roc_depth[:10], indent=2)}
```
"""

        report_path = out / "EXPERIMENT_HALLUCINATION_V2.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"[HallucinationV2] Report written to {report_path}")

        # Auto-generate discovery file if significant
        if result.is_significant and result.discovery:
            discovery_body = f"""# DISCOVERY: Circuit-Level Hallucination Signal (V2)

**Date:** {timestamp}
**Significance:** |r| > 0.3, p < 0.01

## Finding

{result.discovery}

## Evidence

- n = {result.n_samples} model-generated answers scored
- {result.n_correct} correct, {result.n_incorrect} incorrect
- depth_score correlation: r={result.depth_accuracy_pearson_r:.4f}, p={result.depth_accuracy_p_value:.6f}
- weighted_depth correlation: r={result.weighted_depth_accuracy_pearson_r:.4f}, p={result.weighted_depth_accuracy_p_value:.6f}
- entropy correlation: r={result.entropy_accuracy_pearson_r:.4f}, p={result.entropy_accuracy_p_value:.6f}
- AUC (depth): {result.auc_depth:.4f}
- AUC (weighted): {result.auc_weighted_depth:.4f}
- AUC (entropy): {result.auc_entropy:.4f}

## Implication

Models process questions differently at the circuit level when they produce correct
vs incorrect answers. This depth signature can be measured in real time during
generation — making it a potential hallucination detector that operates on circuit
evidence, not heuristics.

## Methodology

V2 design: model generates its own answers (greedy, max 20 tokens). Depth scored
on the full question+answer. Correctness determined by token overlap against
TruthfulQA ground truth (threshold >= 0.5). This tests the model's own reasoning
patterns, not its processing of externally supplied text.
"""
            disc_path = out / "DISCOVERY_HALLUCINATION_V2.md"
            with open(disc_path, "w", encoding="utf-8") as f:
                f.write(discovery_body)
            print(f"[HallucinationV2] *** DISCOVERY FILE WRITTEN: {disc_path} ***")

        return report_path
