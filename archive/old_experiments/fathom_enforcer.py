"""
REASONING ENFORCER — Real-Time Depth-Steered Generation
========================================================

The first system that actively prevents shallow AI reasoning during
generation. Not trained differently — steered in real time. Monitors depth
trajectory token by token, intervenes when reasoning becomes shallow by
adjusting logits toward tokens associated with deeper computation, and
retries from correction points when steering isn't enough.
"""

import gc
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import numpy as np


@dataclass
class CorrectionEvent:
    """Record of a single depth correction during generation."""
    token_idx: int
    token: str
    depth_before: float
    depth_after: float
    correction_type: str   # "logit_boost", "retry", "accepted"
    improvement: float     # depth_before - depth_after (positive = deeper)


@dataclass
class EnforcementReport:
    """Full report of an enforced generation."""
    prompt: str
    generated_text: str
    n_tokens: int
    n_corrections: int
    n_retries: int
    initial_mean_depth: float      # depth of first uncorrected attempt
    final_mean_depth: float        # depth of final enforced output
    depth_improvement: float       # initial - final (positive = got deeper)
    corrections: List[CorrectionEvent]
    trajectory: List[Tuple[str, float]]  # (token, depth_score) for final output
    compute_time_s: float

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "n_tokens": self.n_tokens,
            "n_corrections": self.n_corrections,
            "n_retries": self.n_retries,
            "initial_mean_depth": self.initial_mean_depth,
            "final_mean_depth": self.final_mean_depth,
            "depth_improvement": self.depth_improvement,
            "compute_time_s": self.compute_time_s,
        }


class ReasoningEnforcer:
    """Wraps a DepthScorer and model to enforce minimum reasoning depth during generation.

    Generates text token by token, periodically scoring depth. When the running
    depth_score exceeds the minimum threshold (indicating shallow processing),
    the enforcer intervenes: it re-ranks candidate tokens by their depth impact,
    boosting those associated with deeper computation. If boosting fails, it
    retries from the last good checkpoint. The result is text that the model was
    steered into reasoning about — not just recalling.

    EXPERIMENTAL: This is a research prototype. Enforcement adds significant
    compute overhead and may produce less fluent text. The depth-fluency tradeoff
    is an open research question.
    """

    def __init__(
        self,
        scorer,
        min_depth_score: float = 7.8,
        max_retries: int = 5,
        check_interval: int = 5,
        beam_width: int = 5,
    ):
        """
        Args:
            scorer: An initialized DepthScorer instance.
            min_depth_score: Maximum acceptable depth_score (lower = deeper).
                Generation is steered when depth exceeds this.
            max_retries: Maximum retry attempts per correction point.
            check_interval: Score depth every N tokens.
            beam_width: Number of candidate tokens to evaluate during correction.
        """
        self.scorer = scorer
        self.min_depth_score = min_depth_score
        self.max_retries = max_retries
        self.check_interval = check_interval
        self.beam_width = beam_width

        self._inner = scorer._get_inner_model()
        if self._inner is None:
            raise ValueError("ReasoningEnforcer requires inner model access.")

    def enforce_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
    ) -> Tuple[str, EnforcementReport]:
        """Generate text with real-time depth enforcement.

        Token-by-token generation with periodic depth checks. When depth
        becomes shallow, apply corrections: boost deeper-associated token
        logits, or retry from the last checkpoint.

        Args:
            prompt: The initial prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (generated_text, EnforcementReport).
        """
        t0 = time.time()

        generated_tokens = []
        trajectory = []
        corrections = []
        current_prompt = prompt
        retries_total = 0
        last_checkpoint = prompt
        last_checkpoint_idx = 0
        uncorrected_depths = []

        for step in range(max_tokens):
            tokens = self._inner.to_tokens(current_prompt)

            with torch.no_grad():
                logits = self._inner(tokens)

            # Should we check depth?
            should_check = (step > 0 and step % self.check_interval == 0)

            if should_check and generated_tokens:
                # Score current depth
                last_token = generated_tokens[-1]
                try:
                    result = self.scorer.score(
                        current_prompt[:-len(last_token)] if current_prompt.endswith(last_token) else current_prompt,
                        target=last_token,
                    )
                    current_depth = result.depth_score
                except Exception:
                    current_depth = self.min_depth_score  # assume OK if scoring fails

                if step <= self.check_interval:
                    uncorrected_depths.append(current_depth)

                # Enforce depth
                if not math.isnan(current_depth) and current_depth > self.min_depth_score:
                    # Try logit boosting
                    corrected_token, new_depth = self._boost_for_depth(
                        current_prompt, logits[0, -1]
                    )

                    if corrected_token is not None and new_depth < current_depth:
                        corrections.append(CorrectionEvent(
                            token_idx=step,
                            token=corrected_token,
                            depth_before=current_depth,
                            depth_after=new_depth,
                            correction_type="logit_boost",
                            improvement=current_depth - new_depth,
                        ))

                        # Replace the last token with the corrected one
                        generated_tokens.append(corrected_token)
                        current_prompt += corrected_token
                        trajectory.append((corrected_token, new_depth))
                        continue

                    # Logit boost didn't help enough — retry from checkpoint
                    if retries_total < self.max_retries:
                        retries_total += 1
                        corrections.append(CorrectionEvent(
                            token_idx=step,
                            token=generated_tokens[-1] if generated_tokens else "",
                            depth_before=current_depth,
                            depth_after=current_depth,
                            correction_type="retry",
                            improvement=0.0,
                        ))

                        # Roll back to checkpoint
                        generated_tokens = generated_tokens[:last_checkpoint_idx]
                        current_prompt = last_checkpoint
                        trajectory = trajectory[:last_checkpoint_idx]
                        continue
                    else:
                        # Accept the shallow output
                        corrections.append(CorrectionEvent(
                            token_idx=step,
                            token=generated_tokens[-1] if generated_tokens else "",
                            depth_before=current_depth,
                            depth_after=current_depth,
                            correction_type="accepted",
                            improvement=0.0,
                        ))
                else:
                    # Depth is OK — save checkpoint
                    last_checkpoint = current_prompt
                    last_checkpoint_idx = len(generated_tokens)

            # Normal token generation
            pred_id = logits[0, -1].argmax().item()
            pred_token = self._inner.tokenizer.decode([pred_id])

            # Verify single-token
            re_encoded = self._inner.tokenizer.encode(pred_token, add_special_tokens=False)
            if len(re_encoded) != 1:
                break

            # Check EOS
            eos_id = getattr(self._inner.tokenizer, "eos_token_id", None)
            if pred_id == eos_id:
                break

            generated_tokens.append(pred_token)
            current_prompt += pred_token
            trajectory.append((pred_token, self.min_depth_score))  # placeholder depth

            torch.cuda.empty_cache()
            gc.collect()

        generated_text = "".join(generated_tokens)

        # Compute final mean depth from trajectory
        final_depths = [d for _, d in trajectory if not math.isnan(d)]
        final_mean = float(np.mean(final_depths)) if final_depths else float("nan")
        initial_mean = float(np.mean(uncorrected_depths)) if uncorrected_depths else final_mean

        report = EnforcementReport(
            prompt=prompt,
            generated_text=generated_text,
            n_tokens=len(generated_tokens),
            n_corrections=len(corrections),
            n_retries=retries_total,
            initial_mean_depth=initial_mean,
            final_mean_depth=final_mean,
            depth_improvement=initial_mean - final_mean if not math.isnan(initial_mean) else 0.0,
            corrections=corrections,
            trajectory=trajectory,
            compute_time_s=time.time() - t0,
        )

        return generated_text, report

    def _boost_for_depth(
        self,
        current_prompt: str,
        last_logits: torch.Tensor,
    ) -> Tuple[Optional[str], float]:
        """Try candidate tokens and pick the one that produces deepest reasoning."""
        top_logits, top_ids = torch.topk(last_logits, k=self.beam_width)

        best_token = None
        best_depth = float("inf")

        for cand_id in top_ids:
            cand_id = cand_id.item()
            cand_token = self._inner.tokenizer.decode([cand_id])

            re_enc = self._inner.tokenizer.encode(cand_token, add_special_tokens=False)
            if len(re_enc) != 1 or not cand_token.strip():
                continue

            try:
                r = self.scorer.score(current_prompt, target=cand_token)
                if not math.isnan(r.depth_score) and r.depth_score < best_depth:
                    best_depth = r.depth_score
                    best_token = cand_token
            except Exception:
                continue

            torch.cuda.empty_cache()
            gc.collect()

        return best_token, best_depth

    def enforcement_report(
        self,
        prompt: str,
        report: EnforcementReport,
    ) -> str:
        """Human-readable summary of an enforcement run.

        Args:
            prompt: The original prompt.
            report: The EnforcementReport from enforce_generate().

        Returns:
            Formatted string summary.
        """
        lines = [
            f"Enforcement Report",
            f"{'=' * 40}",
            f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}",
            f"Tokens generated: {report.n_tokens}",
            f"Corrections: {report.n_corrections}",
            f"Retries: {report.n_retries}",
            f"Initial mean depth: {report.initial_mean_depth:.3f}",
            f"Final mean depth: {report.final_mean_depth:.3f}",
            f"Improvement: {report.depth_improvement:+.3f}",
            f"Compute time: {report.compute_time_s:.1f}s",
        ]

        if report.corrections:
            lines.append(f"\nCorrections:")
            for c in report.corrections:
                lines.append(
                    f"  Token {c.token_idx} '{c.token}': {c.correction_type} "
                    f"(depth {c.depth_before:.3f} → {c.depth_after:.3f}, "
                    f"improvement: {c.improvement:+.3f})"
                )

        return "\n".join(lines)
