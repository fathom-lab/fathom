"""
FATHOM AGENT GUARD — Reasoning Depth Guardrail for Multi-Agent Systems
========================================================================

A lightweight guardrail that wraps any agent output and verifies reasoning
depth before allowing execution. Drop into any LangChain, AutoGPT, or
CrewAI pipeline as a single import.

Usage:
    from fathom_agent_guard import AgentGuard

    guard = AgentGuard(scorer, min_grade="C")
    result = guard.check("The agent's reasoning output...")

    if result["pass"]:
        agent.execute()
    else:
        agent.escalate(result["reason"])

API mode (no local scorer):
    guard = AgentGuard.from_api("http://localhost:8400", api_key="...")
    result = guard.check("agent output")
"""

from dataclasses import dataclass
from typing import Optional, Union


# Grade ordering for comparison
GRADE_ORDER = ["F", "D", "C", "B", "B+", "A", "A+", "S"]
GRADE_RANK = {g: i for i, g in enumerate(GRADE_ORDER)}


@dataclass
class GuardResult:
    """Result of a guard check."""
    passed: bool
    grade: str
    score: int
    reason: str
    depth_score: Optional[float] = None
    entropy: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "pass": self.passed,
            "grade": self.grade,
            "score": self.score,
            "reason": self.reason,
            "depth_score": self.depth_score,
            "entropy": self.entropy,
        }

    def __bool__(self) -> bool:
        return self.passed


class AgentGuard:
    """
    Lightweight reasoning depth guardrail for AI agent pipelines.

    Checks whether agent output meets a minimum reasoning depth threshold
    before allowing execution. Wraps the Fathom scoring engine.

    Args:
        scorer: A DepthScorer instance (or None if using API mode).
        min_grade: Minimum acceptable grade ("F", "D", "C", "B", "B+", "A", "A+", "S").
                   Outputs below this grade will fail the check.
        api_url: Fathom API URL (for API mode, when scorer is None).
        api_key: Fathom API key (optional).
    """

    def __init__(
        self,
        scorer=None,
        min_grade: str = "C",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.scorer = scorer
        self.min_grade = min_grade
        self.api_url = api_url
        self.api_key = api_key

        if min_grade not in GRADE_RANK:
            raise ValueError(
                f"Invalid min_grade '{min_grade}'. Must be one of: {GRADE_ORDER}"
            )

        self._check_count = 0
        self._pass_count = 0
        self._fail_count = 0

    @classmethod
    def from_api(cls, api_url: str, api_key: str = "", min_grade: str = "C"):
        """Create an AgentGuard that uses the Fathom API instead of a local scorer."""
        return cls(scorer=None, min_grade=min_grade, api_url=api_url, api_key=api_key)

    def check(self, output_text: str) -> dict:
        """
        Check whether agent output meets the reasoning depth threshold.

        Args:
            output_text: The agent's output text to evaluate.

        Returns:
            dict with keys: pass, grade, score, reason, depth_score, entropy
        """
        if not output_text or len(output_text.strip()) < 5:
            result = GuardResult(
                passed=False,
                grade="F",
                score=0,
                reason="Output too short to evaluate reasoning depth.",
            )
            self._record(result)
            return result.to_dict()

        try:
            if self.scorer is not None:
                return self._check_local(output_text)
            elif self.api_url:
                return self._check_api(output_text)
            else:
                raise ValueError("No scorer or API URL provided.")
        except Exception as e:
            # On error, fail open with warning
            result = GuardResult(
                passed=True,
                grade="?",
                score=-1,
                reason=f"Guard error (failing open): {e}",
            )
            self._record(result)
            return result.to_dict()

    def _check_local(self, text: str) -> dict:
        """Score using local DepthScorer."""
        # Import here to avoid hard dependency
        try:
            from fathom_score import compute_fathom_score
        except ImportError:
            compute_fathom_score = None

        depth_result = self.scorer.score(text)

        if compute_fathom_score:
            fs = compute_fathom_score(depth_result)
            grade = fs.grade
            score = fs.score
            interpretation = fs.interpretation
        else:
            grade, score = self._grade_from_depth(depth_result.depth_score)
            interpretation = f"Depth score: {depth_result.depth_score:.3f}"

        passed = GRADE_RANK.get(grade, 0) >= GRADE_RANK[self.min_grade]

        if passed:
            reason = f"Output passes depth threshold ({grade} >= {self.min_grade}). {interpretation}"
        else:
            reason = (
                f"Output below depth threshold ({grade} < {self.min_grade}). "
                f"{interpretation}. Recommend human review before execution."
            )

        result = GuardResult(
            passed=passed,
            grade=grade,
            score=score,
            reason=reason,
            depth_score=depth_result.depth_score,
            entropy=getattr(depth_result, "layer_entropy", None),
        )
        self._record(result)
        return result.to_dict()

    def _check_api(self, text: str) -> dict:
        """Score using Fathom API."""
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        resp = requests.post(
            f"{self.api_url.rstrip('/')}/v2/fathom_score",
            json={"prompt": text[:4096]},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract from API response
        result_data = data.get("data", data)
        grade = result_data.get("grade", "?")
        score = result_data.get("fathom_score", 0)
        interpretation = result_data.get("interpretation", "")
        depth_score = result_data.get("depth_result", {}).get("depth_score")
        entropy = result_data.get("depth_result", {}).get("layer_entropy")

        passed = GRADE_RANK.get(grade, 0) >= GRADE_RANK[self.min_grade]

        if passed:
            reason = f"Output passes depth threshold ({grade} >= {self.min_grade}). {interpretation}"
        else:
            reason = (
                f"Output below depth threshold ({grade} < {self.min_grade}). "
                f"{interpretation}. Recommend human review before execution."
            )

        result = GuardResult(
            passed=passed,
            grade=grade,
            score=score,
            reason=reason,
            depth_score=depth_score,
            entropy=entropy,
        )
        self._record(result)
        return result.to_dict()

    def _grade_from_depth(self, depth_score: float) -> tuple:
        """Fallback grading when fathom_score module unavailable."""
        if depth_score <= 7.5:
            return "A+", 90
        elif depth_score <= 7.8:
            return "A", 82
        elif depth_score <= 8.0:
            return "B+", 74
        elif depth_score <= 8.2:
            return "B", 66
        elif depth_score <= 8.5:
            return "C", 55
        elif depth_score <= 9.0:
            return "D", 40
        else:
            return "F", 25

    def _record(self, result: GuardResult):
        """Track pass/fail statistics."""
        self._check_count += 1
        if result.passed:
            self._pass_count += 1
        else:
            self._fail_count += 1

    @property
    def stats(self) -> dict:
        """Return guard statistics."""
        return {
            "checks": self._check_count,
            "passed": self._pass_count,
            "failed": self._fail_count,
            "pass_rate": self._pass_count / self._check_count if self._check_count > 0 else 0.0,
            "min_grade": self.min_grade,
        }

    def __repr__(self) -> str:
        mode = "local" if self.scorer else f"api:{self.api_url}"
        return f"AgentGuard(min_grade={self.min_grade!r}, mode={mode}, checks={self._check_count})"


# ── Convenience functions ─────────────────────────────────────────────

def guard(text: str, min_grade: str = "C", api_url: str = "http://localhost:8400") -> dict:
    """
    One-shot guard check via API. No setup needed.

    Args:
        text: Agent output to check.
        min_grade: Minimum acceptable grade.
        api_url: Fathom API URL.

    Returns:
        dict with pass, grade, score, reason.
    """
    g = AgentGuard.from_api(api_url, min_grade=min_grade)
    return g.check(text)
