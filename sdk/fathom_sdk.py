"""
FATHOM SDK — Python Interface for Reasoning Depth Analysis
============================================================

Two modes of operation:
  - Local mode: loads the model directly (requires GPU)
  - API mode: hits a running Fathom API server (no GPU needed)

Usage:
    from fathom_sdk import Fathom

    # Local (GPU)
    f = Fathom()
    result = f.score("Why does fire spread?")

    # Remote (API)
    f = Fathom(api_url="http://localhost:8400", api_key="your-key")
    result = f.score("Why does fire spread?")
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Allow importing from project root
_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


# ── Result dataclasses ───────────────────────────────────────────────

@dataclass
class ScoreResult:
    """Result of a single prompt score."""
    prompt: str
    score: float                # 0-100 fathom score
    grade: str                  # S/A/B/C/D/F
    interpretation: str
    depth_score: float
    weighted_depth_score: Optional[float]
    layer_entropy: Optional[float]
    early_ratio: float
    mid_ratio: float
    late_ratio: float
    feature_count: int
    compute_time_ms: float
    model: str
    components: dict

    def __repr__(self):
        return (f"ScoreResult(grade='{self.grade}', score={self.score:.0f}, "
                f"depth={self.depth_score:.3f})")


@dataclass
class CompareResult:
    """Result of comparing two prompts."""
    surface: ScoreResult
    insight: ScoreResult
    delta_score: float          # insight.score - surface.score
    winner: str                 # "insight", "surface", or "tied"

    def __repr__(self):
        return (f"CompareResult(winner='{self.winner}', "
                f"delta={self.delta_score:+.1f})")


# ── Main SDK class ───────────────────────────────────────────────────

class Fathom:
    """Fathom reasoning depth analyzer.

    Two modes:
    - Local: loads model to GPU, runs attribution directly.
      f = Fathom()
    - API: connects to a running Fathom server.
      f = Fathom(api_url="http://localhost:8400", api_key="key")
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "google/gemma-2-2b",
        max_feature_nodes: int = 150,
    ):
        self._api_url = api_url.rstrip("/") if api_url else None
        self._api_key = api_key
        self._model_name = model_name
        self._max_feature_nodes = max_feature_nodes
        self._scorer = None
        self._mode = "api" if api_url else "local"

    def _ensure_scorer(self):
        """Lazy-load the DepthScorer for local mode."""
        if self._scorer is not None:
            return
        if self._mode == "api":
            return

        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

        from depth_scorer import DepthScorer
        self._scorer = DepthScorer(
            model_name=self._model_name,
            max_feature_nodes=self._max_feature_nodes,
        )

    def _api_post(self, endpoint: str, payload: dict) -> dict:
        """Make a POST request to the Fathom API."""
        import urllib.request
        import urllib.error

        url = f"{self._api_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API error {e.code}: {body}") from None
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Fathom API at {self._api_url}: {e.reason}"
            ) from None

    def _api_get(self, endpoint: str) -> dict:
        """Make a GET request to the Fathom API."""
        import urllib.request
        import urllib.error

        url = f"{self._api_url}{endpoint}"
        headers = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API error {e.code}: {body}") from None

    # ── Score ────────────────────────────────────────────────────────

    def score(self, prompt: str, target: Optional[str] = None) -> ScoreResult:
        """Score a single prompt.

        Args:
            prompt: The text to analyze.
            target: Optional target token to score toward.

        Returns:
            ScoreResult with grade, score, and all metrics.
        """
        if self._mode == "api":
            return self._score_api(prompt, target)
        return self._score_local(prompt, target)

    def _score_api(self, prompt: str, target: Optional[str]) -> ScoreResult:
        payload = {"prompt": prompt}
        if target:
            payload["target"] = target

        resp = self._api_post("/v2/fathom_score", payload)
        d = resp["data"]
        dr = d["depth_result"]
        meta = resp["meta"]

        return ScoreResult(
            prompt=prompt,
            score=d["fathom_score"],
            grade=d["grade"],
            interpretation=d["interpretation"],
            depth_score=dr["depth_score"] or 0.0,
            weighted_depth_score=dr.get("weighted_depth_score"),
            layer_entropy=dr.get("layer_entropy"),
            early_ratio=dr["early_ratio"],
            mid_ratio=dr["mid_ratio"],
            late_ratio=dr["late_ratio"],
            feature_count=dr["feature_count"],
            compute_time_ms=meta["compute_time_ms"],
            model=meta["model"],
            components=d["components"],
        )

    def _score_local(self, prompt: str, target: Optional[str]) -> ScoreResult:
        import gc
        import time
        import torch
        from fathom_score import compute_fathom_score

        self._ensure_scorer()
        t0 = time.time()

        depth_result = self._scorer.score(prompt, target=target)
        fs = compute_fathom_score(depth_result)

        elapsed_ms = (time.time() - t0) * 1000

        torch.cuda.empty_cache()
        gc.collect()

        return ScoreResult(
            prompt=prompt,
            score=fs.score,
            grade=fs.grade,
            interpretation=fs.interpretation,
            depth_score=depth_result.depth_score,
            weighted_depth_score=depth_result.weighted_depth_score,
            layer_entropy=depth_result.layer_entropy,
            early_ratio=depth_result.early_ratio,
            mid_ratio=depth_result.mid_ratio,
            late_ratio=depth_result.late_ratio,
            feature_count=depth_result.feature_count,
            compute_time_ms=elapsed_ms,
            model=self._model_name,
            components=fs.components,
        )

    # ── Batch ────────────────────────────────────────────────────────

    def batch(self, prompts: List[str]) -> List[ScoreResult]:
        """Score multiple prompts.

        In API mode, batches up to 20 per request.
        In local mode, scores sequentially with GPU cleanup between each.

        Args:
            prompts: List of prompts to score.

        Returns:
            List of ScoreResult (None entries for failures).
        """
        results = []

        if self._mode == "api":
            # Chunk into batches of 20
            for i in range(0, len(prompts), 20):
                chunk = prompts[i:i + 20]
                for prompt in chunk:
                    try:
                        results.append(self.score(prompt))
                    except Exception:
                        results.append(None)
        else:
            for prompt in prompts:
                try:
                    results.append(self.score(prompt))
                except Exception:
                    results.append(None)

        return results

    # ── Compare ──────────────────────────────────────────────────────

    def compare(self, surface_prompt: str, insight_prompt: str) -> CompareResult:
        """Compare two prompts — surface (recall) vs insight (reasoning).

        Args:
            surface_prompt: The surface/recall prompt.
            insight_prompt: The insight/reasoning prompt.

        Returns:
            CompareResult with both scores and delta.
        """
        s = self.score(surface_prompt)
        i = self.score(insight_prompt)

        delta = i.score - s.score
        if abs(delta) < 1.0:
            winner = "tied"
        elif delta > 0:
            winner = "insight"
        else:
            winner = "surface"

        return CompareResult(
            surface=s,
            insight=i,
            delta_score=delta,
            winner=winner,
        )

    # ── Health ───────────────────────────────────────────────────────

    def health(self) -> dict:
        """Check API server health (API mode only)."""
        if self._mode != "api":
            return {"status": "local", "model": self._model_name}
        return self._api_get("/v2/health")

    # ── Badge generation ─────────────────────────────────────────────

    def badge(self, prompt: str, output_path: str) -> str:
        """Score a prompt and generate an SVG badge.

        Args:
            prompt: The text to analyze.
            output_path: Path for the SVG file.

        Returns:
            The output path.
        """
        result = self.score(prompt)
        from fathom_score import FathomScore
        fs = FathomScore(
            score=result.score,
            grade=result.grade,
            components=result.components,
            interpretation=result.interpretation,
        )
        from fathom_badge import generate_badge
        return generate_badge(fs, output_path)
