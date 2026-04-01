"""
FATHOM API v2 — Production-Ready Reasoning Depth Scoring Server
================================================================

FastAPI server exposing all fathom capabilities over HTTP/JSON.
Run: uvicorn depth_api_v2:app --host 0.0.0.0 --port 8400
"""

import atexit
import gc
import json
import math
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"

# ── Version ──────────────────────────────────────────────────────────
FATHOM_VERSION = "2.0"


# ── Request / Response Models ────────────────────────────────────────

class ScoreRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    target: Optional[str] = None

class CompareRequest(BaseModel):
    surface_prompt: str = Field(..., min_length=1, max_length=4096)
    insight_prompt: str = Field(..., min_length=1, max_length=4096)
    surface_target: Optional[str] = None
    insight_target: Optional[str] = None

class FingerprintRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    target: Optional[str] = None

class BatchRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=20)

class TrajectoryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(default=20, ge=1, le=50)

class FathomScoreRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    target: Optional[str] = None

class CorrectnessRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    response: str = Field(..., min_length=1, max_length=4096)

class AdversarialRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)

class FilterDatasetRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    min_grade: str = Field(default="B")

class WaitlistRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)


# ── Rate limiter (in-memory) ────────────────────────────────────────

class RateLimiter:
    """Simple in-memory rate limiter: max_requests per window_seconds per key."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window
        # Prune old entries
        self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]
        if len(self._buckets[key]) >= self.max_requests:
            return False
        self._buckets[key].append(now)
        return True

    def remaining(self, key: str) -> int:
        now = time.time()
        cutoff = now - self.window
        self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]
        return max(0, self.max_requests - len(self._buckets[key]))


_rate_limit = 1000 if os.environ.get("FATHOM_MOCK", "0") == "1" else 10
rate_limiter = RateLimiter(max_requests=_rate_limit, window_seconds=60)


# ── Auth + rate limit dependency ────────────────────────────────────

API_KEY = os.environ.get("FATHOM_API_KEY", "")


async def verify_auth_and_rate(request: Request):
    """Validate API key and check rate limit."""
    # Auth
    if API_KEY:
        provided = request.headers.get("x-api-key", "")
        if provided != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")
        rate_key = provided
    else:
        # No key configured — use client IP for rate limiting
        rate_key = request.client.host if request.client else "unknown"

    # Rate limit
    if not rate_limiter.check(rate_key):
        remaining = rate_limiter.remaining(rate_key)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded (10 req/min). Remaining: {remaining}",
        )


# ── App setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="Fathom API",
    description="Reasoning depth scoring for transformer models",
    version=FATHOM_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RuntimeError)
async def cuda_error_handler(request: Request, exc: RuntimeError):
    """Catch CUDA async errors globally — return 503 instead of crashing."""
    msg = str(exc).lower()
    if "cuda" in msg or "out of memory" in msg:
        _gpu_flush()
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"detail": "GPU out of memory. Service recovering."},
        )
    raise exc


# ── Mock mode ─────────────────────────────────────────────────────────

MOCK_MODE = os.environ.get("FATHOM_MOCK", "0") == "1"


class _MockDepthResult:
    """Fake DepthResult for mock mode testing."""
    def __init__(self, prompt="mock"):
        self.prompt = prompt
        self.depth_score = 8.23
        self.weighted_depth_score = 8.19
        self.feature_count = 87
        self.n_layers = 26
        self.layer_profile = {6: 5, 8: 12, 10: 18, 12: 15, 14: 10, 16: 8, 18: 7, 20: 6, 22: 4, 24: 2}
        self.early_ratio = 0.195
        self.mid_ratio = 0.494
        self.late_ratio = 0.311
        self.compute_time_s = 0.05
        self.normalized_score = 0.51
        self.layer_entropy = 3.12
        self.model_id = "mock/gemma-2-2b"
        self.is_anomalous = False
        self.anomaly_reasons = None
        self.depth_score_ci_low = None
        self.depth_score_ci_high = None
        self.depth_score_std = None


class _MockScorer:
    """Fake DepthScorer for API testing without GPU."""
    model_name = "mock/gemma-2-2b"
    max_feature_nodes = 150

    def score(self, prompt, target=None, n_bootstrap=None):
        return _MockDepthResult(prompt)

    def compare(self, surface, insight, surface_target=None, insight_target=None):
        from dataclasses import dataclass
        s = _MockDepthResult(surface)
        i = _MockDepthResult(insight)
        i.depth_score = 8.10
        i.weighted_depth_score = 8.05
        class _CR:
            pass
        cr = _CR()
        cr.surface = s
        cr.insight = i
        cr.delta = s.depth_score - i.depth_score
        cr.surface_deeper = True
        cr.feature_count_ratio = 1.05
        cr.cross_model = None
        return cr

    def fingerprint(self, result):
        import numpy as np
        return np.array([8.19, 3.12, 0.195, 0.494, 0.311, 0.58, 0.42])

    def score_trajectory(self, prompt, max_tokens=20):
        return [(" the", 8.3, 3.0), (" reason", 8.1, 3.2), (" is", 8.25, 2.9)]


# ── Global scorer (loaded once at startup) ───────────────────────────

_scorer = None
_model_name = "google/gemma-2-2b"
_load_time = 0.0


@app.on_event("startup")
async def load_model():
    """Load DepthScorer once at server startup (or mock in test mode)."""
    global _scorer, _load_time

    if MOCK_MODE:
        _scorer = _MockScorer()
        _load_time = 0.0
        print("[Fathom API] MOCK MODE — no GPU, returning hardcoded scores")
        return

    import torch
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,expandable_segments:True")
    os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from depth_scorer import DepthScorer

    t0 = time.time()
    try:
        _scorer = DepthScorer(model_name=_model_name, max_feature_nodes=50, batch_size=8)
    except OSError:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.enable_flash_sdp(False)
        _scorer = DepthScorer(model_name=_model_name, max_feature_nodes=50, batch_size=8)

    _load_time = time.time() - t0
    print(f"[Fathom API] Model loaded in {_load_time:.1f}s")


def get_scorer():
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _scorer


def _gpu_flush():
    if MOCK_MODE:
        return
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        gc.collect()


def _cuda_ok() -> bool:
    """Check if CUDA is in a usable state. Returns False if context is corrupted."""
    if MOCK_MODE:
        return True
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.current_device()
            return True
    except Exception:
        return False
    return True


def _wrap_response(data: dict, compute_time_ms: float) -> dict:
    return {
        "data": data,
        "meta": {
            "fathom_version": FATHOM_VERSION,
            "model": _model_name,
            "compute_time_ms": round(compute_time_ms, 1),
        },
    }


def _depth_result_to_dict(r) -> dict:
    """Convert DepthResult to clean JSON-safe dict."""
    d = {
        "prompt": r.prompt,
        "depth_score": r.depth_score if not math.isnan(r.depth_score) else None,
        "weighted_depth_score": r.weighted_depth_score,
        "feature_count": r.feature_count,
        "n_layers": r.n_layers,
        "layer_profile": {str(k): v for k, v in r.layer_profile.items()},
        "early_ratio": round(r.early_ratio, 4),
        "mid_ratio": round(r.mid_ratio, 4),
        "late_ratio": round(r.late_ratio, 4),
        "normalized_score": round(r.normalized_score, 4),
        "layer_entropy": r.layer_entropy,
        "compute_time_s": round(r.compute_time_s, 3),
        "model_id": r.model_id,
        "is_anomalous": r.is_anomalous,
        "anomaly_reasons": r.anomaly_reasons,
    }
    if r.depth_score_ci_low is not None:
        d["depth_score_ci_low"] = r.depth_score_ci_low
        d["depth_score_ci_high"] = r.depth_score_ci_high
        d["depth_score_std"] = r.depth_score_std
    return d


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/v2/health")
async def health():
    """GPU status, model loaded, version.

    IMPORTANT: Do NOT query CUDA while model is still loading — async CUDA
    errors during model init will crash the health check and corrupt the
    entire CUDA context.  Only report GPU info once the scorer is ready.
    """
    gpu_info = {}
    if _scorer is not None and not MOCK_MODE:
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                gpu_info = {
                    "gpu_name": torch.cuda.get_device_name(dev),
                    "vram_total_gb": round(torch.cuda.get_device_properties(dev).total_memory / 1024**3, 2),
                    "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                }
                try:
                    free = torch.cuda.mem_get_info()[0]
                    gpu_info["vram_free_gb"] = round(free / 1024**3, 2)
                except RuntimeError:
                    gpu_info["vram_free_gb"] = None
        except Exception:
            gpu_info = {"error": "CUDA error querying GPU state"}

    return {
        "status": "ok" if _scorer is not None else "loading",
        "fathom_version": FATHOM_VERSION,
        "model": _model_name,
        "model_loaded": _scorer is not None,
        "model_load_time_s": round(_load_time, 1),
        "gpu": gpu_info,
    }


@app.get("/v2/calibration")
async def calibration(auth=Depends(verify_auth_and_rate)):
    """Return CALIBRATION constants."""
    from depth_scorer import CALIBRATION
    return {"calibration": CALIBRATION, "fathom_version": FATHOM_VERSION}


@app.post("/v2/score")
async def score(req: ScoreRequest, auth=Depends(verify_auth_and_rate)):
    """Score a single prompt."""
    scorer = get_scorer()
    _gpu_flush()
    t0 = time.time()
    try:
        result = scorer.score(req.prompt, target=req.target)
        elapsed_ms = (time.time() - t0) * 1000
        return _wrap_response(_depth_result_to_dict(result), elapsed_ms)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            _gpu_flush()
            raise HTTPException(status_code=503, detail="GPU out of memory. Try again.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _gpu_flush()


@app.post("/v2/compare")
async def compare(req: CompareRequest, auth=Depends(verify_auth_and_rate)):
    """Compare surface vs insight prompts."""
    scorer = get_scorer()
    _gpu_flush()
    t0 = time.time()
    try:
        result = scorer.compare(
            req.surface_prompt, req.insight_prompt,
            surface_target=req.surface_target,
            insight_target=req.insight_target,
        )
        elapsed_ms = (time.time() - t0) * 1000
        data = {
            "surface": _depth_result_to_dict(result.surface),
            "insight": _depth_result_to_dict(result.insight),
            "delta": round(result.delta, 4),
            "surface_deeper": result.surface_deeper,
            "feature_count_ratio": round(result.feature_count_ratio, 4),
        }
        if result.cross_model:
            data["cross_model"] = result.cross_model
        return _wrap_response(data, elapsed_ms)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            _gpu_flush()
            raise HTTPException(status_code=503, detail="GPU out of memory. Try again.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _gpu_flush()


@app.post("/v2/fingerprint")
async def fingerprint(req: FingerprintRequest, auth=Depends(verify_auth_and_rate)):
    """Return cognitive fingerprint vector."""
    scorer = get_scorer()
    t0 = time.time()
    try:
        result = scorer.score(req.prompt, target=req.target)
        fp = scorer.fingerprint(result)
        elapsed_ms = (time.time() - t0) * 1000
        data = {
            "prompt": req.prompt,
            "fingerprint": fp.tolist(),
            "labels": [
                "weighted_depth_score", "layer_entropy",
                "early_ratio", "mid_ratio", "late_ratio",
                "feature_count_normalized", "compute_efficiency",
            ],
        }
        return _wrap_response(data, elapsed_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _gpu_flush()


@app.post("/v2/batch")
async def batch(req: BatchRequest, auth=Depends(verify_auth_and_rate)):
    """Score up to 20 prompts."""
    if len(req.prompts) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 prompts per batch")

    scorer = get_scorer()
    t0 = time.time()
    results = []
    errors = []
    for i, prompt in enumerate(req.prompts):
        try:
            r = scorer.score(prompt)
            results.append(_depth_result_to_dict(r))
        except Exception as e:
            results.append(None)
            errors.append({"index": i, "prompt": prompt[:80], "error": str(e)})
        _gpu_flush()

    elapsed_ms = (time.time() - t0) * 1000
    data = {
        "results": results,
        "n_scored": sum(1 for r in results if r is not None),
        "n_failed": len(errors),
        "errors": errors if errors else None,
    }
    return _wrap_response(data, elapsed_ms)


@app.post("/v2/trajectory")
async def trajectory(req: TrajectoryRequest, auth=Depends(verify_auth_and_rate)):
    """Depth trajectory over generation."""
    scorer = get_scorer()
    t0 = time.time()
    try:
        traj = scorer.score_trajectory(req.prompt, max_tokens=req.max_tokens)
        elapsed_ms = (time.time() - t0) * 1000
        data = {
            "prompt": req.prompt,
            "trajectory": [
                {"token": token, "depth_score": ds, "entropy": ent}
                for token, ds, ent in traj
            ],
            "n_tokens": len(traj),
        }
        return _wrap_response(data, elapsed_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _gpu_flush()


# ── Black-Box Estimation (no GPU needed) ─────────────────────────────

class EstimateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192)

class EstimateCompareRequest(BaseModel):
    text_a: str = Field(..., min_length=1, max_length=8192)
    text_b: str = Field(..., min_length=1, max_length=8192)

_blackbox_scorer = None

def _get_blackbox():
    global _blackbox_scorer
    if _blackbox_scorer is None:
        try:
            from blackbox_scorer import BlackBoxDepthScorer
            _blackbox_scorer = BlackBoxDepthScorer()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Black-box probe not available: {e}. Run: python build_calibration_data.py && python depth_probe.py"
            )
    return _blackbox_scorer


@app.post("/v2/estimate")
async def estimate(req: EstimateRequest, auth=Depends(verify_auth_and_rate)):
    """Estimate depth from text features alone. No GPU needed."""
    bb = _get_blackbox()
    t0 = time.time()
    result = bb.score(req.text)
    elapsed_ms = (time.time() - t0) * 1000
    return _wrap_response(result.to_dict(), elapsed_ms)


@app.post("/v2/estimate_compare")
async def estimate_compare(req: EstimateCompareRequest, auth=Depends(verify_auth_and_rate)):
    """Compare two texts using black-box probe. No GPU needed."""
    bb = _get_blackbox()
    t0 = time.time()
    result = bb.compare(req.text_a, req.text_b)
    elapsed_ms = (time.time() - t0) * 1000
    return _wrap_response(result, elapsed_ms)


@app.post("/v2/fathom_score")
async def fathom_score_endpoint(req: FathomScoreRequest, auth=Depends(verify_auth_and_rate)):
    """Compute the single-number Fathom Score (0-100)."""
    from fathom_score import compute_fathom_score

    scorer = get_scorer()
    _gpu_flush()
    t0 = time.time()
    try:
        depth_result = scorer.score(req.prompt, target=req.target)
        fs = compute_fathom_score(depth_result)
        elapsed_ms = (time.time() - t0) * 1000
        data = {
            "prompt": req.prompt,
            "fathom_score": fs.score,
            "grade": fs.grade,
            "interpretation": fs.interpretation,
            "components": fs.components,
            "depth_result": _depth_result_to_dict(depth_result),
        }
        return _wrap_response(data, elapsed_ms)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            _gpu_flush()
            raise HTTPException(status_code=503, detail="GPU out of memory. Try again.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _gpu_flush()


# ── Correctness predictor (loaded once) ──────────────────────────────

_correctness_predictor = None


@app.on_event("startup")
async def load_correctness_predictor():
    """Load trained CorrectnessPredictor if available."""
    global _correctness_predictor
    from fathom_correctness import CorrectnessPredictor
    from pathlib import Path as P

    pkl_path = P(__file__).resolve().parent / "correctness_predictor.pkl"
    if pkl_path.exists():
        try:
            _correctness_predictor = CorrectnessPredictor.load(str(pkl_path))
            print(f"[Fathom API] Correctness predictor loaded "
                  f"({_correctness_predictor.n_training_samples} samples)")
        except Exception as e:
            print(f"[Fathom API] Could not load correctness predictor: {e}")
            _correctness_predictor = CorrectnessPredictor()
    else:
        _correctness_predictor = CorrectnessPredictor()
        print("[Fathom API] Correctness predictor initialized (untrained)")


@app.post("/v2/predict_correctness")
async def predict_correctness(req: CorrectnessRequest, auth=Depends(verify_auth_and_rate)):
    """Predict whether an AI response is correct using depth signature.

    Concatenates prompt + response, scores with DepthScorer, runs through
    CorrectnessPredictor. Shallow = correct recall. Deep = uncertain/hallucinated.
    """
    scorer = get_scorer()
    t0 = time.time()
    try:
        combined = f"{req.prompt} {req.response}"
        depth_result = scorer.score(combined)

        if _correctness_predictor is not None and _correctness_predictor.is_trained:
            prediction = _correctness_predictor.predict(depth_result)
        else:
            # Fallback: threshold-based
            from fathom_correctness import CorrectnessPredictor as CP
            fallback = CP()
            prediction = fallback.predict(depth_result)

        elapsed_ms = (time.time() - t0) * 1000
        return _wrap_response({
            "depth_score": round(depth_result.depth_score, 4),
            "is_correct_predicted": prediction["is_correct"],
            "confidence": prediction["confidence"],
            "signal": prediction["depth_signal"],
            "caveat": "Experimental. Based on depth signature only.",
        }, elapsed_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _gpu_flush()


@app.get("/v2/correctness_status")
async def correctness_status(auth=Depends(verify_auth_and_rate)):
    """Returns whether the correctness predictor is trained, threshold, sample size."""
    if _correctness_predictor is not None:
        return _correctness_predictor.status()
    return {
        "is_trained": False,
        "threshold": 8.23,
        "n_training_samples": 0,
        "backend": "none",
        "direction": "Not loaded",
    }


# ── Phase 8 endpoints ────────────────────────────────────────────────

_adversarial_detector = None


@app.on_event("startup")
async def load_adversarial_detector():
    """Load adversarial detector if baseline exists."""
    global _adversarial_detector
    from fathom_adversarial import AdversarialDetector
    from pathlib import Path as P

    pkl_path = P(__file__).resolve().parent / "adversarial_baseline.pkl"
    if pkl_path.exists():
        try:
            _adversarial_detector = AdversarialDetector.load(str(pkl_path))
            print("[Fathom API] Adversarial detector loaded")
        except Exception as e:
            print(f"[Fathom API] Could not load adversarial detector: {e}")
            _adversarial_detector = AdversarialDetector()
    else:
        _adversarial_detector = AdversarialDetector()
        print("[Fathom API] Adversarial detector initialized (no baseline)")


@app.post("/v2/detect_adversarial")
async def detect_adversarial(req: AdversarialRequest, auth=Depends(verify_auth_and_rate)):
    """Detect whether a prompt is adversarial using depth signature anomalies.

    Compares the prompt's cognitive fingerprint against the baseline.
    Flags prompts with z-scores > 3 on any dimension.
    """
    if _adversarial_detector is None or not _adversarial_detector.has_baseline:
        raise HTTPException(
            status_code=503,
            detail="Adversarial detector has no baseline. Build one first.",
        )
    scorer = get_scorer()
    t0 = time.time()
    try:
        result = _adversarial_detector.is_adversarial(req.prompt, scorer)
        elapsed_ms = (time.time() - t0) * 1000
        return _wrap_response(result.to_dict(), elapsed_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _gpu_flush()


@app.get("/v2/adversarial_status")
async def adversarial_status(auth=Depends(verify_auth_and_rate)):
    """Returns adversarial detector status."""
    if _adversarial_detector is not None:
        return _adversarial_detector.status()
    return {"has_baseline": False}


@app.post("/v2/filter_dataset")
async def filter_dataset(req: FilterDatasetRequest, auth=Depends(verify_auth_and_rate)):
    """Score up to 100 texts and return depth scores + filter recommendations.

    Returns each text with its depth score, grade, and whether it passes
    the minimum grade threshold.
    """
    from fathom_filter import TrainingDataFilter, grade_depth

    scorer = get_scorer()
    t0 = time.time()
    try:
        filt = TrainingDataFilter()
        scored = filt.score_dataset(req.texts, scorer, batch_size=10)
        elapsed_ms = (time.time() - t0) * 1000

        results = []
        for s in scored:
            results.append({
                "text": s.text[:200],
                "depth_score": s.depth_score,
                "grade": s.grade,
                "passes_filter": s.grade in _acceptable_grades(req.min_grade),
            })

        n_pass = sum(1 for r in results if r["passes_filter"])
        return _wrap_response({
            "results": results,
            "n_total": len(results),
            "n_pass": n_pass,
            "n_fail": len(results) - n_pass,
            "min_grade": req.min_grade,
        }, elapsed_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _gpu_flush()


def _acceptable_grades(min_grade: str) -> set:
    """Return set of grades at or above min_grade."""
    order = ["A+", "A", "B+", "B", "C", "D", "F"]
    try:
        idx = order.index(min_grade)
    except ValueError:
        idx = 3
    return set(order[:idx + 1])


# ── Dashboard routes ─────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the Fathom dashboard."""
    index = DASHBOARD_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.get("/static/{filename}")
async def static_file(filename: str):
    """Serve dashboard static assets."""
    filepath = DASHBOARD_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # Basic content type detection
    ct = "text/plain"
    if filename.endswith(".js"):
        ct = "application/javascript"
    elif filename.endswith(".css"):
        ct = "text/css"
    elif filename.endswith(".svg"):
        ct = "image/svg+xml"
    return FileResponse(filepath, media_type=ct)


# ── Leaderboard ──────────────────────────────────────────────────────

LEADERBOARD_DIR = Path(__file__).resolve().parent / "leaderboard"


@app.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard_page():
    """Serve the Fathom Leaderboard page."""
    lb_html = LEADERBOARD_DIR / "index.html"
    if not lb_html.exists():
        raise HTTPException(status_code=404, detail="Leaderboard not found")
    return HTMLResponse(content=lb_html.read_text(encoding="utf-8"))


@app.get("/api/leaderboard")
async def leaderboard_data():
    """Return leaderboard scores as JSON."""
    scores_path = LEADERBOARD_DIR / "scores.json"
    if not scores_path.exists():
        return {"benchmark_version": "1.0", "models": [], "last_updated": None}
    return json.loads(scores_path.read_text(encoding="utf-8"))


# ── Waitlist ──────────────────────────────────────────────────────────

WAITLIST_PATH = Path(__file__).resolve().parent / "waitlist.json"


def _load_waitlist() -> list:
    if WAITLIST_PATH.exists():
        try:
            return json.loads(WAITLIST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_waitlist(entries: list):
    WAITLIST_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


@app.post("/waitlist")
async def join_waitlist(req: WaitlistRequest):
    """Add email to the Fathom API waitlist."""
    import re
    email = req.email.strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    entries = _load_waitlist()

    # Check for duplicate
    existing_emails = {e["email"] for e in entries}
    if email in existing_emails:
        return {"message": "You're already on the list.", "position": len(entries)}

    entries.append({
        "email": email,
        "joined": datetime.utcnow().isoformat() + "Z",
    })
    _save_waitlist(entries)

    return {"message": "You're on the list.", "position": len(entries)}


@app.get("/waitlist/count")
async def waitlist_count():
    """Return current waitlist size."""
    entries = _load_waitlist()
    return {"count": len(entries)}


# ── Waitlist page route ───────────────────────────────────────────────

WAITLIST_HTML = Path(__file__).resolve().parent / "waitlist.html"
LANDING_HTML = Path(__file__).resolve().parent / "fathom.html"


@app.get("/fathom/waitlist", response_class=HTMLResponse)
async def waitlist_page():
    """Serve the waitlist signup page."""
    if not WAITLIST_HTML.exists():
        raise HTTPException(status_code=404, detail="Waitlist page not found")
    return HTMLResponse(content=WAITLIST_HTML.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Serve the Fathom landing page."""
    if not LANDING_HTML.exists():
        raise HTTPException(status_code=404, detail="Landing page not found")
    return HTMLResponse(content=LANDING_HTML.read_text(encoding="utf-8"))


# ── Graceful shutdown ────────────────────────────────────────────────

_server_ref = None


@app.on_event("shutdown")
async def cleanup_gpu():
    """Release GPU memory on shutdown so restarts don't OOM."""
    global _scorer
    if _scorer is not None and not MOCK_MODE:
        try:
            import torch
            del _scorer
            _scorer = None
            torch.cuda.empty_cache()
            gc.collect()
            print("[Fathom API] GPU memory released on shutdown")
        except Exception:
            pass


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT from PM2 — exit cleanly so port is released."""
    print(f"[Fathom API] Received signal {signum}, shutting down...")
    sys.exit(0)


# ── Standalone run ───────────────────────────────────────────────────

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    config = uvicorn.Config(
        "depth_api_v2:app",
        host="0.0.0.0",
        port=8400,
        log_level="info",
    )
    server = uvicorn.Server(config)
    server.run()
