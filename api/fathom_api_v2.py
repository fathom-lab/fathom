"""
fathom_api_v2.py — Fathom Scan API v2 (Commitment Intensity + Cognitive State)
================================================================================
POST /scan        → generate + full cognitive state per token
POST /score       → score existing text (no generation)
GET  /health      → model status
GET  /             → info

New in v2:
  - S_early (commitment intensity, IPR-derived)
  - logit entropy per token
  - 2D cognitive zone classification (SAFE/UNCERTAIN/RISKY/DANGER)
  - streaming-ready: returns risk at each token

Usage:
    uvicorn fathom_api_v2:app --host 0.0.0.0 --port 8500

Patent pending: US 64/020,489, 64/021,113, 64/026,964
"""

import sys
import os
import asyncio
import logging
import time
from typing import List, Optional
from contextlib import asynccontextmanager

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'probes'))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
import numpy as np
import traceback

# Simple in-memory rate limiter
_request_times = {}
RATE_LIMIT_PER_MINUTE = 10

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fathom_api_v2")

# ── Global state ────────────────────────────────────────────────────────────

steerer = None
model_name_loaded = None
model_loading = False
model_error = None

DEFAULT_MODEL = os.environ.get("FATHOM_MODEL", "google/gemma-2-2b-it")
S_THRESHOLD = 0.005
EARLY_WINDOW = 7  # tokens [0, EARLY_WINDOW) for S_early


async def load_model_async():
    global steerer, model_loading, model_error, model_name_loaded
    model_loading = True
    try:
        log.info(f"Loading CoherenceSteererExt ({DEFAULT_MODEL})...")
        from coherence_steerer_ext import CoherenceSteererExt
        steerer = CoherenceSteererExt(model_name=DEFAULT_MODEL)
        model_name_loaded = DEFAULT_MODEL
        log.info("Model ready.")
    except Exception as e:
        model_error = str(e)
        log.error(f"Failed: {e}")
    finally:
        model_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(load_model_async())
    yield


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fathom Scan API",
    description=(
        "Cognitive state measurement for LLM generations. "
        "Measures commitment intensity (S), logit entropy, and classifies "
        "each generation into cognitive zones (SAFE/UNCERTAIN/RISKY/DANGER). "
        "Patent pending."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Simple per-IP rate limiting."""
    if request.url.path in ("/scan", "/score"):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        times = _request_times.get(ip, [])
        times = [t for t in times if now - t < 60]  # last minute
        if len(times) >= RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit: {RATE_LIMIT_PER_MINUTE} scans/minute"}
            )
        times.append(now)
        _request_times[ip] = times
    return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs."}
    )


# ── Schemas ─────────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=40, ge=1, le=100)
    s_threshold: float = Field(default=S_THRESHOLD, ge=0.001, le=0.1)

class ScoreRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    text: str = Field(..., min_length=1, max_length=4000)
    s_threshold: float = Field(default=S_THRESHOLD, ge=0.001, le=0.1)

class TokenState(BaseModel):
    idx: int
    token: str
    c_delta: float
    entropy: Optional[float] = None
    logprob: Optional[float] = None

class CognitiveZone(BaseModel):
    zone: str          # SAFE | UNCERTAIN | RISKY | DANGER
    s_early: float
    entropy_early: float
    halluc_probability: float  # estimated from zone
    description: str

class ScanResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens: List[TokenState]
    total_tokens: int
    # Cognitive state
    s_early: float
    entropy_early: float
    zone: CognitiveZone
    # Legacy compatibility
    max_c_delta: float
    flag_rate: float
    # Meta
    model: str
    generation_time_s: float
    api_version: str = "2.0.0"

class HealthResponse(BaseModel):
    status: str
    model: Optional[str]
    model_loaded: bool
    model_loading: bool
    error: Optional[str]
    api_version: str = "2.0.0"


# ── Zone classification ────────────────────────────────────────────────────

# Calibrated from TruthfulQA n=94 experiment (Gemma-2-2B-IT)
# Median thresholds: S_early=0.00192, entropy_early=1.033
S_MEDIAN = 0.00192
E_MEDIAN = 1.033

ZONE_MAP = {
    (True, True): ("DANGER", 0.65, "committed + uncertain = confident wrongness"),
    (True, False): ("RISKY", 0.58, "committed + confident"),
    (False, True): ("UNCERTAIN", 0.38, "uncommitted + uncertain = exploring"),
    (False, False): ("SAFE", 0.22, "uncommitted + confident = grounded retrieval"),
}


def classify_zone(s_early: float, entropy_early: float) -> CognitiveZone:
    s_high = s_early > S_MEDIAN
    e_high = entropy_early > E_MEDIAN
    zone_name, halluc_prob, desc = ZONE_MAP[(s_high, e_high)]
    return CognitiveZone(
        zone=zone_name,
        s_early=round(s_early, 6),
        entropy_early=round(entropy_early, 4),
        halluc_probability=halluc_prob,
        description=desc,
    )


# ── Compute S ───────────────────────────────────────────────────────────────

def compute_S(traj, start=0, end=EARLY_WINDOW, threshold=S_THRESHOLD):
    sl = traj[start:end]
    if not sl:
        return 0.0
    spikes = [c for c in sl if c > threshold]
    if not spikes:
        return 0.0
    return max(sl) / len(spikes)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if steerer else ("loading" if model_loading else "error"),
        model=model_name_loaded,
        model_loaded=steerer is not None,
        model_loading=model_loading,
        error=model_error,
    )


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    if steerer is None:
        if model_loading:
            raise HTTPException(503, "Model loading (~60s). Retry.")
        raise HTTPException(503, f"Model failed: {model_error}")

    if not req.prompt or not req.prompt.strip():
        raise HTTPException(400, "prompt must not be empty")

    t_start = time.time()
    last_err = None
    for attempt in range(2):
        try:
            result = steerer.generate_with_entropy(
                req.prompt, max_tokens=req.max_tokens
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            log.warning(f"Generation attempt {attempt+1} failed: {e}")
            try:
                steerer._inner.reset_hooks()
            except Exception:
                pass
    if last_err is not None:
        log.error(f"Generation failed after retries: {last_err}")
        # Schedule model reload — the model is likely in a corrupted state
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(load_model_async())
        except RuntimeError:
            pass  # no event loop in sync context — model stays as-is
        raise HTTPException(500, f"Generation failed — try again in a moment")

    gen_time = time.time() - t_start

    # Build per-token states
    token_states = []
    n = result.total_tokens
    for i in range(n):
        ts = TokenState(
            idx=i,
            token=result.tokens[i] if i < len(result.tokens) else "",
            c_delta=round(result.c_delta_trajectory[i], 6) if i < len(result.c_delta_trajectory) else 0.0,
            entropy=round(result.entropy_trajectory[i], 4) if i < len(result.entropy_trajectory) else None,
            logprob=round(result.logprob_trajectory[i], 4) if i < len(result.logprob_trajectory) else None,
        )
        token_states.append(ts)

    # Compute S_early
    cd_traj = [result.c_delta_trajectory[i] if i < len(result.c_delta_trajectory) else 0.0
               for i in range(min(n, EARLY_WINDOW))]
    s_early = compute_S(cd_traj, 0, EARLY_WINDOW, req.s_threshold)

    # Early entropy
    ent_traj = result.entropy_trajectory[:min(n, EARLY_WINDOW)]
    entropy_early = float(np.mean(ent_traj)) if ent_traj else 0.0

    # Zone
    zone = classify_zone(s_early, entropy_early)

    # Max C_delta for legacy compatibility
    max_cd = max(result.c_delta_trajectory) if result.c_delta_trajectory else 0.0

    # Flag rate (tokens above threshold)
    flagged = sum(1 for c in result.c_delta_trajectory if c > req.s_threshold)
    flag_rate = flagged / max(1, n)

    return ScanResponse(
        prompt=req.prompt,
        generated_text=result.text,
        tokens=token_states,
        total_tokens=n,
        s_early=round(s_early, 6),
        entropy_early=round(entropy_early, 4),
        zone=zone,
        max_c_delta=round(max_cd, 6),
        flag_rate=round(flag_rate, 4),
        model=model_name_loaded or DEFAULT_MODEL,
        generation_time_s=round(gen_time, 2),
    )


@app.post("/score", response_model=ScanResponse)
def score(req: ScoreRequest):
    """Score existing text without generation — measures cognitive state of provided text."""
    if steerer is None:
        if model_loading:
            raise HTTPException(503, "Model loading (~60s). Retry.")
        raise HTTPException(503, f"Model failed: {model_error}")

    if not req.prompt or not req.prompt.strip():
        raise HTTPException(400, "prompt must not be empty")
    if not req.text or not req.text.strip():
        raise HTTPException(400, "text must not be empty")

    t_start = time.time()
    try:
        # Tokenize the provided text and run forward passes to get C_delta + entropy
        inner = steerer._inner
        full_text = req.prompt + req.text
        tokens = inner.to_tokens(full_text)
        prompt_tokens = inner.to_tokens(req.prompt)
        prompt_len = prompt_tokens.shape[1]

        # Run token-by-token scoring on the text portion
        import torch
        import torch.nn.functional as F

        c_delta_trajectory = []
        entropy_trajectory = []
        logprob_trajectory = []
        text_tokens = []

        for pos in range(prompt_len, tokens.shape[1]):
            context = tokens[:, :pos + 1]
            hidden_states = {}
            capture_hooks = steerer._build_capture_hooks(hidden_states)

            with torch.no_grad():
                try:
                    inner.reset_hooks()
                    logits = inner.run_with_hooks(context, fwd_hooks=capture_hooks)
                except (RuntimeError, IndexError, KeyError):
                    inner.reset_hooks()
                    logits = inner(context)
                    hidden_states = {}

            try:
                c_delta, _ = steerer._fast_c_delta(hidden_states)
            except (KeyError, AttributeError):
                c_delta = None
            c_delta_trajectory.append(c_delta if c_delta is not None else 0.0)

            logit_vec = logits[0, -1].float()
            logp = F.log_softmax(logit_vec, dim=-1)
            p = logp.exp()
            entropy = float(-(p * logp).sum().item())
            entropy_trajectory.append(entropy)

            token_id = int(tokens[0, pos].item())
            logprob_trajectory.append(float(logp[token_id].item()))
            text_tokens.append(inner.tokenizer.decode([token_id]))

    except Exception as e:
        log.error(f"Score error: {e}")
        raise HTTPException(500, str(e))

    gen_time = time.time() - t_start
    n = len(text_tokens)

    token_states = []
    for i in range(n):
        ts = TokenState(
            idx=i,
            token=text_tokens[i],
            c_delta=round(c_delta_trajectory[i], 6),
            entropy=round(entropy_trajectory[i], 4),
            logprob=round(logprob_trajectory[i], 4),
        )
        token_states.append(ts)

    cd_traj = c_delta_trajectory[:min(n, EARLY_WINDOW)]
    s_early = compute_S(cd_traj, 0, EARLY_WINDOW, req.s_threshold)
    ent_traj = entropy_trajectory[:min(n, EARLY_WINDOW)]
    entropy_early = float(np.mean(ent_traj)) if ent_traj else 0.0
    zone = classify_zone(s_early, entropy_early)
    max_cd = max(c_delta_trajectory) if c_delta_trajectory else 0.0
    flagged = sum(1 for c in c_delta_trajectory if c > req.s_threshold)
    flag_rate = flagged / max(1, n)

    return ScanResponse(
        prompt=req.prompt,
        generated_text=req.text,
        tokens=token_states,
        total_tokens=n,
        s_early=round(s_early, 6),
        entropy_early=round(entropy_early, 4),
        zone=zone,
        max_c_delta=round(max_cd, 6),
        flag_rate=round(flag_rate, 4),
        model=model_name_loaded or DEFAULT_MODEL,
        generation_time_s=round(gen_time, 2),
    )


# ── Autopilot endpoint ─────────────────────────────────────────────────────

_autopilot = None

class AutopilotRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=40, ge=1, le=100)
    nudge_strength: float = Field(default=0.3, ge=0.0, le=2.0)
    steer_strength: float = Field(default=0.8, ge=0.0, le=2.0)
    redirect_weight: float = Field(default=0.2, ge=0.0, le=1.0)

class InterventionRecord(BaseModel):
    token_idx: int
    escalation_level: int
    lock_in_layers: List[int]
    correction_magnitude: float
    pre_gini: float
    post_gini: Optional[float]
    pre_s_early: float
    token_changed: bool
    old_token: str
    new_token: str

class AutopilotResponse(BaseModel):
    prompt: str
    generated_text: str
    total_tokens: int
    # Trajectories
    gini_trajectory: List[float]
    s_early_trajectory: List[float]
    escalation_history: List[int]
    # Interventions
    interventions: List[InterventionRecord]
    intervention_count: int
    max_escalation: int
    # Health
    health_score: float
    needs_regen: bool
    # Meta
    model: str
    generation_time_s: float
    overhead_pct: float
    api_version: str = "2.0.0"


@app.post("/autopilot", response_model=AutopilotResponse)
def autopilot(req: AutopilotRequest):
    """
    Generate with trajectory-aware cognitive course correction.

    Monitors SAE feature coherence trajectories per-token and applies
    dynamic, generation-specific corrections when attractor lock-in
    is detected at tokens 3-7 (the S_early window).

    Escalation ladder:
      Level 0: monitor only
      Level 1: gentle nudge (alpha=nudge_strength)
      Level 2: strong multi-layer steer (alpha=steer_strength)
      Level 3: flag for regeneration
    """
    global _autopilot

    if steerer is None:
        if model_loading:
            raise HTTPException(503, "Model loading (~60s). Retry.")
        raise HTTPException(503, f"Model failed: {model_error}")

    # Lazy-load autopilot (uses same GPU model but different engine)
    if _autopilot is None:
        try:
            from fathom_autopilot import FathomAutopilot
            _autopilot = FathomAutopilot(
                model_name=model_name_loaded or DEFAULT_MODEL,
                nudge_strength=req.nudge_strength,
                steer_strength=req.steer_strength,
                redirect_weight=req.redirect_weight,
            )
        except Exception as e:
            log.error(f"Autopilot init failed: {e}")
            raise HTTPException(500, f"Autopilot init failed: {e}")

    t_start = time.time()
    try:
        result = _autopilot.generate_autopilot(
            req.prompt, max_tokens=req.max_tokens, verbose=False
        )
    except Exception as e:
        log.error(f"Autopilot generation failed: {e}")
        try:
            _autopilot._inner.reset_hooks()
        except Exception:
            pass
        raise HTTPException(500, f"Generation failed: {e}")

    gen_time = time.time() - t_start

    # Convert interventions to API format
    intv_records = []
    for intv in result.autopilot_interventions:
        intv_records.append(InterventionRecord(
            token_idx=intv.token_idx,
            escalation_level=intv.escalation_level,
            lock_in_layers=intv.lock_in_layers,
            correction_magnitude=round(intv.correction_magnitude, 6),
            pre_gini=round(intv.pre_gini, 4),
            post_gini=round(intv.post_gini, 4) if intv.post_gini is not None else None,
            pre_s_early=round(intv.pre_s_early, 6),
            token_changed=intv.token_changed,
            old_token=intv.old_token,
            new_token=intv.new_token,
        ))

    return AutopilotResponse(
        prompt=req.prompt,
        generated_text=result.text,
        total_tokens=result.total_tokens,
        gini_trajectory=[round(g, 4) for g in result.gini_trajectory],
        s_early_trajectory=[round(s, 6) for s in result.s_early_trajectory],
        escalation_history=result.escalation_history,
        interventions=intv_records,
        intervention_count=len(intv_records),
        max_escalation=result.max_escalation,
        health_score=round(result.health_score, 3),
        needs_regen=result.needs_regen,
        model=model_name_loaded or DEFAULT_MODEL,
        generation_time_s=round(gen_time, 2),
        overhead_pct=round(result.overhead_pct, 1),
    )


# ── Guardian endpoint (cognitive health check) ───────────────────────────

_guardian = None

class GuardianRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=40, ge=1, le=100)
    gini_threshold: float = Field(default=0.45, ge=0.3, le=0.7)
    s_early_threshold: float = Field(default=0.005, ge=0.001, le=0.05)

class GuardianResponse(BaseModel):
    prompt: str
    generated_text: str
    total_tokens: int
    # Core cognitive state (first-pass, no adaptive loop)
    cognitive_health: float          # composite health score
    mean_gini: float                 # spectrum concentration
    max_gini: float
    mean_spread: float
    mean_early_mass: float
    s_early: float                   # commitment intensity (IPR)
    # Decision
    triggered: bool                  # would guardian flag this prompt?
    trigger_reason: Optional[str]    # which signal fired
    confidence: str                  # HIGH | MODERATE | LOW | UNCERTAIN
    warning: Optional[str]           # human-readable cognitive warning
    # Trajectories (for debugging / visualization)
    gini_trajectory: List[float]
    s_early_value: float
    # Meta
    model: str
    generation_time_s: float
    api_version: str = "2.0.0"


def _classify_confidence(mean_gini: float, s_early: float, health: float,
                          gini_threshold: float, s_threshold: float) -> tuple:
    """
    Classify cognitive confidence based on validated thresholds.

    Returns: (confidence_level, trigger_reason, warning)
    """
    gini_fire = mean_gini > gini_threshold
    s_fire = s_early > s_threshold

    if gini_fire and s_fire:
        return (
            "LOW",
            "both_gini_and_s_early",
            "Model shows genuine uncertainty on this topic (elevated concentration AND commitment intensity). Verify externally."
        )
    if gini_fire:
        return (
            "UNCERTAIN",
            "high_gini",
            "Cognitive signature suggests obscure-retrieval confabulation. Model is searching but struggling. Verify externally."
        )
    if s_fire:
        return (
            "MODERATE",
            "high_s_early",
            "Early-token commitment intensity is elevated. Model locked in fast — may be template-filling."
        )
    # Healthy
    if health > 1.2:
        return ("HIGH", None, None)
    return ("MODERATE", None, None)


@app.post("/guardian", response_model=GuardianResponse)
def guardian(req: GuardianRequest):
    """
    Cognitive health check for a prompt.

    Runs a single-pass greedy generation with full SAE trajectory monitoring
    and returns a cognitive health assessment. Uses validated thresholds to
    flag prompts where the model shows signs of obscure-retrieval confabulation.

    Honest limits:
      - 100% precision on obscure-retrieval confabulation (model searching + failing)
      - Does NOT catch template-fill hallucination (confident plausible bullshit)
      - This is cognitive state measurement, not fact checking

    Designed for integration into agent pipelines as a verification layer.
    """
    global _guardian

    if steerer is None:
        if model_loading:
            raise HTTPException(503, "Model loading (~60s). Retry.")
        raise HTTPException(503, f"Model failed: {model_error}")

    # Lazy-load guardian (uses same GPU model)
    if _guardian is None:
        try:
            from fathom_guardian import FathomGuardian
            _guardian = FathomGuardian(
                model_name=model_name_loaded or DEFAULT_MODEL,
                max_candidates=1,  # we only want the first-pass assessment
                gini_threshold=req.gini_threshold,
                s_early_threshold=req.s_early_threshold,
            )
        except Exception as e:
            log.error(f"Guardian init failed: {e}")
            raise HTTPException(500, f"Guardian init failed: {e}")

    t_start = time.time()
    try:
        # Generate a single greedy candidate with full trajectory capture
        candidate = _guardian._generate_candidate(
            req.prompt, max_tokens=req.max_tokens, temperature=0.0
        )
    except Exception as e:
        log.error(f"Guardian generation failed: {e}")
        try:
            _guardian._inner.reset_hooks()
        except Exception:
            pass
        raise HTTPException(500, f"Generation failed: {e}")

    gen_time = time.time() - t_start

    # Classify
    confidence, reason, warning = _classify_confidence(
        mean_gini=candidate.mean_gini,
        s_early=candidate.s_early,
        health=candidate.health_score,
        gini_threshold=req.gini_threshold,
        s_threshold=req.s_early_threshold,
    )

    triggered = confidence in ("LOW", "UNCERTAIN")

    return GuardianResponse(
        prompt=req.prompt,
        generated_text=candidate.text,
        total_tokens=candidate.total_tokens,
        cognitive_health=round(candidate.health_score, 3),
        mean_gini=round(candidate.mean_gini, 4),
        max_gini=round(candidate.max_gini, 4),
        mean_spread=round(candidate.mean_spread, 4),
        mean_early_mass=round(candidate.mean_early_mass, 4),
        s_early=round(candidate.s_early, 6),
        triggered=triggered,
        trigger_reason=reason,
        confidence=confidence,
        warning=warning,
        gini_trajectory=[round(g, 4) for g in candidate.gini_trajectory],
        s_early_value=round(candidate.s_early, 6),
        model=model_name_loaded or DEFAULT_MODEL,
        generation_time_s=round(gen_time, 2),
    )


# ── Static routes ──────────────────────────────────────────────────────────

@app.get("/demo")
def demo():
    """Serve the interactive demo page."""
    from fastapi.responses import FileResponse
    demo_path = os.path.join(os.path.dirname(__file__), "fathom_demo.html")
    if os.path.exists(demo_path):
        return FileResponse(demo_path, media_type="text/html")
    raise HTTPException(404, "Demo page not found")


@app.get("/")
def root():
    """Redirect to demo page."""
    return RedirectResponse(url="/demo")
