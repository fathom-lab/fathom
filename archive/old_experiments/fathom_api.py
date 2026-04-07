"""
fathom_api.py — Fathom Scan API v1
===================================
POST /scan   → per-token C_delta flags on any text
GET  /health → model load status

Usage:
    uvicorn fathom_api:app --host 0.0.0.0 --port 8500
"""

import sys
import os
import asyncio
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add sae-reasoning-depth to path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fathom_api")

# ── Global model state ──────────────────────────────────────────────────────

monitor = None
model_loading = False
model_error = None

async def load_model_async():
    global monitor, model_loading, model_error
    model_loading = True
    try:
        log.info("Loading FathomMonitor (google/gemma-2-2b)...")
        from fathom_monitor import FathomMonitor
        monitor = FathomMonitor(model_name="google/gemma-2-2b")
        log.info("FathomMonitor ready.")
    except Exception as e:
        model_error = str(e)
        log.error(f"Failed to load model: {e}")
    finally:
        model_loading = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(load_model_async())
    yield

# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fathom Scan API",
    description="Per-token hallucination risk detection via SAE coherence geometry.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ─────────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    text: str
    max_new_tokens: int = 50
    threshold: float = 0.010
    high_threshold: float = 0.020

class FlagOut(BaseModel):
    token_idx: int
    token: str
    c_delta: float
    label: str  # UNCERTAIN | HALLUCINATION_RISK

class ScanResponse(BaseModel):
    prompt: str
    generated_text: str
    annotated: str
    flags: List[FlagOut]
    flag_count: int
    flag_rate: float
    max_c_delta: Optional[float]
    model: str = "google/gemma-2-2b"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_loading: bool
    error: Optional[str]

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if monitor is not None else ("loading" if model_loading else "error"),
        model_loaded=monitor is not None,
        model_loading=model_loading,
        error=model_error,
    )

@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    if monitor is None:
        if model_loading:
            raise HTTPException(status_code=503, detail="Model still loading. Try again in ~60s.")
        raise HTTPException(status_code=503, detail=f"Model failed to load: {model_error}")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    # Override thresholds if provided
    monitor.threshold = req.threshold
    monitor.high_threshold = req.high_threshold

    try:
        result = monitor.run(req.text, max_tokens=req.max_new_tokens)
    except Exception as e:
        log.error(f"Scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    flags_out = [
        FlagOut(
            token_idx=f.token_idx,
            token=f.token,
            c_delta=round(f.c_delta, 6),
            label=f.label,
        )
        for f in result.flags
    ]

    return ScanResponse(
        prompt=result.prompt,
        generated_text=result.text,
        annotated=result.flagged_text,
        flags=flags_out,
        flag_count=result.flag_count,
        flag_rate=round(result.flag_rate, 4),
        max_c_delta=round(result.max_c_delta, 6) if result.max_c_delta else None,
    )

@app.get("/")
def root():
    return {
        "name": "Fathom Scan API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {"scan": "POST /scan"},
    }
