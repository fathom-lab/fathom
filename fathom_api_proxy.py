import os, httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Fathom Scan API Proxy")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LOCAL_URL = os.environ.get("LOCAL_FATHOM_URL", "http://localhost:8500")
API_KEY   = os.environ.get("FATHOM_API_KEY", "")

class ScanRequest(BaseModel):
    text: str
    max_new_tokens: int = 50
    threshold: float = 0.010
    high_threshold: float = 0.020

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{LOCAL_URL}/health")
            return {"status": "ok", "local": r.json()}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/scan")
async def scan(req: ScanRequest, x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{LOCAL_URL}/scan", json=req.dict())
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@app.get("/")
def root():
    return {"name": "Fathom Scan API", "version": "1.0.0", "docs": "/docs"}
