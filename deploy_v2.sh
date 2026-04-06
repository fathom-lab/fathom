#!/bin/bash
# ================================================================
# FATHOM SCAN API v2 — Production Deployment
# ================================================================
# Usage: bash deploy_v2.sh
#
# Deploys:
#   - Fathom API v2 (S_early + entropy + cognitive zones)
#   - Interactive demo page
#   - Cloudflare tunnel for public access
#
# Endpoints after deploy:
#   Local:  http://localhost:8500
#   Demo:   http://localhost:8500/demo
#   Docs:   http://localhost:8500/docs
#   Public: https://fathom-gpu.darkflobi.com/demo
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ███████╗ █████╗ ████████╗██╗  ██╗ ██████╗ ███╗   ███╗"
echo "  ██╔════╝██╔══██╗╚══██╔══╝██║  ██║██╔═══██╗████╗ ████║"
echo "  █████╗  ███████║   ██║   ███████║██║   ██║██╔████╔██║"
echo "  ██╔══╝  ██╔══██║   ██║   ██╔══██║██║   ██║██║╚██╔╝██║"
echo "  ██║     ██║  ██║   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║"
echo "  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝"
echo ""
echo "  SCAN API v2 — cognitive state measurement"
echo "  patent pending: US 64/020,489 & 64/021,113"
echo ""

# ── Step 1: Pre-flight checks ──────────────────────────────────
echo "[1/5] pre-flight checks..."

# GPU
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "99999")
if [ "$GPU_MEM" -gt 2000 ]; then
    echo "  !! GPU busy (${GPU_MEM}MB used). API needs ~6GB VRAM."
    echo "  !! Kill other GPU processes first, or wait for them to finish."
    exit 1
fi
echo "  gpu: free (${GPU_MEM}MB used)"

# Python
python --version 2>&1 | head -1
echo "  python: ok"

# Syntax
python -c "import ast; ast.parse(open('fathom_api_v2.py').read())" 2>/dev/null
python -c "import ast; ast.parse(open('coherence_steerer_ext.py').read())" 2>/dev/null
echo "  syntax: ok"

# Dependencies
python -c "import fastapi, uvicorn, torch, transformers" 2>/dev/null
echo "  deps: ok"

echo ""

# ── Step 2: Setup ──────────────────────────────────────────────
echo "[2/5] setup..."
mkdir -p logs
echo "  logs dir: ok"

# ── Step 3: Start API ─────────────────────────────────────────
echo ""
echo "[3/5] starting fathom api..."
pm2 delete fathom-api 2>/dev/null || true
pm2 start ecosystem.config.js
echo "  api process started"
echo "  model loading (takes ~30s on first request)..."

# ── Step 4: Start tunnel ──────────────────────────────────────
echo ""
echo "[4/5] starting cloudflare tunnel..."
pm2 delete fathom-tunnel 2>/dev/null || true
pm2 start "$(which cloudflared)" --name fathom-tunnel -- tunnel run fathom-gpu
echo "  tunnel started"

# ── Step 5: Verify ────────────────────────────────────────────
echo ""
echo "[5/5] verifying..."
sleep 5

# Health check (model might still be loading)
HEALTH=$(curl -s http://localhost:8500/health 2>/dev/null || echo "unreachable")
echo "  health: $HEALTH"

pm2 list

echo ""
echo "=========================================="
echo "  DEPLOYED"
echo "=========================================="
echo ""
echo "  local api:   http://localhost:8500"
echo "  local demo:  http://localhost:8500/demo"
echo "  swagger:     http://localhost:8500/docs"
echo ""
echo "  public api:  https://fathom-gpu.darkflobi.com"
echo "  public demo: https://fathom-gpu.darkflobi.com/demo"
echo ""
echo "  logs:  pm2 logs fathom-api"
echo "  stop:  pm2 delete fathom-api fathom-tunnel"
echo ""
