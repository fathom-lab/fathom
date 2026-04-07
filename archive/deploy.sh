#!/bin/bash
# ================================================================
# FATHOM — One-Command Deploy
# ================================================================
# Usage:
#   ./deploy.sh docker     Build and run locally with GPU
#   ./deploy.sh runpod     Build, push to Docker Hub, deploy to RunPod
#   ./deploy.sh railway    Deploy via Railway CLI
#   ./deploy.sh mock       Run locally without GPU (mock mode)
# ================================================================

set -e

IMAGE_NAME="fathom-api"
DOCKER_HUB_USER="${DOCKER_HUB_USER:-}"
PORT="${FATHOM_PORT:-8400}"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[FATHOM]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Build Docker Image ──────────────────────────────────────────
build() {
    log "Building Docker image: ${IMAGE_NAME}..."
    docker build -t "${IMAGE_NAME}" -f deploy/Dockerfile .
    log "Build complete."
}

# ── Local Docker (GPU) ──────────────────────────────────────────
deploy_docker() {
    build
    log "Starting container with GPU on port ${PORT}..."
    docker run --gpus all \
        -p "${PORT}:8400" \
        -e FATHOM_MOCK="${FATHOM_MOCK:-0}" \
        --name fathom-server \
        --rm \
        "${IMAGE_NAME}"
}

# ── Local Mock (No GPU) ────────────────────────────────────────
deploy_mock() {
    build
    log "Starting container in MOCK MODE (no GPU) on port ${PORT}..."
    docker run \
        -p "${PORT}:8400" \
        -e FATHOM_MOCK=1 \
        --name fathom-mock \
        --rm \
        "${IMAGE_NAME}"
}

# ── RunPod ──────────────────────────────────────────────────────
deploy_runpod() {
    [ -z "${DOCKER_HUB_USER}" ] && err "Set DOCKER_HUB_USER env var first"

    local full_image="${DOCKER_HUB_USER}/${IMAGE_NAME}:latest"

    build
    log "Tagging and pushing to Docker Hub..."
    docker tag "${IMAGE_NAME}" "${full_image}"
    docker push "${full_image}"

    log "Image pushed: ${full_image}"
    echo ""
    echo -e "${CYAN}To deploy on RunPod:${NC}"
    echo "  1. Go to https://www.runpod.io/console/pods"
    echo "  2. Create new Pod with GPU (RTX 4090 or A100 recommended)"
    echo "  3. Set Docker image: ${full_image}"
    echo "  4. Set exposed port: 8400"
    echo "  5. Pod will be ready when health check passes (~2 min)"
    echo ""
    echo "  Your API will be at: https://<pod-id>-8400.proxy.runpod.net"
    echo ""

    # If runpodctl is installed, offer to create directly
    if command -v runpodctl &> /dev/null; then
        log "runpodctl detected. Creating pod..."
        runpodctl create pod \
            --name "fathom-api" \
            --imageName "${full_image}" \
            --gpuType "NVIDIA RTX 4090" \
            --gpuCount 1 \
            --containerDiskSize 40 \
            --ports "8400/http"
        log "Pod created. Check RunPod console for status."
    fi
}

# ── Railway ─────────────────────────────────────────────────────
deploy_railway() {
    if ! command -v railway &> /dev/null; then
        err "Railway CLI not installed. Run: npm install -g @railway/cli"
    fi

    log "Deploying to Railway..."
    railway up --detach
    log "Deployment started. Check Railway dashboard for URL."
    echo ""
    echo -e "${CYAN}After deployment:${NC}"
    echo "  railway open     — Open dashboard"
    echo "  railway logs      — View logs"
    echo "  railway domain    — Get public URL"
}

# ── Health Check ────────────────────────────────────────────────
healthcheck() {
    local url="${1:-http://localhost:${PORT}}"
    log "Checking health at ${url}/v2/health..."

    for i in $(seq 1 30); do
        if curl -sf "${url}/v2/health" > /dev/null 2>&1; then
            log "API is healthy!"
            curl -s "${url}/v2/health" | python -m json.tool 2>/dev/null || true
            return 0
        fi
        echo -n "."
        sleep 5
    done

    err "Health check failed after 150s"
}

# ── Main ────────────────────────────────────────────────────────
case "${1:-help}" in
    docker)   deploy_docker ;;
    mock)     deploy_mock ;;
    runpod)   deploy_runpod ;;
    railway)  deploy_railway ;;
    build)    build ;;
    health)   healthcheck "${2}" ;;
    *)
        echo "FATHOM Deploy"
        echo ""
        echo "Usage: ./deploy.sh <command>"
        echo ""
        echo "Commands:"
        echo "  docker    Build and run locally with GPU"
        echo "  mock      Build and run without GPU (mock mode)"
        echo "  runpod    Push to Docker Hub + deploy to RunPod"
        echo "  railway   Deploy via Railway CLI"
        echo "  build     Build Docker image only"
        echo "  health    Check API health (optional: URL arg)"
        echo ""
        echo "Environment:"
        echo "  DOCKER_HUB_USER  Docker Hub username (required for runpod)"
        echo "  FATHOM_PORT      API port (default: 8400)"
        echo "  FATHOM_MOCK      Set to 1 for mock mode (no GPU)"
        ;;
esac
