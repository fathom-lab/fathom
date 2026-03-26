# FATHOM API — Production Container
# ===================================
# GPU-accelerated reasoning depth scoring
#
# Build:  docker build -t fathom-api .
# Run:    docker run --gpus all -p 8400:8400 fathom-api

FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY depth_scorer.py .
COPY depth_api_v2.py .
COPY fathom_score.py .
COPY fathom_correctness.py .
COPY fathom_adversarial.py .
COPY fathom_filter.py .
COPY fathom_benchmark.py .
COPY fathom_dna.py .
COPY fathom_saturation.py .
COPY fathom_temporal.py .
COPY dashboard/ ./dashboard/

# Pre-download model weights at build time (cached in layer)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('google/gemma-2-2b'); \
    snapshot_download('mwhanna/gemma-scope-transcoders')"

# Expose API port
EXPOSE 8400

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8400/v2/health || exit 1

# Run with single worker (GPU model is not fork-safe)
CMD ["uvicorn", "depth_api_v2:app", "--host", "0.0.0.0", "--port", "8400", \
     "--workers", "1", "--timeout-keep-alive", "60"]
