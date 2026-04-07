"""
FATHOM CLOUD — Deployment Blueprint Generator
===============================================

Generates all configuration files needed to deploy Fathom as a
production cloud API service on any GPU-enabled platform
(Railway, Render, RunPod, AWS, GCP).

Usage: python fathom_cloud.py
Output: deploy/ directory with all config files

Nothing leaves the machine. This generates configs locally.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEPLOY_DIR = PROJECT_ROOT / "deploy"


def get_requirements() -> str:
    """Generate pinned requirements from current environment."""
    # Core dependencies only (not dev tools)
    CORE_PACKAGES = [
        "torch", "transformers", "transformer-lens", "einops",
        "circuit-tracer", "numpy", "scipy", "scikit-learn",
        "fastapi", "uvicorn", "pydantic", "safetensors",
        "huggingface_hub", "tokenizers", "sentencepiece",
        "requests", "PyYAML", "jaxtyping", "nnsight",
        "matplotlib", "seaborn", "pandas",
    ]

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, timeout=30,
        )
        installed = {}
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                pkg, ver = line.split("==", 1)
                installed[pkg.lower()] = ver

        lines = []
        for pkg in CORE_PACKAGES:
            key = pkg.lower()
            if key in installed:
                lines.append(f"{pkg}=={installed[key]}")
            else:
                lines.append(pkg)
        return "\n".join(sorted(lines, key=str.lower))
    except Exception:
        return "\n".join(CORE_PACKAGES)


def generate_dockerfile() -> str:
    return '''# FATHOM API — Production Container
# ===================================
# GPU-accelerated reasoning depth scoring
#
# Build:  docker build -t fathom-api .
# Run:    docker run --gpus all -p 8400:8400 fathom-api

FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.12 python3.12-venv python3-pip git curl \\
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
RUN python -c "from huggingface_hub import snapshot_download; \\
    snapshot_download('google/gemma-2-2b'); \\
    snapshot_download('mwhanna/gemma-scope-transcoders')"

# Expose API port
EXPOSE 8400

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \\
    CMD curl -f http://localhost:8400/v2/health || exit 1

# Run with single worker (GPU model is not fork-safe)
CMD ["uvicorn", "depth_api_v2:app", "--host", "0.0.0.0", "--port", "8400", \\
     "--workers", "1", "--timeout-keep-alive", "60"]
'''


def generate_docker_compose() -> str:
    return '''# FATHOM — Docker Compose Stack
# ==============================
# API + Redis (rate limiting) + Nginx (reverse proxy)
#
# Usage: docker compose up -d

services:
  fathom-api:
    build: .
    container_name: fathom-api
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - FATHOM_API_KEY=${FATHOM_API_KEY:-}
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
      - SAFETENSORS_FAST_GPU=1
    ports:
      - "8400:8400"
    volumes:
      - model-cache:/root/.cache/huggingface
    depends_on:
      - redis
    networks:
      - fathom-net

  redis:
    image: redis:7-alpine
    container_name: fathom-redis
    restart: unless-stopped
    command: redis-server --maxmemory 128mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    networks:
      - fathom-net

  nginx:
    image: nginx:alpine
    container_name: fathom-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - fathom-api
    networks:
      - fathom-net

volumes:
  model-cache:

networks:
  fathom-net:
    driver: bridge
'''


def generate_nginx_conf() -> str:
    return '''# Fathom API — Nginx Reverse Proxy
upstream fathom {
    server fathom-api:8400;
}

server {
    listen 80;
    server_name _;

    # Rate limiting zone
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    # Max request body (for batch endpoints)
    client_max_body_size 1m;

    location /v2/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://fathom;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }

    location /dashboard {
        proxy_pass http://fathom;
        proxy_set_header Host $host;
    }

    location /static/ {
        proxy_pass http://fathom;
        proxy_set_header Host $host;
    }

    location /health {
        proxy_pass http://fathom/v2/health;
        access_log off;
    }
}
'''


def generate_cloud_config() -> str:
    return '''# FATHOM CLOUD — Deployment Configuration
# =========================================
# Platform-agnostic deployment spec for GPU cloud providers

service:
  name: fathom-api
  version: "2.0"
  port: 8400
  protocol: http

# GPU Requirements
compute:
  gpu:
    minimum: "T4"            # NVIDIA T4 (16GB VRAM) — minimum viable
    recommended: "A10G"      # NVIDIA A10G (24GB VRAM) — production
    vram_minimum_gb: 8       # gemma-2-2b in bf16 + transcoders
    vram_recommended_gb: 16  # headroom for batch scoring
  cpu:
    minimum_cores: 4
    recommended_cores: 8
  memory:
    minimum_gb: 16
    recommended_gb: 32
  storage:
    minimum_gb: 20           # model weights + transcoders
    recommended_gb: 50

# Scaling
scaling:
  min_instances: 1
  max_instances: 10
  target_gpu_utilization: 0.7
  scale_up_threshold: 0.8    # GPU util > 80% for 2 min
  scale_down_threshold: 0.3  # GPU util < 30% for 10 min
  cooldown_seconds: 300

# Performance
performance:
  avg_score_time_s: 2.0      # single prompt scoring
  avg_compare_time_s: 4.0    # surface vs insight comparison
  avg_batch_time_s: 40.0     # 20-prompt batch
  startup_time_s: 120        # model loading cold start
  requests_per_hour_per_gpu: 1800  # at ~2s per score

# Cost Analysis (T4 pricing as of March 2026)
cost:
  t4_hourly_rate: 0.50       # $/hr (RunPod/Lambda)
  a10g_hourly_rate: 1.10     # $/hr
  cost_per_score_t4: 0.00028 # $0.50/hr / 1800 req/hr
  cost_per_1000_scores_t4: 0.28
  cost_per_score_a10g: 0.00061
  cost_per_1000_scores_a10g: 0.61
  monthly_fixed_cost:
    t4: 360.00               # $0.50 * 720 hrs
    a10g: 792.00             # $1.10 * 720 hrs
  break_even:
    developer_tier: 104      # customers at $29/mo to cover T4
    professional_tier: 2     # customers at $199/mo to cover T4

# Health & Monitoring
health:
  endpoint: /v2/health
  interval_seconds: 30
  timeout_seconds: 10
  start_period_seconds: 120
  unhealthy_threshold: 3

# Environment Variables
environment:
  required:
    - FATHOM_API_KEY          # API authentication key
  optional:
    - PYTORCH_CUDA_ALLOC_CONF # default: max_split_size_mb:512
    - SAFETENSORS_FAST_GPU    # default: 1
    - FATHOM_MOCK             # set to 1 for testing without GPU
    - HF_HOME                 # HuggingFace cache directory

# Platform-Specific Notes
platforms:
  runpod:
    template: "RunPod Serverless GPU"
    gpu_type: "NVIDIA T4"
    docker_image: "fathom-api:latest"
    idle_timeout: 300
  railway:
    plan: "Pro (GPU)"
    region: "us-west"
    dockerfile: "Dockerfile"
  render:
    plan: "GPU Instance"
    type: "web"
    docker_command: "uvicorn depth_api_v2:app --host 0.0.0.0 --port 8400"
'''


def main():
    """Generate all deployment files."""
    DEPLOY_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print("FATHOM CLOUD — Generating Deployment Blueprint")
    print(f"Output: {DEPLOY_DIR}")
    print(f"Time: {timestamp}")
    print("=" * 60)

    # 1. requirements.txt
    req_content = get_requirements()
    req_path = DEPLOY_DIR / "requirements.txt"
    req_path.write_text(req_content, encoding="utf-8")
    print(f"[OK] {req_path.name} ({len(req_content.splitlines())} packages)")

    # 2. Dockerfile
    dockerfile_path = DEPLOY_DIR / "Dockerfile"
    dockerfile_path.write_text(generate_dockerfile(), encoding="utf-8")
    print(f"[OK] {dockerfile_path.name}")

    # 3. docker-compose.yml
    compose_path = DEPLOY_DIR / "docker-compose.yml"
    compose_path.write_text(generate_docker_compose(), encoding="utf-8")
    print(f"[OK] {compose_path.name}")

    # 4. nginx.conf
    nginx_path = DEPLOY_DIR / "nginx.conf"
    nginx_path.write_text(generate_nginx_conf(), encoding="utf-8")
    print(f"[OK] {nginx_path.name}")

    # 5. fathom_cloud_config.yaml
    config_path = DEPLOY_DIR / "fathom_cloud_config.yaml"
    config_path.write_text(generate_cloud_config(), encoding="utf-8")
    print(f"[OK] {config_path.name}")

    # 6. .env.example
    env_example = """# Fathom API — Environment Variables
# Copy to .env and fill in values

# API authentication (leave empty to disable auth)
FATHOM_API_KEY=

# GPU optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
SAFETENSORS_FAST_GPU=1

# Set to 1 for testing without GPU
FATHOM_MOCK=0
"""
    env_path = DEPLOY_DIR / ".env.example"
    env_path.write_text(env_example, encoding="utf-8")
    print(f"[OK] {env_path.name}")

    # 7. .dockerignore
    dockerignore = """__pycache__/
*.pyc
*.pyo
.git/
.claude/
*.md
*.tex
*.pdf
*.pkl
*.json
!fathom_benchmark_v1.json
deploy/
figures/
results_*/
EXPERIMENT_*.md
DISCOVERY_*.md
FATHOM_CONSTANT*.md
*.zip
"""
    ignore_path = DEPLOY_DIR / ".dockerignore"
    ignore_path.write_text(dockerignore, encoding="utf-8")
    print(f"[OK] {ignore_path.name}")

    print(f"\n{'='*60}")
    print("DEPLOYMENT READY")
    print(f"{'='*60}")
    print(f"Files generated in: {DEPLOY_DIR}")
    print()
    print("Quick start:")
    print(f"  cd {DEPLOY_DIR}")
    print("  cp .env.example .env")
    print("  docker compose up -d")
    print()
    print("Or deploy to RunPod:")
    print("  docker build -t fathom-api .")
    print("  docker push <registry>/fathom-api:latest")
    print()
    print("Cost estimate (T4):")
    print("  $0.28 per 1,000 scores")
    print("  $360/month fixed (24/7)")
    print("  Break-even: ~13 Developer customers or 2 Professional")


if __name__ == "__main__":
    main()
