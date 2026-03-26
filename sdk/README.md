# Fathom SDK

Reasoning depth analysis for transformer models. Measures *how* a model thinks, not just *what* it outputs.

## Install

```bash
pip install torch transformers circuit-tracer
# Clone the repo, then:
cd sae-reasoning-depth
```

## Quick Start

```python
from sdk import Fathom

# Local mode (requires GPU)
f = Fathom()
result = f.score("Why does fire spread once it starts?")
print(result.grade)            # "A"
print(result.score)            # 74
print(result.interpretation)   # "Strong reasoning depth..."
```

## API Mode

Start the server:
```bash
python fathom_cli.py serve
# or: uvicorn depth_api_v2:app --host 0.0.0.0 --port 8400
```

Connect from anywhere:
```python
from sdk import Fathom

f = Fathom(api_url="http://localhost:8400", api_key="your-key")
result = f.score("any prompt")
```

## Batch Scoring

```python
results = f.batch([
    "What is DNA?",
    "Why does DNA replicate?",
    "How does CRISPR modify DNA?",
])
for r in results:
    if r:
        print(f"{r.grade} ({r.score:.0f}) — {r.prompt[:50]}")
```

## Compare Prompts

```python
delta = f.compare(
    "The capital of France is",           # surface recall
    "Democracy requires separation of powers to prevent"  # deep reasoning
)
print(delta.winner)       # "insight"
print(delta.delta_score)  # +18.3
```

## Generate Badges

```python
f.badge("Why does consciousness remain hard to define?", "badge.svg")
```

## ScoreResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | float | 0-100 composite reasoning depth score |
| `grade` | str | S/A/B/C/D/F letter grade |
| `interpretation` | str | One-sentence human summary |
| `depth_score` | float | Mean activated layer index (lower = deeper) |
| `weighted_depth_score` | float | Attribution-weighted depth |
| `layer_entropy` | float | Shannon entropy of layer distribution |
| `early_ratio` | float | Fraction of features in early layers |
| `mid_ratio` | float | Fraction in mid layers |
| `late_ratio` | float | Fraction in late/output layers |
| `feature_count` | int | Total active SAE features |
| `compute_time_ms` | float | Scoring time in milliseconds |
| `model` | str | Model used for analysis |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/score` | POST | Score a single prompt |
| `/v2/fathom_score` | POST | Score with grade + interpretation |
| `/v2/compare` | POST | Compare surface vs insight |
| `/v2/fingerprint` | POST | 7-element cognitive fingerprint |
| `/v2/batch` | POST | Score up to 20 prompts |
| `/v2/trajectory` | POST | Token-level depth trajectory |
| `/v2/health` | GET | Server + GPU status |
| `/v2/calibration` | GET | Calibration constants |
| `/dashboard` | GET | Web UI |
