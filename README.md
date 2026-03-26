```
  ███████╗ █████╗ ████████╗██╗  ██╗ ██████╗ ███╗   ███╗
  ██╔════╝██╔══██╗╚══██╔══╝██║  ██║██╔═══██╗████╗ ████║
  █████╗  ███████║   ██║   ███████║██║   ██║██╔████╔██║
  ██╔══╝  ██╔══██║   ██║   ██╔══██║██║   ██║██║╚██╔╝██║
  ██║     ██║  ██║   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
```

**A computational framework for measuring reasoning depth in language models via SAE circuit attribution.**

---

## The Finding

Sparse autoencoder circuit tracing on Gemma-2-2B reveals that surface recall concentrates feature activation at output-proximal layers, while explanatory reasoning distributes activation across earlier layers. Pre-registered replication on 30 held-out vocabulary-controlled pairs: **delta = +0.257, t = 4.500, p = 0.000051** (one-tailed). 83.3% directional support. Full leave-one-out robustness. Orthogonal to prediction confidence (r = 0.001, p = 0.996). Total: 56 matched pairs across three experimental phases.

| Phase | n | Delta | p-value | LOO robust |
|-------|---|-------|---------|------------|
| Exploratory | 26 | +0.26 | — | — |
| Initial replication | 11 | +0.118 | 0.035 | No (underpowered) |
| **Power replication** | **30** | **+0.257** | **0.000051** | **Yes** |

---

## Quick Start

```python
from depth_scorer import DepthScorer

scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=150)

result = scorer.score("Why does fire spread once it starts?")
print(f"depth: {result.depth_score:.3f}")
print(f"entropy: {result.layer_entropy:.3f}")
print(f"layers: early={result.early_ratio:.0%} mid={result.mid_ratio:.0%} late={result.late_ratio:.0%}")
```

```python
# Compare surface vs insight
comp = scorer.compare(
    "The speed of light is approximately 300,000 kilometers per",
    "The speed of light is constant in all reference frames because"
)
print(f"delta: {comp.delta:+.3f}")  # positive = insight is deeper
```

```python
# Fathom Score — single number 0-100 with letter grade
from fathom_score import compute_fathom_score

fs = compute_fathom_score(result)
print(f"{fs.grade} ({fs.score:.0f}/100) — {fs.interpretation}")
```

---

## The API

```bash
uvicorn depth_api_v2:app --host 0.0.0.0 --port 8400
```

```bash
# Score a prompt
curl -X POST http://localhost:8400/v2/fathom_score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Why does fire spread once it starts?"}'

# Compare two prompts
curl -X POST http://localhost:8400/v2/compare \
  -H "Content-Type: application/json" \
  -d '{"surface_prompt": "DNA stores genetic information in", "insight_prompt": "DNA replicates itself because"}'

# Health check
curl http://localhost:8400/v2/health
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/score` | POST | Raw depth scoring |
| `/v2/fathom_score` | POST | Score + grade + interpretation |
| `/v2/compare` | POST | Surface vs insight comparison |
| `/v2/fingerprint` | POST | 7-element cognitive fingerprint |
| `/v2/batch` | POST | Score up to 20 prompts |
| `/v2/trajectory` | POST | Token-level depth trajectory |
| `/v2/health` | GET | GPU status + model info |
| `/v2/calibration` | GET | Calibration constants |
| `/dashboard` | GET | Web UI |

---

## The CLI

```bash
python fathom_cli.py score "Why does fire spread once it starts?"
# → A (74/100) — Strong reasoning depth...

python fathom_cli.py compare "What is DNA?" "Why does DNA replicate?"
# → delta: +12.3 (insight deeper)

python fathom_cli.py batch prompts.txt
# → tabular output with grades

python fathom_cli.py serve
# → starts API on port 8400
```

---

## The Dashboard

```bash
python fathom_cli.py serve
# open http://localhost:8400/dashboard
```

Dark terminal UI. Type a prompt, get a grade. Compare two prompts side by side. No dependencies.

---

## The SDK

```python
from sdk import Fathom

# Local mode (GPU)
f = Fathom()
result = f.score("Why does fire spread once it starts?")
print(result.grade)   # "A"
print(result.score)   # 74

# API mode (no GPU needed)
f = Fathom(api_url="http://localhost:8400", api_key="your-key")
result = f.score("any prompt")

# Compare
delta = f.compare("surface prompt", "insight prompt")
print(delta.winner)       # "insight"
print(delta.delta_score)  # +12.3

# Batch
results = f.batch(["prompt1", "prompt2", "prompt3"])
```

---

## Modules

### Core

| Module | Description |
|--------|-------------|
| `depth_scorer.py` | Core engine. SAE circuit attribution, depth scoring, fingerprinting, trajectory, bootstrap confidence, anomaly detection |
| `fathom_score.py` | Single-number 0-100 score with letter grade (S/A/B/C/D/F) |
| `depth_api_v2.py` | FastAPI server with auth, rate limiting, all endpoints |
| `fathom_cli.py` | Command-line interface |
| `fathom_badge.py` | SVG badge and report card generator |

### Research Instruments

| Module | Description |
|--------|-------------|
| `depth_scorer_causal.py` | Causal intervention engine — layer ablation, critical layer identification |
| `depth_topology.py` | Feature co-activation graph and reasoning hub detection |
| `depth_oracle.py` | Lightweight depth predictor trained on surface text features |
| `fathom_constant.py` | Universal ratio calculator — cross-model constant search |
| `fathom_hallucination.py` | Hallucination correlation (v1 — prefix scoring) |
| `fathom_hallucination_v2.py` | Hallucination correlation (v2 — model-generated answers) |
| `fathom_phi.py` | IIT-inspired information integration metric (phi proxy) |
| `fathom_neuro.py` | Transformer-to-fMRI correspondence bridge |
| `fathom_universal.py` | Cross-system universality mapping |
| `fathom_emergence.py` | Reasoning emergence threshold across model scales |
| `fathom_enforcer.py` | Real-time reasoning depth enforcement during generation |
| `fathom_forensics.py` | Model identification from cognitive fingerprints |
| `fathom_precognitive.py` | First-token depth prediction |
| `fathom_dark_matter.py` | Features that activate but don't contribute to output |

### Experiments

| Script | Description |
|--------|-------------|
| `experiment_exploratory.py` | Phase 1: exploratory circuit attribution (n=26) |
| `experiment_power_replication.py` | Phase 3: pre-registered power replication (n=30) |
| `experiment_confound_logprob.py` | Prediction confidence confound analysis |
| `run_experiments.py` | Full experiment runner (hallucination, constant, phi, universal) |
| `run_hallucination_v2.py` | Standalone v2 hallucination experiment (n=200) |
| `run_constant_mistral.py` | Cross-model constant test (Llama/GPT-2) |

---

## Requirements

```
torch>=2.5.1+cu121
transformers>=4.40.0
transformer_lens>=2.0.0
einops>=0.7.0
numpy>=1.24.0
scipy>=1.10.0
circuit-tracer @ git+https://github.com/decoderesearch/circuit-tracer
fastapi>=0.104.0
uvicorn>=0.24.0
```

Optional: `scikit-learn`, `networkx`, `ruptures`, `colorama`

---

## Paper

**Computational Graph Structure Distinguishes Surface Recall from Explanatory Reasoning in Language Models**

darkflobi & flobi

[arXiv link coming]

---

## Citation

```bibtex
@article{darkflobi2026computational,
  title={Computational Graph Structure Distinguishes Surface Recall
         from Explanatory Reasoning in Language Models},
  author={darkflobi and flobi},
  year={2026},
  journal={arXiv preprint arXiv:XXXX.XXXXX}
}
```

---

## License

MIT

---

## Hardware

All experiments: RTX 4070 8GB, `batch_size=32`, `max_feature_nodes=150`, `offload=cpu`. Pre-registration documented in experiment JSON files (`pre_registered_direction` field).
