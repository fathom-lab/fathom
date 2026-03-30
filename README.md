```
  ███████╗ █████╗ ████████╗██╗  ██╗ ██████╗ ███╗   ███╗
  ██╔════╝██╔══██╗╚══██╔══╝██║  ██║██╔═══██╗████╗ ████║
  █████╗  ███████║   ██║   ███████║██║   ██║██╔████╔██║
  ██╔══╝  ██╔══██║   ██║   ██╔══██║██║   ██║██║╚██╔╝██║
  ██║     ██║  ██║   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
```

**Measure how deeply AI thinks.** Reasoning depth scoring via SAE circuit attribution.

US Provisional Patent 64/020,489 | [Paper](https://arxiv.org/abs/XXXX.XXXXX) | [Blog](https://darkflobi.com/fathom-blog) | [Demo](https://darkflobi.com/fathom/demo) | [API Waitlist](https://darkflobi.com/fathom/waitlist)

---

## What Fathom Does

When a transformer retrieves a memorized fact, computation concentrates at output-proximal layers.
When it reasons, computation distributes across mid-layers.
Fathom measures this distinction. One prompt in, one number out.

```
> fathom score "The capital of France is"
  depth: 8.71  entropy: 2.41  grade: D
  signal: shallow_recall

> fathom score "Why do economies inflate during monetary expansion"
  depth: 7.64  entropy: 3.52  grade: A
  signal: deep_reasoning
```

**Pre-registered replication (n=30):** delta = +0.257, t = 4.500, **p = 0.000051**. 83.3% directional. Leave-one-out robust. Orthogonal to prediction confidence (r = 0.001).

---

## Install

```bash
git clone https://github.com/darkflobi/fathom.git
cd fathom
pip install -r requirements.txt
```

Requires: Python 3.10+, PyTorch 2.5+, CUDA GPU (tested on RTX 4070 8GB).

---

## 30 Seconds to Your First Score

```python
from depth_scorer import DepthScorer

scorer = DepthScorer(model_name="google/gemma-2-2b")

# Score a single prompt
result = scorer.score("Why does fire spread once it starts?")
print(f"depth: {result.depth_score:.3f}  grade: {result.grade}")

# Compare surface vs insight
comp = scorer.compare(
    "The speed of light is approximately 300,000 km/s",
    "The speed of light is constant in all reference frames because"
)
print(f"delta: {comp.delta:+.3f}")  # positive = insight is deeper
```

---

## The Fathom Scale

Empirically calibrated on 10 cognitive task types (Gemma-2-2B, T=0.0):

| Depth | Task | Category |
|------:|------|----------|
| 0.5723 | Godel incompleteness | Meta-logical self-reference |
| 0.4749 | Exercise > mental health | Causal web synthesis |
| 0.3905 | Mirror reversal | Causal synthesis |
| 0.3799 | Logical syllogism | Logical integration |
| 0.3216 | Free will / compatibilism | Partial synthesis |
| 0.1753 | Heraclitus paradox | Explanatory |
| 0.1579 | Time perception | Listed explanation |
| 0.0870 | Probability (marbles) | Structured calculation |
| 0.0000 | Algebra (age problem) | Sequential computation |
| 0.0000 | Arithmetic (bat-and-ball) | Sequential computation |

---

## Key Discoveries

### The Decomposability Law
Depth measures integration *structure*, not content volume. A shorter answer with tighter causal integration scores higher than a longer answer with modular structure. Formally analogous to Tononi's Integrated Information (phi).

### Task-Specific Stochastic Resonance
Concept-level integration tasks score *deeper* at T=0.3 than T=0.0 (+0.107 for syllogisms). Formal-precision tasks score *shallower* (-0.149 for Godel). Same pattern as biological stochastic resonance (Wiesenfeld & Moss 1995).

### The Fathom Constant
Surface/insight depth ratio across architectures: Gemma-2-2B = 1.0302, Qwen3-0.6B = 1.0020. Mean = **1.0212**. The structural pattern holds across labs and parameter counts.

### fMRI Topology Convergence
Universality coefficient **0.830** between transformer layer distributions and human prefrontal/hippocampal activation patterns. Correlation, not causal claim.

---

## API

```bash
uvicorn depth_api_v2:app --host 0.0.0.0 --port 8400
```

| Endpoint | Description |
|----------|-------------|
| `POST /v2/score` | Raw depth scoring |
| `POST /v2/fathom_score` | Score + grade + interpretation |
| `POST /v2/compare` | Surface vs insight comparison |
| `POST /v2/fingerprint` | 7-element cognitive fingerprint |
| `POST /v2/batch` | Score up to 20 prompts |
| `POST /v2/trajectory` | Token-level depth trajectory |
| `GET /v2/health` | GPU status + model info |
| `GET /dashboard` | Interactive web UI |

```bash
curl -X POST http://localhost:8400/v2/fathom_score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Why does fire spread?"}'
```

---

## CLI

```bash
python fathom_cli.py score "Why does fire spread once it starts?"
python fathom_cli.py compare "What is DNA?" "Why does DNA replicate?"
python fathom_cli.py batch prompts.txt
python fathom_cli.py serve  # starts API + dashboard
```

---

## What We Don't Know Yet

**Hallucination prediction was null.** Three experiments (n=140 total). Depth measures reasoning *type*, not factual *correctness*. p=0.62. This is in the paper.

**Cross-model universality is preliminary.** The Fathom Constant on 2 architectures is a teaser, not a law. Pre-registration for 5+ architectures filed at OSF.

**One GPU, one researcher.** All results from RTX 4070 8GB. Replication on independent hardware welcome.

---

## Reproduce Everything

```bash
# Phase 3 pre-registered replication (the main result)
python experiment_power_replication.py

# Temperature sweep
python experiment_temperature.py

# Cross-architecture constant
python fathom_constant_v2.py

# Full experiment suite
python run_experiments.py
```

All raw data in JSON: `results_power_replication.json`, `temperature_experiment_data.json`, `fathom_constant_v2_results.json`.

---

## Citation

```bibtex
@article{rodabaugh2026fathom,
  title={Fathom: Measuring Reasoning Depth and Task-Specific Optimal Noise
         in Language Models via SAE Circuit Attribution},
  author={Rodabaugh, Alexander},
  year={2026},
  journal={arXiv preprint arXiv:XXXX.XXXXX}
}
```

---

## License

MIT

---

*Built at 5am, alone, on a single GPU. Because the question mattered and nobody was asking it.*
