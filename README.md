```
  ███████╗ █████╗ ████████╗██╗  ██╗ ██████╗ ███╗   ███╗
  ██╔════╝██╔══██╗╚══██╔══╝██║  ██║██╔═══██╗████╗ ████║
  █████╗  ███████║   ██║   ███████║██║   ██║██╔████╔██║
  ██╔══╝  ██╔══██║   ██║   ██╔══██║██║   ██║██║╚██╔╝██║
  ██║     ██║  ██║   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
```

**Measure how deeply AI thinks.** Reasoning depth scoring via SAE circuit attribution.

US Provisional Patents 64/020,489 & 64/021,113 | [Paper](https://doi.org/10.5281/zenodo.19326175) | [Blog](https://darkflobi.com/fathom-blog) | [Demo](https://darkflobi.com/fathom/demo) | [API Waitlist](https://darkflobi.com/fathom/waitlist)

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
Surface/insight depth ratio: Gemma-2-2B = 1.0302, Gemma-2-2B-IT = 1.0384. Weighted mean = **1.0343** (spread 0.0082, convergent). RLHF amplifies the signal (86.7% directional accuracy vs 76.7% base). Cross-family validation pending (Llama device error on retest).

### RLHF Safety: Adversarial Depth Amplification (v3, April 2026)
RLHF safety training deepens adversarial processing through additive early-layer circuits. n=30 matched pairs, base vs instruct Gemma-2-2B:
- **Instruct model:** adversarial prompts route DEEPER (delta=-0.124, p=0.0028, Cohen's d=-0.596)
- **Base model:** no significant depth difference (delta=+0.061, p=0.128)
- **Both models:** 1.61-1.66x feature count explosion on adversarial inputs (p<0.000001)
- Layer profile cosine similarity 0.996 = flooding/additive, not substitutive rerouting
- Extra features concentrate 58-61% in early layers — front-loaded detection circuits
- ICML submission target: Neel Nanda's Mech Interp Workshop (May 8, 2026)

### Negative Results
- **Entropy proxy does not detect K:** K_proxy=1.0015 vs SAE-based K=1.027 on Gemma. Activation entropy measures spread, not concept routing. SAE attribution is strictly necessary.
- **Pythia-160m too small for depth routing:** 12-layer model shows no K signal even with SAE proxy. K emergence scan parked for larger models (OLMo).

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

```bash
# RLHF selective amplification (white-box, GPU required)
python score_remaining_pairs.py --model base
python score_remaining_pairs.py --model instruct
python score_remaining_pairs.py --analyze

# Selective amplification cross-validation (black-box, API keys only)
export ANTHROPIC_API_KEY=sk-ant-...
python experiment_selective_blackbox.py --claude
```

---

## Citation

```bibtex
@article{rodabaugh2026fathom,
  title={Fathom: Measuring Reasoning Depth and Task-Specific Optimal Noise
         in Language Models via SAE Circuit Attribution},
  author={Rodabaugh, Alexander},
  year={2026},
  doi={10.5281/zenodo.19326175},
  url={https://doi.org/10.5281/zenodo.19326175},
  note={Zenodo preprint. arXiv submission pending.}
}
```

---

## License

MIT

---

*Built at 5am, alone, on a single GPU. Because the question mattered and nobody was asking it.*
