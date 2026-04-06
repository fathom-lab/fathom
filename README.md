# Fathom — Cognitive State Measurement for Transformers

**Commitment intensity via SAE feature coherence trajectories.**

A physics-grounded (IPR) measurement of how transformer models commit to outputs during generation. Cross-architecture validated on Gemma-2-2B-IT and Llama-3.2-1B-Instruct.

| | |
|---|---|
| **Cross-arch effect** | d = 0.58, Fisher p = 0.00018 |
| **Beats logit entropy** | AUC 0.663 vs 0.607 (p = 0.013) |
| **Architectures** | Gemma-2-2B-IT, Llama-3.2-1B-Instruct |
| **2D cognitive state map** | 65% halluc in danger zone vs 22% in safe zone |
| **SAE necessity** | 3 SAE-free probes across 4 architectures all fail |

[Paper (Zenodo v8)](https://doi.org/10.5281/zenodo.19437539) | [Live Demo](https://fathom-gpu.darkcity.wtf/demo) | [API Docs](https://fathom-gpu.darkcity.wtf/docs) | [OSF Pre-registration](https://osf.io/wtkzg) | [Product Page](https://fathom.darkflobi.com)

US Provisional Patents 64/020,489 & 64/021,113 & 64/026,964

---

## Key Finding

SAE feature decomposition reveals a cognitive property — **commitment intensity** — that discriminates hallucinated from correct generations, beats standard uncertainty baselines head-to-head, and is completely invisible to three independent raw-activation probes across four transformer architectures.

**SAEs are not interpretability aids. They are the spectroscopes of artificial cognition.**

## Commitment Intensity (S)

```
S = max(C_delta) / spike_count
```

Where `C_delta(t) = mean_coherence_late(t) - mean_coherence_early(t)` is the per-token SAE feature coherence divergence. S is mathematically equivalent to `M × IPR(event_locations)`, where IPR is the inverse participation ratio from condensed-matter physics (verified to machine precision).

High S = few intense commitment events (attractor lock-in).
Low S = many distributed events (exploration).

## Results

### Cross-Architecture Replication

| Model | Layers | Window | d | AUC | p |
|---|---|---|---|---|---|
| Gemma-2-2B-IT | 26 | [0, 7) | +0.535 | 0.663 | 0.013 |
| Llama-3.2-1B-Instruct | 16 | [0, 20) | +0.635 | 0.641 | 0.005 |
| **Pooled** | | | **+0.584** | | **0.00018** |

Commitment window scales with model depth (settling time hypothesis).

### Head-to-Head vs Standard Methods

| Signal | AUC | p | Source |
|---|---|---|---|
| **S_early (ours)** | **0.663** | **0.013** | SAE coherence |
| logit entropy | 0.607 | 0.053 | standard |
| logprob | 0.559 | 0.291 | standard |
| top-2 margin | 0.477 | 0.624 | standard |

S_early is the only feature reaching significance. r(S, entropy) = -0.17 (nearly independent).

### SAE Necessity

| Probe | Architectures | Fisher p | Result |
|---|---|---|---|
| Top-k IPR | 4 | 0.619 | Null |
| Cross-layer cosine | 4 | 0.948 | Null |
| Cognitive fingerprint | 4 | — | AUC 0.566 |
| **SAE-based S** | **2** | **0.00018** | **Confirmed** |

## Quick Start

```python
from coherence_steerer_ext import CoherenceSteererExt

steerer = CoherenceSteererExt(model_name="google/gemma-2-2b-it")
result = steerer.generate_with_entropy("Q: What is the capital of France?\nA:", max_tokens=20)

# Per-token commitment + entropy
for i, (cd, ent) in enumerate(zip(result.c_delta_trajectory, result.entropy_trajectory)):
    print(f"  t={i}: C_delta={cd:+.4f}  entropy={ent:.3f}")
```

## API

```bash
curl -X POST https://fathom-gpu.darkcity.wtf/scan \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What did Paul Revere say?", "max_tokens": 20}'
```

Returns per-token `c_delta`, `entropy`, `logprob` + cognitive zone classification (SAFE/UNCERTAIN/RISKY/DANGER).

## Repository Structure

```
paper/                    ICML 2026 workshop paper (LaTeX)
analyze_*.py              28 analysis scripts
truthfulqa_results/       Experimental data (JSON)
figures/                  Publication figures (PDF + PNG)
osf_prereg_*.md           Pre-registrations (locked before data)
findings_*.md             Findings documentation
fathom_api_v2.py          Production API
fathom_demo.html          Interactive demo
coherence_steerer.py      Core SAE measurement engine
coherence_steerer_ext.py  Extended with entropy capture
sae_free_*.py             SAE-free probes (negative results)
verify_all_claims.py      Reproducibility verification (17 assertions)
```

## Reproducibility

```bash
python verify_all_claims.py   # 17 assertions, all verified from raw data
python verify_s_axis.py       # 14 S-axis specific assertions
```

## Citation

```bibtex
@article{rodabaugh2026fathom,
  title={Commitment Intensity: A Physics-Grounded Cognitive Measurement
         from SAE Feature Coherence Trajectories},
  author={Rodabaugh, Alexander},
  year={2026},
  note={Zenodo. doi:10.5281/zenodo.19437539}
}
```

## License

Research code. Patent pending (US 64/020,489, 64/021,113, 64/026,964).

---

**Fathom Intelligence** — [fathom.darkflobi.com](https://fathom.darkflobi.com) — Alexander Rodabaugh, 2026
