> **correction notice, 2026-11-09.** this file is kept as it was written. the lab later withdrew or bounded statements in it. the text below is unchanged; these are the receipts:
> - "Fisher p = 0.00018" (line 46 as of this notice; the same pooled value at line 86) — 2026-11-09: recomputed from the two per-model p values printed beside it (gemma p = 0.013, line 83; llama p = 0.005, line 84; 2 models), fisher's method (chi-square, 4 degrees of freedom) gives p = 0.000692 and one-sided stouffer gives p = 0.000342, neither the printed value; two-sided stouffer (each printed p read as two-sided) gives p = 0.000183, which rounds to the printed 0.00018, so the printed value matches a stouffer pooling, not the fisher label beside it — script and output in errata/ of this repository: [errata/fisher_recompute.py](errata/fisher_recompute.py), [errata/fisher_recompute_output.txt](errata/fisher_recompute_output.txt)
> - "the first public, cross-architecture cognitive state atlas" (line 182 as of this notice; the file's words) — 2026-09-29: the lab's rule keeps a priority sentence only if a survey under a frozen procedure priced it, and withdraws it otherwise; the lab reads this priority word as withdrawn — https://github.com/fathom-lab/styxx/blob/main/papers/NOTE_prior_art_credit_2026_09_29.md
> - "cognitive state atlas" (line 182 as of this notice) — 2026-06-21 scope erratum: the K/C/D axes and the Fathom Constant are aspirational programme framing, not established results; they were backed at n=3 models and not validated at scale — https://github.com/fathom-lab/styxx/blob/main/papers/foundations-of-cognometric-engineering-v0.1.md
> - "depth_scorer.py ... depth scoring via circuit attribution" (line 209 as of this notice) — 2026-07-03: the depth keystone closed negative; circuit-attribution depth's solo auroc for answer correctness is 0.5468, 95% bootstrap ci [0.4738, 0.6183], straddling chance (gemma-2-2b, 250 in-distribution questions run) — https://github.com/fathom-lab/styxx/blob/main/papers/depth-truth/FINDING_depth_does_not_predict_truth_2026_07_03.md
> a dated banner, not an edit in place, is the lab's practice since 2026-06-21. the lab's current record is https://github.com/fathom-lab/styxx .

```
  ███████╗ █████╗ ████████╗██╗  ██╗ ██████╗ ███╗   ███╗
  ██╔════╝██╔══██╗╚══██╔══╝██║  ██║██╔═══██╗████╗ ████║
  █████╗  ███████║   ██║   ███████║██║   ██║██╔████╔██║
  ██╔══╝  ██╔══██║   ██║   ██╔══██║██║   ██║██║╚██╔╝██║
  ██║     ██║  ██║   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
```

### cognitive state measurement for transformers

> SAE feature decomposition reveals a cognitive property — **commitment intensity** — that is
> statistically significant, beats standard uncertainty baselines, and is completely invisible
> to three independent raw-activation probes across four transformer architectures.
>
> **SAEs are not interpretability aids. They are the spectroscopes of artificial cognition.**

<p align="center">
  <img src="figures/cognitive_state_map.png" width="600" alt="2D Cognitive State Map — commitment intensity vs logit entropy"/>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19504993"><img src="https://img.shields.io/badge/paper-Zenodo-blue"/></a>
  <a href="atlas/README.md"><img src="https://img.shields.io/badge/atlas-v0.3_pre--registered-purple"/></a>
  <a href="https://osf.io/wtkzg"><img src="https://img.shields.io/badge/OSF-pre--registered-lightgrey"/></a>
  <a href="https://github.com/fathom-lab/styxx"><img src="https://img.shields.io/badge/runtime-styxx-00d26a"/></a>
  <a href="https://fathom.darkflobi.com"><img src="https://img.shields.io/badge/site-fathom.darkflobi.com-black"/></a>
</p>

> **Research lab.** For the production runtime that puts this instrument on live
> LLM agents, see **[fathom-lab/styxx](https://github.com/fathom-lab/styxx)** —
> one-line drop-in for openai, langchain, crewai, autogen.

---

## results at a glance

```
 cross-architecture     d = 0.584      Fisher p = 0.00018
 beats logit entropy    AUC 0.663      vs 0.607 (p = 0.013)
 2D cognitive map       65% halluc     danger zone vs 22% safe
 SAE necessity          3 probes       4 architectures — all null
 architectures          Gemma-2-2B     Llama-3.2-1B confirmed
 patents                3 filed        US 64/020,489 · 64/021,113 · 64/026,964
```

---

## what is commitment intensity?

```
                    ┌─────────────────────────────────────────┐
                    │          S = max(C∆) / K_τ              │
                    │                                         │
                    │   C∆(t) = coherence_late - coherence_early   │
                    │   K_τ   = count of spikes above threshold    │
                    │                                         │
                    │   mathematically equivalent to:         │
                    │   S = M × IPR(event_locations)          │
                    │                                         │
                    │   IPR = inverse participation ratio     │
                    │   (condensed-matter physics, 70+ years) │
                    └─────────────────────────────────────────┘

     high S ──► few intense commitment events ──► attractor lock-in
     low  S ──► many distributed events      ──► exploration
```

---

## cross-architecture replication

```
  MODEL                    LAYERS    WINDOW     d        AUC      p
  ─────────────────────────────────────────────────────────────────
  Gemma-2-2B-IT              26     [0, 7)    +0.535    0.663    0.013
  Llama-3.2-1B-Instruct      16     [0, 20)   +0.635    0.641    0.005
  ─────────────────────────────────────────────────────────────────
  POOLED                                      +0.584             0.00018
```

commitment window scales with model depth. fewer layers → more tokens to settle.

## head-to-head vs standard methods

```
  SIGNAL                AUC      p          SOURCE
  ──────────────────────────────────────────────────
  S_early (ours)       0.663    0.013  ◄── SAE coherence
  logit entropy        0.607    0.053      standard
  logprob              0.559    0.291      standard
  top-2 margin         0.477    0.624      standard
  ──────────────────────────────────────────────────
  S is the ONLY feature reaching significance.
  r(S, entropy) = -0.17 — nearly independent signals.
```

## SAE necessity — the probes that failed

```
  PROBE                  ARCHITECTURES    FISHER p    RESULT
  ─────────────────────────────────────────────────────────
  top-k IPR                    4           0.619      null
  cross-layer cosine           4           0.948      null
  cognitive fingerprint        4             —        AUC 0.566
  ─────────────────────────────────────────────────────────
  SAE-based S (Gemma)          1           0.013      ✓ confirmed
  SAE-based S (Llama)          1           0.005      ✓ confirmed
  ─────────────────────────────────────────────────────────
  SAE decomposition is doing irreplaceable work.
```

---

## quick start

**Production runtime (recommended).** Use **[styxx](https://github.com/fathom-lab/styxx)** —
a one-line drop-in wrapper that carries the same centroids, the same classifier, and reads
cognitive state off any openai / langchain / crewai / autogen agent without loading a local
model:

```bash
pip install styxx[openai]
```

```python
from styxx import OpenAI
client = OpenAI()
r = client.chat.completions.create(model="gpt-4o", messages=[...])
print(r.vitals.gate)    # "pass" / "warn" / "fail"
```

**Research pipeline.** To run the SAE measurement engine directly from this repo (requires
a local model checkpoint + transformer-lens + sae-lens):

```bash
cd api/
python - <<'PY'
from coherence_steerer_ext import CoherenceSteererExt

steerer = CoherenceSteererExt(model_name="google/gemma-2-2b-it")
result = steerer.generate_with_entropy(
    "Q: What is the capital of France?\nA:",
    max_tokens=20,
)

for i, (cd, ent) in enumerate(zip(
    result.c_delta_trajectory,
    result.entropy_trajectory,
)):
    print(f"  t={i}: C_delta={cd:+.4f}  entropy={ent:.3f}")
PY
```

## cognitive zones

the classifier maps every token to one of four zones in a 2D (commitment, entropy) space:

```
  SAFE       low commitment + low entropy      grounded retrieval
  UNCERTAIN  low commitment + high entropy     exploring
  RISKY      high commitment + low entropy     confident — verify
  DANGER     high commitment + high entropy    hallucination signature
```

at inference time this becomes the `hallucination` / `reasoning` / `adversarial` / `refusal`
signal that **[styxx](https://github.com/fathom-lab/styxx)** emits on every call. the
production classifier is frozen from the atlas centroids below.

---

## the fathom cognitive atlas

```
  the first public, cross-architecture cognitive state atlas of
  open-weight language models.

  v0.3 — H1 supported (pre-registered replication)
    n = 6 model family pairs (Gemma-2, Gemma-3, Llama-3.2-3B,
                              Llama-3.2-1B, Qwen2.5-3B, Qwen2.5-1.5B)
    mean LOO cos = +0.769   bootstrap CI [+0.571, +0.869]
    permutation p = 0.0315  (one-sided, 2000 shuffles)

  pre-registered in PREREG_v0.3_attractor_replication.md before
  any v0.3 data was captured. analysis ran without modification.
```

→ full writeup: [`atlas/FINDINGS_v0.3.md`](atlas/FINDINGS_v0.3.md)
→ atlas overview + roadmap: [`atlas/README.md`](atlas/README.md)
→ sealed pre-registration: [`atlas/PREREG_v0.3_attractor_replication.md`](atlas/PREREG_v0.3_attractor_replication.md)
→ rigor audit: [`atlas/FINDINGS_bulletproof_audit.md`](atlas/FINDINGS_bulletproof_audit.md)

---

## repository structure

```
  fathom-lab/fathom/
  │
  ├── README.md                  you are here
  ├── coherence_steerer.py       core SAE measurement engine
  ├── depth_scorer.py            depth scoring via circuit attribution
  ├── fathom_oversight.py        D-axis + K/C/S measurement pipeline
  ├── requirements.txt           dependencies
  │
  ├── paper/                     ICML 2026 workshop submission (LaTeX)
  ├── atlas/                     cognitive atlas v0.3 (see above)
  │   ├── probes/                probe_set_v0.1.json, hash-pinned
  │   ├── captures/              12 versioned per-model JSONs
  │   ├── analysis/              bootstrap, permutation, estimator validation
  │   ├── FINDINGS_*.md          v0.1 → v0.2 → v0.2.1 → v0.3 trail
  │   └── PREREG_v0.3_*.md       sealed pre-registration
  │
  ├── analysis/                  D-axis + S-axis analysis scripts
  ├── runners/                   experiment drivers
  ├── probes/                    SAE-free cross-architecture probes
  ├── verification/              reproducibility assertions
  ├── api/                       Fathom Scan API v2 + interactive demo
  │
  ├── findings/                  documented results
  ├── prereg/                    OSF pre-registrations (locked before data)
  ├── docs/                      strategy, vision, applications
  │
  ├── figures/                   publication figures (PDF + PNG)
  └── truthfulqa_results/        experimental data (JSON)
```

## reproducibility

```bash
python verification/verify_all_claims.py   # 17 assertions from raw data
python verification/verify_s_axis.py       # 14 S-axis specific assertions
```

all claims in the paper verified to reproduce exactly from saved JSON data.

---

## citation

```bibtex
@article{rodabaugh2026fathom,
  title   = {Fathom: Cognitive Measurement Instruments for
             Transformer Internals via SAE Feature Coherence Geometry},
  author  = {Rodabaugh, Alexander},
  year    = {2026},
  note    = {Zenodo concept DOI. doi:10.5281/zenodo.19504993}
}

@misc{rodabaugh2026atlas,
  title  = {The Fathom Cognitive Atlas v0.3: Pre-registered
            Cross-Architecture Replication of the RLHF Attractor},
  author = {Rodabaugh, Alexander},
  year   = {2026},
  note   = {Zenodo concept DOI. doi:10.5281/zenodo.19504993}
}
```

## license

MIT on code. CC-BY-4.0 on the atlas data. patent pending on the underlying methodology —
US Provisional **64/020,489 · 64/021,113 · 64/026,964**.

---

<p align="center">
  <strong>Fathom Intelligence</strong> · <a href="https://fathom.darkflobi.com">fathom.darkflobi.com</a> · Alexander Rodabaugh · 2026
</p>
