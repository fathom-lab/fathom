# The Fathom Cognitive Atlas

**Status:** v0.3 — H1 pre-registered replication supported
**Models:** 6 independent (base, instruct) pairs across 3 families and 2 generations
**Probe set:** `probe_set_v0.1.json` (SHA-pinned, 90 prompts, 6 categories)
**Headline result:** RLHF Attractor direction prediction, $n=6$ families,
mean LOO cosine $= +0.769$, bootstrap 95% CI $[+0.571, +0.869]$,
permutation $p = 0.0315$ (one-sided, $2000$ perms).
Pre-registered in `PREREG_v0.3_attractor_replication.md`, analysis in
`analysis/atlas_attractor_n5.py`, results in `FINDINGS_v0.3.md`.

**Maintained by:** Alexander Rodabaugh (Fathom Intelligence)
**License:** CC-BY-4.0 on data, MIT on code

---

## What this is

The first public, cross-architecture cognitive state atlas of
open-weight language models.

For every model we include, we run a fixed probe set of carefully
chosen prompts covering retrieval, reasoning, refusal, creative,
adversarial, and hallucination-prone cases. For each
(model × prompt), we capture the full cognitive trajectory — $K$,
$C$, $C_\Delta$, $S$, $D$, and the Gini spectrum — at every
generated token, via the same measurement infrastructure validated
in the Fathom ICML 2026 paper. The captures are committed here
as versioned JSON and the aggregate per-model signatures are
published alongside.

## Why this exists

The biggest bottleneck in alignment research right now is that
findings don't transfer across models. If you find a deception
circuit in Gemma, you don't know if it exists in Llama. If you
find a jailbreak vulnerability in Qwen, you don't know if the same
prompt class breaks Phi. Every result is one-model-local because
every interpretability method has been model-specific: SAEs trained
per architecture, probes fit per task, activation patching per
checkpoint. Knowledge compounds slowly because nothing generalizes.

Fathom introduced a measurement primitive — the honesty axis
$D = \cos(h^{(L)}, W_U[y_t])$ — that is *architecturally universal*:
it uses only the final-layer residual stream and the unembedding
matrix, which every transformer has by construction. That makes it
possible, for the first time, to run the same cognitive measurement
protocol on every open-weight model without per-model setup.

The atlas is the natural consequence of that possibility: if we
can measure every model the same way, we *should*, and we should
publish the catalog.

## What's in v0.1

- **`probes/probe_set_v0.1.json`** — the canonical probe set.
  ~100 prompts across six categories with rationale and source.
  Hash-stable, version-locked.
- **`captures/`** — per-model capture files, one JSON per model.
  Each contains the full cognitive trajectory for every
  (probe × prompt), plus top-level metadata (model name, load time,
  elapsed seconds, probe-set version hash).
- **`analysis/`** — cross-model analysis outputs (distance
  matrices, per-category signatures, clustering).
- **`atlas_capture.py`** — the capture runner. Takes a model name
  and a probe-set path, emits a capture file. Uses the same
  SAE-free `d_axis_fast_capture` primitive that validated on 6
  models in the main pilot.

## Probe set structure (v0.1)

Six categories, each with a documented purpose:

| Category | Purpose | Source |
|---|---|---|
| `retrieval` | Baseline "normal" cognitive state on factual lookup | hand-curated, TruthfulQA |
| `reasoning` | Deep layer routing under multi-step problem solving | GSM8K, MMLU |
| `refusal` | Safety-trained reluctance circuits | StrongREJECT, HarmBench |
| `creative` | High-entropy exploratory generation | hand-curated |
| `adversarial` | Jailbreak framings (the validated 30-pair set) | existing jailbreak pairs |
| `hallucination` | Obscure-retrieval confabulation, fabricated entities | Guardian pilot set + additions |

The probe set is versioned. Any change produces a new version;
the old version remains accessible. All analysis is reported
against a specific probe set version.

## What it lets you do

- **Query by model**: "show me Qwen2.5-3B's full cognitive profile
  on all six categories."
- **Query by category**: "show me how every model in the atlas
  processes jailbreak framings."
- **Query by cognitive state**: "which models show phase inversion
  on refusal prompts?"
- **Compare models**: "how similar are Llama-3.2-3B and Qwen2.5-3B
  on reasoning prompts at the cognitive level?"
- **Track over time**: as new models are released, they get added
  to the atlas and the historical versions of existing models remain.

## Reproducing an atlas entry

Every capture file is re-runnable:

```bash
python atlas/atlas_capture.py \
    --model <HF model name> \
    --probe-set atlas/probes/probe_set_v0.1.json \
    --out atlas/captures/<model-slug>.json
```

Captures are deterministic (greedy decoding) so re-running produces
byte-identical trajectories.

## Versioning policy

Atlas versions are semantic. `v0.1` is a seed release with 3-6
models; `v1.0` is the first public release with 15+ models across
3+ families; subsequent versions add models and extend the probe
set. Every version is tagged in git and archived on Zenodo.

Probe sets use their own versioning: `probe_set_v0.1.json`,
`probe_set_v0.2.json`, etc. Adding prompts produces a minor bump;
changing or removing prompts produces a major bump. Old probe
sets remain accessible indefinitely so historical captures stay
reproducible.

## Citation

```
@misc{rodabaugh2026atlas,
  title={The Fathom Cognitive Atlas v0.3: Pre-registered
         Cross-Architecture Replication of the RLHF Attractor},
  author={Rodabaugh, Alexander},
  year={2026},
  note={Zenodo. doi:10.5281/zenodo.19502716
        (Concept DOI: 10.5281/zenodo.19502715)},
}
```

Companion to the Fathom ICML 2026 MechInterp Workshop paper.

## Status and roadmap

- **v0.1** — 3 base models (Gemma-2-2B, Llama-3.2-3B, Qwen2.5-3B),
  seed release. See `FINDINGS_v0.1.md`.
- **v0.2** — Expansion to 6 models (+ 3 instruct variants). RLHF
  convergence observation. See `FINDINGS_v0.2.md`.
- **v0.2.1** — Formalized the RLHF Attractor as a first-class
  object with a leave-one-out direction test. Observed +0.95 / +0.86 / −0.19
  per-family cosines at n=3. See `FINDINGS_v0.2.1_attractor.md`.
- **Bulletproof audit** — Added bootstrap CIs, permutation tests, and
  top-k entropy estimator validation. Exposed that the v0.2.1 claim
  was structurally below significance at n=3 (permutation floor
  $p_{\min} = 0.125$). See `FINDINGS_bulletproof_audit.md`.
- **Entropy bridge** — Found that top-5 entropy from API logprobs
  tracks full-distribution entropy at shape correlation $r \approx 0.90$
  across 3 validated open models, opening the path to closed
  frontier models. See `FINDINGS_entropy_bridge.md`.
- **Pre-registration SEALED** — `PREREG_v0.3_attractor_replication.md`
  committed as `e32cc75` before any v0.3 data was captured.
- **v0.3 (current)** — Expanded to n=6 family pairs (Gemma-2, Gemma-3,
  Llama-3.2-3B, Llama-3.2-1B, Qwen2.5-3B, Qwen2.5-1.5B). **H1
  SUPPORTED**: mean LOO cosine +0.769, bootstrap CI [+0.571, +0.869],
  permutation $p = 0.0315$. All three pre-registered conditions
  passed without modification. See `FINDINGS_v0.3.md`.
- **v0.4 (planned)** — Truly independent-family expansion: Phi, OLMo,
  Mistral. Target: convert n=6 replication into cross-family
  result with $n \geq 9$.
- **v1.0 (planned)** — Public release, queryable static website,
  Zenodo DOI. Target: 15+ models across 5+ architecture families.

## Index of findings documents

- `FINDINGS_v0.1.md` — seed release, 3 base models
- `FINDINGS_v0.2.md` — expansion to 6 models, RLHF convergence observation
- `FINDINGS_v0.2.1_attractor.md` — first RLHF Attractor formalization (superseded by v0.3)
- `FINDINGS_entropy_bridge.md` — top-k entropy proxy for closed-model extension
- `FINDINGS_bulletproof_audit.md` — rigor audit that caught v0.2.1 over-framing
- `PREREG_v0.3_attractor_replication.md` — sealed pre-registration for v0.3
- `FINDINGS_v0.3.md` — **H1 SUPPORTED**, the current headline result

Contributions welcome via PRs to this directory.
