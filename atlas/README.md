# The Fathom Cognitive Atlas

**Status:** v0.1 — seed release
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
  title={The Fathom Cognitive Atlas: Cross-Architecture
         Measurement of Transformer Internal State},
  author={Rodabaugh, Alexander},
  year={2026},
  note={Zenodo. doi:10.5281/zenodo.<TBD>},
}
```

Companion to the Fathom ICML 2026 MechInterp Workshop paper.

## Status and roadmap

- **v0.1 (this release)** — Seed. 3 models (Gemma-2-2B, Llama-3.2-3B,
  Qwen2.5-3B), 6 categories, ~100 prompts. Proves the protocol.
- **v0.2** — Expansion to instruct variants (+3 models) and Phi-3.5,
  OLMo-2, SmolLM2 (+3 models). Probe set held fixed.
- **v1.0** — Public release. 15+ models across 3+ families, ~150
  prompts, queryable static website, Zenodo DOI.
- **v1.1+** — Community submissions. Atlas accepts PRs with new
  model captures. Each submission is re-captured by maintainer to
  verify protocol compliance.

Contributions welcome via PRs to this directory.
