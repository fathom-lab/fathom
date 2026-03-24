# Computational Graph Structure Distinguishes Surface Recall from Explanatory Reasoning in Language Models

**Paper:** [Computational Graph Structure Distinguishes Surface Recall from Explanatory Reasoning in Language Models](https://arxiv.org/abs/XXXX.XXXXX) *(arXiv link pending)*

## Summary

We show that sparse autoencoder (SAE) circuit attribution reveals a robust difference in computational graph structure between surface-recall and explanatory-reasoning prompts in language models.

**Key finding:** Surface-recall prompts concentrate SAE feature activation at output-proximal layers; explanatory-reasoning prompts distribute activation across middle layers. This difference survives pre-registered replication (n=30, p=0.000051, LOO robust) on held-out matched pairs.

## Results

| Phase | n valid | Δ (surface − insight) | p-value | LOO robust |
|-------|---------|----------------------|---------|------------|
| Exploratory | ~30 | +0.26 | — | — |
| Pre-registered replication | 11 | +0.118 | 0.035 | No (power issue) |
| Power replication | 30 | +0.257 | **0.000051** | **Yes** |

## Repository Structure

- `experiment_exploratory.py` — Exploratory circuit attribution experiment (Gemma-2-2B)
- `experiment_power_replication.py` — Power replication script (n=30, pre-registered)
- `depth_scorer.py` — Core depth scoring engine using SAE circuit attribution
- `results_power_replication.json` — Full per-pair results, power replication (n=30)
- `results_initial_replication.json` — Full per-pair results, initial replication (n=11 valid)
- `results_exploratory.json` — Exploratory results
- `paper_draft.md` — Paper draft (pre-final)

## Requirements

```
torch>=2.5.1+cu121
transformers>=4.40.0
transformer_lens>=2.0.0
einops>=0.7.0
numpy>=1.24.0
scipy>=1.10.0
git+https://github.com/decoderesearch/circuit-tracer
```

## Citation

```bibtex
@article{darkflobi2026computational,
  title={Computational Graph Structure Distinguishes Surface Recall from Explanatory Reasoning in Language Models},
  author={darkflobi and flobi},
  year={2026},
  journal={arXiv preprint}
}
```

## Notes

- All experiments run on Gemma-2-2B via TransformerLens + circuit-tracer
- GPU: RTX 4070 8GB, batch_size=32, max_feature_nodes=500, offload=cpu
- Pre-registration documented in experiment JSON files (field: `pre_registered_direction`)
- Raw attribution outputs available on request
