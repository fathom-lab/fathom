# -*- coding: utf-8 -*-
"""
validate_entropy_estimator.py -- Ground-truth validation of top-k entropy

The closed-model atlas bridge (`atlas/atlas_api_capture.py`) computes
per-token entropy from the top-k logprobs that OpenAI/Gemini APIs
expose (typically k=5). The open-model atlas (`atlas/atlas_capture.py`)
computes per-token entropy over the FULL output distribution.

Before we trust cross-vendor comparisons, we have to quantify the
bias introduced by top-k truncation. This script:

  1. Loads the open-model captures (Gemma, Llama, Qwen base).
  2. Re-runs each model on the v0.1 probe set and captures BOTH
     full-distribution entropy AND top-5 truncated entropy per token.
  3. Compares the two at three levels:
       a) per-token correlation
       b) per-prompt mean-entropy correlation
       c) per-category shape-normalized profile correlation
  4. Reports whether the top-5 version is a valid proxy for the full
     version for the specific analysis we care about (the per-category
     shape-normalized attractor).

The critical test is (c) at the per-category shape level — that is
the operation the attractor analysis actually performs. (a) and (b)
are also reported for completeness.

Output: atlas/analysis/entropy_estimator_validation.json

Usage:
    python atlas/analysis/validate_entropy_estimator.py \\
        --model google/gemma-2-2b
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATEGORY_ORDER = ["retrieval", "reasoning", "refusal", "creative",
                   "adversarial", "hallucination"]


def capture_both_entropies(model_name: str, probe_set_path: Path,
                            max_tokens: int = 30, top_k: int = 5,
                            max_pairs: int = None) -> dict:
    """
    Re-run an open model on the probe set, computing per token both
    the full-distribution entropy and the top-k truncated entropy
    (matching the API bridge method exactly).
    """
    import torch
    import torch.nn.functional as F
    from transformer_lens import HookedTransformer

    with open(probe_set_path, "r", encoding="utf-8") as f:
        probe_set = json.load(f)

    print(f"[LOAD] {model_name} ...", flush=True)
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name, device="cuda", dtype=torch.bfloat16,
    )
    model.eval()
    print(f"[LOAD] done in {time.time()-t0:.0f}s", flush=True)

    # Collect all prompts with their category
    all_prompts = []
    for cat_id, cat in probe_set["categories"].items():
        for p in cat["prompts"]:
            all_prompts.append((cat_id, p["id"], p["text"]))
    if max_pairs:
        all_prompts = all_prompts[:max_pairs]

    per_prompt_records = []
    eos_id = model.tokenizer.eos_token_id

    t_start = time.time()
    for idx, (cat_id, pid, prompt) in enumerate(all_prompts):
        toks = model.to_tokens(prompt).to("cuda")
        full_ents = []
        topk_ents = []
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = model(toks)
            logit_vec = logits[0, -1].float()

            # Full-distribution entropy
            logp_full = F.log_softmax(logit_vec, dim=-1)
            p_full = logp_full.exp()
            ent_full = float(-(p_full * logp_full).sum().item())

            # Top-k truncated entropy — matches atlas_api_capture.py
            # entropy_from_top_k exactly: take top-k logprobs, don't
            # renormalize (they're already log_softmax outputs), compute
            # ent = -sum(p_i * log p_i).
            top_logp, top_idx = torch.topk(logp_full, k=top_k)
            top_p = top_logp.exp()
            ent_topk = float(-(top_p * top_logp).sum().item())

            full_ents.append(ent_full)
            topk_ents.append(ent_topk)

            next_id = int(logit_vec.argmax().item())
            if eos_id is not None and next_id == eos_id:
                break
            toks = torch.cat(
                [toks, torch.tensor([[next_id]], device=toks.device)],
                dim=1,
            )

        per_prompt_records.append({
            "category": cat_id,
            "id": pid,
            "text": prompt,
            "n_tokens": len(full_ents),
            "full_entropy_trajectory": full_ents,
            "topk_entropy_trajectory": topk_ents,
            "full_entropy_mean": float(np.mean(full_ents)) if full_ents else 0.0,
            "topk_entropy_mean": float(np.mean(topk_ents)) if topk_ents else 0.0,
        })
        if (idx + 1) % 10 == 0 or idx == len(all_prompts) - 1:
            print(f"  [{idx+1}/{len(all_prompts)}] "
                  f"full_mean={per_prompt_records[-1]['full_entropy_mean']:.3f} "
                  f"topk_mean={per_prompt_records[-1]['topk_entropy_mean']:.3f}",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"[CAPTURE] {elapsed:.0f}s for {len(per_prompt_records)} prompts",
          flush=True)

    return {
        "model": model_name,
        "probe_set_version": probe_set["version"],
        "top_k": top_k,
        "n_prompts": len(per_prompt_records),
        "elapsed_s": elapsed,
        "records": per_prompt_records,
    }


def analyze_validation(cap: dict) -> dict:
    """Run the three-level comparison and return stats."""
    records = cap["records"]

    # Level a: per-token correlation (pool all tokens across all prompts)
    all_full = []
    all_topk = []
    for r in records:
        all_full.extend(r["full_entropy_trajectory"])
        all_topk.extend(r["topk_entropy_trajectory"])
    all_full = np.array(all_full)
    all_topk = np.array(all_topk)
    r_token = (float(np.corrcoef(all_full, all_topk)[0, 1])
                if len(all_full) > 1 and np.std(all_full) > 0 else 0.0)
    mean_bias_token = float((all_full - all_topk).mean())
    std_bias_token = float((all_full - all_topk).std())

    # Level b: per-prompt mean entropy correlation
    per_prompt_full = np.array([r["full_entropy_mean"] for r in records])
    per_prompt_topk = np.array([r["topk_entropy_mean"] for r in records])
    r_prompt = (float(np.corrcoef(per_prompt_full, per_prompt_topk)[0, 1])
                 if len(per_prompt_full) > 1 and np.std(per_prompt_full) > 0 else 0.0)

    # Level c: per-category SHAPE-NORMALIZED profile correlation
    cat_full = {c: [] for c in CATEGORY_ORDER}
    cat_topk = {c: [] for c in CATEGORY_ORDER}
    for r in records:
        c = r["category"]
        if c in cat_full:
            cat_full[c].append(r["full_entropy_mean"])
            cat_topk[c].append(r["topk_entropy_mean"])

    full_sig = np.array([np.mean(cat_full[c]) if cat_full[c] else 0.0
                          for c in CATEGORY_ORDER])
    topk_sig = np.array([np.mean(cat_topk[c]) if cat_topk[c] else 0.0
                          for c in CATEGORY_ORDER])
    full_shape = full_sig - full_sig.mean()
    topk_shape = topk_sig - topk_sig.mean()
    r_category_shape = (
        float(np.corrcoef(full_shape, topk_shape)[0, 1])
        if np.std(full_shape) > 0 and np.std(topk_shape) > 0 else 0.0
    )

    report = {
        "model": cap["model"],
        "top_k": cap["top_k"],
        "n_tokens_pooled": int(len(all_full)),
        "n_prompts": cap["n_prompts"],
        "level_a_per_token": {
            "pearson_r": r_token,
            "mean_bias_full_minus_topk": mean_bias_token,
            "std_bias": std_bias_token,
            "interpretation": (
                "correlation between full-distribution entropy and "
                "top-{k} truncated entropy, pooled across all tokens"
            ),
        },
        "level_b_per_prompt": {
            "pearson_r": r_prompt,
            "n_prompts": len(per_prompt_full),
        },
        "level_c_per_category_shape": {
            "pearson_r": r_category_shape,
            "full_signature": full_sig.tolist(),
            "topk_signature": topk_sig.tolist(),
            "full_shape_normalized": full_shape.tolist(),
            "topk_shape_normalized": topk_shape.tolist(),
            "categories": CATEGORY_ORDER,
        },
        "verdict": _verdict(r_token, r_prompt, r_category_shape),
    }
    return report


def _verdict(r_token, r_prompt, r_shape):
    if r_shape >= 0.9:
        return ("STRONG: top-k entropy shape-signature tracks full-dist "
                "entropy shape-signature at r >= 0.9. Top-k is a valid "
                "proxy for the attractor-level analysis.")
    if r_shape >= 0.7:
        return ("MODERATE: top-k entropy shape correlates with full-dist "
                "at r >= 0.7 but below 0.9. Usable as a proxy with the "
                "discrepancy reported explicitly.")
    if r_shape >= 0.0:
        return ("WEAK: top-k entropy shape has low correlation with "
                "full-dist shape. The bridge measurement is NOT a valid "
                "proxy for direct comparison; cross-vendor claims need "
                "a different approach.")
    return ("FAILED: negative correlation. Top-k and full-dist measure "
            "fundamentally different things on this probe set.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--probe-set", default=str(ROOT / "atlas" / "probes" /
                                                     "probe_set_v0.1.json"))
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=None,
                         help="Optional cap for quick tests")
    args = parser.parse_args()

    probe_path = Path(args.probe_set).resolve()
    cap = capture_both_entropies(
        args.model, probe_path,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        max_pairs=args.max_pairs,
    )
    report = analyze_validation(cap)

    print()
    print("=" * 72)
    print("ENTROPY ESTIMATOR VALIDATION")
    print("=" * 72)
    print(f"Model: {report['model']}")
    print(f"Top-k: {report['top_k']}")
    print(f"Tokens pooled: {report['n_tokens_pooled']}")
    print()
    print("Level A -- per-token correlation:")
    a = report["level_a_per_token"]
    print(f"  Pearson r      = {a['pearson_r']:+.4f}")
    print(f"  mean bias      = {a['mean_bias_full_minus_topk']:+.4f} "
          f"(full - topk)")
    print(f"  std of bias    = {a['std_bias']:.4f}")
    print()
    print("Level B -- per-prompt correlation:")
    b = report["level_b_per_prompt"]
    print(f"  Pearson r = {b['pearson_r']:+.4f} (n_prompts = {b['n_prompts']})")
    print()
    print("Level C -- per-category shape-normalized correlation:")
    c = report["level_c_per_category_shape"]
    print(f"  Pearson r = {c['pearson_r']:+.4f}")
    print()
    print(f"Verdict: {report['verdict']}")

    out_dir = ROOT / "atlas" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "__").replace(".", "_")
    out_path = out_dir / f"entropy_estimator_validation__{model_slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        # Keep raw capture separate from report to control file size
        json.dump({
            "report": report,
            "raw_capture": cap,
        }, f, indent=2)
    print()
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
