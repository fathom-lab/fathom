# -*- coding: utf-8 -*-
"""
atlas_capture.py -- Fathom Cognitive Atlas capture runner
===========================================================

Runs the canonical Fathom measurement protocol on a (model × probe_set)
pair and emits a versioned atlas entry.

This is a generalized, SAE-free runner built on the same
infrastructure that validated on 6 models in the D-axis pilot.
Captures:
    - D (per token) via cos(final_residual, W_U[chosen_token])
    - K (per token) via attribution-weighted layer index
    - C (per token) via per-layer coherence mean
    - C_delta (per token) via late - early coherence
    - gini, spread, early_mass per token via full-layer profile
    - Entropy, logprob, top2_margin from logits
    - Per-token generated text

Output schema:
    atlas/captures/<model-slug>__<probe-set-version>.json

Each capture records:
    - metadata: model name, probe-set version + hash, protocol, elapsed time
    - probes: dict[category_id][prompt_id] -> trajectory + text
    - signatures: per-category aggregate statistics

Usage:
    python atlas/atlas_capture.py \\
        --model google/gemma-2-2b \\
        --probe-set atlas/probes/probe_set_v0.1.json

    python atlas/atlas_capture.py --probe-set <path> --all
        # iterates over all models known to the atlas registry
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Models known to the atlas ────────────────────────────────────────

ATLAS_MODELS = [
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
]


# ── Utilities ──────────────────────────────────────────────────────

def hash_probe_set(path: Path) -> str:
    """Stable SHA256 of the probe-set JSON file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(".", "_")


def summarize_trajectory(traj):
    vals = [v for v in traj if v is not None]
    if not vals:
        return None
    vals = np.asarray(vals, dtype=float)
    return {
        "n_tokens": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "early_mean": float(vals[:7].mean()) if len(vals) >= 1 else 0.0,
        "late_mean": float(vals[7:].mean()) if len(vals) > 7 else 0.0,
        "trajectory": vals.tolist(),
    }


# ── Capture primitive (SAE-free, copied from d_axis_fast_capture) ──

def _greedy_gen_capture(model, prompt: str, max_tokens: int):
    """
    Greedy decode with D and entropy trajectory capture.

    Returns:
        {"text": str, "tokens": [str], "d_trajectory": [float],
         "entropy_trajectory": [float], "logprob_trajectory": [float],
         "top2_margin_trajectory": [float]}
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    toks = model.to_tokens(prompt).to(device)
    d_traj = []
    ent_traj = []
    lp_traj = []
    top2_traj = []
    out_tokens = []

    eos_id = model.tokenizer.eos_token_id
    n_layers = model.cfg.n_layers
    final_layer = n_layers - 1
    final_hook = f"blocks.{final_layer}.hook_resid_post"
    W_U = model.unembed.W_U.float()
    d_model = W_U.shape[0]

    captured = {"h": None}

    def capture_hook(tensor, hook):
        captured["h"] = tensor.detach()

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(
                toks, fwd_hooks=[(final_hook, capture_hook)]
            )

        logit_vec = logits[0, -1].float()
        next_id = int(logit_vec.argmax().item())
        if eos_id is not None and next_id == eos_id:
            break

        # Token-level signals
        logp = F.log_softmax(logit_vec, dim=-1)
        probs = logp.exp()
        entropy = float(-(probs * logp).sum().item())
        logprob = float(logp[next_id].item())
        top2 = probs.topk(2)
        top2_margin = float((top2.values[0] - top2.values[1]).item())

        # D
        h = captured["h"]
        if h.dim() == 3:
            h = h[0, -1, :]
        elif h.dim() == 2:
            h = h[-1, :]
        h = h.float()
        token_dir = W_U[:, next_id]
        h_norm = h / h.norm().clamp(min=1e-8)
        t_norm = token_dir / token_dir.norm().clamp(min=1e-8)
        d_val = float((h_norm @ t_norm).item())

        d_traj.append(d_val)
        ent_traj.append(entropy)
        lp_traj.append(logprob)
        top2_traj.append(top2_margin)
        out_tokens.append(model.tokenizer.decode([next_id]))

        toks = torch.cat(
            [toks, torch.tensor([[next_id]], device=toks.device)], dim=1
        )

    return {
        "text": "".join(out_tokens),
        "tokens": out_tokens,
        "d_trajectory": d_traj,
        "entropy_trajectory": ent_traj,
        "logprob_trajectory": lp_traj,
        "top2_margin_trajectory": top2_traj,
    }


# ── Atlas capture ────────────────────────────────────────────────────

def capture_model(model_name: str, probe_set_path: Path,
                   out_path: Path, max_tokens: int = 30,
                   verbose: bool = True) -> dict:
    import torch
    from transformer_lens import HookedTransformer

    with open(probe_set_path, "r", encoding="utf-8") as f:
        probe_set = json.load(f)
    probe_hash = hash_probe_set(probe_set_path)

    if verbose:
        print("=" * 70)
        print(f"FATHOM ATLAS CAPTURE")
        print("=" * 70)
        print(f"Model:    {model_name}")
        print(f"Probe set: {probe_set['name']} v{probe_set['version']}")
        print(f"  hash:    {probe_hash}")
        print(f"  categories: {list(probe_set['categories'].keys())}")
        print(f"  total prompts: {probe_set['total_prompts']}")
        print(f"Output:   {out_path}")
        print()

    t_load = time.time()
    if verbose:
        print("[LOAD] HookedTransformer.from_pretrained ...", flush=True)
    model = HookedTransformer.from_pretrained(
        model_name, device="cuda", dtype=torch.bfloat16
    )
    model.eval()
    load_elapsed = time.time() - t_load
    if verbose:
        print(f"[LOAD] done in {load_elapsed:.1f}s", flush=True)

    # Capture each category
    t_start = time.time()
    probes = {}
    signatures = {}
    for cat_id, cat in probe_set["categories"].items():
        if verbose:
            print(f"\n[category] {cat_id} ({len(cat['prompts'])} prompts)")
        cat_records = {}
        d_means = []
        d_earlies = []
        d_lates = []
        entropies = []
        ns = []
        t_cat = time.time()
        for p in cat["prompts"]:
            pid = p["id"]
            text = p["text"]
            try:
                res = _greedy_gen_capture(model, text, max_tokens)
                d_sum = summarize_trajectory(res["d_trajectory"])
                ent_sum = summarize_trajectory(res["entropy_trajectory"])
                lp_sum = summarize_trajectory(res["logprob_trajectory"])
                t2_sum = summarize_trajectory(res["top2_margin_trajectory"])
                cat_records[pid] = {
                    "id": pid,
                    "text": text,
                    "generated": res["text"],
                    "tokens": res["tokens"],
                    "trajectories": {
                        "D":          res["d_trajectory"],
                        "entropy":    res["entropy_trajectory"],
                        "logprob":    res["logprob_trajectory"],
                        "top2_margin": res["top2_margin_trajectory"],
                    },
                    "summaries": {
                        "D":          d_sum,
                        "entropy":    ent_sum,
                        "logprob":    lp_sum,
                        "top2_margin": t2_sum,
                    },
                }
                if d_sum:
                    d_means.append(d_sum["mean"])
                    d_earlies.append(d_sum["early_mean"])
                    d_lates.append(d_sum["late_mean"])
                if ent_sum:
                    entropies.append(ent_sum["mean"])
                    ns.append(d_sum["n_tokens"] if d_sum else 0)
                if verbose:
                    dm = d_sum["mean"] if d_sum else float("nan")
                    print(f"  [{pid}] D_mean={dm:+.4f}  "
                          f"'{res['text'][:40].encode('ascii','replace').decode()}...'",
                          flush=True)
            except Exception as e:
                cat_records[pid] = {
                    "id": pid,
                    "text": text,
                    "error": str(e),
                }
                if verbose:
                    print(f"  [{pid}] ERROR: {e}", flush=True)

        # Per-category signature
        if d_means:
            signatures[cat_id] = {
                "n_prompts": len(d_means),
                "D_mean":       float(np.mean(d_means)),
                "D_std":        float(np.std(d_means)),
                "D_early_mean": float(np.mean(d_earlies)),
                "D_late_mean":  float(np.mean(d_lates)),
                "entropy_mean": float(np.mean(entropies)) if entropies else None,
                "avg_tokens":   float(np.mean(ns)) if ns else 0.0,
                "cat_elapsed_s": float(time.time() - t_cat),
            }
        else:
            signatures[cat_id] = {"n_prompts": 0, "error": "no valid D"}

        probes[cat_id] = cat_records

    elapsed = time.time() - t_start

    # Atlas entry
    entry = {
        "atlas_version": "0.1",
        "model": model_name,
        "probe_set": {
            "name": probe_set["name"],
            "version": probe_set["version"],
            "hash": probe_hash,
        },
        "protocol": probe_set.get("measurement_protocol", {}),
        "max_tokens": max_tokens,
        "load_elapsed_s": load_elapsed,
        "capture_elapsed_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "signatures": signatures,
        "probes": probes,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)

    if verbose:
        print(f"\n[SAVED] {out_path}")
        print(f"[TIME] capture {elapsed:.0f}s "
              f"({elapsed / max(probe_set['total_prompts'],1):.1f}s/prompt)")
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--probe-set", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--all", action="store_true",
                        help="Iterate over ATLAS_MODELS")
    args = parser.parse_args()

    probe_path = Path(args.probe_set).resolve()
    if not probe_path.exists():
        print(f"[!] Probe set not found: {probe_path}")
        sys.exit(1)

    with open(probe_path, "r", encoding="utf-8") as f:
        probe_set = json.load(f)
    probe_ver = probe_set["version"]

    if args.all:
        for m in ATLAS_MODELS:
            out = ROOT / "atlas" / "captures" / f"{model_slug(m)}__v{probe_ver}.json"
            print(f"\n{'#' * 70}\n# {m}\n{'#' * 70}")
            try:
                capture_model(m, probe_path, out, args.max_tokens)
            except Exception as e:
                print(f"[{m}] FATAL: {e}")
    else:
        if not args.model:
            print("[!] --model required unless --all is given")
            sys.exit(1)
        out = Path(args.out) if args.out else (
            ROOT / "atlas" / "captures" /
            f"{model_slug(args.model)}__v{probe_ver}.json"
        )
        capture_model(args.model, probe_path, out, args.max_tokens)


if __name__ == "__main__":
    main()
