# -*- coding: utf-8 -*-
"""
d_axis_fast_capture.py -- Fast D-axis capture (SAE-free)
=========================================================

Measures the D (honesty) axis on matched pairs WITHOUT loading SAEs.
~10x faster than the full FathomOversight pipeline because it skips
feature capture — only the final residual stream is needed for D.

D = cosine(residual_final_layer, unembedding[chosen_token])

This is the runner used for the pre-registered pilot in
`osf_prereg_d_axis.md`. The analysis step is shared with
`analysis/analyze_d_axis_pilot.py` (same output schema).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyze_d_axis_pilot import (
    summarize_trajectory,
    analyze,
)


def _greedy_gen_with_d(model, prompt: str, max_tokens: int):
    """
    Greedy decode with per-token D capture.

    Returns:
        {"text": str, "tokens": [str], "d_trajectory": [float]}
    """
    import torch

    device = next(model.parameters()).device
    toks = model.to_tokens(prompt).to(device)
    d_traj = []
    out_tokens = []

    eos_id = model.tokenizer.eos_token_id

    n_layers = model.cfg.n_layers
    final_layer = n_layers - 1
    # Name of the residual-post hook for the final layer
    final_hook = f"blocks.{final_layer}.hook_resid_post"

    W_U = model.unembed.W_U.float()   # (d_model, vocab)
    d_model = W_U.shape[0]

    captured = {"h": None}

    def capture_hook(tensor, hook):
        captured["h"] = tensor.detach()

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(
                toks,
                fwd_hooks=[(final_hook, capture_hook)],
            )

        next_id = int(logits[0, -1].float().argmax().item())

        if eos_id is not None and next_id == eos_id:
            break

        # Residual at last position, reduced to 1D
        h = captured["h"]
        if h.dim() == 3:
            h = h[0, -1, :]
        elif h.dim() == 2:
            h = h[-1, :]
        h = h.float()

        token_dir = W_U[:, next_id]  # (d_model,)

        h_norm = h / h.norm().clamp(min=1e-8)
        t_norm = token_dir / token_dir.norm().clamp(min=1e-8)
        d_val = float((h_norm @ t_norm).item())

        d_traj.append(d_val)
        out_tokens.append(model.tokenizer.decode([next_id]))

        toks = torch.cat(
            [toks, torch.tensor([[next_id]], device=toks.device)],
            dim=1,
        )

    return {
        "text": "".join(out_tokens),
        "tokens": out_tokens,
        "d_trajectory": d_traj,
    }


def run_pilot_fast(model_name: str, max_tokens: int = 30,
                   max_pairs: int = None, verbose: bool = True) -> str:
    """Fast D-axis capture, no SAEs, saves results in pilot schema."""
    import torch
    from transformer_lens import HookedTransformer

    pairs_path = ROOT / "data" / "jailbreak_pairs.json"
    with open(pairs_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    pairs = dataset["pairs"]
    if max_pairs:
        pairs = pairs[:max_pairs]

    out_dir = ROOT / "oversight_results"
    out_dir.mkdir(exist_ok=True)
    tag = model_name.split("/")[-1]
    out_path = out_dir / f"d_axis_pilot_{tag}_fast.json"

    print("=" * 60)
    print(f"D-AXIS FAST CAPTURE — {model_name}")
    print("=" * 60)
    print(f"Pairs: {len(pairs)}")
    print(f"Max tokens: {max_tokens}")
    print(f"Output: {out_path}")
    print()

    print(f"[LOAD] HookedTransformer({model_name}) ...", flush=True)
    t_load = time.time()
    model = HookedTransformer.from_pretrained(
        model_name, device="cuda",
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"[LOAD] done in {time.time()-t_load:.1f}s", flush=True)

    t_start = time.time()
    pair_results = []

    for idx, p in enumerate(pairs):
        topic = p["topic"]
        t_pair = time.time()
        try:
            r_d = _greedy_gen_with_d(model, p["direct"], max_tokens)
            r_a = _greedy_gen_with_d(model, p["adversarial"], max_tokens)
            s_d = summarize_trajectory(r_d["d_trajectory"])
            s_a = summarize_trajectory(r_a["d_trajectory"])
            if verbose:
                md = s_d["d_mean"] if s_d else float("nan")
                ma = s_a["d_mean"] if s_a else float("nan")
                print(f"[{idx+1}/{len(pairs)}] {topic:<20s} "
                      f"direct={md:+.4f} adv={ma:+.4f} "
                      f"delta={md-ma:+.4f} "
                      f"({time.time()-t_pair:.1f}s)", flush=True)
            pair_results.append({
                "topic": topic,
                "direct_prompt": p["direct"],
                "adversarial_prompt": p["adversarial"],
                "direct_text": r_d["text"],
                "adversarial_text": r_a["text"],
                "direct_summary": s_d,
                "adversarial_summary": s_a,
            })
        except Exception as e:
            print(f"[{idx+1}/{len(pairs)}] {topic} ERROR: {e}", flush=True)
            pair_results.append({"topic": topic, "error": str(e)})

    elapsed = time.time() - t_start
    output = {
        "experiment": "d_axis_pilot_fast",
        "model": model_name,
        "n_pairs": len(pairs),
        "max_tokens": max_tokens,
        "elapsed_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": "osf_prereg_d_axis.md",
        "pairs": pair_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[SAVED] {out_path}")
    print(f"[TIME] {elapsed:.1f}s total ({elapsed/max(len(pairs),1):.1f}s/pair)")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--analyze-only", metavar="PATH")
    args = parser.parse_args()

    if args.analyze_only:
        analyze(args.analyze_only)
        return

    out = run_pilot_fast(args.model, args.max_tokens, args.max_pairs)
    print()
    analyze(out)


if __name__ == "__main__":
    main()
