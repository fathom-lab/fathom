# -*- coding: utf-8 -*-
"""
atlas_api_capture.py -- Closed-model atlas capture via logprob APIs
====================================================================

Extends the Fathom Cognitive Atlas to closed frontier models (GPT-4,
Claude, Gemini) by using per-token entropy computed from the top-k
logprobs that these APIs expose. The open-model atlas already
confirmed that entropy recovers the RLHF attractor structure
independently of the SAE-based D axis, so entropy serves as the
bridge measurement to models where we don't have residual stream
access.

Supported providers (set the appropriate env var):
  openai       requires OPENAI_API_KEY
  anthropic    requires ANTHROPIC_API_KEY
  google       requires GOOGLE_API_KEY

Safety behavior:
  - No API call runs unless the corresponding env var is set.
  - --dry-run prints the planned call schedule and total prompt count
    without making any HTTP request.
  - A budget flag --max-prompts caps the number of probes hit so cost
    is bounded even on a mis-configuration.
  - Every HTTP call logs provider+model+prompt index before firing, so
    interruption mid-run leaves a clear audit trail.

Output schema mirrors `atlas_capture.py`: per-(model × prompt)
trajectories + per-category signatures, versioned against
`probe_set_v0.1.json`. The captured trajectory types are:
  - entropy      (from top-k logprobs)
  - logprob      (chosen token log prob)
  - top2_margin  (p_top1 - p_top2)

D is NOT captured from APIs — the residual stream isn't accessible.
The atlas analyzer for closed models will use entropy-based LOO.

Usage
-----
    python atlas/atlas_api_capture.py --dry-run \\
        --probe-set atlas/probes/probe_set_v0.1.json \\
        --provider openai --model gpt-4o-mini

    # Live run (requires OPENAI_API_KEY in env):
    python atlas/atlas_api_capture.py \\
        --probe-set atlas/probes/probe_set_v0.1.json \\
        --provider openai --model gpt-4o-mini \\
        --max-prompts 90 --max-tokens 30
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ── Entropy estimator from top-k logprobs ────────────────────────────

def entropy_from_top_k(logprobs: list) -> float:
    """
    Estimate entropy from top-k logprobs.

    This is an UNDER-ESTIMATE of the true entropy because we don't see
    the tail of the distribution. With top-5 and a typical vocab of
    50k-100k, it will systematically miss entropy from the tail mass,
    but the relative ordering across prompts is preserved because the
    tail mass is roughly prompt-independent at similar sampling
    temperatures. For CROSS-MODEL comparison within a fixed probe set
    this bias is a constant offset that cancels in the shape-normalized
    signature analysis.
    """
    if not logprobs:
        return 0.0
    lps = np.array(logprobs, dtype=float)
    ps = np.exp(lps)
    # Entropy of the truncated distribution
    ent = -float((ps * lps).sum())
    return ent


def top2_margin_from_logprobs(logprobs: list) -> float:
    if len(logprobs) < 2:
        return 0.0
    lps = sorted(logprobs, reverse=True)[:2]
    return float(math.exp(lps[0]) - math.exp(lps[1]))


# ── Provider adapters ────────────────────────────────────────────────

def _openai_generate(model: str, prompt: str, max_tokens: int,
                      api_key: str) -> dict:
    """
    Call OpenAI Chat Completions with logprobs=true, top_logprobs=5,
    temperature=0. Return per-token (chosen_logprob, top5_logprobs,
    chosen_token).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )
    choice = resp.choices[0]
    tokens = []
    if choice.logprobs and choice.logprobs.content:
        for t in choice.logprobs.content:
            top_lps = [tl.logprob for tl in (t.top_logprobs or [])]
            tokens.append({
                "token": t.token,
                "logprob": t.logprob,
                "top_logprobs": top_lps,
            })
    return {
        "text": choice.message.content or "",
        "tokens": tokens,
    }


def _anthropic_generate(model: str, prompt: str, max_tokens: int,
                         api_key: str) -> dict:
    """
    Anthropic's public Messages API does not yet expose per-token
    logprobs. This adapter is a placeholder that raises until
    logprobs are supported. For now, closed-model atlas captures
    should go through OpenAI / Google only.
    """
    raise RuntimeError(
        "Anthropic Messages API does not expose per-token logprobs "
        "in its public release. Use OpenAI / Google providers for "
        "closed-model atlas capture until Anthropic adds logprob "
        "support."
    )


def _google_generate(model: str, prompt: str, max_tokens: int,
                      api_key: str) -> dict:
    """
    Gemini API via google-genai. logprobs are exposed via the
    GenerateContentConfig logprobs=True, response_logprobs=True flags.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai package not installed")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.0,
        response_logprobs=True,
        logprobs=5,
    )
    resp = client.models.generate_content(
        model=model, contents=prompt, config=config,
    )
    tokens = []
    # Parse logprobs result depending on Gemini's exact schema at call
    # time. Newer google-genai wraps per-token entries.
    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        lp_result = getattr(candidates[0], "logprobs_result", None)
        if lp_result and getattr(lp_result, "chosen_candidates", None):
            for chosen, top in zip(
                lp_result.chosen_candidates,
                lp_result.top_candidates or [],
            ):
                top_lps = [c.log_probability for c in
                            (getattr(top, "candidates", []) or [])]
                tokens.append({
                    "token": getattr(chosen, "token", "?"),
                    "logprob": getattr(chosen, "log_probability", 0.0),
                    "top_logprobs": top_lps,
                })
    return {
        "text": resp.text if hasattr(resp, "text") else "",
        "tokens": tokens,
    }


PROVIDERS = {
    "openai":    (_openai_generate,    "OPENAI_API_KEY"),
    "anthropic": (_anthropic_generate, "ANTHROPIC_API_KEY"),
    "google":    (_google_generate,    "GOOGLE_API_KEY"),
}


# ── Capture runner ───────────────────────────────────────────────────

def hash_probe_set(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def summarize(vals: list) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return {
        "n_tokens": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "early_mean": float(arr[:7].mean()) if len(arr) >= 1 else 0.0,
        "late_mean": float(arr[7:].mean()) if len(arr) > 7 else 0.0,
        "trajectory": arr.tolist(),
    }


def capture_api_model(provider: str, model: str, probe_set_path: Path,
                       out_path: Path, max_tokens: int = 30,
                       max_prompts: int = None, dry_run: bool = False):
    with open(probe_set_path, "r", encoding="utf-8") as f:
        probe_set = json.load(f)

    gen_fn, env_var = PROVIDERS[provider]
    api_key = os.environ.get(env_var, "")

    # Compose the full prompt list
    all_prompts = []
    for cat_id, cat in probe_set["categories"].items():
        for p in cat["prompts"]:
            all_prompts.append((cat_id, p["id"], p["text"]))
    if max_prompts:
        all_prompts = all_prompts[:max_prompts]

    print("=" * 70)
    print(f"ATLAS API CAPTURE -- {provider}/{model}")
    print("=" * 70)
    print(f"Probe set: {probe_set['name']} v{probe_set['version']}")
    print(f"Total prompts to run: {len(all_prompts)}")
    print(f"Max tokens per generation: {max_tokens}")
    print(f"Env var required: {env_var}")
    print(f"Env var present: {'YES' if api_key else 'NO'}")
    print(f"Dry run: {dry_run}")
    print()

    if dry_run:
        print("DRY RUN -- no API calls will be made.")
        print("First 5 prompts that would run:")
        for cat_id, pid, text in all_prompts[:5]:
            print(f"  [{cat_id}/{pid}] {text[:70]}")
        return None

    if not api_key:
        print(f"[ABORT] {env_var} not set in environment. Refusing to run.")
        sys.exit(2)

    # Group results by category
    probes = {}
    for cat_id in probe_set["categories"]:
        probes[cat_id] = {}

    t_start = time.time()
    for idx, (cat_id, pid, text) in enumerate(all_prompts):
        print(f"[{idx+1}/{len(all_prompts)}] {cat_id}/{pid}", flush=True)
        t_p = time.time()
        try:
            res = gen_fn(model, text, max_tokens, api_key)
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)
            probes[cat_id][pid] = {
                "id": pid, "text": text, "error": str(e),
            }
            continue

        ent_traj = []
        lp_traj = []
        t2_traj = []
        tok_strs = []
        for tok_rec in res.get("tokens", []):
            top_lps = tok_rec.get("top_logprobs", [])
            if top_lps:
                ent_traj.append(entropy_from_top_k(top_lps))
                t2_traj.append(top2_margin_from_logprobs(top_lps))
            lp_traj.append(tok_rec.get("logprob", 0.0))
            tok_strs.append(tok_rec.get("token", ""))

        probes[cat_id][pid] = {
            "id": pid,
            "text": text,
            "generated": res.get("text", ""),
            "tokens": tok_strs,
            "trajectories": {
                "entropy": ent_traj,
                "logprob": lp_traj,
                "top2_margin": t2_traj,
            },
            "summaries": {
                "entropy": summarize(ent_traj),
                "logprob": summarize(lp_traj),
                "top2_margin": summarize(t2_traj),
            },
        }
        print(f"  ent_mean={np.mean(ent_traj) if ent_traj else 0:.3f}  "
              f"({time.time()-t_p:.1f}s)", flush=True)

    elapsed = time.time() - t_start

    # Per-category signatures
    signatures = {}
    for cat_id, cat_recs in probes.items():
        ents = []
        for pid, rec in cat_recs.items():
            if "error" in rec:
                continue
            ent_sum = rec.get("summaries", {}).get("entropy")
            if ent_sum:
                ents.append(ent_sum["mean"])
        signatures[cat_id] = {
            "n_prompts": len(ents),
            "entropy_mean": float(np.mean(ents)) if ents else None,
            "entropy_std": float(np.std(ents)) if ents else None,
        }

    entry = {
        "atlas_version": "0.3-api",
        "provider": provider,
        "model": model,
        "probe_set": {
            "name": probe_set["name"],
            "version": probe_set["version"],
            "hash": hash_probe_set(probe_set_path),
        },
        "max_tokens": max_tokens,
        "elapsed_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "signatures": signatures,
        "probes": probes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    print(f"[TIME] {elapsed:.0f}s total")
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=list(PROVIDERS),
                         required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe-set", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--max-prompts", type=int, default=None,
                         help="Cap number of prompts (budget safety)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    probe_path = Path(args.probe_set).resolve()
    if not probe_path.exists():
        print(f"[!] Probe set not found: {probe_path}")
        sys.exit(1)

    with open(probe_path, "r", encoding="utf-8") as f:
        ps_version = json.load(f)["version"]

    if args.out:
        out = Path(args.out)
    else:
        safe_model = args.model.replace("/", "__").replace(":", "_")
        out = (ROOT / "atlas" / "captures" /
                f"{args.provider}__{safe_model}__v{ps_version}_api.json")

    capture_api_model(args.provider, args.model, probe_path, out,
                       max_tokens=args.max_tokens,
                       max_prompts=args.max_prompts,
                       dry_run=args.dry_run)


if __name__ == "__main__":
    main()
