"""
FATHOM CLI — Command-Line Reasoning Depth Scorer
==================================================

Quick scoring without running the full API server.

Usage:
  python fathom_cli.py score "Why does fire spread once it starts?"
  python fathom_cli.py compare "What is DNA?" "Why does DNA replicate?"
  python fathom_cli.py batch prompts.txt
  python fathom_cli.py serve
"""

import gc
import math
import os
import sys
import time
from pathlib import Path

# ── Memory env vars before any torch import ─────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Color helpers (graceful fallback) ────────────────────────────────

try:
    import colorama
    colorama.init()
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False


def _c(text: str, code: str) -> str:
    if not _HAS_COLOR:
        return text
    return f"{code}{text}\033[0m"

def _bold(t):   return _c(t, "\033[1m")
def _green(t):  return _c(t, "\033[32m")
def _yellow(t): return _c(t, "\033[33m")
def _red(t):    return _c(t, "\033[31m")
def _cyan(t):   return _c(t, "\033[36m")
def _dim(t):    return _c(t, "\033[2m")

def _grade_color(grade: str, text: str) -> str:
    if grade == "S":
        return _c(text, "\033[1;35m")  # bold magenta
    elif grade == "A":
        return _c(text, "\033[1;32m")  # bold green
    elif grade == "B":
        return _green(text)
    elif grade == "C":
        return _yellow(text)
    elif grade == "D":
        return _c(text, "\033[33m")
    else:
        return _red(text)


# ── Model loader ────────────────────────────────────────────────────

_scorer = None

def _load_scorer():
    global _scorer
    if _scorer is not None:
        return _scorer

    import torch

    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from depth_scorer import DepthScorer

    print(_dim("Loading model..."), flush=True)
    t0 = time.time()

    try:
        _scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=150)
    except OSError:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.enable_flash_sdp(False)
        _scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=150)

    elapsed = time.time() - t0
    print(_dim(f"Model loaded in {elapsed:.1f}s\n"), flush=True)
    return _scorer


def _gpu_flush():
    import torch
    torch.cuda.empty_cache()
    gc.collect()


# ── Display helpers ──────────────────────────────────────────────────

def _print_score_result(depth_result, fathom_s):
    """Print a scored result with formatting."""
    prompt_display = depth_result.prompt
    if len(prompt_display) > 80:
        prompt_display = prompt_display[:77] + "..."

    grade_str = _grade_color(fathom_s.grade, f" {fathom_s.grade} ")
    score_str = _grade_color(fathom_s.grade, f"{fathom_s.score:.0f}/100")

    print(f"\n  {_bold('Prompt:')}  {prompt_display}")
    print(f"  {_bold('Score:')}   {score_str}  [{grade_str}]")
    print(f"  {_bold('Depth:')}   {depth_result.depth_score:.3f}  "
          f"{_dim('(lower = deeper)')}")

    if depth_result.weighted_depth_score is not None:
        print(f"  {_bold('W.Depth:')} {depth_result.weighted_depth_score:.3f}  "
              f"{_dim('(attribution-weighted)')}")

    if depth_result.layer_entropy is not None:
        print(f"  {_bold('Entropy:')} {depth_result.layer_entropy:.3f}  "
              f"{_dim('(higher = more distributed)')}")

    print(f"  {_bold('Layers:')}  "
          f"early={depth_result.early_ratio:.0%}  "
          f"mid={depth_result.mid_ratio:.0%}  "
          f"late={depth_result.late_ratio:.0%}")
    print(f"  {_bold('Features:')} {depth_result.feature_count}  "
          f"{_dim(f'({depth_result.compute_time_s:.1f}s)')}")

    if depth_result.is_anomalous:
        reasons = ", ".join(depth_result.anomaly_reasons or [])
        print(f"  {_red('ANOMALOUS:')} {reasons}")

    print(f"\n  {fathom_s.interpretation}")


# ── Commands ─────────────────────────────────────────────────────────

def cmd_score(prompt: str):
    """Score a single prompt."""
    from fathom_score import compute_fathom_score

    scorer = _load_scorer()
    result = scorer.score(prompt)
    fs = compute_fathom_score(result)
    _print_score_result(result, fs)
    _gpu_flush()


def cmd_compare(surface: str, insight: str):
    """Compare two prompts."""
    from fathom_score import compute_fathom_score

    scorer = _load_scorer()
    comp = scorer.compare(surface, insight)
    fs_surface = compute_fathom_score(comp.surface)
    fs_insight = compute_fathom_score(comp.insight)

    print(f"\n  {_bold('=== SURFACE ===')}")
    _print_score_result(comp.surface, fs_surface)

    print(f"\n  {_bold('=== INSIGHT ===')}")
    _print_score_result(comp.insight, fs_insight)

    print(f"\n  {'─' * 50}")
    delta_str = f"{comp.delta:+.3f}"
    if comp.delta > 0:
        print(f"  {_bold('Delta:')} {_green(delta_str)}  "
              f"{_dim('(positive = insight is deeper)')}")
    else:
        print(f"  {_bold('Delta:')} {_yellow(delta_str)}  "
              f"{_dim('(negative = surface is deeper — unexpected)')}")
    print(f"  {_bold('Feature ratio:')} {comp.feature_count_ratio:.2f}x")

    _gpu_flush()


def cmd_batch(filepath: str):
    """Score prompts from a file (one per line)."""
    from fathom_score import compute_fathom_score

    path = Path(filepath)
    if not path.exists():
        print(f"{_red('Error:')} File not found: {filepath}")
        sys.exit(1)

    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.strip().startswith("#")]

    if not prompts:
        print(f"{_red('Error:')} No prompts found in {filepath}")
        sys.exit(1)

    print(f"\n  Scoring {len(prompts)} prompts from {filepath}\n")

    scorer = _load_scorer()

    # Header
    print(f"  {'#':>3}  {'Score':>5}  {'Grade':>5}  {'Depth':>7}  "
          f"{'Entropy':>7}  {'Prompt'}")
    print(f"  {'─' * 3}  {'─' * 5}  {'─' * 5}  {'─' * 7}  "
          f"{'─' * 7}  {'─' * 40}")

    scores = []
    for i, prompt in enumerate(prompts):
        try:
            result = scorer.score(prompt)
            fs = compute_fathom_score(result)
            scores.append(fs.score)

            prompt_short = prompt[:45] + "..." if len(prompt) > 45 else prompt
            grade_str = _grade_color(fs.grade, f"  {fs.grade}  ")

            depth_str = f"{result.depth_score:.3f}" if not math.isnan(result.depth_score) else "  N/A"
            entropy_str = f"{result.layer_entropy:.3f}" if result.layer_entropy is not None else "  N/A"

            print(f"  {i+1:>3}  {fs.score:>5.0f}  {grade_str}  "
                  f"{depth_str:>7}  {entropy_str:>7}  {prompt_short}")
        except Exception as e:
            print(f"  {i+1:>3}  {_red('FAIL')}                          "
                  f"{prompt[:40]}... ({e})")
        _gpu_flush()

    if scores:
        import numpy as np
        arr = np.array(scores)
        print(f"\n  {'─' * 60}")
        print(f"  {_bold('Summary:')} {len(scores)} scored  |  "
              f"mean={arr.mean():.1f}  std={arr.std():.1f}  "
              f"min={arr.min():.0f}  max={arr.max():.0f}")


def cmd_serve():
    """Start the Fathom API server."""
    print(f"\n  Starting Fathom API server on port 8400...")
    print(f"  {_dim('uvicorn depth_api_v2:app --host 0.0.0.0 --port 8400')}\n")

    import uvicorn
    uvicorn.run(
        "depth_api_v2:app",
        host="0.0.0.0",
        port=8400,
        log_level="info",
    )


# ── Usage ────────────────────────────────────────────────────────────

USAGE = """
  Fathom CLI — Reasoning Depth Scorer

  Usage:
    python fathom_cli.py score   "prompt text"
    python fathom_cli.py compare "surface prompt" "insight prompt"
    python fathom_cli.py batch   prompts.txt
    python fathom_cli.py serve
"""


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "score":
        if len(sys.argv) < 3:
            print(f"{_red('Error:')} score requires a prompt argument")
            print(f"  Usage: python fathom_cli.py score \"your prompt here\"")
            sys.exit(1)
        cmd_score(sys.argv[2])

    elif command == "compare":
        if len(sys.argv) < 4:
            print(f"{_red('Error:')} compare requires two prompt arguments")
            print(f'  Usage: python fathom_cli.py compare "surface" "insight"')
            sys.exit(1)
        cmd_compare(sys.argv[2], sys.argv[3])

    elif command == "batch":
        if len(sys.argv) < 3:
            print(f"{_red('Error:')} batch requires a file path")
            print(f"  Usage: python fathom_cli.py batch prompts.txt")
            sys.exit(1)
        cmd_batch(sys.argv[2])

    elif command == "serve":
        cmd_serve()

    else:
        print(f"{_red('Error:')} Unknown command '{command}'")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
