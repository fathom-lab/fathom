"""
Fathom — Reasoning Depth Scoring via SAE Circuit Attribution
=============================================================

Measures how deeply a language model processes a prompt by analyzing
sparse autoencoder feature activation distributions across layers.

Quick start:

    from fathom import Fathom

    # Remote (API — no GPU needed)
    f = Fathom(api_url="http://localhost:8400")
    result = f.score("Why does fire spread once it starts?")
    print(result.grade, result.score)

    # Local (GPU required)
    f = Fathom()
    result = f.score("Why does fire spread once it starts?")
    print(result.grade, result.score)

    # Compare surface vs insight
    comparison = f.compare(
        "What is DNA?",
        "Why does DNA replication prevent errors?"
    )
    print(comparison.delta, comparison.direction)

For the low-level scorer:

    from fathom import DepthScorer
    scorer = DepthScorer(model_name="google/gemma-2-2b")
    result = scorer.score("prompt text")
    print(result.depth_score, result.layer_entropy)
"""

__version__ = "0.1.0"
__author__ = "darkflobi"

import sys
from pathlib import Path

# Allow importing from project root (for development installs)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Public API — lazy imports to keep startup fast when GPU modules not needed
# ---------------------------------------------------------------------------

def __getattr__(name):
    """Lazy-load heavy modules on first access."""

    if name == "DepthScorer":
        from depth_scorer import DepthScorer as _DS
        globals()["DepthScorer"] = _DS
        return _DS

    if name == "DepthResult":
        from depth_scorer import DepthResult as _DR
        globals()["DepthResult"] = _DR
        return _DR

    if name in ("FathomScore", "compute_fathom_score"):
        from fathom_score import FathomScore as _FS, compute_fathom_score as _cfs
        globals()["FathomScore"] = _FS
        globals()["compute_fathom_score"] = _cfs
        return globals()[name]

    if name in ("Fathom", "ScoreResult", "CompareResult"):
        from sdk.fathom_sdk import Fathom as _F, ScoreResult as _SR, CompareResult as _CR
        globals()["Fathom"] = _F
        globals()["ScoreResult"] = _SR
        globals()["CompareResult"] = _CR
        return globals()[name]

    if name == "CorrectnessPredictor":
        from fathom_correctness import CorrectnessPredictor as _CP
        globals()["CorrectnessPredictor"] = _CP
        return _CP

    if name == "BenchmarkRunner":
        from fathom_benchmark import BenchmarkRunner as _BR
        globals()["BenchmarkRunner"] = _BR
        return _BR

    raise AttributeError(f"module 'fathom' has no attribute {name!r}")


__all__ = [
    # Core
    "DepthScorer",
    "DepthResult",
    # Score
    "FathomScore",
    "compute_fathom_score",
    # SDK / high-level
    "Fathom",
    "ScoreResult",
    "CompareResult",
    # Correctness
    "CorrectnessPredictor",
    # Benchmark
    "BenchmarkRunner",
    # Meta
    "__version__",
]
