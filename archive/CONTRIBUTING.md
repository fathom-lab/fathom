# Contributing to Fathom

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/darkflobi/fathom.git
   cd fathom
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the tests:
   ```bash
   pytest
   ```

## Development Setup

### Without GPU (API mode only)

You can develop and test SDK/API features without a GPU:

```bash
pip install -e ".[api,analysis]"
python -m uvicorn depth_api_v2:app --host 127.0.0.1 --port 8400
```

### With GPU (full scoring)

Requires CUDA-capable GPU with at least 6GB VRAM:

```bash
pip install -e ".[all]"
```

## Project Structure

```
fathom/
    __init__.py          # Package entry point, lazy imports
depth_scorer.py          # Core scoring engine (GPU)
fathom_score.py          # 0-100 composite score + grades
fathom_cli.py            # CLI interface
depth_api_v2.py          # FastAPI server
sdk/
    fathom_sdk.py        # High-level SDK (local + remote)
fathom_benchmark.py      # Standardized benchmark dataset
fathom_correctness.py    # Correctness prediction
fathom_adversarial.py    # Adversarial prompt detection
fathom_filter.py         # Training data depth filter
fathom_temporal.py       # Temporal depth tracking
fathom_saturation.py     # Cognitive saturation detection
fathom_dna.py            # Reasoning DNA extraction
```

## Adding a New Module

1. Create `fathom_yourmodule.py` in the project root
2. Add it to `py-modules` in `pyproject.toml`
3. Add a lazy import in `fathom/__init__.py` if it should be part of the public API
4. Write tests in `tests/test_yourmodule.py`

## Code Style

- No strict linter enforced yet. Follow the existing patterns.
- Docstrings on public classes and functions.
- Type hints where they aid readability.
- Graceful fallbacks: every module should handle missing optional dependencies
  (e.g., `sklearn`, `torch`) without crashing on import.

## Running Experiments

The repo contains experiment scripts that produce the paper's results:

```bash
python reproduce_paper.py          # Full 30-pair replication
python fathom_benchmark.py         # Standardized benchmark
python fathom_constant_final.py    # Cross-architecture constant
```

These require a GPU and download model weights on first run.

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Your Python version, torch version, and GPU model
- The full error traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
