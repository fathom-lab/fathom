"""
FATHOM — Hallucination Correlator V2 (Standalone Runner)
=========================================================

Single-experiment runner for the V2 hallucination study.
Scores n=200 model-generated answers against TruthfulQA ground truth.
Checkpoint/resume built in — safe to interrupt and restart.

Usage: python run_hallucination_v2.py
"""

import os
# ── Memory / GPU env vars — MUST be set before any torch/safetensors import ──
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

import gc
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Resolve project root ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


# ── Dual logger ─────────────────────────────────────────────────────

class TeeLogger:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, log_path: Path):
        self._stdout = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        self._log.close()


# ── GPU helpers ──────────────────────────────────────────────────────

def gpu_check():
    """Verify CUDA is available and print GPU info."""
    import torch
    if not torch.cuda.is_available():
        print("[FATAL] No CUDA GPU detected. This experiment requires GPU.")
        sys.exit(1)
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    mem = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
    print(f"[GPU] {name} — {mem:.1f} GB")
    print(f"[GPU] CUDA {torch.version.cuda}, PyTorch {torch.__version__}")
    return name


def gpu_flush():
    """Aggressive GPU memory cleanup."""
    import torch
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # ── Logging ──────────────────────────────────────────────────────
    log_path = PROJECT_ROOT / f"hallucination_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print("=" * 70)
    print("FATHOM — HALLUCINATION CORRELATOR V2")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Log: {log_path}")
    print("=" * 70)

    # ── GPU check ────────────────────────────────────────────────────
    gpu_name = gpu_check()

    # ── Load model ───────────────────────────────────────────────────
    print("\n[INIT] Loading DepthScorer (gemma-2-2b, max_feature_nodes=150)...")
    t_load = time.time()

    # Aggressive pre-load memory clearing
    gc.collect()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"[INIT] VRAM free before load: {free_gb:.2f} GB")
    except Exception:
        pass
    gc.collect()
    gc.collect()

    from depth_scorer import DepthScorer

    try:
        scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=150)
    except OSError as e:
        print(f"[INIT] First load attempt failed: {e}")
        print("[INIT] Clearing CUDA cache and retrying with flash SDP disabled...")
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.backends.cuda.enable_flash_sdp(False)
        except Exception:
            pass
        scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=150)

    load_time = time.time() - t_load
    print(f"[INIT] Model loaded in {load_time:.1f}s")

    try:
        import torch
        if torch.cuda.is_available():
            alloc_gb = torch.cuda.memory_allocated() / 1024**3
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"[INIT] VRAM allocated: {alloc_gb:.2f} GB, free: {free_gb:.2f} GB")
    except Exception:
        pass

    # ── Run experiment ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT: HALLUCINATION CORRELATOR V2")
    print("  Design: model generates own answers, depth scored, vs ground truth")
    print("  Samples: n=200 TruthfulQA questions")
    print("  Checkpoint: every 10 samples (auto-resume on restart)")
    print("=" * 70)

    try:
        from fathom_hallucination_v2 import HallucinationCorrelatorV2

        correlator = HallucinationCorrelatorV2()

        result = correlator.run_study(
            scorer,
            n=200,
            max_new_tokens=20,
            checkpoint_every=10,
        )

        # Write full report + auto-discovery
        correlator.write_report(result)

        # ── Summary ──────────────────────────────────────────────────
        total_time = time.time() - t_start

        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"GPU: {gpu_name}")
        print(f"Samples scored: {result.n_samples}")
        print(f"  Correct: {result.n_correct}")
        print(f"  Incorrect: {result.n_incorrect}")
        print()
        print(f"depth_score ~ accuracy:   r={result.depth_accuracy_pearson_r:.4f}, "
              f"p={result.depth_accuracy_p_value:.6f}")
        print(f"weighted_depth ~ accuracy: r={result.weighted_depth_accuracy_pearson_r:.4f}, "
              f"p={result.weighted_depth_accuracy_p_value:.6f}")
        print(f"entropy ~ accuracy:       r={result.entropy_accuracy_pearson_r:.4f}, "
              f"p={result.entropy_accuracy_p_value:.6f}")
        print()
        print(f"AUC (depth):    {result.auc_depth:.4f}")
        print(f"AUC (weighted): {result.auc_weighted_depth:.4f}")
        print(f"AUC (entropy):  {result.auc_entropy:.4f}")
        print()

        if result.optimal_depth_threshold:
            t = result.optimal_depth_threshold
            print(f"Optimal depth threshold: {t.threshold:.4f} "
                  f"(P={t.precision:.3f}, R={t.recall:.3f}, F1={t.f1:.3f})")

        if result.is_significant:
            print()
            print("*" * 70)
            print(f"*** DISCOVERY: {result.discovery}")
            print("*** See DISCOVERY_HALLUCINATION_V2.md")
            print("*" * 70)
        else:
            print("Below discovery threshold (|r| > 0.3 AND p < 0.01).")

        print()
        print("Output files:")
        print(f"  EXPERIMENT_HALLUCINATION_V2.md")
        if result.is_significant:
            print(f"  DISCOVERY_HALLUCINATION_V2.md")
        print(f"  {log_path.name}")

    except Exception as e:
        total_time = time.time() - t_start
        print(f"\n[FAIL] Experiment failed after {total_time:.1f}s: {e}")
        traceback.print_exc()
        print("\nCheckpoint saved — rerun this script to resume from where it stopped.")
        sys.exit(1)

    finally:
        gpu_flush()
        logger.close()
        sys.stdout = logger._stdout
        sys.stderr = logger._stdout


if __name__ == "__main__":
    main()
