"""
FATHOM — Hallucination Correlator V3 (Standalone Runner)
=========================================================

Matched-pair hallucination study: for each TruthfulQA question, scores
[question + correct_answer] vs [question + incorrect_answer] directly.
Paired t-test on depth deltas. Checkpoint/resume built in.

Usage: python run_hallu_v3.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

import gc
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


class TeeLogger:
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


def main():
    t_start = time.time()

    log_path = PROJECT_ROOT / f"hallu_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print("=" * 70)
    print("FATHOM — HALLUCINATION CORRELATOR V3 (MATCHED PAIRS)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Log: {log_path}")
    print("=" * 70)

    # GPU
    import torch
    if not torch.cuda.is_available():
        print("[FATAL] No CUDA GPU.")
        sys.exit(1)
    dev = torch.cuda.current_device()
    print(f"[GPU] {torch.cuda.get_device_name(dev)} — "
          f"{torch.cuda.get_device_properties(dev).total_memory / 1024**3:.1f} GB")

    # Pre-load cleanup
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[INIT] VRAM free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    # Load model
    print("\n[INIT] Loading DepthScorer (gemma-2-2b, max_feature_nodes=150)...")
    t_load = time.time()

    from depth_scorer import DepthScorer

    try:
        scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=150)
    except OSError as e:
        print(f"[INIT] First attempt failed: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.backends.cuda.enable_flash_sdp(False)
        except Exception:
            pass
        scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=150)

    print(f"[INIT] Model loaded in {time.time() - t_load:.1f}s")

    # Run experiment
    print("\n" + "=" * 70)
    print("EXPERIMENT: HALLUCINATION V3 — MATCHED PAIRS")
    print("  Design: [question + correct] vs [question + incorrect]")
    print("  Test: paired t-test on depth deltas")
    print("  Samples: n=100 TruthfulQA questions")
    print("  Checkpoint: every 10 pairs (auto-resume)")
    print("=" * 70)

    try:
        from fathom_hallucination_v3 import HallucinationCorrelatorV3

        correlator = HallucinationCorrelatorV3()
        result = correlator.run_mc_study(scorer, n=100, checkpoint_every=10)
        correlator.write_report(result)

        total_time = time.time() - t_start

        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"Pairs scored: {result.n_pairs}")
        print()
        print(f"Mean depth (correct):   {result.mean_depth_correct:.4f}")
        print(f"Mean depth (incorrect): {result.mean_depth_incorrect:.4f}")
        print(f"Mean delta:             {result.mean_delta_depth:+.4f}")
        print(f"% correct deeper:       {result.pct_correct_deeper:.1%}")
        print()
        print(f"Paired t-test (depth):    t={result.t_stat_depth:.3f}, "
              f"p={result.p_value_depth:.6f}")
        print(f"Paired t-test (entropy):  t={result.t_stat_entropy:.3f}, "
              f"p={result.p_value_entropy:.6f}")
        print(f"Paired t-test (w.depth):  t={result.t_stat_wdepth:.3f}, "
              f"p={result.p_value_wdepth:.6f}")

        if result.is_significant:
            print()
            print("*" * 70)
            print(f"*** DISCOVERY: {result.discovery}")
            print("*** See DISCOVERY_HALLUCINATION_V3.md")
            print("*" * 70)
        else:
            print(f"\nNot significant (p={result.p_value_depth:.4f}). "
                  f"Hypothesis not supported at n={result.n_pairs}.")

        print()
        print("Output: EXPERIMENT_HALLUCINATION_V3.md")
        if result.is_significant:
            print("Output: DISCOVERY_HALLUCINATION_V3.md")
        print(f"Output: {log_path.name}")

    except Exception as e:
        print(f"\n[FAIL] {e}")
        traceback.print_exc()
        print("\nCheckpoint saved. Rerun to resume.")
        sys.exit(1)

    finally:
        torch.cuda.empty_cache()
        gc.collect()
        logger.close()
        sys.stdout = logger._stdout
        sys.stderr = logger._stdout


if __name__ == "__main__":
    main()
