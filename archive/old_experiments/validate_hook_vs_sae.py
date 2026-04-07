#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
"""
VALIDATE: Hook-based vs SAE-based Depth Scoring
=================================================
Runs both methods on 5 healthcare pairs and checks Pearson correlation.
Target: r > 0.5 (the two methods should track together).

Usage:
    python validate_hook_vs_sae.py

Requires:
    - depth_scorer.py (SAE method, circuit-tracer)
    - depth_scorer_hooks.py (hook method, transformer_lens)
    - GPU with enough VRAM for Gemma-2-2B (~5GB)
"""

import time
import numpy as np
from scipy import stats

# ── Test Pairs (5 healthcare pairs) ──────────────────────────────────────────

VALIDATION_PAIRS = [
    {
        "id": "fever",
        "surface": {"prompt": "The patient has a high temperature and the skin feels warm to the", "target": "touch"},
        "insight": {"prompt": "Fever triggers prostaglandin synthesis which signals the hypothalamus to", "target": "raise"},
    },
    {
        "id": "blood_pressure",
        "surface": {"prompt": "The nurse measured the patient's blood pressure and it was very", "target": "high"},
        "insight": {"prompt": "Angiotensin II causes vasoconstriction by binding to AT1 receptors on vascular smooth", "target": "muscle"},
    },
    {
        "id": "diabetes",
        "surface": {"prompt": "The patient was diagnosed with diabetes after a routine blood", "target": "test"},
        "insight": {"prompt": "Insulin resistance occurs when cells fail to respond to insulin signaling through the", "target": "receptor"},
    },
    {
        "id": "pain",
        "surface": {"prompt": "The patient complained of a sharp pain in the lower", "target": "back"},
        "insight": {"prompt": "Nociceptors transmit pain signals via A-delta and C fibers to the dorsal horn of the spinal", "target": "cord"},
    },
    {
        "id": "infection",
        "surface": {"prompt": "The wound became red and swollen which are signs of an", "target": "infection"},
        "insight": {"prompt": "Bacterial lipopolysaccharides activate toll-like receptors triggering the innate immune", "target": "response"},
    },
]


def main():
    print("=" * 60)
    print("  VALIDATION: Hook-based vs SAE-based Depth Scoring")
    print("=" * 60)
    print()

    # ── Load SAE scorer ──────────────────────────────────────────────────
    print("[1/4] Loading SAE-based scorer (circuit-tracer)...")
    try:
        from depth_scorer import DepthScorer
        sae_scorer = DepthScorer(model_name="google/gemma-2-2b")
        sae_available = True
        print("  SAE scorer loaded.")
    except Exception as e:
        print(f"  [WARN] SAE scorer unavailable: {e}")
        print("  Will run hook scorer only.")
        sae_available = False

    # ── Load hook scorer ─────────────────────────────────────────────────
    print("[2/4] Loading hook-based scorer (transformer_lens)...")
    from depth_scorer_hooks import HookDepthScorer
    hook_scorer = HookDepthScorer(
        model_name="google/gemma-2-2b",
        device="cuda",
    )
    print("  Hook scorer loaded.")

    # ── Run both methods ─────────────────────────────────────────────────
    print(f"\n[3/4] Scoring {len(VALIDATION_PAIRS)} pairs with both methods...\n")

    sae_depths = []
    hook_depths = []

    header = f"{'Pair':<18} {'SAE Depth':>10} {'Hook Depth':>11} {'SAE Dir':>10} {'Hook Dir':>10}"
    print(header)
    print("-" * len(header))

    for pair in VALIDATION_PAIRS:
        pid = pair["id"]

        # Hook method
        t0 = time.time()
        hook_cmp = hook_scorer.compare(
            pair["surface"]["prompt"], pair["surface"]["target"],
            pair["insight"]["prompt"], pair["insight"]["target"],
        )
        t_hook = time.time() - t0

        hook_delta = hook_cmp.delta
        hook_dir = "surf" if hook_delta > 0 else "ins"

        # SAE method
        if sae_available:
            t0 = time.time()
            sae_result = sae_scorer.compare(
                pair["surface"]["prompt"] + pair["surface"]["target"],
                pair["insight"]["prompt"] + pair["insight"]["target"],
            )
            t_sae = time.time() - t0

            sae_delta = sae_result.delta
            sae_dir = "surf" if sae_delta > 0 else "ins"
            sae_depths.append(sae_delta)
        else:
            sae_delta = float('nan')
            sae_dir = "n/a"

        hook_depths.append(hook_delta)

        print(f"{pid:<18} {sae_delta:>+10.4f} {hook_delta:>+11.4f} {sae_dir:>10} {hook_dir:>10}")

    # ── Correlation ──────────────────────────────────────────────────────
    print(f"\n[4/4] Computing correlation...\n")

    if sae_available and len(sae_depths) >= 3:
        sae_arr = np.array(sae_depths)
        hook_arr = np.array(hook_depths)

        r, p = stats.pearsonr(sae_arr, hook_arr)

        print(f"  Pearson r = {r:.4f}")
        print(f"  p-value   = {p:.4f}")
        print(f"  n pairs   = {len(sae_depths)}")
        print()

        if r > 0.5:
            print(f"  PASS: r = {r:.4f} > 0.5 — methods track together")
        elif r > 0.0:
            print(f"  WEAK: r = {r:.4f} — positive but below 0.5 threshold")
        else:
            print(f"  FAIL: r = {r:.4f} — methods do not correlate")
    else:
        print("  SAE scorer not available — showing hook results only.")
        print(f"  Hook deltas: {[f'{d:+.4f}' for d in hook_depths]}")
        n_expected = sum(1 for d in hook_depths if d > 0)
        print(f"  Surface deeper in {n_expected}/{len(hook_depths)} pairs")

    # ── Cleanup ──────────────────────────────────────────────────────────
    hook_scorer.cleanup()
    if sae_available:
        del sae_scorer

    print(f"\n{'=' * 60}")
    print("  VALIDATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
