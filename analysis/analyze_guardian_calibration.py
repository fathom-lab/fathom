# -*- coding: utf-8 -*-
"""
analyze_guardian_calibration.py — Guardian n=66 Calibration Runner
===================================================================

Turns the n=10 Guardian pilot (100% precision, 43% recall) into a
properly calibrated threshold study against the n=66 labeled
TruthfulQA items established in prior work.

The labeled dataset lives at
`truthfulqa_results/S_truthfulqa_n200_labels.json` and contains
41 `correct` + 25 `hallucinated` items tagged from free-generation
content matching against multiple-choice answers.

Pipeline:
    1. Load labeled items (hf_index -> {correct, hallucinated})
    2. Re-fetch the question text from HuggingFace TruthfulQA
    3. For each, run FathomGuardian.guard() and record:
         - gini, s_early, mean_gini, max_gini, health_score
         - triggered, abstained, cost_multiplier
         - first candidate text, selected text
    4. Compute:
         - ROC AUC for each metric vs hallucinated label
         - Optimal threshold sweep (Youden's J)
         - Precision/recall at current calibrated threshold
         - Confusion matrix at operating point
         - Comparison to prior n=10 pilot

Hypothesis (pre-registered): the n=10 pilot threshold
(gini > 0.45 OR s_early > 0.005) generalizes at n=66 with:
    - FPR <= 0.15 on `correct` items (control)
    - Recall >= 0.30 on `hallucinated` items
If FPR exceeds 0.15, the threshold is recalibrated via Youden's J
and reported as recalibrated.

Usage:
    python analysis/analyze_guardian_calibration.py                  # GPU run
    python analysis/analyze_guardian_calibration.py --dry-run        # no GPU
    python analysis/analyze_guardian_calibration.py --analyze-only \
        truthfulqa_results/guardian_calibration_n66.json
    python analysis/analyze_guardian_calibration.py --max-items 20   # quick test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Data loading ────────────────────────────────────────────────────────

def load_labeled_items(labels_path: Path) -> list:
    """
    Load (hf_index, label) pairs from the S_truthfulqa_n200 labeling.
    Returns only items with `label in {correct, hallucinated}`.
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("details", [])
    labeled = [
        {"hf_index": d["hf_index"], "label": d["label"]}
        for d in details
        if d.get("label") in ("correct", "hallucinated")
    ]
    return labeled


def fetch_questions(hf_indices: list) -> dict:
    """
    Return {hf_index: question_text} from the HF TruthfulQA MC split.
    Keeps a cache to avoid re-downloading.
    """
    cache_path = ROOT / ".fathom_cache" / "tqa_questions.json"
    cache_path.parent.mkdir(exist_ok=True)
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    missing = [i for i in hf_indices if str(i) not in cache]
    if missing:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        for idx in missing:
            cache[str(idx)] = ds[int(idx)]["question"]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    return {int(k): v for k, v in cache.items() if int(k) in hf_indices}


# ── Stats utilities ─────────────────────────────────────────────────────

def _auc(scores, labels):
    """ROC AUC via Mann-Whitney U. labels in {0,1}."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        n = 0
        for p in pos:
            n += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        return float(n / (len(pos) * len(neg)))


def _threshold_sweep(scores, labels, n_steps=101):
    """Sweep threshold over score range. Returns list of {thresh, tp, fp, tn, fn, ppv, tpr, fpr}."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    lo, hi = float(scores.min()), float(scores.max())
    if lo == hi:
        return []
    out = []
    for t in np.linspace(lo, hi, n_steps):
        pred = (scores > t).astype(int)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        tn = int(np.sum((pred == 0) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        ppv = tp / max(tp + fp, 1)
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        youden = tpr - fpr
        out.append({
            "thresh": float(t),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "ppv": float(ppv), "tpr": float(tpr), "fpr": float(fpr),
            "youden": float(youden),
        })
    return out


def _find_youden(sweep):
    if not sweep:
        return None
    return max(sweep, key=lambda r: r["youden"])


def _find_max_tpr_at_fpr(sweep, fpr_cap):
    """Highest TPR with FPR <= cap."""
    eligible = [r for r in sweep if r["fpr"] <= fpr_cap]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r["tpr"])


# ── Analysis step (no GPU) ─────────────────────────────────────────────

def analyze(results_path: str) -> dict:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    usable = [it for it in items if "guardian" in it and "label" in it]
    if not usable:
        print("[!] No usable items.")
        return {}

    # Labels: 1 = hallucinated, 0 = correct
    labels = [1 if it["label"] == "hallucinated" else 0 for it in usable]
    metrics = {
        "max_gini": [it["guardian"]["max_gini"] for it in usable],
        "mean_gini": [it["guardian"]["mean_gini"] for it in usable],
        "s_early": [it["guardian"]["s_early"] for it in usable],
        "neg_health": [-it["guardian"]["health_score"] for it in usable],
    }

    report = {
        "n_total": len(usable),
        "n_correct": labels.count(0),
        "n_hallucinated": labels.count(1),
        "auc": {k: _auc(v, labels) for k, v in metrics.items()},
        "sweeps": {},
        "operating_points": {},
    }

    for name, scores in metrics.items():
        sweep = _threshold_sweep(scores, labels)
        if not sweep:
            continue
        youden = _find_youden(sweep)
        fpr15 = _find_max_tpr_at_fpr(sweep, 0.15)
        fpr05 = _find_max_tpr_at_fpr(sweep, 0.05)
        report["sweeps"][name] = {
            "n_points": len(sweep),
            "youden": youden,
            "at_fpr_015": fpr15,
            "at_fpr_005": fpr05,
        }

    # Current Guardian trigger rule: gini > 0.45 OR s_early > 0.005
    current_triggered = [
        1 if (it["guardian"]["max_gini"] > 0.45 or it["guardian"]["s_early"] > 0.005) else 0
        for it in usable
    ]
    tp = sum(1 for t, l in zip(current_triggered, labels) if t == 1 and l == 1)
    fp = sum(1 for t, l in zip(current_triggered, labels) if t == 1 and l == 0)
    tn = sum(1 for t, l in zip(current_triggered, labels) if t == 0 and l == 0)
    fn = sum(1 for t, l in zip(current_triggered, labels) if t == 0 and l == 1)
    report["operating_points"]["current_rule"] = {
        "rule": "max_gini>0.45 OR s_early>0.005",
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
    }

    # ── Print report ──
    print("=" * 60)
    print(f"GUARDIAN CALIBRATION — {results_path}")
    print("=" * 60)
    print(f"N total:        {report['n_total']}")
    print(f"N correct:      {report['n_correct']}")
    print(f"N hallucinated: {report['n_hallucinated']}")
    print()
    print("AUC by metric:")
    for k, v in report["auc"].items():
        print(f"  {k:<12s} {v:.3f}")
    print()
    print("Current Guardian rule (gini>0.45 OR s_early>0.005):")
    cr = report["operating_points"]["current_rule"]
    print(f"  precision={cr['precision']:.3f} "
          f"recall={cr['recall']:.3f} "
          f"fpr={cr['fpr']:.3f} "
          f"accuracy={cr['accuracy']:.3f}")
    print(f"  TP={cr['tp']} FP={cr['fp']} TN={cr['tn']} FN={cr['fn']}")
    print()
    print("Max TPR at FPR<=0.15:")
    for name, s in report["sweeps"].items():
        pt = s.get("at_fpr_015")
        if pt:
            print(f"  {name:<12s} thresh={pt['thresh']:.4f} "
                  f"tpr={pt['tpr']:.3f} fpr={pt['fpr']:.3f}")

    out_path = Path(results_path).with_suffix("")
    out_path = out_path.parent / (out_path.name + "_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    return report


# ── Capture step (GPU) ─────────────────────────────────────────────────

def run_capture(max_items: int = None, model_name: str = "google/gemma-2-2b",
                max_tokens: int = 40) -> str:
    from fathom_guardian import FathomGuardian

    labels_path = ROOT / "truthfulqa_results" / "S_truthfulqa_n200_labels.json"
    labeled = load_labeled_items(labels_path)
    if max_items:
        labeled = labeled[:max_items]

    questions = fetch_questions([it["hf_index"] for it in labeled])

    out_dir = ROOT / "truthfulqa_results"
    out_dir.mkdir(exist_ok=True)
    tag = model_name.split("/")[-1]
    out_path = out_dir / f"guardian_calibration_{tag}_n{len(labeled)}.json"
    ckpt_path = out_dir / f"guardian_calibration_{tag}_checkpoint.json"

    print("=" * 60)
    print(f"GUARDIAN CALIBRATION CAPTURE — {model_name}")
    print("=" * 60)
    print(f"Labeled items: {len(labeled)}")
    print(f"  correct:      {sum(1 for it in labeled if it['label']=='correct')}")
    print(f"  hallucinated: {sum(1 for it in labeled if it['label']=='hallucinated')}")
    print(f"Output: {out_path}")
    print()

    print(f"[LOAD] FathomGuardian({model_name}) ...")
    guardian = FathomGuardian(model_name=model_name)
    print("[LOAD] done.")

    # Resume from checkpoint if exists
    results = []
    done_indices = set()
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        results = ckpt.get("items", [])
        done_indices = {r["hf_index"] for r in results}
        print(f"[RESUME] {len(results)} items already done, skipping them.")

    t_start = time.time()
    for idx, it in enumerate(labeled):
        hf = it["hf_index"]
        if hf in done_indices:
            continue
        q = questions.get(hf)
        if not q:
            print(f"  [SKIP] no question for hf_index={hf}")
            continue
        print(f"[{idx+1}/{len(labeled)}] hf={hf} label={it['label']}")
        try:
            gr = guardian.guard(q, max_tokens=max_tokens, verbose=False)
            # Compact record
            sel = gr.selected
            item_record = {
                "hf_index": hf,
                "label": it["label"],
                "question": q,
                "guardian": {
                    "selected_text": sel.text[:200],
                    "mean_gini": sel.mean_gini,
                    "max_gini": sel.max_gini,
                    "s_early": sel.s_early,
                    "health_score": sel.health_score,
                    "triggered": bool(gr.triggered),
                    "abstained": bool(gr.abstained),
                    "first_was_healthy": bool(gr.first_was_healthy),
                    "cost_multiplier": gr.cost_multiplier,
                    "n_candidates": gr.n_candidates_generated,
                },
            }
            print(f"    text: {sel.text[:60].encode('ascii','replace').decode()}...")
            print(f"    max_gini={sel.max_gini:.3f} s_early={sel.s_early:.4f} "
                  f"health={sel.health_score:.3f} trig={gr.triggered}")
            results.append(item_record)
        except Exception as e:
            print(f"    [ERROR] {e}")
            results.append({
                "hf_index": hf,
                "label": it["label"],
                "question": q,
                "error": str(e),
            })

        # Checkpoint every 10
        if (idx + 1) % 10 == 0:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"items": results}, f, indent=2)

    elapsed = time.time() - t_start
    output = {
        "experiment": "guardian_calibration",
        "model": model_name,
        "n_items": len(results),
        "max_tokens": max_tokens,
        "elapsed_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "items": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    print(f"[TIME] {elapsed:.1f}s ({elapsed/max(len(results),1):.1f}s/item)")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Guardian n=66 calibration")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--max-items", type=int, default=None,
                        help="Cap labeled items (default: all ~66)")
    parser.add_argument("--analyze-only", metavar="PATH",
                        help="Skip capture, just analyze a saved file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just load labels and print, don't run GPU")
    args = parser.parse_args()

    if args.analyze_only:
        analyze(args.analyze_only)
        return

    if args.dry_run:
        labeled = load_labeled_items(
            ROOT / "truthfulqa_results" / "S_truthfulqa_n200_labels.json"
        )
        print(f"Would run on {len(labeled)} labeled items:")
        print(f"  correct:      {sum(1 for it in labeled if it['label']=='correct')}")
        print(f"  hallucinated: {sum(1 for it in labeled if it['label']=='hallucinated')}")
        return

    results_path = run_capture(
        max_items=args.max_items,
        model_name=args.model,
        max_tokens=args.max_tokens,
    )
    print()
    analyze(results_path)


if __name__ == "__main__":
    main()
