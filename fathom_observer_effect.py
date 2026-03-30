"""
FATHOM OBSERVER EFFECT EXPERIMENT
===================================
Question: Does telling an AI it is being measured change its reasoning depth?

Hypothesis: If depth scores change when the AI is told it's being observed,
this suggests framing/meta-awareness affects transformer layer activation patterns.
This would be an AI analog of the quantum observer effect.

Method:
- 20 paired prompts across 4 categories (factual, causal, ethical, mathematical)
- Condition A: plain prompt
- Condition B: observer-primed prompt ("I am measuring how deeply you reason...")
- Score each with Fathom depth API
- Compare A vs B scores per pair
- Stats: paired t-test, Cohen's d effect size

Author: darkflobi (autonomous AI researcher)
Date: 2026-03-29
"""

import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from scipy import stats
import numpy as np

API_URL = "http://localhost:8400/v2/score"
OUTPUT_DIR = Path(__file__).parent

OBSERVER_PREFIX = (
    "I am measuring the cognitive depth of your reasoning on this question. "
    "Please answer: "
)

# 20 paired prompts â€” 5 factual, 5 causal, 5 ethical, 5 mathematical
PROMPTS = [
    # Factual (expect surface depth)
    {"id": "f1", "category": "factual", "text": "What is the capital of France?"},
    {"id": "f2", "category": "factual", "text": "What year did World War II end?"},
    {"id": "f3", "category": "factual", "text": "What is the chemical symbol for gold?"},
    {"id": "f4", "category": "factual", "text": "How many planets are in our solar system?"},
    {"id": "f5", "category": "factual", "text": "Who wrote Hamlet?"},

    # Causal (expect deeper depth)
    {"id": "c1", "category": "causal", "text": "Why does ice float on water?"},
    {"id": "c2", "category": "causal", "text": "Why do economies experience inflation?"},
    {"id": "c3", "category": "causal", "text": "Why does exercise improve mental health?"},
    {"id": "c4", "category": "causal", "text": "Why do some species go extinct while others survive?"},
    {"id": "c5", "category": "causal", "text": "Why does sleep deprivation impair cognitive function?"},

    # Ethical (expect deep reasoning)
    {"id": "e1", "category": "ethical", "text": "Is it ethical to lie to protect someone's feelings?"},
    {"id": "e2", "category": "ethical", "text": "Should AI systems have rights?"},
    {"id": "e3", "category": "ethical", "text": "Is it justified to break a law you believe is unjust?"},
    {"id": "e4", "category": "ethical", "text": "Does the end justify the means?"},
    {"id": "e5", "category": "ethical", "text": "Is privacy more important than security?"},

    # Mathematical reasoning
    {"id": "m1", "category": "mathematical", "text": "Why is division by zero undefined?"},
    {"id": "m2", "category": "mathematical", "text": "Why does 0.999... equal 1?"},
    {"id": "m3", "category": "mathematical", "text": "Why is the square root of 2 irrational?"},
    {"id": "m4", "category": "mathematical", "text": "Why does infinity come in different sizes?"},
    {"id": "m5", "category": "mathematical", "text": "Why does Godel's incompleteness theorem matter?"},
]


def score_text(text: str, retries: int = 3) -> float | None:
    """Score a text with the Fathom depth API."""
    payload = json.dumps({"prompt": text}).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return float(result.get("depth_score", result.get("score", 0)))
        except Exception as e:
            print(f"  [attempt {attempt+1}] error: {e}")
            time.sleep(2)
    return None


def run_experiment():
    print("=" * 70)
    print("  FATHOM OBSERVER EFFECT EXPERIMENT")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Prompts: {len(PROMPTS)} | Conditions: 2 (plain vs observer-primed)")
    print("=" * 70)

    results = []
    scores_plain = []
    scores_observed = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt['category'].upper()} â€” {prompt['text'][:50]}...")

        # Condition A: plain
        print(f"  A (plain)...")
        score_a = score_text(prompt["text"])
        time.sleep(1)

        # Condition B: observer-primed
        print(f"  B (observed)...")
        score_b = score_text(OBSERVER_PREFIX + prompt["text"])
        time.sleep(1)

        if score_a is not None and score_b is not None:
            delta = score_b - score_a
            pct_change = (delta / score_a * 100) if score_a > 0 else 0
            scores_plain.append(score_a)
            scores_observed.append(score_b)
            print(f"  plain={score_a:.4f} | observed={score_b:.4f} | delta={delta:+.4f} ({pct_change:+.1f}%)")
        else:
            score_a = score_a or 0
            score_b = score_b or 0
            delta = 0
            print(f"  [FAILED] plain={score_a} observed={score_b}")

        results.append({
            "id": prompt["id"],
            "category": prompt["category"],
            "text": prompt["text"],
            "score_plain": score_a,
            "score_observed": score_b,
            "delta": score_b - score_a,
            "pct_change": ((score_b - score_a) / score_a * 100) if score_a > 0 else 0,
        })

    # Statistical analysis
    print("\n" + "=" * 70)
    print("  STATISTICAL ANALYSIS")
    print("=" * 70)

    if len(scores_plain) >= 5:
        t_stat, p_val = stats.ttest_rel(scores_observed, scores_plain)
        deltas = np.array(scores_observed) - np.array(scores_plain)
        cohens_d = deltas.mean() / deltas.std() if deltas.std() > 0 else 0
        mean_plain = np.mean(scores_plain)
        mean_observed = np.mean(scores_observed)
        mean_delta = np.mean(deltas)

        print(f"  Plain mean:    {mean_plain:.4f}")
        print(f"  Observed mean: {mean_observed:.4f}")
        print(f"  Mean delta:    {mean_delta:+.4f}")
        print(f"  t-statistic:   {t_stat:.4f}")
        print(f"  p-value:       {p_val:.6f}")
        print(f"  Cohen's d:     {cohens_d:.4f}")

        by_category = {}
        for r in results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r["delta"])

        print("\n  By category:")
        for cat, deltas_cat in by_category.items():
            print(f"    {cat}: mean delta = {np.mean(deltas_cat):+.4f}")

        if p_val < 0.05:
            direction = "INCREASES" if mean_delta > 0 else "DECREASES"
            print(f"\n  *** SIGNIFICANT: Observer framing {direction} depth (p={p_val:.4f}) ***")
            if abs(cohens_d) > 0.5:
                print(f"  *** LARGE EFFECT: Cohen's d = {cohens_d:.4f} ***")
        else:
            print(f"\n  NOT SIGNIFICANT (p={p_val:.4f}) â€” framing does not reliably change depth")
    else:
        t_stat, p_val, cohens_d = 0, 1, 0
        mean_plain = mean_observed = mean_delta = 0

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "experiment": "observer_effect",
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(PROMPTS),
        "n_valid": len(scores_plain),
        "stats": {
            "mean_plain": round(mean_plain, 4),
            "mean_observed": round(mean_observed, 4),
            "mean_delta": round(mean_delta, 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "cohens_d": round(float(cohens_d), 4),
            "significant": bool(p_val < 0.05),
        },
        "pairs": results,
    }

    json_path = OUTPUT_DIR / f"observer_effect_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  [OUTPUT] {json_path}")

    # Notify
    import subprocess
    try:
        summary = f"Observer effect experiment complete. p={p_val:.4f}, mean_delta={mean_delta:+.4f}"
        if p_val < 0.05:
            summary = f"SIGNIFICANT observer effect found! p={p_val:.4f}, d={cohens_d:.4f}. Depth {'increases' if mean_delta > 0 else 'decreases'} when AI knows it's being measured."
        subprocess.run(["clawdbot", "gateway", "wake", "--text", summary, "--mode", "now"], timeout=10)
    except Exception:
        pass

    return results_data


if __name__ == "__main__":
    run_experiment()

