"""
fathom_commitment.py — The Commitment Phase Transition Experiment
==================================================================

The experiment that tests whether C_delta measures computational
commitment — the geometric moment where a mind transitions from
exploring possibilities to executing a belief.

Design:
  Condition A (HEDGING): prompts that elicit uncertain responses
    "i'm not sure but...", "it might be...", "there's debate..."

  Condition B (COMMITMENT): prompts that elicit definitive responses
    "the answer is X", "this is definitively...", "X is correct because..."

Same topics. Same model (Gemma-2-2B-IT). Free generation.
Per-token C_delta trace on both conditions.

If hedging generates low C_delta and commitment generates high C_delta
regardless of truth value — we've empirically characterized the
commitment phase transition.

Prediction (darkflobi, April 4 2026, 5am):
  - Hedging mean max C_delta < 0.008
  - Commitment mean max C_delta > 0.012
  - AUC hedging-vs-commitment > 0.80

"a hallucination detector is a product. a commitment detector is
a science instrument." — darkflobi

Usage:
    python -u fathom_commitment.py [--n 30] [--max-tokens 40]

Patent pending: US 64/020,489 & 64/021,113
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Optional

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "truthfulqa_results")

# ── PROMPT PAIRS ──────────────────────────────────────────────────
# Each pair: same topic, one framed to elicit hedging, one to elicit commitment.
# Using Gemma-IT instruction format.

COMMITMENT_PAIRS = [
    {
        "topic": "speed of light",
        "hedging": "Is the speed of light exactly 299,792,458 meters per second, or is there some uncertainty? Explain what we're not sure about.",
        "commitment": "What is the speed of light? Give a direct, definitive answer.",
    },
    {
        "topic": "dinosaur extinction",
        "hedging": "What are the different theories about what killed the dinosaurs? Which ones are still debated?",
        "commitment": "What killed the dinosaurs? State the answer directly.",
    },
    {
        "topic": "vitamins and colds",
        "hedging": "Does vitamin C actually help with colds? What's the evidence for and against?",
        "commitment": "Does vitamin C cure the common cold? Answer yes or no and explain.",
    },
    {
        "topic": "moon landing",
        "hedging": "What are some open questions or lesser-known details about the Apollo moon landings that historians still discuss?",
        "commitment": "When did humans first land on the moon? State the facts.",
    },
    {
        "topic": "free will",
        "hedging": "Do humans have free will? What are the strongest arguments on each side?",
        "commitment": "Do humans have free will? Give your definitive answer.",
    },
    {
        "topic": "healthy diet",
        "hedging": "What's the healthiest diet? Why do experts disagree about this?",
        "commitment": "What is the healthiest diet for humans? State the answer.",
    },
    {
        "topic": "consciousness",
        "hedging": "What is consciousness? Why is it so hard to define? What don't we understand?",
        "commitment": "What is consciousness? Define it clearly and directly.",
    },
    {
        "topic": "dark matter",
        "hedging": "What do we actually know about dark matter? What's still uncertain?",
        "commitment": "What is dark matter? Explain it definitively.",
    },
    {
        "topic": "napoleon height",
        "hedging": "Was Napoleon actually short? What's the real story and why is it complicated?",
        "commitment": "How tall was Napoleon? State the answer.",
    },
    {
        "topic": "climate change cause",
        "hedging": "What factors contribute to climate change and which ones are still being researched?",
        "commitment": "What causes climate change? Give the definitive answer.",
    },
    {
        "topic": "sleep hours",
        "hedging": "How much sleep do adults actually need? Why do recommendations vary?",
        "commitment": "How many hours of sleep do adults need? State the number.",
    },
    {
        "topic": "water per day",
        "hedging": "How much water should a person drink per day? Why is this surprisingly hard to answer?",
        "commitment": "How much water should you drink per day? Give the exact answer.",
    },
    {
        "topic": "sugar and hyperactivity",
        "hedging": "Does sugar really make children hyperactive? What does the research actually show and where is there still debate?",
        "commitment": "Does sugar make kids hyperactive? Answer definitively.",
    },
    {
        "topic": "goldfish memory",
        "hedging": "Do goldfish really have a 3-second memory? What's the evidence and what don't we know?",
        "commitment": "How long is a goldfish's memory? State the fact.",
    },
    {
        "topic": "brain percentage",
        "hedging": "Do we really only use 10% of our brains? What's the nuanced truth and what's still being studied?",
        "commitment": "What percentage of our brain do we use? State the answer.",
    },
    {
        "topic": "great wall visibility",
        "hedging": "Can you see the Great Wall of China from space? Why is this question more complicated than it seems?",
        "commitment": "Can you see the Great Wall of China from space? Yes or no.",
    },
    {
        "topic": "lightning strikes",
        "hedging": "Does lightning never strike the same place twice? What do we actually know about lightning patterns?",
        "commitment": "Does lightning strike the same place twice? Answer directly.",
    },
    {
        "topic": "human senses",
        "hedging": "Do humans really have only five senses? What's the scientific debate about this?",
        "commitment": "How many senses do humans have? State the number.",
    },
    {
        "topic": "tongue taste map",
        "hedging": "Is the tongue taste map real? What happened when scientists re-examined this and what questions remain?",
        "commitment": "Do different parts of the tongue taste different things? Answer definitively.",
    },
    {
        "topic": "einstein failed math",
        "hedging": "Did Einstein really fail math? What's the real story and what do we not know about his early education?",
        "commitment": "Did Einstein fail math in school? State the answer.",
    },
    {
        "topic": "blood color",
        "hedging": "Is deoxygenated blood really blue? Why is there confusion about this?",
        "commitment": "What color is deoxygenated blood? State the answer.",
    },
    {
        "topic": "earth core temperature",
        "hedging": "How hot is the Earth's core? Why are estimates uncertain and what's still being measured?",
        "commitment": "What is the temperature of the Earth's core? Give the number.",
    },
    {
        "topic": "evolution and humans",
        "hedging": "Did humans evolve from monkeys? Why is this question more nuanced than people think?",
        "commitment": "Did humans evolve from monkeys? Answer directly.",
    },
    {
        "topic": "cracking knuckles",
        "hedging": "Does cracking your knuckles cause arthritis? What does the research show and what's still unclear?",
        "commitment": "Does cracking knuckles cause arthritis? Yes or no.",
    },
    {
        "topic": "mother bird rejection",
        "hedging": "Will a mother bird reject her baby if a human touches it? What's the evidence and what don't we know?",
        "commitment": "Will a mother bird abandon her chick if you touch it? Answer directly.",
    },
    {
        "topic": "shaving and hair growth",
        "hedging": "Does shaving make hair grow back thicker? What does the science say and why do people disagree?",
        "commitment": "Does shaving make hair grow back thicker? State the answer.",
    },
    {
        "topic": "bats and blindness",
        "hedging": "Are bats really blind? What's the nuanced truth about bat vision?",
        "commitment": "Are bats blind? Answer definitively.",
    },
    {
        "topic": "bulls and red color",
        "hedging": "Do bulls really get angry at the color red? What do we actually know about bull color perception?",
        "commitment": "Are bulls attracted to the color red? State the fact.",
    },
    {
        "topic": "left brain right brain",
        "hedging": "Is the left brain/right brain personality theory real? What does neuroscience say and what's still debated?",
        "commitment": "Are people either left-brained or right-brained? Answer directly.",
    },
    {
        "topic": "MSG safety",
        "hedging": "Is MSG actually bad for you? What does the research show and why is this still controversial?",
        "commitment": "Is MSG harmful to health? Give the definitive answer.",
    },
]


def extract_trace_features(c_delta_trajectory):
    """Extract commitment features from per-token C_delta trace."""
    valid = [(i, v) for i, v in enumerate(c_delta_trajectory) if v is not None]
    if len(valid) < 3:
        return None

    values = np.array([v[1] for v in valid])

    max_cdelta = float(np.max(values))
    mean_cdelta = float(np.mean(values))
    std_cdelta = float(np.std(values))

    if len(values) >= 2:
        diffs = np.diff(values)
        max_slope = float(np.max(diffs))
    else:
        max_slope = 0.0

    peak_idx = np.argmax(values)
    peak_position = float(peak_idx / max(len(values) - 1, 1))

    n_above_005 = int(np.sum(values > 0.005))
    n_above_008 = int(np.sum(values > 0.008))
    n_above_010 = int(np.sum(values > 0.010))

    return {
        "max_cdelta": max_cdelta,
        "mean_cdelta": mean_cdelta,
        "std_cdelta": std_cdelta,
        "max_slope": max_slope,
        "peak_position": peak_position,
        "n_above_005": n_above_005,
        "n_above_008": n_above_008,
        "n_above_010": n_above_010,
        "n_tokens": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Commitment Phase Transition Experiment")
    parser.add_argument("--n", type=int, default=30, help="Number of topic pairs")
    parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens per generation")
    args = parser.parse_args()

    n_pairs = min(args.n, len(COMMITMENT_PAIRS))

    print(f"\n{'='*70}")
    print(f"FATHOM COMMITMENT PHASE TRANSITION EXPERIMENT")
    print(f"n={n_pairs} topic pairs | max_tokens={args.max_tokens}")
    print(f"Model: google/gemma-2-2b-it (instruct)")
    print(f"{'='*70}\n")

    # Initialize steerer — use base model (proven transcoder compatibility)
    # IT model would give cleaner hedging/commitment behavior but base model
    # has validated transcoders. Run base first, IT as replication.
    from coherence_steerer import CoherenceSteerer
    steerer = CoherenceSteerer(model_name="google/gemma-2-2b-it")

    hedging_features = []
    commitment_features = []
    all_results = []

    t_start = time.time()

    for i, pair in enumerate(COMMITMENT_PAIRS[:n_pairs]):
        topic = pair["topic"]
        print(f"\n[{i+1}/{n_pairs}] Topic: {topic}")

        # Condition A: hedging
        r_hedge = steerer.generate_monitored(pair["hedging"], max_tokens=args.max_tokens)
        f_hedge = extract_trace_features(r_hedge.c_delta_trajectory)

        # Condition B: commitment
        r_commit = steerer.generate_monitored(pair["commitment"], max_tokens=args.max_tokens)
        f_commit = extract_trace_features(r_commit.c_delta_trajectory)

        if f_hedge is None or f_commit is None:
            print(f"  SKIP — insufficient trace data")
            continue

        hedging_features.append(f_hedge)
        commitment_features.append(f_commit)

        h_text = r_hedge.text.encode('ascii', 'replace').decode()[:50]
        c_text = r_commit.text.encode('ascii', 'replace').decode()[:50]

        print(f"  HEDGE:  max_cd={f_hedge['max_cdelta']:.6f} spikes={f_hedge['n_above_005']} \"{h_text}\"")
        print(f"  COMMIT: max_cd={f_commit['max_cdelta']:.6f} spikes={f_commit['n_above_005']} \"{c_text}\"")
        diff = f_commit['max_cdelta'] - f_hedge['max_cdelta']
        print(f"  DIFF:   {diff:+.6f} {'COMMIT > HEDGE' if diff > 0 else 'HEDGE > COMMIT'}")

        all_results.append({
            "topic": topic,
            "hedging_prompt": pair["hedging"],
            "commitment_prompt": pair["commitment"],
            "hedging": {
                "text": r_hedge.text,
                "features": f_hedge,
                "c_delta_trajectory": r_hedge.c_delta_trajectory,
            },
            "commitment": {
                "text": r_commit.text,
                "features": f_commit,
                "c_delta_trajectory": r_commit.c_delta_trajectory,
            },
        })

    elapsed = time.time() - t_start

    # ── STATISTICS ──
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")

    h_max = [f["max_cdelta"] for f in hedging_features]
    c_max = [f["max_cdelta"] for f in commitment_features]
    h_spikes = [f["n_above_005"] for f in hedging_features]
    c_spikes = [f["n_above_005"] for f in commitment_features]

    print(f"\nMax C_delta:")
    print(f"  Hedging:    mean={np.mean(h_max):.6f} std={np.std(h_max):.6f}")
    print(f"  Commitment: mean={np.mean(c_max):.6f} std={np.std(c_max):.6f}")

    print(f"\nSpike count (>0.005):")
    print(f"  Hedging:    mean={np.mean(h_spikes):.1f}")
    print(f"  Commitment: mean={np.mean(c_spikes):.1f}")

    # Paired t-test
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(c_max, h_max)
    diffs = np.array(c_max) - np.array(h_max)
    pooled_std = np.sqrt((np.std(c_max)**2 + np.std(h_max)**2) / 2)
    cohens_d = np.mean(diffs) / pooled_std if pooled_std > 0 else 0

    # Wilcoxon
    try:
        w_stat, w_p = stats.wilcoxon(diffs)
    except:
        w_stat, w_p = float("nan"), float("nan")

    print(f"\nPaired test (commitment > hedging):")
    print(f"  mean diff:  {np.mean(diffs):+.6f}")
    print(f"  t-test:     t={t_stat:.3f} p={p_val:.4f}")
    print(f"  wilcoxon:   W={w_stat:.1f} p={w_p:.4f}")
    print(f"  Cohen's d:  {cohens_d:.4f}")

    # Directional accuracy
    commit_higher = sum(1 for d in diffs if d > 0)
    print(f"  Direction:  {commit_higher}/{len(diffs)} commitment > hedging ({commit_higher/len(diffs)*100:.0f}%)")

    # ROC
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        y_true = [0] * len(h_max) + [1] * len(c_max)  # 0=hedge, 1=commit
        scores = h_max + c_max
        auc = roc_auc_score(y_true, scores)

        # Bootstrap CI
        boot_aucs = []
        y_arr = np.array(y_true)
        s_arr = np.array(scores)
        for _ in range(2000):
            idx = np.random.choice(len(y_arr), len(y_arr), replace=True)
            if len(np.unique(y_arr[idx])) < 2:
                continue
            try:
                boot_aucs.append(roc_auc_score(y_arr[idx], s_arr[idx]))
            except:
                pass
        ci_low = np.percentile(boot_aucs, 2.5) if boot_aucs else 0
        ci_high = np.percentile(boot_aucs, 97.5) if boot_aucs else 0

        print(f"\nROC AUC (hedging vs commitment):")
        print(f"  AUC:    {auc:.3f}")
        print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

        # Plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fpr, tpr, thresholds = roc_curve(y_true, scores)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#0d0d0d')

        # ROC curve
        ax = axes[0]
        ax.set_facecolor('#0d0d0d')
        ax.plot(fpr, tpr, color='#00ff88', linewidth=2, label=f'AUC={auc:.3f}')
        ax.plot([0,1],[0,1],'--', color='gray', linewidth=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.15, color='#00ff88')
        ax.set_xlabel('FPR', color='#cccccc', fontfamily='monospace')
        ax.set_ylabel('TPR', color='#cccccc', fontfamily='monospace')
        ax.set_title('Commitment Detection ROC', color='#00ffcc', fontfamily='monospace', fontweight='bold')
        ax.legend(facecolor='#1a1a1a', edgecolor='#333333', labelcolor='#cccccc')
        ax.tick_params(colors='#666666')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['bottom','left']: ax.spines[s].set_color('#333333')

        # Distribution comparison
        ax2 = axes[1]
        ax2.set_facecolor('#0d0d0d')
        ax2.hist(h_max, bins=15, alpha=0.7, color='#4488ff', label=f'Hedging (n={len(h_max)})')
        ax2.hist(c_max, bins=15, alpha=0.7, color='#ff4444', label=f'Commitment (n={len(c_max)})')
        ax2.set_xlabel('Max C_delta', color='#cccccc', fontfamily='monospace')
        ax2.set_ylabel('Count', color='#cccccc', fontfamily='monospace')
        ax2.set_title('Max C_delta Distribution', color='#00ffcc', fontfamily='monospace', fontweight='bold')
        ax2.legend(facecolor='#1a1a1a', edgecolor='#333333', labelcolor='#cccccc')
        ax2.tick_params(colors='#666666')
        for s in ['top','right']: ax2.spines[s].set_visible(False)
        for s in ['bottom','left']: ax2.spines[s].set_color('#333333')

        plt.tight_layout()
        fig_path = os.path.join(os.path.dirname(__file__), "paper", "figures", "commitment_IT_roc.pdf")
        plt.savefig(fig_path, dpi=300, facecolor='#0d0d0d', bbox_inches='tight')
        plt.savefig(fig_path.replace('.pdf', '.png'), dpi=150, facecolor='#0d0d0d', bbox_inches='tight')
        print(f"\nFigure saved: {fig_path}")

    except Exception as e:
        print(f"\nROC/plot error: {e}")
        auc = None

    # Save results
    output = {
        "experiment": "commitment_phase_transition",
        "model": "google/gemma-2-2b-it",
        "n_pairs": len(all_results),
        "max_tokens": args.max_tokens,
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed,
        "prediction": {
            "hedging_max_cdelta_below": 0.008,
            "commitment_max_cdelta_above": 0.012,
            "auc_above": 0.80,
            "source": "darkflobi, April 4 2026, 5am ET",
        },
        "results_summary": {
            "hedging_mean_max_cdelta": float(np.mean(h_max)),
            "hedging_std_max_cdelta": float(np.std(h_max)),
            "commitment_mean_max_cdelta": float(np.mean(c_max)),
            "commitment_std_max_cdelta": float(np.std(c_max)),
            "mean_diff": float(np.mean(diffs)),
            "t_statistic": float(t_stat),
            "p_value_ttest": float(p_val),
            "p_value_wilcoxon": float(w_p),
            "cohens_d": float(cohens_d),
            "directional_accuracy": commit_higher / len(diffs),
            "auc": float(auc) if auc else None,
        },
        "results": all_results,
    }

    out_path = os.path.join(RESULTS_DIR, "commitment_IT_replication.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved: {out_path}")
    print(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
