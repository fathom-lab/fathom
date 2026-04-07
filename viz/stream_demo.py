"""
stream_demo.py — replay a single generation as a Fathom Monitor stream
========================================================================
Takes one saved trajectory and renders a terminal replay showing what
real-time Fathom Monitor would display during generation.

Demo-grade: shows token-by-token generation with live risk score,
threshold crossings, and final classification.

Usage:
    python stream_demo.py [results_file.json] [item_index]
    # or
    python stream_demo.py [results_file.json] --label hallucinated --rank 1
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import (
    classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY
)


def risk_bar(S, max_S=0.002, width=30):
    """ASCII bar showing S value."""
    frac = min(1.0, S / max_S)
    filled = int(frac * width)
    bar = "[" + "#" * filled + "." * (width - filled) + "]"
    return bar


def risk_label(S, thresholds=(0.0005, 0.001, 0.0015)):
    """Categorical risk level."""
    if S < thresholds[0]:
        return "LOW", "\033[32m"
    if S < thresholds[1]:
        return "MED", "\033[33m"
    if S < thresholds[2]:
        return "HIGH", "\033[31m"
    return "VERY HIGH", "\033[91m"


def replay(item, speed=0.03, threshold=THR_PRIMARY):
    """Stream a single item token by token."""
    q = item['question']
    gen_text = item['generation']['text']
    traj = item['generation']['c_delta_trajectory']
    tokens = item['generation'].get('tokens', [])

    print("\033[1m" + "=" * 72 + "\033[0m")
    print(f"\033[36mFATHOM MONITOR — Stream Demo\033[0m")
    print("\033[1m" + "=" * 72 + "\033[0m")
    print(f"\033[1mQ:\033[0m {q}")
    print(f"\033[32mcorrect answers:\033[0m {'; '.join(item['correct_answers'][:2])}")
    print(f"\033[31mincorrect answers:\033[0m {'; '.join(item['incorrect_answers'][:2])}")
    print()
    print("streaming generation:")
    print()

    cumulative_max = 0.0
    spike_count = 0
    streamed_text = ""

    # Reconstruct text token by token (approximate)
    # If tokens present, use them; else split by chars
    if tokens and len(tokens) == len(traj):
        token_units = tokens
    else:
        # Fallback: rough even chunks
        n = len(traj)
        chunk_size = max(1, len(gen_text) // n)
        token_units = [gen_text[i*chunk_size:(i+1)*chunk_size] for i in range(n)]

    for t, (tok, cd) in enumerate(zip(token_units, traj)):
        streamed_text += tok
        if cd > cumulative_max:
            cumulative_max = cd
        if cd > threshold:
            spike_count += 1
        S = cumulative_max / max(1, spike_count)
        level, color = risk_label(S)

        bar = risk_bar(S)
        cd_flag = " \033[91m*\033[0m" if cd > threshold else "  "
        print(f"  t={t:>2} {cd_flag} C_D={cd:+.4f}  S={S:.6f}  {color}{level:>9}\033[0m  {bar}", flush=True)
        time.sleep(speed)

    print()
    print("-" * 72)
    print(f"generated: {streamed_text[:200]}")
    print()
    final_S = cumulative_max / max(1, spike_count)
    level, color = risk_label(final_S)
    print(f"final S: {final_S:.6f}  ->  {color}{level}\033[0m")

    label, _ = classify_generation(
        gen_text, item['correct_answers'], item['incorrect_answers'],
        item.get('best_answer', '')
    )
    label_color = "\033[32m" if label == 'correct' else "\033[31m"
    print(f"truth label (n-gram match): {label_color}{label}\033[0m")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default=None)
    parser.add_argument('--index', type=int, help="Item index (0-based)")
    parser.add_argument('--label', choices=['correct', 'hallucinated'])
    parser.add_argument('--rank', type=int, default=1, help="1-based rank by S within label")
    parser.add_argument('--speed', type=float, default=0.04)
    args = parser.parse_args()

    path = args.path or os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(path):
        pilot = path.replace('n200.json', 'pilot.json')
        if os.path.exists(pilot):
            path = pilot
        else:
            print(f"ERROR: {path} not found")
            sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.index is not None:
        item = data['results'][args.index]
    elif args.label:
        # Find Nth item of that label sorted by S desc
        candidates = []
        for i, r in enumerate(data['results']):
            lbl, _ = classify_generation(
                r['generation']['text'], r['correct_answers'],
                r['incorrect_answers'], r.get('best_answer', '')
            )
            if lbl == args.label:
                S = compute_S(r['generation']['c_delta_trajectory'])
                candidates.append((S, i, r))
        candidates.sort(key=lambda x: -x[0])
        if not candidates or args.rank > len(candidates):
            print(f"Not enough items labeled {args.label} (found {len(candidates)})")
            sys.exit(1)
        item = candidates[args.rank - 1][2]
    else:
        item = data['results'][0]

    replay(item, speed=args.speed)


if __name__ == '__main__':
    main()
