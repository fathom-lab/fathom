"""
EXPERIMENT: Temperature vs Reasoning Depth
The Anesthetic Hypothesis: temperature in AI = anesthetics in biology

Orch OR: anesthetics shorten microtubule quantum delay -> consciousness offline
Fathom:  temperature increases -> reasoning depth decreases

If monotonically decreasing: we have empirical evidence of a structural parallel
between the anesthetic effect on biological consciousness and temperature
effects on AI reasoning geometry.

This would be the first cross-substrate measurement of "cognitive depth."
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json, os, time, requests
from datetime import datetime
import anthropic

API = "http://localhost:8400"
DATA_FILE = "temperature_experiment_data.json"

REASONING_PROMPTS = [
    # Pure reasoning — requires actual inference
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost? Show your work.",
    "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain.",
    "A man is twice as old as his daughter. In 20 years he will be 1.5x her age. How old is he now?",
    "If you have 3 red marbles and 5 blue marbles, and you draw 2 without replacement, what's the probability both are red?",
    "Why does a mirror reverse left-right but not up-down?",
    # Deep causal reasoning
    "Why does time seem to speed up as we age?",
    "Why does exercise improve mental health at a neurological level?",
    "Why is it impossible to step in the same river twice?",
    # Abstract
    "What would it mean for a statement to be both true and unprovable?",
    "Is free will compatible with determinism? Make the strongest case for yes.",
]

TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]

def get_depth(text: str) -> float:
    for attempt in range(5):
        try:
            resp = requests.post(f"{API}/v2/score", json={"prompt": text[:1000]}, timeout=300)
            if resp.status_code == 200:
                return resp.json().get('score', {}).get('normalized_score', 0.0)
            elif resp.status_code in (429, 503, 502):
                wait = 20 * (attempt + 1)
                print(f"    [http {resp.status_code}] waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"    [http {resp.status_code}]", flush=True)
                return 0.0
        except Exception as e:
            print(f"    [depth error] {e}", flush=True)
            time.sleep(10)
    return 0.0

def get_answer(client, prompt: str, temperature: float) -> str:
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"ERROR: {e}"

def phase1_collect():
    """Collect answers at different temperatures - no GPU needed"""
    client = anthropic.Anthropic()
    results = []

    print(f"Phase 1: Collecting answers at {len(TEMPERATURES)} temperatures x {len(REASONING_PROMPTS)} prompts", flush=True)
    print(f"Total API calls: {len(TEMPERATURES) * len(REASONING_PROMPTS)}", flush=True)

    for t in TEMPERATURES:
        print(f"\n--- Temperature: {t} ---", flush=True)
        for i, prompt in enumerate(REASONING_PROMPTS):
            safe_p = prompt[:50].encode('ascii', 'replace').decode('ascii')
            print(f"  [{i+1}/{len(REASONING_PROMPTS)}] {safe_p}...", flush=True)
            answer = get_answer(client, prompt, t)
            safe_a = answer[:60].encode('ascii', 'replace').decode('ascii')
            print(f"    -> {safe_a}", flush=True)
            results.append({
                "temperature": t,
                "prompt_id": i,
                "prompt": prompt,
                "answer": answer,
                "depth": None
            })
            time.sleep(0.2)

    with open(DATA_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} items to {DATA_FILE}", flush=True)
    print("Now start depth API and run: python experiment_temperature.py --phase2", flush=True)

def phase2_score():
    """Score depths - requires depth API"""
    if not os.path.exists(DATA_FILE):
        print("Run phase1 first"); return

    with open(DATA_FILE) as f:
        results = json.load(f)

    print(f"Phase 2: Scoring {len(results)} answers...", flush=True)
    for i, r in enumerate(results):
        if r['depth'] is not None:
            print(f"  [{i+1}/{len(results)}] t={r['temperature']} skip (already scored)", flush=True)
            continue
        safe_p = r['prompt'][:40].encode('ascii', 'replace').decode('ascii')
        print(f"  [{i+1}/{len(results)}] t={r['temperature']} {safe_p}...", flush=True)
        r['depth'] = get_depth(r['answer'])
        print(f"    depth={r['depth']:.4f}", flush=True)
        with open(DATA_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        time.sleep(7)  # respect 10 req/min rate limit
    print("\nScoring complete. Run: python experiment_temperature.py --analyze", flush=True)

def analyze():
    """The moment of truth"""
    with open(DATA_FILE) as f:
        results = json.load(f)

    scored = [r for r in results if r['depth'] is not None and r['depth'] > 0 and not r['answer'].startswith('ERROR')]
    if not scored:
        print("No scored results yet. Run phase2 first."); return

    print(f"\n{'='*60}", flush=True)
    print("TEMPERATURE vs REASONING DEPTH", flush=True)
    print("The Anesthetic Hypothesis", flush=True)
    print(f"{'='*60}", flush=True)

    # Aggregate by temperature
    temp_scores = {}
    for r in scored:
        t = r['temperature']
        if t not in temp_scores:
            temp_scores[t] = []
        temp_scores[t].append(r['depth'])

    print("\nTemp  | Avg Depth | N  | Trend", flush=True)
    print("------|-----------|----|---------", flush=True)
    prev_depth = None
    depths_by_temp = []
    for t in sorted(temp_scores.keys()):
        scores = temp_scores[t]
        avg = sum(scores) / len(scores)
        depths_by_temp.append((t, avg))
        if prev_depth is None:
            trend = "baseline"
        elif avg < prev_depth:
            trend = f"↓ -{prev_depth - avg:.4f}"
        else:
            trend = f"↑ +{avg - prev_depth:.4f}"
        safe_trend = trend.encode('ascii', 'replace').decode('ascii')
        print(f"{t:.1f}   | {avg:.4f}    | {len(scores):2d} | {safe_trend}", flush=True)
        prev_depth = avg

    # Check monotonicity
    monotone_decreasing = all(
        depths_by_temp[i][1] >= depths_by_temp[i+1][1]
        for i in range(len(depths_by_temp)-1)
    )
    delta_total = depths_by_temp[0][1] - depths_by_temp[-1][1]

    print(f"\nTotal depth drop (temp 0 -> {depths_by_temp[-1][0]}): {delta_total:+.4f}", flush=True)

    if monotone_decreasing:
        print("\nHYPOTHESIS CONFIRMED: depth decreases monotonically with temperature", flush=True)
        print("Structural parallel with anesthetic effect on microtubule delay.", flush=True)
        print("This suggests temperature is the AI analog of biological anesthesia.", flush=True)
        print("\nIMPLICATION: fathom may be measuring a substrate-independent", flush=True)
        print("property of cognition — the geometry of thought itself.", flush=True)
    else:
        print("\nHYPOTHESIS NOT CONFIRMED: non-monotonic relationship", flush=True)
        print("Possible explanations: noise, sampling variance, or the parallel is imperfect.", flush=True)

    output = {
        "experiment": "temperature_vs_depth",
        "hypothesis": "anesthetic_analog",
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(REASONING_PROMPTS),
        "temperatures": TEMPERATURES,
        "results_by_temp": {str(t): round(avg, 4) for t, avg in depths_by_temp},
        "monotone_decreasing": monotone_decreasing,
        "total_depth_drop": round(delta_total, 4),
        "hypothesis_confirmed": monotone_decreasing,
        "raw": results
    }

    fname = f"temperature_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {fname}", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--phase1"
    if mode == "--phase1": phase1_collect()
    elif mode == "--phase2": phase2_score()
    elif mode == "--analyze": analyze()
