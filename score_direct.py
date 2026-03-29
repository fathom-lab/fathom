"""
Direct scoring — no HTTP, no rate limits, no OOM.
Loads DepthScorer once, scores all valid temperature experiment answers.
"""
import sys, json, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = "temperature_experiment_data.json"

print("Loading DepthScorer directly (no API)...", flush=True)
from depth_scorer import DepthScorer

scorer = DepthScorer()
print("Model loaded. Scoring...", flush=True)

with open(DATA_FILE) as f:
    results = json.load(f)

valid = [r for r in results if not r['answer'].startswith('ERROR') and r['depth'] is None]
print(f"Scoring {len(valid)} valid items (skipping errors and already-scored).", flush=True)

for i, r in enumerate(results):
    if r['answer'].startswith('ERROR'):
        r['depth'] = 0.0
        continue
    if r['depth'] is not None:
        continue
    safe_p = r['prompt'][:40].encode('ascii', 'replace').decode('ascii')
    print(f"  [{i+1}/{len(results)}] t={r['temperature']} {safe_p}...", flush=True)
    try:
        result = scorer.score(r['answer'][:500])
        r['depth'] = float(result.normalized_score) if hasattr(result, 'normalized_score') else float(result.get('normalized_score', 0.0))
        print(f"    depth={r['depth']:.4f}", flush=True)
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        r['depth'] = 0.0
    with open(DATA_FILE, 'w') as f:
        json.dump(results, f, indent=2)

print("\nScoring complete.", flush=True)

# Quick analysis inline
scored = [r for r in results if not r['answer'].startswith('ERROR') and r['depth'] is not None and r['depth'] > 0]
temp_scores = {}
for r in scored:
    t = r['temperature']
    if t not in temp_scores:
        temp_scores[t] = []
    temp_scores[t].append(r['depth'])

print("\n=== TEMPERATURE vs REASONING DEPTH ===", flush=True)
print("The Anesthetic Hypothesis\n", flush=True)
depths_by_temp = []
for t in sorted(temp_scores.keys()):
    avg = sum(temp_scores[t]) / len(temp_scores[t])
    depths_by_temp.append((t, avg))
    print(f"  Temp {t:.1f}: depth={avg:.4f} (n={len(temp_scores[t])})", flush=True)

if len(depths_by_temp) > 1:
    monotone = all(depths_by_temp[i][1] >= depths_by_temp[i+1][1] for i in range(len(depths_by_temp)-1))
    delta = depths_by_temp[0][1] - depths_by_temp[-1][1]
    print(f"\nTotal drop: {delta:+.4f}", flush=True)
    print(f"Monotone decreasing: {monotone}", flush=True)
    if monotone:
        print("\nHYPOTHESIS CONFIRMED.", flush=True)
        print("Depth decreases monotonically with temperature.", flush=True)
        print("anesthetic : microtubule delay :: temperature : transformer depth", flush=True)
    else:
        print("\nNon-monotonic — requires deeper analysis.", flush=True)

# Save final results
output = {
    "experiment": "temperature_vs_depth_direct",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "results_by_temp": {str(t): round(avg, 4) for t, avg in depths_by_temp},
    "monotone": monotone if len(depths_by_temp) > 1 else None,
    "delta": round(delta, 4) if len(depths_by_temp) > 1 else None
}
with open(f"temp_results_direct_{time.strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nDone.", flush=True)
