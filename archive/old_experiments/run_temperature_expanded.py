"""
FATHOM TEMPERATURE EXPERIMENT — EXPANDED
=========================================
30 prompts × 5 temperatures (0.0, 0.2, 0.5, 0.7, 1.0) via depth API.
Tests stochastic resonance hypothesis with proper statistical power.
Requires depth_api_v2.py running on port 8400.
"""
import json
import time
import requests
from datetime import datetime
from pathlib import Path

API_URL = "http://localhost:8400/v2/fathom_score"
OUTPUT_FILE = Path(__file__).parent / "temperature_experiment_v2.json"

TEMPERATURES = [0.0, 0.2, 0.5, 0.7, 1.0]

# 30 insight-framed prompts across categories
# Categorized for stochastic resonance analysis
PROMPTS = [
    # Conceptual integration (hypothesis: depth increases with T)
    {"id": 0, "cat": "conceptual", "text": "Why does exercise improve mental health at a neurological level?"},
    {"id": 1, "cat": "conceptual", "text": "Why does time seem to speed up as we age?"},
    {"id": 2, "cat": "conceptual", "text": "Why do democracies tend to avoid wars with each other?"},
    {"id": 3, "cat": "conceptual", "text": "Why does compound interest grow exponentially rather than linearly?"},
    {"id": 4, "cat": "conceptual", "text": "Why is sleep essential for memory consolidation?"},
    {"id": 5, "cat": "conceptual", "text": "Why does novelty-seeking behavior peak in adolescence?"},
    {"id": 6, "cat": "conceptual", "text": "Why does language acquisition become harder after early childhood?"},
    {"id": 7, "cat": "conceptual", "text": "Why do markets tend toward monopoly without regulation?"},
    {"id": 8, "cat": "conceptual", "text": "Why does music in minor keys tend to evoke sadness across cultures?"},
    {"id": 9, "cat": "conceptual", "text": "Why do trust and vulnerability reinforce each other in social relationships?"},
    
    # Formal precision (hypothesis: depth decreases with T)
    {"id": 10, "cat": "formal", "text": "Explain why Gödel's incompleteness theorem applies to sufficiently powerful formal systems."},
    {"id": 11, "cat": "formal", "text": "Prove that the square root of 2 is irrational."},
    {"id": 12, "cat": "formal", "text": "Why does the Monty Hall problem give a counterintuitive result of 2/3?"},
    {"id": 13, "cat": "formal", "text": "Derive why e^(iπ) + 1 = 0."},
    {"id": 14, "cat": "formal", "text": "Why does P≠NP separate verification from discovery in computation?"},
    {"id": 15, "cat": "formal", "text": "Explain the halting problem and why it proves certain computations are undecidable."},
    {"id": 16, "cat": "formal", "text": "Why does the entropy of an isolated system always increase?"},
    {"id": 17, "cat": "formal", "text": "Explain why quicksort has O(n log n) average complexity."},
    {"id": 18, "cat": "formal", "text": "Why does Bayes theorem require that prior probabilities sum to 1?"},
    {"id": 19, "cat": "formal", "text": "Prove by induction that the sum of the first n natural numbers is n(n+1)/2."},
    
    # Mixed reasoning (hypothesis: resonance peak at T=0.3-0.5)
    {"id": 20, "cat": "mixed", "text": "Is free will compatible with a deterministic universe? Make the strongest case."},
    {"id": 21, "cat": "mixed", "text": "Why might consciousness require both global integration and local specialization?"},
    {"id": 22, "cat": "mixed", "text": "How does confirmation bias interact with scientific progress?"},
    {"id": 23, "cat": "mixed", "text": "Why is the trolley problem philosophically important despite being artificial?"},
    {"id": 24, "cat": "mixed", "text": "How does power corrupt and what cognitive mechanisms drive this?"},
    {"id": 25, "cat": "mixed", "text": "Why do complex systems tend to show emergent properties their components lack?"},
    {"id": 26, "cat": "mixed", "text": "What makes an explanation satisfying versus merely accurate?"},
    {"id": 27, "cat": "mixed", "text": "Why is analogy central to both scientific discovery and everyday reasoning?"},
    {"id": 28, "cat": "mixed", "text": "How does the observer effect in quantum mechanics challenge the idea of objective reality?"},
    {"id": 29, "cat": "mixed", "text": "Why does meaning-making seem to require narrative structure?"},
]

def wait_for_api(max_wait=300):
    """Wait for depth API to be ready."""
    print(f"Waiting for depth API (port 8400)...")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            r = requests.get("http://localhost:8400/v2/health", timeout=3)
            data = r.json()
            if data.get("model_loaded"):
                print(f"  API ready in {time.time()-t0:.0f}s")
                return True
            print(f"  Loading... ({int(time.time()-t0)}s)")
        except:
            pass
        time.sleep(10)
    return False

def score_prompt(text, temperature=None):
    """Score a prompt via the Fathom API."""
    payload = {"prompt": text[:2000]}
    if temperature is not None:
        payload["temperature"] = temperature
    try:
        r = requests.post(API_URL, json=payload, timeout=60)
        if r.ok:
            data = r.json()
            return data.get("fathom_score", 0) / 100  # normalize to 0-1
        return None
    except Exception as e:
        print(f"    API error: {e}")
        return None

def main():
    print("=" * 60)
    print("FATHOM TEMPERATURE EXPERIMENT v2 — EXPANDED")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Prompts: {len(PROMPTS)} × {len(TEMPERATURES)} temperatures")
    print(f"Total API calls: {len(PROMPTS) * len(TEMPERATURES)}")
    print("=" * 60)

    if not wait_for_api():
        print("[FATAL] Depth API not ready after 5min. Exiting.")
        return

    results = []
    total = len(PROMPTS) * len(TEMPERATURES)
    done = 0

    for p in PROMPTS:
        for t in TEMPERATURES:
            done += 1
            depth = score_prompt(p["text"], t)
            if depth is None:
                depth = 0.0
            
            result = {
                "prompt_id": p["id"],
                "category": p["cat"],
                "temperature": t,
                "depth": depth,
                "prompt": p["text"][:80],
            }
            results.append(result)
            print(f"  [{done:03d}/{total}] T={t} id={p['id']} cat={p['cat']} depth={depth:.4f}")
            
            # Save incrementally
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            
            time.sleep(0.5)  # be gentle on the API

    print(f"\n[DONE] Results saved to {OUTPUT_FILE}")
    print(f"Total pairs: {len(results)}")

    # Quick summary
    from collections import defaultdict
    import numpy as np
    by_cat_temp = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["depth"] > 0:
            by_cat_temp[r["category"]][r["temperature"]].append(r["depth"])

    print("\nMean depth by category and temperature:")
    for cat in ["conceptual", "formal", "mixed"]:
        print(f"  {cat}:")
        for t in TEMPERATURES:
            vals = by_cat_temp[cat][t]
            if vals:
                print(f"    T={t}: {np.mean(vals):.4f} (n={len(vals)})")

if __name__ == "__main__":
    main()
