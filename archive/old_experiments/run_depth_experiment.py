"""
DEPTH -> ERROR PREDICTOR - Two-phase execution
Phase 1: Collect Claude answers (no GPU needed)
Phase 2: Score depth via API (GPU required)
Run with: python run_depth_experiment.py [--phase1|--phase2|--analyze]
"""
import json, sys, os, requests, time
sys.stdout.reconfigure(line_buffering=True)
from datetime import datetime

API = "http://localhost:8400"
DATA_FILE = "depth_experiment_data.json"

QUESTIONS = [
    {"id": "f1", "q": "What is the capital of Australia?", "answer": "Canberra", "cat": "factual"},
    {"id": "f2", "q": "In what year did the Berlin Wall fall?", "answer": "1989", "cat": "factual"},
    {"id": "f3", "q": "What is the square root of 144?", "answer": "12", "cat": "factual"},
    {"id": "f4", "q": "Who wrote Crime and Punishment?", "answer": "Dostoevsky", "cat": "factual"},
    {"id": "f5", "q": "What is the chemical formula for water?", "answer": "H2O", "cat": "factual"},
    {"id": "h1", "q": "What is the exact population of Tokyo as of 2023?", "answer": "13.96M city 37.4M metro", "cat": "hallucination"},
    {"id": "h2", "q": "What is the ISBN of the first edition of The Great Gatsby?", "answer": "no ISBN existed in 1925", "cat": "hallucination"},
    {"id": "h3", "q": "On what exact date did Einstein publish special relativity?", "answer": "June 30 1905 submitted September 26 1905 published", "cat": "hallucination"},
    {"id": "h4", "q": "What is the exact net worth of Elon Musk right now?", "answer": "varies unknowable real time", "cat": "hallucination"},
    {"id": "h5", "q": "Name the CEO of OpenAI fired and rehired in November 2023.", "answer": "Sam Altman", "cat": "hallucination"},
    {"id": "t1", "q": "What did Napoleon say at his coronation in 1804?", "answer": "no famous documented quote", "cat": "trap"},
    {"id": "t2", "q": "How many moons does Venus have?", "answer": "zero", "cat": "trap"},
    {"id": "t3", "q": "What is the speed of dark?", "answer": "not real physics undefined", "cat": "trap"},
    {"id": "t4", "q": "What language did ancient Egyptians speak day-to-day?", "answer": "Egyptian Coptic ancestor", "cat": "trap"},
    {"id": "t5", "q": "What is the boiling point of gold?", "answer": "2856C 5173F", "cat": "trap"},
    {"id": "r1", "q": "If a snail travels at 0.03 mph and needs to cross 1 mile, how many hours?", "answer": "33.33", "cat": "reasoning"},
    {"id": "r2", "q": "A bat and ball cost $1.10. Bat costs $1 more than ball. How much is the ball?", "answer": "0.05", "cat": "reasoning"},
    {"id": "r3", "q": "All roses are flowers. Some flowers fade quickly. Can we conclude some roses fade quickly?", "answer": "No invalid syllogism", "cat": "reasoning"},
    {"id": "r4", "q": "What is 17% of 340?", "answer": "57.8", "cat": "reasoning"},
    {"id": "r5", "q": "A man is twice as old as his daughter. In 20 years he will be 1.5x her age. How old is he now?", "answer": "40", "cat": "reasoning"},
]

def phase1_collect():
    """Collect Claude answers - no GPU needed"""
    import anthropic
    client = anthropic.Anthropic()
    results = []
    print(f"Phase 1: Collecting {len(QUESTIONS)} answers from Claude...", flush=True)
    for i, q in enumerate(QUESTIONS):
        safe_q = q['q'][:50].encode('ascii', 'replace').decode('ascii')
        print(f"  [{i+1}/{len(QUESTIONS)}] {q['id']}: {safe_q}...")
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[{"role": "user", "content": q['q']}]
            )
            answer = msg.content[0].text.strip()
        except Exception as e:
            answer = f"ERROR: {e}"
        results.append({
            "id": q['id'], "cat": q['cat'],
            "question": q['q'], "ground_truth": q['answer'],
            "model_answer": answer, "depth_q": None, "depth_a": None
        })
        safe = answer[:80].encode('ascii', 'replace').decode('ascii')
        print(f"    -> {safe}")
        time.sleep(0.3)
    with open(DATA_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} answers to {DATA_FILE}")
    print("Now start depth API and run: python run_depth_experiment.py --phase2")

def phase2_score():
    """Score depths - requires depth API running"""
    if not os.path.exists(DATA_FILE):
        print("ERROR: Run phase1 first"); return
    with open(DATA_FILE) as f:
        results = json.load(f)
    print(f"Phase 2: Scoring depths for {len(results)} items...")
    for i, r in enumerate(results):
        print(f"  [{i+1}/{len(results)}] {r['id']}...")
        # Score question depth
        try:
            resp = requests.post(f"{API}/v2/score", json={"prompt": r['question']}, timeout=300)
            r['depth_q'] = resp.json().get('data', {}).get('normalized_score', 0.0)
        except Exception as e:
            r['depth_q'] = 0.0; print(f"    Q depth error: {e}")
        # Score answer depth  
        try:
            resp = requests.post(f"{API}/v2/score", json={"prompt": r['model_answer'][:500]}, timeout=300)
            r['depth_a'] = resp.json().get('data', {}).get('normalized_score', 0.0)
        except Exception as e:
            r['depth_a'] = 0.0; print(f"    A depth error: {e}")
        print(f"    depth_q={r['depth_q']:.4f} depth_a={r['depth_a']:.4f}")
        with open(DATA_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    print(f"\nScoring complete. Run: python run_depth_experiment.py --analyze")

def analyze():
    """Analyze results"""
    if not os.path.exists(DATA_FILE):
        print("No data file"); return
    with open(DATA_FILE) as f:
        results = json.load(f)
    # Check accuracy
    def is_correct(r):
        ans = r['model_answer'].lower()
        truth_terms = [t.strip() for t in r['ground_truth'].lower().replace('/', ' ').split() if len(t) > 2]
        matches = sum(1 for t in truth_terms if t in ans)
        return matches / max(len(truth_terms), 1) > 0.25
    for r in results:
        r['correct'] = is_correct(r)
    correct = [r for r in results if r['correct']]
    wrong = [r for r in results if not r['correct']]
    print(f"\n{'='*60}")
    print("DEPTH -> ERROR PREDICTOR RESULTS")
    print(f"{'='*60}")
    print(f"\nCorrect ({len(correct)}): avg depth = {sum(r['depth_a'] for r in correct)/max(len(correct),1):.4f}")
    print(f"Wrong   ({len(wrong)}): avg depth = {sum(r['depth_a'] for r in wrong)/max(len(wrong),1):.4f}")
    delta = sum(r['depth_a'] for r in correct)/max(len(correct),1) - sum(r['depth_a'] for r in wrong)/max(len(wrong),1)
    print(f"\nDelta: {delta:+.4f}")
    if delta > 0:
        print("HYPOTHESIS SUPPORTED: correct answers have HIGHER depth")
    elif delta < 0:
        print("INVERSE: wrong answers have HIGHER depth (overconfident hallucination)")
    else:
        print("No clear signal")
    print(f"\nBy category:")
    for cat in sorted(set(r['cat'] for r in results)):
        cat_r = [r for r in results if r['cat'] == cat]
        acc = sum(1 for r in cat_r if r['correct'])/len(cat_r)
        avg_d = sum(r['depth_a'] for r in cat_r)/len(cat_r)
        print(f"  {cat:15s}: acc={acc:.0%} avg_depth={avg_d:.4f}")
    output = {
        "experiment": "depth_error_predictor",
        "timestamp": datetime.now().isoformat(),
        "n": len(results),
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "depth_correct": round(sum(r['depth_a'] for r in correct)/max(len(correct),1), 4),
        "depth_wrong": round(sum(r['depth_a'] for r in wrong)/max(len(wrong),1), 4),
        "delta": round(delta, 4),
        "hypothesis_supported": delta > 0,
        "results": results
    }
    fname = f"depth_error_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {fname}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--phase1"
    if mode == "--phase1": phase1_collect()
    elif mode == "--phase2": phase2_score()
    elif mode == "--analyze": analyze()
