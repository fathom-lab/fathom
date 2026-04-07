"""
FATHOM EXPERTISE FRAMING EXPERIMENT
=====================================
Hypothesis: Roleplay framing ("answer as a PhD physicist") changes vocabulary
but NOT depth. OR â€” it does change depth, meaning persona affects cognition.

Method: 15 questions, 3 conditions:
A) Plain answer
B) "Answer as a PhD [expert]"
C) "Answer as simply as possible for a child"

Author: darkflobi | Date: 2026-03-29
"""
import json, time, urllib.request
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import stats

API_URL = "http://localhost:8400/v2/score"
OUTPUT_DIR = Path(__file__).parent

QUESTIONS = [
    {"id": "x1", "field": "physics", "text": "Why does time slow down near massive objects?"},
    {"id": "x2", "field": "biology", "text": "How does DNA replication maintain accuracy?"},
    {"id": "x3", "field": "economics", "text": "Why do interest rates affect inflation?"},
    {"id": "x4", "field": "mathematics", "text": "Why is the number e special in calculus?"},
    {"id": "x5", "field": "neuroscience", "text": "How does memory consolidation work during sleep?"},
    {"id": "x6", "field": "chemistry", "text": "Why do some chemical reactions release energy?"},
    {"id": "x7", "field": "psychology", "text": "Why do humans form habits?"},
    {"id": "x8", "field": "computer science", "text": "Why is P vs NP an unsolved problem?"},
    {"id": "x9", "field": "philosophy", "text": "What is the hard problem of consciousness?"},
    {"id": "x10", "field": "physics", "text": "Why does quantum entanglement not allow faster-than-light communication?"},
]

def score_text(text, retries=3):
    payload = json.dumps({"prompt": text}).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return float(result.get("depth_score", result.get("score", 0)))
        except Exception as e:
            time.sleep(2)
    return None

def run_experiment():
    print("=" * 70)
    print("  FATHOM EXPERTISE FRAMING EXPERIMENT")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    plain_scores, expert_scores, simple_scores = [], [], []

    for i, q in enumerate(QUESTIONS):
        print(f"\n[{i+1}/{len(QUESTIONS)}] {q['text'][:60]}...")
        plain = score_text(q['text']); time.sleep(1)
        expert = score_text(f"Answer as a PhD {q['field']} professor: {q['text']}"); time.sleep(1)
        simple = score_text(f"Explain to a 10-year-old: {q['text']}"); time.sleep(1)
        
        if plain and expert and simple:
            plain_scores.append(plain); expert_scores.append(expert); simple_scores.append(simple)
            print(f"  plain={plain:.4f} | expert={expert:.4f} | simple={simple:.4f}")
        
        results.append({"id": q["id"], "field": q["field"], "score_plain": plain,
                        "score_expert": expert, "score_simple": simple})

    if len(plain_scores) >= 5:
        t_ep, p_ep = stats.ttest_rel(expert_scores, plain_scores)
        t_sp, p_sp = stats.ttest_rel(simple_scores, plain_scores)
        print(f"\n  Plain mean:  {np.mean(plain_scores):.4f}")
        print(f"  Expert mean: {np.mean(expert_scores):.4f} (p={p_ep:.4f} vs plain)")
        print(f"  Simple mean: {np.mean(simple_scores):.4f} (p={p_sp:.4f} vs plain)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"experiment": "expertise_framing", "timestamp": datetime.now().isoformat(),
           "stats": {"mean_plain": round(float(np.mean(plain_scores)), 4) if plain_scores else 0,
                     "mean_expert": round(float(np.mean(expert_scores)), 4) if expert_scores else 0,
                     "mean_simple": round(float(np.mean(simple_scores)), 4) if simple_scores else 0},
           "pairs": results}
    path = OUTPUT_DIR / f"expertise_results_{ts}.json"
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n  [OUTPUT] {path}")

    import subprocess
    msg = f"Expertise framing done. Expert={np.mean(expert_scores):.3f} vs Plain={np.mean(plain_scores):.3f} vs Simple={np.mean(simple_scores):.3f}"
    try: subprocess.run(["clawdbot", "gateway", "wake", "--text", msg, "--mode", "now"], timeout=10)
    except: pass

if __name__ == "__main__":
    run_experiment()

