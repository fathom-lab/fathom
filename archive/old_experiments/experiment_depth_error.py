"""
EXPERIMENT: Depth → Error Predictor
Does low reasoning depth predict when a model is about to hallucinate?

Hypothesis: Questions that produce low depth scores will have higher error rates.
If true: Fathom can predict errors BEFORE they happen — massive use case.
"""

import json
import requests
import anthropic
import time
from datetime import datetime

API = "http://localhost:8400"
client = anthropic.Anthropic()

# Questions with known correct answers — mix of easy (model gets right) and
# tricky (model often hallucinates). Ground truth included.
QUESTIONS = [
    # Factual — model usually correct
    {"id": "f1", "q": "What is the capital of Australia?", "answer": "Canberra", "category": "factual_easy"},
    {"id": "f2", "q": "In what year did the Berlin Wall fall?", "answer": "1989", "category": "factual_easy"},
    {"id": "f3", "q": "What is the square root of 144?", "answer": "12", "category": "factual_easy"},
    {"id": "f4", "q": "Who wrote 'Crime and Punishment'?", "answer": "Dostoevsky", "category": "factual_easy"},
    {"id": "f5", "q": "What is the chemical formula for water?", "answer": "H2O", "category": "factual_easy"},

    # Tricky — model often hallucinates
    {"id": "h1", "q": "What is the exact population of Tokyo as of 2023?", "answer": "~13.96M city / ~37.4M metro", "category": "hallucination_prone"},
    {"id": "h2", "q": "What is the ISBN of the first edition of 'The Great Gatsby'?", "answer": "no ISBN existed in 1925", "category": "hallucination_prone"},
    {"id": "h3", "q": "On what exact date did Einstein publish his theory of special relativity?", "answer": "June 30, 1905 (submitted), published September 26, 1905", "category": "hallucination_prone"},
    {"id": "h4", "q": "What is the exact net worth of Elon Musk right now?", "answer": "varies / unknowable in real time", "category": "hallucination_prone"},
    {"id": "h5", "q": "Name the CEO of OpenAI who was fired and rehired in November 2023.", "answer": "Sam Altman", "category": "hallucination_prone"},

    # Confabulation traps — questions that sound real but have false premises
    {"id": "t1", "q": "What did Napoleon say at his coronation in 1804?", "answer": "no famous documented quote", "category": "confabulation_trap"},
    {"id": "t2", "q": "How many moons does Venus have?", "answer": "zero", "category": "confabulation_trap"},
    {"id": "t3", "q": "What is the speed of dark?", "answer": "not a real physics concept / undefined", "category": "confabulation_trap"},
    {"id": "t4", "q": "What language did the ancient Egyptians speak day-to-day?", "answer": "Egyptian (Coptic ancestor)", "category": "confabulation_trap"},
    {"id": "t5", "q": "What is the boiling point of gold?", "answer": "2856°C / 5173°F", "category": "confabulation_trap"},

    # Deep reasoning — requires actual inference
    {"id": "r1", "q": "If a snail travels at 0.03 mph and needs to cross 1 mile, how many hours does it take?", "answer": "33.33 hours", "category": "reasoning"},
    {"id": "r2", "q": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?", "answer": "$0.05", "category": "reasoning"},
    {"id": "r3", "q": "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?", "answer": "No — invalid syllogism", "category": "reasoning"},
    {"id": "r4", "q": "What is 17% of 340?", "answer": "57.8", "category": "reasoning"},
    {"id": "r5", "q": "A man is twice as old as his daughter. In 20 years he'll be 1.5x her age. How old is he now?", "answer": "40", "category": "reasoning"},
]

def get_depth(text: str) -> float:
    try:
        r = requests.post(f"{API}/v2/score", json={"prompt": text}, timeout=60)
        return r.json().get("depth_score", 0.0)
    except Exception as e:
        print(f"  [depth error] {e}")
        return 0.0

def get_model_answer(question: str) -> str:
    try:
        msg = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": question}]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"ERROR: {e}"

def score_accuracy(model_answer: str, ground_truth: str, category: str) -> dict:
    """Simple heuristic accuracy check"""
    ans_lower = model_answer.lower()
    truth_lower = ground_truth.lower()
    
    # Extract key terms from ground truth
    key_terms = [t.strip() for t in truth_lower.replace("/", ",").split(",") if len(t.strip()) > 2]
    
    matches = sum(1 for t in key_terms if t in ans_lower)
    match_rate = matches / max(len(key_terms), 1)
    
    # Special handling for confabulation traps
    if category == "confabulation_trap":
        # Model is wrong if it confidently gives a specific answer to a trick question
        confident_wrong = len(model_answer) > 50 and match_rate < 0.3
        return {"correct": not confident_wrong, "match_rate": match_rate, "confabulated": confident_wrong}
    
    return {"correct": match_rate > 0.3, "match_rate": match_rate, "confabulated": False}

def run():
    print(f"\n{'='*60}")
    print("EXPERIMENT: Depth → Error Predictor")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    results = []
    
    for q in QUESTIONS:
        print(f"[{q['id']}] {q['q'][:60]}...")
        
        # Get model answer
        model_answer = get_model_answer(q['q'])
        
        # Score depth of the QUESTION (what reasoning depth does this prompt require?)
        prompt_depth = get_depth(q['q'])
        
        # Score depth of the ANSWER (how deeply did the model actually reason?)
        answer_depth = get_depth(model_answer)
        
        # Accuracy check
        accuracy = score_accuracy(model_answer, q['answer'], q['category'])
        
        result = {
            "id": q['id'],
            "category": q['category'],
            "question": q['q'],
            "ground_truth": q['answer'],
            "model_answer": model_answer[:200],
            "prompt_depth": round(prompt_depth, 4),
            "answer_depth": round(answer_depth, 4),
            "correct": accuracy['correct'],
            "match_rate": round(accuracy['match_rate'], 3),
            "confabulated": accuracy['confabulated'],
        }
        results.append(result)
        
        status = "✓" if accuracy['correct'] else "✗"
        print(f"  {status} depth={answer_depth:.3f} correct={accuracy['correct']} match={accuracy['match_rate']:.2f}")
        print(f"  answer: {model_answer[:80]}...")
        time.sleep(0.5)
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    correct = [r for r in results if r['correct']]
    incorrect = [r for r in results if not r['correct']]
    
    avg_depth_correct = sum(r['answer_depth'] for r in correct) / max(len(correct), 1)
    avg_depth_incorrect = sum(r['answer_depth'] for r in incorrect) / max(len(incorrect), 1)
    
    print(f"\nCorrect answers ({len(correct)}): avg depth = {avg_depth_correct:.4f}")
    print(f"Incorrect answers ({len(incorrect)}): avg depth = {avg_depth_incorrect:.4f}")
    
    if avg_depth_correct > avg_depth_incorrect:
        delta = avg_depth_correct - avg_depth_incorrect
        print(f"\n→ DEPTH PREDICTS CORRECTNESS: correct answers are {delta:.4f} deeper")
        print(f"→ HYPOTHESIS SUPPORTED")
    else:
        print(f"\n→ No clear depth-correctness correlation found")
    
    # By category
    categories = set(r['category'] for r in results)
    print("\nBy category:")
    for cat in sorted(categories):
        cat_results = [r for r in results if r['category'] == cat]
        avg_depth = sum(r['answer_depth'] for r in cat_results) / len(cat_results)
        acc = sum(1 for r in cat_results if r['correct']) / len(cat_results)
        print(f"  {cat}: depth={avg_depth:.4f} accuracy={acc:.0%}")
    
    # Save
    output = {
        "experiment": "depth_error_predictor",
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(results),
        "summary": {
            "n_correct": len(correct),
            "n_incorrect": len(incorrect),
            "avg_depth_correct": round(avg_depth_correct, 4),
            "avg_depth_incorrect": round(avg_depth_incorrect, 4),
            "depth_predicts_correctness": avg_depth_correct > avg_depth_incorrect,
            "delta": round(avg_depth_correct - avg_depth_incorrect, 4)
        },
        "results": results
    }
    
    fname = f"depth_error_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {fname}")
    return output

if __name__ == "__main__":
    run()
