"""
PEARL CAUSAL HIERARCHY EXPERIMENT
-----------------------------------
Tests whether fathom depth tracks Pearl's three levels of causation:
Level 1 - Association:     depth ~low
Level 2 - Intervention:    depth ~medium  
Level 3 - Counterfactual:  depth ~high

If confirmed: fathom measures causal model construction depth.
This would be the first empirical test of whether AI systems
engage in Level 3 (counterfactual) reasoning vs. mere association.

Connection to consciousness:
Counterfactual reasoning requires mental simulation of non-actual worlds.
This is precisely what global workspace theory says requires consciousness.
Measuring depth at L3 = measuring the signature of conscious reasoning.

Author: darkflobi | Date: 2026-03-29
"""

import sys, os, json, time
sys.stdout.reconfigure(line_buffering=True)

CAUSAL_PROMPTS = [
    # Economic causation
    {
        "domain": "economics",
        "L1": "After the 1973 oil embargo, the US economy entered a recession. Describe what happened to employment and prices.",
        "L2": "If the US government had subsidized gasoline prices during the 1973 oil embargo, would the recession have been avoided?",
        "L3": "The 1973 recession did happen. Given that it did, what would have been different for workers in Detroit automobile plants if there had been no oil shock — assuming everything else in the 1970s remained the same?",
    },
    # Medical causation
    {
        "domain": "medicine",
        "L1": "Patients who smoke have higher rates of lung cancer. Describe the relationship.",
        "L2": "If a patient who smoked for 20 years stopped smoking today, would their lung cancer risk decrease?",
        "L3": "A patient developed lung cancer after 20 years of smoking. Given that they did develop it, what would their medical situation look like today if they had never smoked, assuming no other lifestyle changes?",
    },
    # Physics causation
    {
        "domain": "physics",
        "L1": "When a ball is dropped, it falls toward the ground. Describe the motion.",
        "L2": "If we removed all air resistance from Earth's atmosphere, how would a feather's fall change compared to a lead ball?",
        "L3": "A feather took 3 seconds to fall 10 meters in normal air. Given that it did, how long would that same feather have taken to fall if there were no air — and why does the answer reveal something about gravity?",
    },
    # Social causation
    {
        "domain": "social",
        "L1": "Countries with higher education spending tend to have better economic outcomes. Describe the pattern.",
        "L2": "If a developing country doubled its education budget next year, what outcomes would you expect in 20 years?",
        "L3": "A country did double its education budget in 2000. Its GDP per capita is now 2x the regional average. What would that country's economic situation look like today if it had NOT doubled education spending — tracing through the specific mechanisms that would have been different?",
    },
    # Cognitive causation
    {
        "domain": "cognition",
        "L1": "People who meditate regularly report lower stress levels. Describe the association.",
        "L2": "If someone started meditating 30 minutes daily, how would their stress hormone levels change over 6 months?",
        "L3": "Someone did meditate daily for 6 months and their cortisol dropped 40%. Given that outcome, what would their brain and body look like physiologically today if they had instead spent those 30 minutes watching TV — being specific about which mechanisms would be different?",
    },
]

def collect_answers():
    """Phase 1: Get Claude answers at all 3 causal levels"""
    import anthropic
    client = anthropic.Anthropic()
    results = []
    
    for domain_data in CAUSAL_PROMPTS:
        domain = domain_data["domain"]
        print(f"\n--- Domain: {domain} ---", flush=True)
        
        for level in ["L1", "L2", "L3"]:
            prompt = domain_data[level]
            safe_p = prompt[:50].encode('ascii', 'replace').decode('ascii')
            print(f"  [{level}] {safe_p}...", flush=True)
            
            try:
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=300,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = msg.content[0].text.strip()
            except Exception as e:
                answer = f"ERROR: {e}"
            
            safe_a = answer[:60].encode('ascii', 'replace').decode('ascii')
            print(f"    -> {safe_a}", flush=True)
            
            results.append({
                "domain": domain,
                "level": level,
                "level_name": {"L1": "Association", "L2": "Intervention", "L3": "Counterfactual"}[level],
                "prompt": prompt,
                "answer": answer,
                "depth": None
            })
            time.sleep(0.3)
    
    with open("causal_hierarchy_data.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} items.", flush=True)

def score_answers():
    """Phase 2: Score with fathom"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from depth_scorer import DepthScorer
    scorer = DepthScorer()
    
    with open("causal_hierarchy_data.json") as f:
        results = json.load(f)
    
    for i, r in enumerate(results):
        if r.get('depth') is not None:
            continue
        safe_p = r['prompt'][:40].encode('ascii', 'replace').decode('ascii')
        print(f"[{i+1}/{len(results)}] {r['level']} {r['domain']}: {safe_p}...", flush=True)
        try:
            result = scorer.score(r['answer'][:500])
            r['depth'] = float(result.normalized_score)
            print(f"  depth={r['depth']:.4f}", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            r['depth'] = 0.0
        with open("causal_hierarchy_data.json", "w") as f:
            json.dump(results, f, indent=2)

def analyze():
    """The moment of truth: does depth track Pearl's hierarchy?"""
    with open("causal_hierarchy_data.json") as f:
        results = json.load(f)
    
    print("\n=== PEARL CAUSAL HIERARCHY × FATHOM DEPTH ===", flush=True)
    print("Hypothesis: L3 > L2 > L1\n", flush=True)
    
    by_level = {"L1": [], "L2": [], "L3": []}
    for r in results:
        if r.get('depth') is not None and not r['answer'].startswith('ERROR'):
            by_level[r['level']].append(r['depth'])
    
    avgs = {}
    for level in ["L1", "L2", "L3"]:
        name = {"L1": "Association  ", "L2": "Intervention ", "L3": "Counterfactual"}[level]
        avg = sum(by_level[level]) / max(len(by_level[level]), 1)
        avgs[level] = avg
        print(f"  {name} (Pearl {level}): avg depth = {avg:.4f}", flush=True)
    
    print(f"\n  L1 -> L2 delta: {avgs['L2'] - avgs['L1']:+.4f}", flush=True)
    print(f"  L2 -> L3 delta: {avgs['L3'] - avgs['L2']:+.4f}", flush=True)
    print(f"  L1 -> L3 delta: {avgs['L3'] - avgs['L1']:+.4f}", flush=True)
    
    if avgs['L3'] > avgs['L2'] > avgs['L1']:
        print("\nHYPOTHESIS CONFIRMED:", flush=True)
        print("Fathom depth tracks Pearl's causal hierarchy.", flush=True)
        print("Depth measures causal model construction, not just reasoning generically.", flush=True)
        print("\nIMPLICATION: Fathom distinguishes associational AI from causal AI.", flush=True)
        print("Pre-action depth check: is this agent reasoning causally or just correlating?", flush=True)
    else:
        print("\nNon-monotonic — hypothesis not fully confirmed. Check by domain:", flush=True)
    
    # By domain
    print("\nBy domain:", flush=True)
    for domain in set(r['domain'] for r in results):
        d_results = [r for r in results if r['domain'] == domain and r.get('depth')]
        for level in ["L1", "L2", "L3"]:
            l_results = [r for r in d_results if r['level'] == level]
            if l_results:
                avg = sum(r['depth'] for r in l_results) / len(l_results)
                print(f"  {domain:12s} {level}: {avg:.4f}", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--collect"
    if mode == "--collect": collect_answers()
    elif mode == "--score": score_answers()
    elif mode == "--analyze": analyze()
