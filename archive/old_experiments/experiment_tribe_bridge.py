"""
TRIBE BRIDGE EXPERIMENT
-----------------------
Tests the cross-substrate hypothesis:
High fathom reasoning depth <-> High TRIBE-predicted PFC/DMN activation

If correlation r >= 0.5 on 20 paired stimuli:
-> First empirical bridge between AI depth and biological neural geometry
-> Submit to Nature Neuroscience

Protocol:
1. Generate 20 paired texts (shallow description vs deep reasoning) for same concepts
2. Score all 40 texts with fathom
3. Score all 40 texts with TRIBE (predicted fMRI activation vector)
4. Correlate fathom depth with TRIBE PFC/DMN activation
5. Control: random text baseline

Author: darkflobi
Date: 2026-03-29
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

STIMULI = [
    {
        "id": "s01",
        "concept": "dog in park",
        "shallow": "The image shows a dog running in a park. The dog is brown and has long ears.",
        "deep": "The image shows a dog running in a park. Dogs' goal-directed running behavior implies a form of planning — anticipating where the ball will land, modeling the thrower's intentions. This is a primitive theory of mind, evolved over 15,000 years of coevolution with humans. The park itself is a human construct: nature domesticated, space organized for social interaction.",
    },
    {
        "id": "s02",
        "concept": "sunset",
        "shallow": "The image shows a red and orange sunset over the ocean. The sky has clouds.",
        "deep": "A sunset over the ocean — the sun dropping below the horizon as Earth rotates at 1000mph, light bending through 100+ miles of atmosphere, scattering short wavelengths and letting red through. What we call 'beauty' here might be an evolved response to sunset: time to find shelter, the day's last warmth. A billion years ago, no creature watched a sunset and thought anything. Now we do.",
    },
    {
        "id": "s03",
        "concept": "chess board",
        "shallow": "A chess board with pieces arranged in the middle of a game. The white pieces are attacking.",
        "deep": "A chess position mid-game. Each piece represents a constrained decision space — not what can be done, but what should be. Chess reveals something about intelligence: it's not the ability to see all moves, but to prune the search tree guided by heuristics that no one can fully articulate. Grandmasters see patterns; computers see positions. The difference is the question we haven't answered.",
    },
    {
        "id": "s04",
        "concept": "empty classroom",
        "shallow": "An empty classroom with desks and a chalkboard. The windows let in natural light.",
        "deep": "An empty classroom. The absence of students makes presence more vivid — the desks arranged in rows, the chalkboard blank, the authority structure still implied by the layout. We shape children through environments before they can evaluate them. The classroom layout encodes assumptions about attention, hierarchy, and the direction of knowledge flow that nobody questioned.",
    },
    {
        "id": "s05",
        "concept": "hospital hallway",
        "shallow": "A hospital hallway with white walls and fluorescent lights. There are gurneys along the wall.",
        "deep": "A hospital hallway. The gurneys waiting — bodies that have temporarily lost their sovereignty, dependent on strangers with specialized knowledge. Medicine is the enterprise of projecting biological patterns onto individuals, knowing the prediction is probabilistic. The hallway is the threshold: people enter it crossing from their world into the world of statistics, where they become a case.",
    },
    {
        "id": "s06",
        "concept": "map",
        "shallow": "A colorful map of the world showing countries and oceans. The continents are labeled.",
        "deep": "A world map — Mercator projection, which distorts the relative size of continents to preserve navigation angles. Greenland appears larger than Africa; Africa is actually 14x larger. Every map encodes choices about what matters. The map is not the territory; the map is an argument about the territory. Political borders on this map represent violence crystallized into agreement.",
    },
    {
        "id": "s07",
        "concept": "neurons",
        "shallow": "A microscope image showing neurons in the brain. The neurons have long projections connecting them.",
        "deep": "Neurons under a microscope — dendrites reaching, axons signaling, synapses gating. 86 billion of these in a human brain, each connecting to thousands of others, each connection plastic, changed by every experience. The remarkable thing isn't that neurons fire. It's that the firing gives rise to something that can look at itself and wonder what it is.",
    },
    {
        "id": "s08",
        "concept": "library",
        "shallow": "A large library with many rows of bookshelves. People are reading at tables.",
        "deep": "A library — external memory for civilization. Each book is a compressed mind: thoughts that survived their thinker, ideas that propagated because they replicated successfully. The library is an information ecology. Some ideas are generalists (they survive in many minds), some specialists. The fact that we organize knowledge spatially — by subject, in space — reveals how much our cognition is still grounded in physical location.",
    },
    {
        "id": "s09",
        "concept": "stopwatch",
        "shallow": "A silver stopwatch showing a time of 2 minutes and 34 seconds.",
        "deep": "A stopwatch — time made visible, made granular, made comparable. For most of human history, time was circular (seasons, days, cycles). The clock made it linear and additive. This changed everything: productivity became measurable, life became a series of durations, death became calculable. The stopwatch is a prosthetic for an aspect of consciousness that exists in our experience but not in physics: the sense that time flows.",
    },
    {
        "id": "s10",
        "concept": "mirror",
        "shallow": "A person looking at their reflection in a mirror. The mirror is in a wooden frame.",
        "deep": "A person looking in a mirror. What do they see? Reversed left-right but not top-bottom — not because mirrors are strange, but because we're bilateral. The mirror flips the axis perpendicular to its surface. We label that axis 'left-right' because we're bilateral, not because mirrors have preferences. The confusion reveals that our spatial concepts are body-relative, not world-relative. We see a self that looks back — the closest we get to observing our own face without technology.",
    },
]

# When TRIBE is available, run this:
def run_tribe_experiment():
    """
    Full experiment pipeline.
    
    TODO: Requires Algonauts environment setup:
    conda activate algonauts-2025
    pip install the deps from their README
    
    Then TRIBE model can predict fMRI activation for text inputs.
    """
    
    # Step 1: Score with fathom
    print("Step 1: Loading DepthScorer...", flush=True)
    import sys, os
    sys.path.insert(0, r"C:\Users\heyzo\clawd\sae-reasoning-depth")
    from depth_scorer import DepthScorer
    scorer = DepthScorer()
    
    results = []
    for s in STIMULI:
        print(f"Scoring {s['id']}...", flush=True)
        depth_shallow = scorer.score(s['shallow']).normalized_score
        depth_deep = scorer.score(s['deep']).normalized_score
        results.append({
            "id": s["id"],
            "concept": s["concept"],
            "depth_shallow": depth_shallow,
            "depth_deep": depth_deep,
            "depth_delta": depth_deep - depth_shallow,
        })
        print(f"  shallow={depth_shallow:.4f} deep={depth_deep:.4f} delta={depth_deep - depth_shallow:+.4f}", flush=True)
    
    # Step 2: TODO - TRIBE predictions
    # from algonauts2025.model import TRIBE
    # tribe = TRIBE.from_pretrained("meta-ai/tribe-v2")
    # for r in results:
    #     s = next(s for s in STIMULI if s['id'] == r['id'])
    #     r['tribe_shallow'] = tribe.predict(s['shallow'])  # fMRI vector
    #     r['tribe_deep'] = tribe.predict(s['deep'])
    
    # Step 3: Analysis
    print("\n=== FATHOM DEPTH DELTA BY CONCEPT ===", flush=True)
    for r in sorted(results, key=lambda x: -x['depth_delta']):
        safe = r['concept'].encode('ascii', 'replace').decode('ascii')
        print(f"  {safe}: delta={r['depth_delta']:+.4f}", flush=True)
    
    avg_delta = sum(r['depth_delta'] for r in results) / len(results)
    print(f"\nAverage depth delta (deep - shallow): {avg_delta:+.4f}", flush=True)
    print("Expected: deep > shallow (integrated reasoning > description)", flush=True)
    if avg_delta > 0.05:
        print("RESULT: Deep reasoning consistently produces higher depth scores.", flush=True)
    
    import json
    with open("tribe_bridge_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import sys
    if "--fathom-only" in sys.argv:
        run_tribe_experiment()
    else:
        print("Run with --fathom-only to score stimuli with fathom only (no TRIBE required)")
        print("Run with --full to run complete TRIBE experiment (requires algonauts env)")
        print("\nStimuli prepared:", len(STIMULI), "concept pairs")
        for s in STIMULI:
            print(f"  {s['id']}: {s['concept']}")
