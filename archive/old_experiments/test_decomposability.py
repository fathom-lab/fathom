"""
DECOMPOSABILITY LAW VALIDATION
Same content, different integration structure.
Does fathom score the WEB version higher than the LIST version?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

LIST_VERSION = """Exercise improves mental health through multiple mechanisms:
1. BDNF release increases neuroplasticity and promotes new neuron formation.
2. Serotonin levels increase through regular aerobic exercise.
3. Cortisol is reduced through HPA axis regulation.
4. Sleep quality improves with regular physical activity.
5. Social engagement from group exercise reduces isolation.
Each mechanism independently contributes to better mood and reduced anxiety."""

WEB_VERSION = """Exercise initiates a self-reinforcing cascade where each mechanism amplifies the others: physical activity triggers BDNF release, which enhances synaptic plasticity, which reshapes the HPA axis stress circuitry, reducing baseline cortisol. Lower cortisol enables deeper sleep architecture, which in turn amplifies overnight BDNF synthesis. Meanwhile, the social components of exercise further reduce cortisol through oxytocin pathways. The mechanisms cannot be separated -- disrupt any one and the entire cascade weakens. This irreducible interdependence is why exercise has uniquely broad and robust effects on mental health compared to any single pharmaceutical intervention."""

SELF_REF_VERSION = """The question of what fathom measures when scoring this response creates an interesting loop: to answer accurately, I must simultaneously hold the definition of fathom (layer activation distribution during inference), the observer effect (does knowing I am being measured change my response?), and the meta-question (does my awareness of the measurement change what the measurement captures?). The honest answer requires integrating all three levels at once: fathom measures the irreducibility of my computation, my awareness of being measured may increase that irreducibility (by engaging meta-cognitive circuits), and the meta-question itself demonstrates the irreducibility by requiring all levels to be held simultaneously."""

print("Loading DepthScorer...", flush=True)
from depth_scorer import DepthScorer
scorer = DepthScorer()
print("Scoring...", flush=True)

list_score = scorer.score(LIST_VERSION)
print(f"\nLIST  version: {list_score.normalized_score:.4f}", flush=True)

web_score = scorer.score(WEB_VERSION)
print(f"WEB   version: {web_score.normalized_score:.4f}", flush=True)

self_score = scorer.score(SELF_REF_VERSION)
print(f"SELF-REF version: {self_score.normalized_score:.4f}", flush=True)

print(f"\nList -> Web delta: {web_score.normalized_score - list_score.normalized_score:+.4f}", flush=True)
print(f"Web -> Self-ref delta: {self_score.normalized_score - web_score.normalized_score:+.4f}", flush=True)

if web_score.normalized_score > list_score.normalized_score:
    print("\nDECOMPOSABILITY LAW: CONFIRMED", flush=True)
    print("Same content, integrated structure = higher depth.", flush=True)
else:
    print("\nDECOMPOSABILITY LAW: NOT CONFIRMED", flush=True)

if self_score.normalized_score > web_score.normalized_score:
    print("SELF-REFERENCE HYPOTHESIS: CONFIRMED", flush=True)
    print("Meta-cognitive self-reference produces highest depth.", flush=True)
