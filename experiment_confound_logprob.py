"""
Experiment: Prediction Uncertainty Confound Test
=================================================
Tests whether the depth effect (surface > insight mean layer) is explained by
prediction uncertainty (surface prompts having higher confidence / lower perplexity).

Method: For each of the 30 power-replication pairs, compute:
  - log P(target | surface_prompt) using Gemma-2-2B
  - log P(target | insight_prompt) using Gemma-2-2B

If prediction confidence drives depth: surface should have HIGHER confidence (higher log_prob)
AND higher mean layer. The correlation between (surface_logprob - insight_logprob) and
(surface_depth - insight_depth) would be positive and strong.

If confidence doesn't explain depth: the correlation should be near zero.

Uses the already-loaded Gemma-2-2B from the research venv.
"""

import json
import torch
import numpy as np
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

PAIRS_PATH = r"C:\Users\heyzo\clawd\research\experiment_12_power_pairs.json"
DEPTH_RESULTS_PATH = r"C:\Users\heyzo\clawd\research\experiment_12_power_results.json"
OUTPUT_PATH = r"C:\Users\heyzo\clawd\research\experiment_confound_results.json"

MODEL_NAME = "google/gemma-2-2b"

print("Loading tokenizer + model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Model loaded.")

# Load pairs
with open(PAIRS_PATH) as f:
    data = json.load(f)
pairs = data["pairs"]

# Load depth results for correlation analysis
with open(DEPTH_RESULTS_PATH) as f:
    depth_data = json.load(f)
depth_by_topic = {p["topic"]: p for p in depth_data["pairs"]}

def get_log_prob(prompt, target):
    """Get log P(target | prompt) using the model."""
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if len(target_ids) != 1:
        return None, f"multi-token target: {target} -> {target_ids}"
    
    target_id = target_ids[0]
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # last position
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_log_prob = log_probs[target_id].item()
    
    return target_log_prob, None

results = []
log_prob_deltas = []  # surface_lp - insight_lp (positive = surface more confident)
depth_deltas = []     # from depth results (surface_depth - insight_depth, positive = surface deeper)

print("\nRunning log-probability analysis on 30 pairs...")
print("=" * 60)

for i, pair in enumerate(pairs):
    topic = pair["topic"]
    surface_prompt = pair["surface"]
    insight_prompt = pair["insight"]
    surface_target = pair.get("surface_target", "")
    insight_target = pair.get("insight_target", "")
    
    print(f"\nPair {i+1}/30: {topic}")
    
    # Get log probs
    s_lp, s_err = get_log_prob(surface_prompt, surface_target)
    i_lp, i_err = get_log_prob(insight_prompt, insight_target)
    
    if s_err or i_err:
        print(f"  SKIP - surface_err: {s_err}, insight_err: {i_err}")
        continue
    
    lp_delta = s_lp - i_lp  # positive = surface more confident
    
    # Get depth delta from stored results
    depth_pair = depth_by_topic.get(topic)
    if not depth_pair:
        print(f"  SKIP - no depth data for {topic}")
        continue
    
    depth_delta = depth_pair["delta"]  # surface_depth - insight_depth (positive = surface higher)
    
    print(f"  surface log_prob: {s_lp:.4f}, insight log_prob: {i_lp:.4f}")
    print(f"  lp_delta (s-i): {lp_delta:+.4f} | depth_delta (s-i): {depth_delta:+.4f}")
    
    log_prob_deltas.append(lp_delta)
    depth_deltas.append(depth_delta)
    
    results.append({
        "topic": topic,
        "surface_log_prob": s_lp,
        "insight_log_prob": i_lp,
        "lp_delta": lp_delta,
        "depth_delta": depth_delta,
    })

print("\n" + "=" * 60)
print(f"\nN valid pairs: {len(results)}")

if len(results) < 10:
    print("Too few pairs to analyze.")
else:
    lp_arr = np.array(log_prob_deltas)
    depth_arr = np.array(depth_deltas)
    
    # Correlation between log_prob_delta and depth_delta
    r, p_r = stats.pearsonr(lp_arr, depth_arr)
    rho, p_rho = stats.spearmanr(lp_arr, depth_arr)
    
    # Is surface more confident on average?
    t_lp, p_lp = stats.ttest_1samp(lp_arr, 0)
    pct_surface_more_confident = np.mean(lp_arr > 0) * 100
    
    print(f"\n--- PREDICTION CONFIDENCE ANALYSIS ---")
    print(f"Surface more confident (lp_delta > 0): {pct_surface_more_confident:.1f}% of pairs")
    print(f"Mean lp_delta: {np.mean(lp_arr):+.4f} (SD={np.std(lp_arr):.4f})")
    print(f"t-test (lp_delta vs 0): t={t_lp:.3f}, p={p_lp:.4f}")
    
    print(f"\n--- CONFOUND CORRELATION ---")
    print(f"Pearson r(lp_delta, depth_delta): r={r:.3f}, p={p_r:.4f}")
    print(f"Spearman rho: rho={rho:.3f}, p={p_rho:.4f}")
    
    if abs(r) < 0.3 and p_r > 0.1:
        verdict = "CONFOUND RULED OUT — prediction confidence does not explain depth effect"
    elif r > 0.5 and p_r < 0.05:
        verdict = "CONFOUND PRESENT — prediction confidence correlated with depth effect"
    else:
        verdict = "INCONCLUSIVE — weak correlation, further analysis needed"
    
    print(f"\nVERDICT: {verdict}")
    
    output = {
        "n_pairs": len(results),
        "pct_surface_more_confident": pct_surface_more_confident,
        "mean_lp_delta": float(np.mean(lp_arr)),
        "sd_lp_delta": float(np.std(lp_arr)),
        "t_lp_delta": float(t_lp),
        "p_lp_delta": float(p_lp),
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "verdict": verdict,
        "pairs": results,
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {OUTPUT_PATH}")
