"""
Experiment 11: Circuit-Tracer Depth Metric Across 30 Matched Pairs
===================================================================
For each pair (surface vs insight):
  1. Run attribution graph via circuit-tracer
  2. Extract active_features: [layer, token_pos, feature_id, activation]
  3. Compute mean_activated_layer per prompt
  4. Compare surface vs insight mean layer depth

Result: 30 paired observations → paired t-test
H0: no difference in mean activation depth between surface and insight prompts
"""

import json, os, sys, time, torch, traceback
import numpy as np
from scipy import stats

sys.path.insert(0, r"C:\Users\heyzo\clawd\research\circuit-tracer")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from circuit_tracer import ReplacementModel, attribute

MODEL_NAME = "google/gemma-2-2b"
TRANSCODER = "gemma"
PAIRS_PATH = r"C:\Users\heyzo\clawd\research\experiment_11_depth_pairs.json"
OUTPUT_PATH = r"C:\Users\heyzo\clawd\research\experiment_11_results.json"
LOG_PATH = r"C:\Users\heyzo\clawd\research\experiment_11_run.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

# ── LOAD PAIRS ────────────────────────────────────────────────────────────────

with open(PAIRS_PATH, encoding="utf-8") as f:
    data = json.load(f)
pairs = data["pairs"]
log(f"loaded {len(pairs)} pairs")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

log(f"loading model: {MODEL_NAME}")
t0 = time.time()
model = ReplacementModel.from_pretrained(
    MODEL_NAME,
    TRANSCODER,
    dtype=torch.bfloat16,
    backend="transformerlens",
)
log(f"model loaded in {time.time()-t0:.1f}s")

# ── ATTRIBUTION FUNCTION ──────────────────────────────────────────────────────

def get_mean_depth(prompt, target_token):
    """Run attribution graph, return mean layer of activated features.
    Falls back to auto-target if the string token is multi-token in Gemma vocab."""
    targets = [target_token]
    for attempt in range(2):
        try:
            graph = attribute(
                prompt,
                model,
                attribution_targets=(targets if attempt == 0 else None),
                batch_size=32,
                max_feature_nodes=500,
                offload="cpu",
            )
            # active_features: tensor of shape (n_features, 3) → (layer, pos, feature_idx)
            af = graph.active_features
            if af is None or len(af) == 0:
                return None, 0
            layers = af[:, 0].float().cpu().numpy()
            return float(np.mean(layers)), len(layers)
        except ValueError as e:
            if "encoded to" in str(e) and attempt == 0:
                log(f"  multi-token target '{target_token}', falling back to auto-target")
                continue
            log(f"  ERROR: {e}")
            return None, 0
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()
            return None, 0
    return None, 0

# ── RUN EXPERIMENT ────────────────────────────────────────────────────────────

results = []
surface_depths = []
insight_depths = []

log("=" * 60)
log("RUNNING 30 MATCHED PAIRS")
log("=" * 60)

for i, pair in enumerate(pairs):
    topic = pair["topic"]
    log(f"\nPair {i+1}/{len(pairs)}: {topic}")
    
    # Surface
    log(f"  surface: {pair['surface']!r} → {pair['surface_target']!r}")
    t0 = time.time()
    s_depth, s_n = get_mean_depth(pair["surface"], pair["surface_target"])
    s_time = time.time() - t0
    s_depth_str = f"{s_depth:.3f}" if s_depth is not None else "None"
    log(f"  surface depth: {s_depth_str} ({s_n} features, {s_time:.1f}s)")
    
    # Insight
    log(f"  insight: {pair['insight']!r} → {pair['insight_target']!r}")
    t0 = time.time()
    i_depth, i_n = get_mean_depth(pair["insight"], pair["insight_target"])
    i_time = time.time() - t0
    i_depth_str = f"{i_depth:.3f}" if i_depth is not None else "None"
    log(f"  insight depth: {i_depth_str} ({i_n} features, {i_time:.1f}s)")
    
    pair_result = {
        "topic": topic,
        "surface_prompt": pair["surface"],
        "surface_target": pair["surface_target"],
        "insight_prompt": pair["insight"],
        "insight_target": pair["insight_target"],
        "surface_mean_layer": s_depth,
        "insight_mean_layer": i_depth,
        "delta": (i_depth - s_depth) if (s_depth is not None and i_depth is not None) else None,
        "surface_n_features": s_n,
        "insight_n_features": i_n,
    }
    results.append(pair_result)
    
    if s_depth is not None and i_depth is not None:
        surface_depths.append(s_depth)
        insight_depths.append(i_depth)
        delta = i_depth - s_depth
        log(f"  DELTA: {delta:+.3f} ({'insight deeper' if delta > 0 else 'surface deeper'})")
    
    # Save intermediate results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"pairs": results, "n_complete": i+1}, f, indent=2)

# ── STATISTICAL TEST ──────────────────────────────────────────────────────────

log("\n" + "=" * 60)
log("STATISTICAL TEST")
log("=" * 60)

surface_arr = np.array(surface_depths)
insight_arr = np.array(insight_depths)
deltas = insight_arr - surface_arr

log(f"  n valid pairs: {len(surface_arr)}")
log(f"  surface mean layer: {surface_arr.mean():.4f} ± {surface_arr.std():.4f}")
log(f"  insight mean layer: {insight_arr.mean():.4f} ± {insight_arr.std():.4f}")
log(f"  mean delta (insight - surface): {deltas.mean():+.4f}")
log(f"  % pairs where insight > surface: {(deltas > 0).mean()*100:.1f}%")

t_stat, p_value = stats.ttest_rel(insight_arr, surface_arr, alternative="greater")
log(f"  paired t-test (one-tailed, insight > surface):")
log(f"    t = {t_stat:.4f}")
log(f"    p = {p_value:.6f}")

if p_value < 0.05:
    verdict = f"SIGNIFICANT — insight prompts activate deeper layers (p={p_value:.4f})"
elif p_value < 0.10:
    verdict = f"MARGINAL — weak signal (p={p_value:.4f})"
else:
    verdict = f"NOT SIGNIFICANT — no depth difference detected (p={p_value:.4f})"

log(f"  VERDICT: {verdict}")

# ── SAVE FINAL RESULTS ────────────────────────────────────────────────────────

final = {
    "n_pairs": len(pairs),
    "n_valid": len(surface_arr),
    "surface_mean_layer": float(surface_arr.mean()),
    "insight_mean_layer": float(insight_arr.mean()),
    "mean_delta": float(deltas.mean()),
    "pct_insight_deeper": float((deltas > 0).mean() * 100),
    "t_statistic": float(t_stat),
    "p_value": float(p_value),
    "verdict": verdict,
    "pairs": results,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2)

log(f"\nsaved to {OUTPUT_PATH}")
log("experiment 11 complete 😁")
