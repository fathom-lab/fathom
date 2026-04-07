import gc, json, os, sys, time, torch
import numpy as np
from scipy import stats

sys.path.insert(0, r"C:\Users\heyzo\clawd\research\circuit-tracer")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from circuit_tracer import ReplacementModel, attribute

MODEL_NAME = "google/gemma-2-2b"
PAIRS_PATH = r"C:\Users\heyzo\clawd\research\experiment_12_power_pairs.json"
OUTPUT_PATH = r"C:\Users\heyzo\clawd\research\experiment_12_power_results.json"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

with open(PAIRS_PATH) as f:
    data = json.load(f)
pairs = data["pairs"]
log(f"loaded {len(pairs)} held-out replication pairs")
log(f"pre-registered hypothesis: {data['pre_registered_direction']}")

log("loading model...")
t0 = time.time()
model = ReplacementModel.from_pretrained(MODEL_NAME, "gemma", dtype=torch.bfloat16, backend="transformerlens")
log(f"loaded in {time.time()-t0:.1f}s")

def get_mean_depth(prompt, target):
    try:
        graph = attribute(prompt, model, attribution_targets=[target], batch_size=16, max_feature_nodes=500, offload="cpu")
        # active_features is tensor of shape (n_active_features, 3): (layer, pos, feature_idx)
        af = graph.active_features
        result = (None, 0)
        if af is not None and af.shape[0] > 0:
            layers = af[:, 0].float().cpu().numpy()
            result = (float(np.mean(layers)), len(layers))
        del graph, af
        gc.collect()
        torch.cuda.empty_cache()
        return result
    except Exception as e:
        log(f"  ERROR: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return None, 0

# Resume from checkpoint if exists
results = []
surface_depths, insight_depths = [], []
start_idx = 0
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH) as f:
            saved = json.load(f)
        if "pairs" in saved and isinstance(saved["pairs"], list):
            results = saved["pairs"]
            start_idx = len(results)
            for r in results:
                if r.get("surface_mean_layer") is not None and r.get("insight_mean_layer") is not None:
                    surface_depths.append(r["surface_mean_layer"])
                    insight_depths.append(r["insight_mean_layer"])
            log(f"resuming from checkpoint: {start_idx}/{len(pairs)} pairs already done")
    except Exception as e:
        log(f"could not load checkpoint: {e}, starting fresh")
        results = []
        start_idx = 0

for i, pair in enumerate(pairs):
    if i < start_idx:
        continue
    log(f"\nPair {i+1}/{len(pairs)}: {pair['topic']}")
    s_d, s_n = get_mean_depth(pair["surface"], pair["surface_target"])
    i_d, i_n = get_mean_depth(pair["insight"], pair["insight_target"])
    s_str = f"{s_d:.3f}" if s_d is not None else "None"
    i_str = f"{i_d:.3f}" if i_d is not None else "None"
    delta = (s_d - i_d) if (s_d is not None and i_d is not None) else None
    delta_str = f"{delta:.3f}" if delta is not None else "N/A"
    log(f"  surface: {s_str} ({s_n} features) | insight: {i_str} ({i_n} features) | delta: {delta_str}")
    results.append({"topic": pair["topic"], "surface_mean_layer": s_d, "insight_mean_layer": i_d, "delta": delta, "surface_n": s_n, "insight_n": i_n})
    if s_d is not None and i_d is not None:
        surface_depths.append(s_d)
        insight_depths.append(i_d)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"pairs": results, "n_complete": i+1}, f, indent=2)
    gc.collect()
    torch.cuda.empty_cache()

# STATISTICAL TEST - PRE-REGISTERED DIRECTION: surface > insight
s_arr, i_arr = np.array(surface_depths), np.array(insight_depths)
t_stat, p_value = stats.ttest_rel(s_arr, i_arr, alternative="greater")
deltas = s_arr - i_arr
pct_surface_deeper = (deltas > 0).mean() * 100

verdict = f"REPLICATION {'CONFIRMED' if p_value < 0.05 else 'FAILED'} — surface > insight (p={p_value:.6f})"

log(f"\n{'='*50}")
log(f"REPLICATION RESULTS (pre-registered: surface > insight)")
log(f"surface mean: {s_arr.mean():.4f} | insight mean: {i_arr.mean():.4f}")
log(f"mean delta: {deltas.mean():+.4f}")
log(f"% surface deeper: {pct_surface_deeper:.1f}%")
log(f"t = {t_stat:.4f}, p = {p_value:.6f}")
log(f"VERDICT: {verdict}")

final = {"n_pairs": len(pairs), "n_valid": len(s_arr), "surface_mean_layer": float(s_arr.mean()), "insight_mean_layer": float(i_arr.mean()), "mean_delta": float(deltas.mean()), "pct_surface_deeper": float(pct_surface_deeper), "t_statistic": float(t_stat), "p_value": float(p_value), "verdict": verdict, "pairs": results}
with open(OUTPUT_PATH, "w") as f:
    json.dump(final, f, indent=2)
log(f"saved to {OUTPUT_PATH}")
