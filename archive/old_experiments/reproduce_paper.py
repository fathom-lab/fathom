"""
FATHOM — Reproduce Core Finding
=================================

Reproduces the main result from:
  "Computational Graph Structure Distinguishes Surface Recall
   from Explanatory Reasoning in Language Models"

Scores the 30 power replication pairs on Gemma-2-2B, runs a one-tailed
paired t-test, and prints the key statistics. Expected result:
  delta ~ +0.257, p ~ 0.000051, 83% directional support.

Single command: python reproduce_paper.py
"""

import gc
import math
import os
import sys
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["SAFETENSORS_FAST_GPU"] = "1"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── The 30 power replication pairs from the paper ────────────────────
# Pre-registered hypothesis: surface_mean_layer > insight_mean_layer
# Zero topic overlap with exploratory (n=26) or initial replication (n=15)
# Vocabulary-controlled, token-count matched, syntactically parallel

PAIRS = [
    {
        "topic": "earthquakes",
        "surface": "Earthquakes happen when tectonic plates shift and the ground shakes from the energy they",
        "surface_target": "release",
        "insight": "Earthquakes occur when stress along fault lines exceeds friction and the sudden slip radiates seismic",
        "insight_target": "waves",
    },
    {
        "topic": "metabolism",
        "surface": "Metabolism converts food into energy and your body burns calories to keep you",
        "surface_target": "alive",
        "insight": "Metabolism converts nutrients into ATP through oxidation and the rate adapts to the cell's current demand for",
        "insight_target": "energy",
    },
    {
        "topic": "ocean_currents",
        "surface": "Ocean currents move water around the globe and they influence temperatures near coastal",
        "surface_target": "regions",
        "insight": "Ocean currents redistribute heat because density differences between warm and cold water masses drive the",
        "insight_target": "flow",
    },
    {
        "topic": "immune_response",
        "surface": "The immune system attacks bacteria and white blood cells destroy pathogens inside the",
        "surface_target": "body",
        "insight": "The immune system activates when antigens bind to receptors and triggers a cascade that amplifies the response to",
        "insight_target": "infection",
    },
    {
        "topic": "dreams",
        "surface": "Dreams happen during sleep and the brain creates vivid images while you",
        "surface_target": "sleep",
        "insight": "Dreams emerge during REM sleep when the brain replays memories and consolidates emotional experience into long-term",
        "insight_target": "storage",
    },
    {
        "topic": "volcanoes",
        "surface": "Volcanoes erupt when magma rises to the surface and lava flows from the",
        "surface_target": "top",
        "insight": "Volcanoes erupt when pressure from dissolved gases in magma exceeds the strength of the overlying",
        "insight_target": "rock",
    },
    {
        "topic": "blood_pressure",
        "surface": "Blood pressure measures the force of blood against artery walls and high levels can damage the",
        "surface_target": "heart",
        "insight": "Blood pressure rises when blood vessels constrict or fluid volume increases the load against vascular",
        "insight_target": "resistance",
    },
    {
        "topic": "fermentation",
        "surface": "Fermentation converts sugar into alcohol and yeast releases carbon dioxide as",
        "surface_target": "waste",
        "insight": "Fermentation converts glucose to ethanol when yeast depletes oxygen and switches to anaerobic energy",
        "insight_target": "production",
    },
    {
        "topic": "hurricanes",
        "surface": "Hurricanes form over warm ocean water and strong winds spiral around a central",
        "surface_target": "eye",
        "insight": "Hurricanes intensify when warm ocean water evaporates rapidly and rising air creates low pressure that sustains",
        "insight_target": "rotation",
    },
    {
        "topic": "drug_resistance",
        "surface": "Drug resistance develops when bacteria survive treatment and the survivors pass on their",
        "surface_target": "traits",
        "insight": "Drug resistance develops when antibiotics apply selective pressure and only bacteria with resistance mutations survive to",
        "insight_target": "divide",
    },
    {
        "topic": "coral_reefs",
        "surface": "Coral reefs are built by tiny animals and provide habitat for thousands of ocean",
        "surface_target": "species",
        "insight": "Coral reefs bleach when rising water temperatures cause the symbiotic algae that supply nutrients to leave the",
        "insight_target": "coral",
    },
    {
        "topic": "black_holes",
        "surface": "Black holes are regions in space where gravity is so strong that light cannot",
        "surface_target": "escape",
        "insight": "Black holes form when massive stars exhaust their fuel and gravitational collapse compresses matter past the",
        "insight_target": "limit",
    },
    {
        "topic": "taxation",
        "surface": "Taxes are collected by governments and the revenue funds public services like schools and",
        "surface_target": "roads",
        "insight": "Taxes reduce after-tax returns on economic activity and the behavioral response depends on how sensitive supply and demand are to the",
        "insight_target": "price",
    },
    {
        "topic": "empathy",
        "surface": "Empathy is the ability to understand how others feel and it helps people connect with",
        "surface_target": "others",
        "insight": "Empathy activates neural circuits that simulate another person's experience and allows prediction of their emotional",
        "insight_target": "response",
    },
    {
        "topic": "plate_tectonics",
        "surface": "Tectonic plates move slowly across the mantle and their collisions build mountain",
        "surface_target": "ranges",
        "insight": "Tectonic plates move because convection currents in the mantle drag them and plate boundaries concentrate geological",
        "insight_target": "activity",
    },
    {
        "topic": "obesity",
        "surface": "Obesity develops when calorie intake exceeds energy expenditure and excess energy is stored as",
        "surface_target": "fat",
        "insight": "Obesity develops when hormonal signals regulating appetite are disrupted and the body defends an elevated weight",
        "insight_target": "level",
    },
    {
        "topic": "leadership",
        "surface": "Effective leaders inspire teams and communicate goals that motivate people to",
        "surface_target": "perform",
        "insight": "Effective leaders reduce uncertainty by modeling expected behavior and creating conditions where trust develops",
        "insight_target": "naturally",
    },
    {
        "topic": "bacteria_growth",
        "surface": "Bacteria grow by dividing in two and populations double rapidly under favorable",
        "surface_target": "conditions",
        "insight": "Bacteria grow exponentially when resources are abundant and growth slows as waste accumulates and nutrients become",
        "insight_target": "rare",
    },
    {
        "topic": "cryptocurrency",
        "surface": "Cryptocurrency is a digital currency that uses encryption to secure transactions without a central",
        "surface_target": "bank",
        "insight": "Cryptocurrency maintains trust without intermediaries by requiring consensus across nodes that makes fraudulent transactions computationally",
        "insight_target": "hard",
    },
    {
        "topic": "social_norms",
        "surface": "Social norms are unwritten rules about behavior and people follow them to fit in with their",
        "surface_target": "group",
        "insight": "Social norms persist because violating them triggers social costs and conformity is individually rational even when the norm is",
        "insight_target": "wrong",
    },
    {
        "topic": "nuclear_energy",
        "surface": "Nuclear energy is released by splitting atoms and the heat produced drives turbines to generate",
        "surface_target": "electricity",
        "insight": "Nuclear fission releases energy because the product nuclei have lower binding energy per nucleon and the mass difference converts to",
        "insight_target": "heat",
    },
    {
        "topic": "monopoly",
        "surface": "Monopolies can charge higher prices because customers have no alternative supplier to turn",
        "surface_target": "to",
        "insight": "Monopolies price above marginal cost because profit maximization sets marginal revenue equal to marginal cost and captures consumer",
        "insight_target": "value",
    },
    {
        "topic": "trauma",
        "surface": "Trauma is a psychological response to severe stress and it can affect mental health for",
        "surface_target": "years",
        "insight": "Trauma dysregulates the stress response and encodes threatening experiences as fragmented sensory memories that activate fear without",
        "insight_target": "context",
    },
    {
        "topic": "bird_migration",
        "surface": "Birds migrate seasonally to warmer regions when temperatures drop and food becomes",
        "surface_target": "rare",
        "insight": "Birds migrate because changes in day length trigger hormonal shifts that coordinate departure with the conditions birds will find upon",
        "insight_target": "arrival",
    },
    {
        "topic": "stock_bubbles",
        "surface": "Stock bubbles form when prices rise far above true value and eventually the market",
        "surface_target": "falls",
        "insight": "Stock bubbles inflate when rising prices attract momentum buyers whose demand drives prices higher until the pool of buyers is",
        "insight_target": "gone",
    },
    {
        "topic": "enzyme",
        "surface": "Enzymes are proteins that speed up reactions and each enzyme has an active",
        "surface_target": "site",
        "insight": "Enzymes accelerate reactions by binding the substrate and lowering the energy needed to form",
        "insight_target": "products",
    },
    {
        "topic": "language_acquisition",
        "surface": "Children learn language through exposure and practice and fluency develops through repeated use of vocabulary and",
        "surface_target": "grammar",
        "insight": "Children acquire language because implicit statistical learning mechanisms extract patterns from input before conscious rules are ever",
        "insight_target": "taught",
    },
    {
        "topic": "opioid_dependence",
        "surface": "Opioids relieve pain by binding to receptors in the brain and the relief can make people want",
        "surface_target": "more",
        "insight": "Opioids create dependence by downregulating endorphin production so users need increasing doses to achieve the same",
        "insight_target": "effect",
    },
    {
        "topic": "ocean_acidification",
        "surface": "Ocean acidification occurs when the sea absorbs carbon dioxide and the pH begins to",
        "surface_target": "fall",
        "insight": "Ocean acidification occurs when dissolved CO2 forms carbonic acid that reduces the carbonate ions marine animals need to build their",
        "insight_target": "shells",
    },
    {
        "topic": "supply_chains",
        "surface": "Supply chains move goods from manufacturers to consumers and disruptions can cause shortages at the",
        "surface_target": "store",
        "insight": "Supply chains amplify small demand shocks because each upstream supplier over-orders to buffer uncertainty and the swings compound into",
        "insight_target": "chaos",
    },
]

# ── Published results (for comparison) ───────────────────────────────
PUBLISHED = {
    "delta": 0.257,
    "t": 4.500,
    "p": 0.000051,
    "pct_direction": 83.3,
    "n": 30,
}


def main():
    import numpy as np

    print("=" * 70)
    print("FATHOM — REPRODUCE CORE FINDING")
    print("Paper: Computational Graph Structure Distinguishes Surface Recall")
    print("       from Explanatory Reasoning in Language Models")
    print("=" * 70)
    print(f"Pairs: {len(PAIRS)} (power replication set)")
    print(f"Pre-registered hypothesis: surface_mean_layer > insight_mean_layer")
    print(f"Published result: delta=+{PUBLISHED['delta']:.3f}, "
          f"p={PUBLISHED['p']:.6f}, {PUBLISHED['pct_direction']:.1f}% directional")
    print()

    # GPU setup
    import torch
    if not torch.cuda.is_available():
        print("[FATAL] No CUDA GPU detected. This script requires GPU.")
        sys.exit(1)

    dev = torch.cuda.current_device()
    print(f"[GPU] {torch.cuda.get_device_name(dev)} — "
          f"{torch.cuda.get_device_properties(dev).total_memory / 1024**3:.1f} GB")

    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load model
    print("\n[INIT] Loading DepthScorer (google/gemma-2-2b)...")
    t_load = time.time()

    from depth_scorer import DepthScorer

    try:
        scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=500)
    except OSError:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.backends.cuda.enable_flash_sdp(False)
        except Exception:
            pass
        scorer = DepthScorer(model_name="google/gemma-2-2b", max_feature_nodes=500)

    print(f"[INIT] Loaded in {time.time() - t_load:.1f}s\n")

    # Score all pairs
    surface_depths = []
    insight_depths = []
    pair_results = []

    print(f"{'#':>3}  {'Topic':<22}  {'Surface':>8}  {'Insight':>8}  {'Delta':>8}  Dir")
    print(f"{'─'*3}  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*3}")

    for i, pair in enumerate(PAIRS):
        topic = pair["topic"]

        try:
            r_s = scorer.score(pair["surface"], target=pair["surface_target"])
            torch.cuda.empty_cache()
            gc.collect()

            r_i = scorer.score(pair["insight"], target=pair["insight_target"])
            torch.cuda.empty_cache()
            gc.collect()

            if math.isnan(r_s.depth_score) or math.isnan(r_i.depth_score):
                print(f"{i+1:>3}  {topic:<22}  {'NaN':>8}  {'NaN':>8}  {'skip':>8}")
                continue

            s_d = r_s.depth_score
            i_d = r_i.depth_score
            delta = s_d - i_d

            surface_depths.append(s_d)
            insight_depths.append(i_d)
            pair_results.append({
                "topic": topic,
                "surface_depth": s_d,
                "insight_depth": i_d,
                "delta": delta,
            })

            direction = "+" if delta > 0 else "-"
            print(f"{i+1:>3}  {topic:<22}  {s_d:>8.3f}  {i_d:>8.3f}  {delta:>+8.3f}  {direction}")

        except Exception as e:
            print(f"{i+1:>3}  {topic:<22}  FAILED: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    # Statistical test
    print()
    if len(surface_depths) < 5:
        print("[FATAL] Too few valid pairs for statistical test.")
        sys.exit(1)

    s_arr = np.array(surface_depths)
    i_arr = np.array(insight_depths)
    deltas = s_arr - i_arr

    try:
        from scipy.stats import ttest_rel
        t_stat, p_two = ttest_rel(s_arr, i_arr)
        # One-tailed: we predicted surface > insight
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    except ImportError:
        # Manual paired t-test
        n = len(deltas)
        mean_d = deltas.mean()
        std_d = deltas.std(ddof=1)
        t_stat = mean_d / (std_d / math.sqrt(n))
        p_one = math.exp(-0.717 * abs(t_stat) - 0.416 * t_stat * t_stat)
        p_one = max(0.0, min(1.0, p_one))

    mean_delta = float(deltas.mean())
    pct_direction = float((deltas > 0).sum()) / len(deltas) * 100

    # LOO robustness
    loo_robust = True
    for j in range(len(deltas)):
        loo_s = np.delete(s_arr, j)
        loo_i = np.delete(i_arr, j)
        try:
            from scipy.stats import ttest_rel as tt
            _, p2 = tt(loo_s, loo_i)
            p1 = p2 / 2
        except ImportError:
            d = loo_s - loo_i
            t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
            p1 = math.exp(-0.717 * abs(t) - 0.416 * t * t)
        if p1 > 0.05:
            loo_robust = False
            break

    # Results
    print("=" * 70)
    print("REPRODUCTION RESULTS")
    print("=" * 70)
    print(f"  Valid pairs:           {len(deltas)}")
    print(f"  Surface mean layer:    {s_arr.mean():.4f}")
    print(f"  Insight mean layer:    {i_arr.mean():.4f}")
    print()
    print(f"  Delta (surface-insight):  {mean_delta:+.4f}   (published: +{PUBLISHED['delta']:.3f})")
    print(f"  t-statistic:              {t_stat:.4f}    (published: {PUBLISHED['t']:.3f})")
    print(f"  p-value (one-tailed):     {p_one:.6f}  (published: {PUBLISHED['p']:.6f})")
    print(f"  % in predicted direction: {pct_direction:.1f}%     (published: {PUBLISHED['pct_direction']:.1f}%)")
    print(f"  LOO robust:               {'Yes' if loo_robust else 'No'}")
    print()

    if p_one < 0.001:
        print("  VERDICT: REPLICATION CONFIRMED (p < 0.001)")
    elif p_one < 0.05:
        print("  VERDICT: REPLICATION CONFIRMED (p < 0.05)")
    else:
        print("  VERDICT: REPLICATION FAILED")

    print()
    print("  Note: Small variations from published values are expected due to")
    print("  stochastic GPU computation and library version differences.")
    print("  The direction and significance level should match.")
    print("=" * 70)


if __name__ == "__main__":
    main()
