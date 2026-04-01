"""
K_PROXY VALIDATION — Gemma-2-2B Ground Truth Test
==================================================
Run the entropy proxy on Gemma-2-2B where we KNOW K=1.027 from SAE measurement.
If K_proxy ≈ 1.027 → proxy works, Pythia-160m was just too small.
If K_proxy ≈ 1.0   → proxy is blind, need SAE-based approach for checkpoint scan.
"""

import torch
import numpy as np
import json
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "k_emergence_results")

# Same probe pairs as the Pythia experiment
PROBE_PAIRS = [
    {
        "surface": "Earthquakes happen when tectonic plates shift and the ground shakes from the energy they",
        "surface_target": " release",
        "insight": "Earthquakes occur when accumulated stress along fault lines exceeds friction and the sudden slip radiates seismic",
        "insight_target": " waves",
    },
    {
        "surface": "Black holes are regions in space where gravity is so strong that light cannot",
        "surface_target": " escape",
        "insight": "Black holes form when massive stars exhaust their fuel and gravitational collapse compresses matter past the",
        "insight_target": " limit",
    },
    {
        "surface": "Hurricanes form over warm ocean water and strong winds spiral around a central",
        "surface_target": " eye",
        "insight": "Hurricanes intensify because warm ocean water evaporates rapidly and rising air creates low pressure that sustains",
        "insight_target": " rotation",
    },
    {
        "surface": "Nuclear energy is released by splitting atoms and the heat produced drives turbines to generate",
        "surface_target": " electricity",
        "insight": "In fission the product nuclei have lower binding energy per nucleon and the mass difference converts to",
        "insight_target": " heat",
    },
    {
        "surface": "Birds migrate seasonally to warmer regions when temperatures drop and food becomes",
        "surface_target": " rare",
        "insight": "Migration is triggered because changes in day length trigger hormonal shifts that coordinate departure with the conditions birds will find upon",
        "insight_target": " arrival",
    },
    {
        "surface": "Enzymes are proteins that speed up reactions and each enzyme has an active",
        "surface_target": " site",
        "insight": "Enzymes catalyze reactions by binding the substrate and lowering the energy needed to form",
        "insight_target": " products",
    },
    {
        "surface": "Volcanoes erupt when magma rises to the surface and lava flows from the",
        "surface_target": " top",
        "insight": "Eruptions occur when pressure from dissolved gases in magma exceeds the strength of the overlying",
        "insight_target": " rock",
    },
    {
        "surface": "Coral reefs are built by tiny animals and provide habitat for thousands of ocean",
        "surface_target": " species",
        "insight": "Bleaching occurs when rising temperatures cause the symbiotic algae that supply nutrients to leave the",
        "insight_target": " coral",
    },
    {
        "surface": "Drug resistance develops when bacteria survive treatment and the survivors pass on their",
        "surface_target": " traits",
        "insight": "Antibiotics apply selective pressure and only bacteria with resistance mutations survive to",
        "insight_target": " divide",
    },
    {
        "surface": "Supply chains move goods from manufacturers to consumers and disruptions can cause shortages at the",
        "surface_target": " store",
        "insight": "Supply chains amplify small demand shocks because each upstream supplier over-orders to buffer uncertainty and the swings compound into",
        "insight_target": " chaos",
    },
]


def get_layer_activations_gemma(model, tokenizer, text, device="cuda"):
    """Capture hidden states at every layer for Gemma-2-2B."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    hidden_states = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach())
        else:
            hidden_states.append(output.detach())

    hooks = []
    # Gemma-2-2B uses model.layers
    for layer in model.model.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return torch.stack(hidden_states)


def compute_layer_entropy(activations):
    """Shannon entropy of activation magnitude distribution per layer."""
    magnitudes = activations.squeeze(1).abs().mean(dim=1)
    probs = magnitudes / (magnitudes.sum(dim=1, keepdim=True) + 1e-10)
    entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=1)
    return entropy.cpu().numpy()


def compute_depth_proxy(model, tokenizer, prompt, target, device="cuda"):
    """Compute depth proxy for a single prompt on Gemma."""
    full_text = prompt + target
    activations = get_layer_activations_gemma(model, tokenizer, full_text, device)
    entropy = compute_layer_entropy(activations)

    n_layers = len(entropy)
    early = entropy[:n_layers // 3]
    mid = entropy[n_layers // 3: 2 * n_layers // 3]
    late = entropy[2 * n_layers // 3:]

    return {
        "mean_entropy": float(np.mean(entropy)),
        "early_entropy": float(np.mean(early)),
        "mid_entropy": float(np.mean(mid)),
        "late_entropy": float(np.mean(late)),
        "layer_entropies": entropy.tolist(),
        "late_early_ratio": float(np.mean(late) / (np.mean(early) + 1e-10)),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VALIDATION] Device: {device}")
    print(f"[VALIDATION] Model: google/gemma-2-2b")
    print(f"[VALIDATION] Ground truth K: 1.027 (from SAE measurement)")
    print()

    print("[VALIDATION] Loading Gemma-2-2B...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        dtype=torch.float16,
    ).to(device)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Layers: {len(model.model.layers)}")

    surface_ratios = []
    insight_ratios = []

    for idx, pair in enumerate(PROBE_PAIRS):
        s = compute_depth_proxy(model, tokenizer, pair["surface"], pair["surface_target"], device)
        i = compute_depth_proxy(model, tokenizer, pair["insight"], pair["insight_target"], device)
        surface_ratios.append(s["late_early_ratio"])
        insight_ratios.append(i["late_early_ratio"])
        print(f"  Pair {idx+1}: surface={s['late_early_ratio']:.4f}  insight={i['late_early_ratio']:.4f}  "
              f"delta={s['late_early_ratio'] - i['late_early_ratio']:+.4f}")

    s_mean = np.mean(surface_ratios)
    i_mean = np.mean(insight_ratios)
    K_proxy = s_mean / i_mean if i_mean > 0 else 0.0
    delta = s_mean - i_mean
    t_stat, p_value = stats.ttest_rel(surface_ratios, insight_ratios)

    print()
    print("=" * 60)
    print("PROXY VALIDATION RESULTS — Gemma-2-2B")
    print("=" * 60)
    print(f"  K_proxy (entropy):  {K_proxy:.4f}")
    print(f"  K (SAE, ground):    1.027")
    print(f"  Delta:              {delta:+.4f}")
    print(f"  Surface mean:       {s_mean:.4f}")
    print(f"  Insight mean:       {i_mean:.4f}")
    print(f"  t-stat:             {t_stat:.4f}")
    print(f"  p-value:            {p_value:.6f}")
    print()
    if abs(K_proxy - 1.027) < 0.05 and p_value < 0.05:
        print("  >>> PROXY VALIDATED — entropy captures the K signal")
        print("  >>> Pythia-160m was too small. Scale to 410m/1b.")
    elif K_proxy > 1.005 and p_value < 0.1:
        print("  >>> PROXY PARTIALLY VALIDATED — signal present but weaker")
        print("  >>> May need refinement before checkpoint scan.")
    else:
        print("  >>> PROXY FAILED — entropy does not capture K")
        print("  >>> Need SAE-based measurement for checkpoint scan.")
    print("=" * 60)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "model": "google/gemma-2-2b",
        "K_proxy": float(K_proxy),
        "K_ground_truth": 1.027,
        "delta": float(delta),
        "surface_mean": float(s_mean),
        "insight_mean": float(i_mean),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "surface_ratios": [float(x) for x in surface_ratios],
        "insight_ratios": [float(x) for x in insight_ratios],
    }
    with open(os.path.join(RESULTS_DIR, "gemma_proxy_validation.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
