"""
K EMERGENCE EXPERIMENT — Pythia-160m Training Checkpoint Scan
=============================================================
Measures K_proxy (layer-wise activation entropy ratio) across Pythia-160m
training checkpoints to detect phase transition in depth routing.

Hypothesis: K snaps discontinuously during training (phase transition),
not gradually (crossover). dK/dt spike = true transition.

Uses entropy proxy instead of SAE-based depth scoring (no pre-trained
SAEs available for Pythia). Proxy is monotonically related to true K.

Authors: darkflobi + claude code
Date: 2026-03-31
"""

import torch
import numpy as np
import json
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "EleutherAI/pythia-160m"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "pythia_checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "k_emergence_results")

# Coarse scan: logarithmically spaced through training
# Pythia-160m has 143000 steps total
CHECKPOINTS = [
    "step1",
    "step16",
    "step64",
    "step256",
    "step1000",
    "step2000",
    "step4000",
    "step8000",
    "step16000",
    "step32000",
    "step64000",
    "step128000",
    "step143000",
]

# Strategy: hit late ones first to confirm K is stable at maturity,
# then scan backwards (per darkflobi's recommendation)
SCAN_ORDER = [
    "step143000",  # final
    "step128000",  # near-final
    "step64000",   # late training
    "step16000",   # mid training
    "step8000",    # early-mid
    "step4000",    # early
    "step2000",    # early
    "step1000",    # very early
    "step256",     # initialization region
    "step64",      # initialization
    "step16",      # near-random
    "step1",       # random init
]

# Probe prompts — matched pairs from the Fathom paper (subset for speed)
# Surface = shallow retrieval, Insight = deep reasoning
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


# ── Core Measurement Functions ───────────────────────────────────────────────

def get_layer_activations(model, tokenizer, text, device="cuda"):
    """
    Run forward pass with hooks to capture hidden state at every layer.
    Returns tensor of shape (n_layers, seq_len, hidden_dim).
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    hidden_states = []

    def hook_fn(module, input, output):
        # GPTNeoX layers output a tuple: (hidden_state, ...)
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach())
        else:
            hidden_states.append(output.detach())

    hooks = []
    for layer in model.gpt_neox.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return torch.stack(hidden_states)  # (n_layers, 1, seq_len, hidden_dim)


def compute_layer_entropy(activations):
    """
    Compute Shannon entropy of activation magnitude distribution per layer.
    Returns array of shape (n_layers,) — entropy at each layer.
    """
    # activations: (n_layers, 1, seq_len, hidden_dim)
    # Take magnitude of activations, average over sequence
    magnitudes = activations.squeeze(1).abs().mean(dim=1)  # (n_layers, hidden_dim)

    # Normalize to probability distribution per layer
    probs = magnitudes / (magnitudes.sum(dim=1, keepdim=True) + 1e-10)

    # Shannon entropy
    entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=1)  # (n_layers,)

    return entropy.cpu().numpy()


def compute_depth_proxy(model, tokenizer, prompt, target, device="cuda"):
    """
    Compute depth proxy for a single prompt.
    Returns the mean layer entropy (higher entropy = more distributed computation).

    For surface prompts: expect late-layer concentration (lower entropy in early, higher in late)
    For insight prompts: expect early-distributed computation (higher entropy in early/mid)
    """
    full_text = prompt + target
    activations = get_layer_activations(model, tokenizer, full_text, device)
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


def measure_K_proxy(model, tokenizer, pairs, device="cuda"):
    """
    Measure K_proxy across all probe pairs.
    K_proxy = mean(late_early_ratio for surface) / mean(late_early_ratio for insight)

    If K_proxy > 1: surface prompts concentrate more in late layers than insight prompts
    (same direction as the original K finding).
    """
    surface_ratios = []
    insight_ratios = []

    for pair in pairs:
        s = compute_depth_proxy(model, tokenizer, pair["surface"], pair["surface_target"], device)
        i = compute_depth_proxy(model, tokenizer, pair["insight"], pair["insight_target"], device)
        surface_ratios.append(s["late_early_ratio"])
        insight_ratios.append(i["late_early_ratio"])

    s_mean = np.mean(surface_ratios)
    i_mean = np.mean(insight_ratios)
    K_proxy = s_mean / i_mean if i_mean > 0 else 0.0
    delta = s_mean - i_mean

    # Statistical test
    t_stat, p_value = stats.ttest_rel(surface_ratios, insight_ratios)

    return {
        "K_proxy": float(K_proxy),
        "delta": float(delta),
        "surface_mean": float(s_mean),
        "insight_mean": float(i_mean),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_pairs": len(pairs),
        "surface_ratios": [float(x) for x in surface_ratios],
        "insight_ratios": [float(x) for x in insight_ratios],
    }


# ── Experiment Runner ────────────────────────────────────────────────────────

def run_coarse_scan():
    """Run K_proxy measurement across all checkpoints in strategic order."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[K-EMERGENCE] Device: {device}")
    print(f"[K-EMERGENCE] Model: {MODEL_NAME}")
    print(f"[K-EMERGENCE] Checkpoints: {len(SCAN_ORDER)}")
    print(f"[K-EMERGENCE] Probe pairs: {len(PROBE_PAIRS)}")
    print()

    # Load tokenizer once (same across checkpoints)
    print("[K-EMERGENCE] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    results_file = os.path.join(RESULTS_DIR, "coarse_scan.json")

    # Resume from partial results if they exist
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"[K-EMERGENCE] Resuming — {len(results)} checkpoints already done")

    for idx, step in enumerate(SCAN_ORDER):
        if step in results:
            print(f"[{idx+1}/{len(SCAN_ORDER)}] {step} — CACHED (K_proxy={results[step]['K_proxy']:.4f})")
            continue

        print(f"[{idx+1}/{len(SCAN_ORDER)}] Loading {MODEL_NAME} @ {step}...")
        t0 = time.time()

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                revision=step,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16,
            ).to(device)
            model.eval()
            load_time = time.time() - t0
            print(f"  Loaded in {load_time:.1f}s")

            t1 = time.time()
            result = measure_K_proxy(model, tokenizer, PROBE_PAIRS, device)
            measure_time = time.time() - t1
            result["load_time"] = load_time
            result["measure_time"] = measure_time
            result["step"] = step
            result["step_num"] = int(step.replace("step", ""))

            results[step] = result
            print(f"  K_proxy={result['K_proxy']:.4f}  delta={result['delta']:+.4f}  "
                  f"p={result['p_value']:.6f}  ({measure_time:.1f}s)")

            # Save after each checkpoint (crash-safe)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            results[step] = {"error": str(e), "step": step, "step_num": int(step.replace("step", ""))}

        finally:
            # Free VRAM
            if 'model' in dir():
                del model
            torch.cuda.empty_cache()

    print("\n[K-EMERGENCE] Coarse scan complete!")
    return results


def plot_results(results):
    """Plot K_proxy vs training step and dK/dt."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Filter valid results and sort by step
    valid = [(r["step_num"], r["K_proxy"], r.get("p_value", 1.0))
             for r in results.values() if "K_proxy" in r]
    valid.sort(key=lambda x: x[0])

    if len(valid) < 3:
        print("[PLOT] Not enough data points to plot")
        return

    steps = np.array([v[0] for v in valid])
    K_values = np.array([v[1] for v in valid])
    p_values = np.array([v[2] for v in valid])

    # ── Figure 1: K_proxy vs training step ───────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#040407')

    # Top: K_proxy curve
    ax1 = axes[0]
    ax1.set_facecolor('#040407')
    ax1.plot(steps, K_values, 'o-', color='#00ffa8', linewidth=2, markersize=8, zorder=5)
    ax1.axhline(y=1.027, color='#FFB800', linestyle='--', alpha=0.5, label='K=1.027 (cross-arch constant)')
    ax1.axhline(y=1.0, color='#666666', linestyle=':', alpha=0.3)

    # Mark significant points
    for s, k, p in valid:
        if p < 0.05:
            ax1.plot(s, k, 'o', color='#00ffa8', markersize=12, zorder=6)
        else:
            ax1.plot(s, k, 'o', color='#ff3333', markersize=8, alpha=0.5, zorder=4)

    ax1.set_xscale('log')
    ax1.set_xlabel('Training Step', color='#e0e0e0', fontsize=12)
    ax1.set_ylabel('K_proxy', color='#e0e0e0', fontsize=12)
    ax1.set_title('K EMERGENCE — Pythia-160m Depth Routing Through Training',
                   color='#00ffa8', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#0a0a0a', edgecolor='#1a1a1a', labelcolor='#e0e0e0')
    ax1.tick_params(colors='#666666')
    ax1.spines['bottom'].set_color('#1a1a1a')
    ax1.spines['left'].set_color('#1a1a1a')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.1, color='#00ffa8')

    # Bottom: dK/dt (numerical derivative)
    ax2 = axes[1]
    ax2.set_facecolor('#040407')

    if len(steps) > 1:
        # Compute dK/dt using finite differences on log scale
        log_steps = np.log10(steps + 1)
        dK_dt = np.gradient(K_values, log_steps)

        ax2.bar(steps, dK_dt, width=steps * 0.3, color='#00ffa8', alpha=0.7, edgecolor='#00ffa8')
        ax2.axhline(y=0, color='#666666', linestyle=':', alpha=0.3)

        # Highlight the peak — this is the critical point candidate
        peak_idx = np.argmax(np.abs(dK_dt))
        ax2.bar(steps[peak_idx], dK_dt[peak_idx], width=steps[peak_idx] * 0.3,
                color='#FFB800', alpha=0.9, edgecolor='#FFB800', label=f'Peak at step {steps[peak_idx]}')

    ax2.set_xscale('log')
    ax2.set_xlabel('Training Step', color='#e0e0e0', fontsize=12)
    ax2.set_ylabel('dK/d(log step)', color='#e0e0e0', fontsize=12)
    ax2.set_title('dK/dt — Phase Transition Discriminator (spike = transition, smooth = crossover)',
                   color='#FFB800', fontsize=11)
    ax2.legend(facecolor='#0a0a0a', edgecolor='#1a1a1a', labelcolor='#e0e0e0')
    ax2.tick_params(colors='#666666')
    ax2.spines['bottom'].set_color('#1a1a1a')
    ax2.spines['left'].set_color('#1a1a1a')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.1, color='#FFB800')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "k_emergence_curve.png")
    plt.savefig(plot_path, dpi=150, facecolor='#040407', bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {plot_path}")

    # ── Summary stats ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("K EMERGENCE RESULTS SUMMARY")
    print("=" * 60)
    for s, k, p in valid:
        sig = "*" if p < 0.05 else " "
        print(f"  step {s:>6d}:  K_proxy = {k:.4f}  p = {p:.6f} {sig}")

    if len(steps) > 1:
        log_steps = np.log10(steps + 1)
        dK_dt = np.gradient(K_values, log_steps)
        peak_idx = np.argmax(np.abs(dK_dt))
        print(f"\n  Peak dK/dt at step {steps[peak_idx]} (magnitude: {dK_dt[peak_idx]:.4f})")
        print(f"  Final K_proxy: {K_values[-1]:.4f}")
        print(f"  K range: [{K_values.min():.4f}, {K_values.max():.4f}]")

        # Spike vs crossover heuristic
        peak_mag = np.abs(dK_dt[peak_idx])
        mean_mag = np.mean(np.abs(dK_dt))
        ratio = peak_mag / (mean_mag + 1e-10)
        print(f"\n  Spike ratio (peak/mean dK/dt): {ratio:.2f}")
        if ratio > 3.0:
            print("  >>> SPIKE DETECTED — consistent with phase transition")
        elif ratio > 1.5:
            print("  >>> MODERATE PEAK — ambiguous, need finer resolution")
        else:
            print("  >>> SMOOTH — consistent with crossover")

    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("K EMERGENCE EXPERIMENT")
    print("Pythia-160m — Coarse Checkpoint Scan")
    print("Hypothesis: K snaps as phase transition during training")
    print("=" * 60)
    print()

    results = run_coarse_scan()
    plot_results(results)
