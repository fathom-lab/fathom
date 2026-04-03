"""
visualize_governor.py — Publication-Ready Governor Pilot Figures
================================================================

Generates Figure 2 (slide 4) for the ICML paper and seed deck:
- Panel A: C_delta trace for unsteered (hallucinated) generation
- Panel B: C_delta trace after directional steering (reasoning failure governor)
- Panel C: C_delta trace after noise injection (confabulation governor)
- Panel D: Bar chart comparing false claim presence across conditions

Also generates the detection latency curve (Figure 3):
- X axis: number of tokens observed
- Y axis: paired directional accuracy (%)
- Shows when hallucination becomes detectable

Run after cognitive_governor_experiment.py produces results.
Can also be run standalone with pre-saved JSON data.

Patent pending: US 64/020,489 & 64/021,113
"""

import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available, generating ASCII plots only")


# ── Color scheme (darkflobi aesthetic) ───────────────────────────────────

COLORS = {
    "correct": "#00FF00",       # matrix green — grounded
    "hallucinated": "#FF4444",  # red — false
    "steered": "#00CCFF",       # cyan — directional governor
    "noised": "#FFaa00",        # amber — confabulation governor
    "bg": "#000000",            # black
    "grid": "#1a1a1a",          # dark grid
    "text": "#CCCCCC",          # light gray text
    "accent": "#00FFFF",        # cyan accent
}


def plot_governor_comparison(
    unsteered_trace,
    steered_traces,    # dict: {strength: trace}
    noised_traces,     # dict: {noise_scale: trace}
    rewind_point,
    question,
    output_path="figures/governor_comparison.png",
):
    """
    Generate the 4-panel governor comparison figure.

    Each trace is a list of (token_idx, c_delta) tuples.
    """
    if not HAS_MPL:
        _ascii_governor(unsteered_trace, steered_traces, noised_traces, rewind_point)
        return

    fig = plt.figure(figsize=(16, 10), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Unsteered (hallucinated) C_delta trace
    ax1 = fig.add_subplot(gs[0, 0])
    _style_axis(ax1, "A: Unsteered (hallucinated)")
    tokens, cds = zip(*unsteered_trace) if unsteered_trace else ([], [])
    ax1.plot(tokens, cds, color=COLORS["hallucinated"], linewidth=2, label="hallucinated")
    ax1.axvline(x=rewind_point, color=COLORS["accent"], linestyle="--", alpha=0.7, label=f"rewind @ {rewind_point}")
    ax1.set_xlabel("Token", color=COLORS["text"])
    ax1.set_ylabel("C_delta", color=COLORS["text"])
    ax1.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # Panel B: Directional steering (reasoning failure governor)
    ax2 = fig.add_subplot(gs[0, 1])
    _style_axis(ax2, "B: Directional Steering (reasoning governor)")
    # Show unsteered as ghost
    ax2.plot(tokens, cds, color=COLORS["hallucinated"], linewidth=1, alpha=0.3, label="unsteered")
    for strength, trace in sorted(steered_traces.items()):
        if trace:
            t, c = zip(*trace)
            alpha = 0.4 + 0.6 * (strength / max(steered_traces.keys()))
            ax2.plot(t, c, color=COLORS["steered"], linewidth=2, alpha=alpha, label=f"steer {strength}x")
    ax2.axvline(x=rewind_point, color=COLORS["accent"], linestyle="--", alpha=0.7)
    ax2.set_xlabel("Token", color=COLORS["text"])
    ax2.set_ylabel("C_delta", color=COLORS["text"])
    ax2.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=8)

    # Panel C: Noise injection (confabulation governor)
    ax3 = fig.add_subplot(gs[1, 0])
    _style_axis(ax3, "C: Noise Injection (confabulation governor)")
    ax3.plot(tokens, cds, color=COLORS["hallucinated"], linewidth=1, alpha=0.3, label="unsteered")
    for scale, trace in sorted(noised_traces.items()):
        if trace:
            t, c = zip(*trace)
            alpha = 0.4 + 0.6 * (scale / max(noised_traces.keys()))
            ax3.plot(t, c, color=COLORS["noised"], linewidth=2, alpha=alpha, label=f"noise {scale}x")
    ax3.axvline(x=rewind_point, color=COLORS["accent"], linestyle="--", alpha=0.7)
    ax3.set_xlabel("Token", color=COLORS["text"])
    ax3.set_ylabel("C_delta", color=COLORS["text"])
    ax3.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=8)

    # Panel D: Summary bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    _style_axis(ax4, "D: False Claim Eliminated?")
    conditions = ["Unsteered"]
    values = [1]  # 1 = contains false claim
    bar_colors = [COLORS["hallucinated"]]

    for strength in sorted(steered_traces.keys()):
        conditions.append(f"Steer {strength}x")
        values.append(0.5)  # placeholder — fill from actual results
        bar_colors.append(COLORS["steered"])

    for scale in sorted(noised_traces.keys()):
        conditions.append(f"Noise {scale}x")
        values.append(0.5)  # placeholder
        bar_colors.append(COLORS["noised"])

    bars = ax4.bar(range(len(conditions)), values, color=bar_colors, alpha=0.8)
    ax4.set_xticks(range(len(conditions)))
    ax4.set_xticklabels(conditions, rotation=45, ha="right", fontsize=8, color=COLORS["text"])
    ax4.set_ylabel("Contains false claim", color=COLORS["text"])
    ax4.set_ylim(0, 1.2)

    # Title
    fig.suptitle(
        f"Cognitive Governor A/B Test — \"{question[:50]}...\"",
        color=COLORS["accent"], fontsize=14, fontweight="bold", y=0.98
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"[+] Saved governor comparison to {output_path}")
    plt.close(fig)


def plot_detection_latency(
    window_accuracies,  # list of (window_size, accuracy, p_value)
    output_path="figures/detection_latency.png",
):
    """
    Generate the detection latency curve (Figure 3).

    Shows paired directional accuracy vs. number of tokens observed.
    """
    if not HAS_MPL:
        _ascii_latency(window_accuracies)
        return

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
    _style_axis(ax, "Detection Latency Curve — K_autocorr Paired Accuracy")

    windows = [w for w, a, p in window_accuracies]
    accs = [a * 100 for w, a, p in window_accuracies]
    pvals = [p for w, a, p in window_accuracies]

    # Color by significance
    colors = [COLORS["correct"] if p < 0.05 else COLORS["text"] for p in pvals]

    ax.plot(windows, accs, color=COLORS["accent"], linewidth=2, zorder=2)
    for i, (w, a, p) in enumerate(zip(windows, accs, pvals)):
        ax.scatter(w, a, color=colors[i], s=60, zorder=3, edgecolors="white", linewidth=0.5)

    ax.axhline(y=50, color=COLORS["hallucinated"], linestyle="--", alpha=0.5, label="chance (50%)")
    ax.axhline(y=65, color=COLORS["noised"], linestyle=":", alpha=0.5, label="signal threshold (65%)")

    ax.set_xlabel("Tokens observed", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("Paired accuracy (%)", color=COLORS["text"], fontsize=12)
    ax.set_ylim(40, 90)
    ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # Annotate first significant point
    for w, a, p in window_accuracies:
        if p < 0.05:
            ax.annotate(
                f"p={p:.4f}\n{a*100:.0f}%",
                xy=(w, a * 100), xytext=(w + 2, a * 100 + 5),
                color=COLORS["correct"], fontsize=9,
                arrowprops=dict(arrowstyle="->", color=COLORS["correct"]),
            )
            break

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"[+] Saved detection latency curve to {output_path}")
    plt.close(fig)


def plot_kce_space(
    correct_points,     # list of (K, C, E) tuples
    halluc_points,      # list of (K, C, E) tuples
    output_path="figures/kce_space.png",
):
    """
    Generate the K-C-E 3D scatter plot showing separation.
    """
    if not HAS_MPL:
        print("[WARN] 3D plot requires matplotlib")
        return

    fig = plt.figure(figsize=(10, 8), facecolor=COLORS["bg"])
    ax = fig.add_subplot(111, projection="3d", facecolor=COLORS["bg"])

    if correct_points:
        ck, cc, ce = zip(*correct_points)
        ax.scatter(ck, cc, ce, c=COLORS["correct"], s=30, alpha=0.7, label="correct")
    if halluc_points:
        hk, hc, he = zip(*halluc_points)
        ax.scatter(hk, hc, he, c=COLORS["hallucinated"], s=30, alpha=0.7, label="hallucinated")

    ax.set_xlabel("K (depth)", color=COLORS["text"])
    ax.set_ylabel("C (coherence)", color=COLORS["text"])
    ax.set_zlabel("E (grounding)", color=COLORS["text"])
    ax.set_title("K-C-E Cognitive Space", color=COLORS["accent"], fontsize=14)
    ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # Style 3D axes
    ax.tick_params(colors=COLORS["text"])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"[+] Saved K-C-E space to {output_path}")
    plt.close(fig)


# ── Styling helpers ──────────────────────────────────────────────────────

def _style_axis(ax, title):
    """Apply darkflobi aesthetic to axis."""
    ax.set_facecolor(COLORS["bg"])
    ax.set_title(title, color=COLORS["accent"], fontsize=11, fontweight="bold")
    ax.tick_params(colors=COLORS["text"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color=COLORS["grid"])


# ── ASCII fallbacks ──────────────────────────────────────────────────────

def _ascii_governor(unsteered, steered, noised, rewind):
    """ASCII visualization when matplotlib unavailable."""
    print("\n=== GOVERNOR COMPARISON (ASCII) ===\n")
    print("Unsteered C_delta trace:")
    for t, cd in unsteered:
        bar = "#" * int(cd * 500) if cd else ""
        flag = " <<<" if t == rewind else ""
        print(f"  [{t:>2}] {bar:40s} {cd:+.4f}{flag}" if cd else f"  [{t:>2}] —")

    for label, traces, color_name in [("STEERED", steered, "cyan"), ("NOISED", noised, "amber")]:
        for strength, trace in sorted(traces.items()):
            print(f"\n{label} ({strength}x):")
            for t, cd in trace:
                bar = "=" * int(cd * 500) if cd else ""
                print(f"  [{t:>2}] {bar:40s} {cd:+.4f}" if cd else f"  [{t:>2}] —")


def _ascii_latency(window_accs):
    """ASCII detection latency curve."""
    print("\n=== DETECTION LATENCY (ASCII) ===\n")
    print(f"{'Window':>6} {'Accuracy':>8} {'Bar':>30}")
    for w, a, p in window_accs:
        bar = "|" * int(a * 50)
        sig = " *" if p < 0.05 else ""
        print(f"{w:>6} {a:>8.1%} {bar}{sig}")


if __name__ == "__main__":
    # Demo with placeholder data
    print("Visualization module loaded. Run after governor experiment for real figures.")
    print(f"matplotlib available: {HAS_MPL}")

    # Create figures directory
    Path("figures").mkdir(exist_ok=True)

    if HAS_MPL:
        # Demo detection latency curve with n=50 data
        demo_data = [
            (5, 0.56, 0.48), (7, 0.60, 0.20), (10, 0.66, 0.033),
            (12, 0.66, 0.033), (14, 0.74, 0.0009), (15, 0.72, 0.003),
            (20, 0.68, 0.015), (25, 0.82, 0.00001), (30, 0.64, 0.065),
        ]
        plot_detection_latency(demo_data, "figures/detection_latency_demo.png")
        print("Demo figure generated: figures/detection_latency_demo.png")
