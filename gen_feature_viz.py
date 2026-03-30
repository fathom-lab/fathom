"""Generate feature layer distribution visualization for the Fathom paper.

Runs vocabulary-matched surface/insight pairs through the DepthScorer,
captures per-layer feature activation profiles, and produces fig6_feature_layers.png.
"""
import json
import sys
import numpy as np

# Vocabulary-matched surface/insight pairs (from paper methodology)
PAIRS = [
    {
        "surface": "The speed of light in a vacuum is approximately",
        "insight": "Light travels at that speed in a vacuum because",
    },
    {
        "surface": "Water freezes at zero degrees Celsius under normal",
        "insight": "Water freezes at that temperature because molecular",
    },
    {
        "surface": "The largest organ in the human body is the",
        "insight": "The skin became the largest organ because evolutionary",
    },
    {
        "surface": "Photosynthesis converts sunlight into chemical energy in",
        "insight": "Photosynthesis evolved to convert sunlight because organisms",
    },
    {
        "surface": "The Great Wall of China was built during the",
        "insight": "The Great Wall was built because northern invasions",
    },
    {
        "surface": "DNA carries genetic information using four nucleotide",
        "insight": "DNA uses four nucleotides because information storage",
    },
    {
        "surface": "The boiling point of water at sea level is",
        "insight": "Water boils at that temperature because molecular",
    },
    {
        "surface": "Gravity on Earth accelerates objects at approximately nine",
        "insight": "Gravity accelerates objects at that rate because planetary",
    },
]


def main():
    from depth_scorer import DepthScorer

    print("Loading model...")
    scorer = DepthScorer()
    n_layers = scorer._n_layers

    all_surface_profiles = []
    all_insight_profiles = []

    for i, pair in enumerate(PAIRS):
        print(f"\nPair {i+1}/{len(PAIRS)}")
        print(f"  Surface: {pair['surface'][:50]}...")
        s_result = scorer.score(pair["surface"])
        print(f"  Insight: {pair['insight'][:50]}...")
        i_result = scorer.score(pair["insight"])

        # Convert layer_profile to full array (0..n_layers-1)
        s_prof = np.zeros(n_layers)
        for layer, count in s_result.layer_profile.items():
            if 0 <= layer < n_layers:
                s_prof[layer] = count
        # Normalize to proportions
        if s_prof.sum() > 0:
            s_prof = s_prof / s_prof.sum()

        i_prof = np.zeros(n_layers)
        for layer, count in i_result.layer_profile.items():
            if 0 <= layer < n_layers:
                i_prof[layer] = count
        if i_prof.sum() > 0:
            i_prof = i_prof / i_prof.sum()

        all_surface_profiles.append(s_prof)
        all_insight_profiles.append(i_prof)

        print(f"  Surface depth={s_result.depth_score:.3f}, features={s_result.feature_count}")
        print(f"  Insight depth={i_result.depth_score:.3f}, features={i_result.feature_count}")

    # Save raw data
    data = {
        "n_layers": n_layers,
        "n_pairs": len(PAIRS),
        "surface_profiles": [p.tolist() for p in all_surface_profiles],
        "insight_profiles": [p.tolist() for p in all_insight_profiles],
    }
    with open("feature_viz_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\nSaved raw data to feature_viz_data.json")

    # Generate figure
    generate_figure(data)


def generate_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n_layers = data["n_layers"]
    surface_profiles = np.array(data["surface_profiles"])
    insight_profiles = np.array(data["insight_profiles"])

    # Mean and SEM across pairs
    s_mean = surface_profiles.mean(axis=0)
    s_sem = surface_profiles.std(axis=0) / np.sqrt(len(surface_profiles))
    i_mean = insight_profiles.mean(axis=0)
    i_sem = insight_profiles.std(axis=0) / np.sqrt(len(insight_profiles))

    layers = np.arange(n_layers)

    # Layer bands for Gemma-2-2B
    early_bound, mid_bound = 8, 16

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("white")

    # --- Panel A: Layer-wise feature activation ---
    bar_width = 0.35
    ax1.bar(layers - bar_width/2, s_mean, bar_width, yerr=s_sem,
            color="#d35400", alpha=0.85, label="Surface recall",
            capsize=2, error_kw={"linewidth": 0.8})
    ax1.bar(layers + bar_width/2, i_mean, bar_width, yerr=i_sem,
            color="#2980b9", alpha=0.85, label="Insight reasoning",
            capsize=2, error_kw={"linewidth": 0.8})

    # Band shading
    ax1.axvspan(-0.5, early_bound - 0.5, alpha=0.06, color="green", zorder=0)
    ax1.axvspan(early_bound - 0.5, mid_bound - 0.5, alpha=0.06, color="gold", zorder=0)
    ax1.axvspan(mid_bound - 0.5, n_layers - 0.5, alpha=0.06, color="red", zorder=0)

    # Band labels
    ax1.text(early_bound/2 - 0.5, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.12,
             "Early", ha="center", va="bottom", fontsize=8, color="green", alpha=0.7,
             fontweight="bold")
    ax1.text((early_bound + mid_bound)/2 - 0.5, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.12,
             "Mid", ha="center", va="bottom", fontsize=8, color="#b8860b", alpha=0.7,
             fontweight="bold")
    ax1.text((mid_bound + n_layers)/2 - 0.5, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.12,
             "Late", ha="center", va="bottom", fontsize=8, color="red", alpha=0.7,
             fontweight="bold")

    ax1.set_xlabel("Transformer Layer", fontsize=11)
    ax1.set_ylabel("Feature Activation Proportion", fontsize=11)
    ax1.set_title("(a) Per-Layer SAE Feature Distribution", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(0, n_layers, 2))
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_xlim(-0.5, n_layers - 0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Panel B: Band-aggregated ratios ---
    s_early = s_mean[:early_bound].sum()
    s_mid = s_mean[early_bound:mid_bound].sum()
    s_late = s_mean[mid_bound:].sum()

    i_early = i_mean[:early_bound].sum()
    i_mid = i_mean[early_bound:mid_bound].sum()
    i_late = i_mean[mid_bound:].sum()

    x_bands = np.arange(3)
    ax2.bar(x_bands - bar_width/2, [s_early, s_mid, s_late], bar_width,
            color="#d35400", alpha=0.85, label="Surface")
    ax2.bar(x_bands + bar_width/2, [i_early, i_mid, i_late], bar_width,
            color="#2980b9", alpha=0.85, label="Insight")

    ax2.set_xticks(x_bands)
    ax2.set_xticklabels(["Early\n(0-7)", "Mid\n(8-15)", "Late\n(16-25)"], fontsize=9)
    ax2.set_ylabel("Cumulative Proportion", fontsize=11)
    ax2.set_title("(b) Band Aggregation", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = r"C:\Users\heyzo\clawd\fathom-arxiv-submission\fig6_feature_layers.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    # If data already exists, just regenerate figure
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        with open("feature_viz_data.json") as f:
            data = json.load(f)
        generate_figure(data)
    else:
        main()
