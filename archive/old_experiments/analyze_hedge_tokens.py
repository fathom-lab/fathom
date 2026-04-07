"""
analyze_hedge_tokens.py — Hypothesis B: Performed Uncertainty
==============================================================

Tests whether RLHF-trained models perform uncertainty while internally
committed. Extracts C_delta at hedge tokens ("perhaps", "maybe", etc.)
and compares to surrounding non-hedge tokens.

Hypothesis B prediction: hedge tokens have LOWER C_delta than surrounding
tokens = the model's geometry is quieter (committed) at the exact moment
it expresses uncertainty. Geometry and language contradict = performed uncertainty.

Null prediction: hedge tokens have EQUAL or HIGHER C_delta = genuine uncertainty.

Uses saved traces from the commitment experiment — no GPU needed.

Usage:
    python analyze_hedge_tokens.py <results_json>
"""

import json
import sys
import numpy as np
from scipy import stats

HEDGE_WORDS = {
    "perhaps", "maybe", "possibly", "might", "could",
    "uncertain", "unclear", "debat", "argu", "think",
    "suggest", "possible", "likely", "unlikely",
    "not sure", "it's possible", "it might", "i think",
    "some", "may", "however", "although", "yet",
}

def is_hedge_token(token_text):
    """Check if a token contains hedge language."""
    t = token_text.lower().strip()
    for hw in HEDGE_WORDS:
        if hw in t:
            return True
    return False

def analyze(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    # Collect C_delta at hedge vs non-hedge tokens in HEDGING responses
    hedge_token_cdeltas = []
    nonhedge_token_cdeltas = []

    # Also collect all C_deltas from COMMITMENT responses as baseline
    commit_all_cdeltas = []

    # Track positions
    hedge_positions = []
    nonhedge_positions = []

    n_hedging_with_tokens = 0
    n_hedge_tokens_found = 0

    for r in results:
        # Hedging response — look for hedge tokens
        h = r["hedging"]
        h_traj = h.get("c_delta_trajectory", [])
        h_text = h.get("text", "")

        # We need per-token text to identify hedge tokens
        # The trajectory is per-token C_delta, text is the full generation
        # We'll split text into rough token estimates
        # Better: check if tokens are saved separately
        h_tokens = h_text.split() if h_text else []

        if len(h_traj) > 0 and len(h_tokens) > 0:
            n_hedging_with_tokens += 1
            # Align tokens to trajectory (rough — trajectory may have more entries than words)
            # Use min of both lengths
            n_align = min(len(h_traj), len(h_tokens))

            for i in range(n_align):
                cd = h_traj[i]
                if cd is None:
                    continue
                # Skip first 3 tokens (positional confound per darkflobi)
                if i < 3:
                    continue

                if is_hedge_token(h_tokens[i]):
                    hedge_token_cdeltas.append(cd)
                    hedge_positions.append(i / max(n_align - 1, 1))
                    n_hedge_tokens_found += 1
                else:
                    nonhedge_token_cdeltas.append(cd)
                    nonhedge_positions.append(i / max(n_align - 1, 1))

        # Commitment response — collect all C_deltas as baseline
        c = r["commitment"]
        c_traj = c.get("c_delta_trajectory", [])
        for i, cd in enumerate(c_traj):
            if cd is not None and i >= 3:  # skip first 3 tokens
                commit_all_cdeltas.append(cd)

    # Results
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS B: PERFORMED UNCERTAINTY ANALYSIS")
    print(f"{'='*60}")
    print(f"\nData: {len(results)} topic pairs")
    print(f"Hedging responses with tokens: {n_hedging_with_tokens}")
    print(f"Hedge tokens found: {n_hedge_tokens_found}")
    print(f"Non-hedge tokens: {len(nonhedge_token_cdeltas)}")
    print(f"Commitment tokens (baseline): {len(commit_all_cdeltas)}")

    if not hedge_token_cdeltas or not nonhedge_token_cdeltas:
        print("\nInsufficient hedge tokens found. Cannot analyze.")
        return

    h_cd = np.array(hedge_token_cdeltas)
    nh_cd = np.array(nonhedge_token_cdeltas)
    c_cd = np.array(commit_all_cdeltas)

    print(f"\n--- THE FOUR NUMBERS ---")
    print(f"1. Mean C_delta at HEDGE tokens (hedging responses):     {np.mean(h_cd):.6f} (n={len(h_cd)})")
    print(f"2. Mean C_delta at NON-HEDGE tokens (hedging responses): {np.mean(nh_cd):.6f} (n={len(nh_cd)})")
    print(f"3. Mean C_delta ALL tokens (commitment responses):       {np.mean(c_cd):.6f} (n={len(c_cd)})")
    print(f"4. Hedge vs non-hedge difference:                        {np.mean(h_cd) - np.mean(nh_cd):+.6f}")

    # Statistical test: hedge vs non-hedge within hedging responses
    t_stat, p_val = stats.ttest_ind(h_cd, nh_cd)
    d = (np.mean(h_cd) - np.mean(nh_cd)) / np.sqrt((np.std(h_cd)**2 + np.std(nh_cd)**2) / 2)

    print(f"\n--- HYPOTHESIS B TEST ---")
    print(f"Hedge vs non-hedge tokens (within hedging responses):")
    print(f"  t = {t_stat:.3f}")
    print(f"  p = {p_val:.4f}")
    print(f"  d = {d:.4f}")

    if np.mean(h_cd) < np.mean(nh_cd) and p_val < 0.05:
        print(f"\n  *** HYPOTHESIS B SUPPORTED ***")
        print(f"  Hedge tokens show LOWER C_delta than surrounding tokens.")
        print(f"  The model's geometry is quieter when it expresses uncertainty.")
        print(f"  This is consistent with performed uncertainty.")
    elif np.mean(h_cd) >= np.mean(nh_cd):
        print(f"\n  NULL: hedge tokens show equal or higher C_delta.")
        print(f"  Genuine uncertainty — geometry matches language.")
    else:
        print(f"\n  TREND: hedge tokens lower but not significant (p={p_val:.4f}).")

    # Positional analysis
    if hedge_positions:
        print(f"\n--- POSITIONAL CHECK ---")
        print(f"Hedge token mean position: {np.mean(hedge_positions):.2f} (0=start, 1=end)")
        print(f"Non-hedge token mean position: {np.mean(nonhedge_positions):.2f}")

    # Hedge tokens vs commitment baseline
    t2, p2 = stats.ttest_ind(h_cd, c_cd)
    print(f"\n--- HEDGE vs COMMITMENT BASELINE ---")
    print(f"Hedge token C_delta:     {np.mean(h_cd):.6f}")
    print(f"Commitment all C_delta:  {np.mean(c_cd):.6f}")
    print(f"t = {t2:.3f}, p = {p2:.4f}")

    if np.mean(h_cd) < np.mean(c_cd):
        print(f"Hedge tokens are QUIETER than commitment tokens.")
    else:
        print(f"Hedge tokens are LOUDER than commitment tokens.")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to IT replication
        path = "C:/Users/heyzo/clawd/sae-reasoning-depth/truthfulqa_results/commitment_IT_replication.json"
    else:
        path = sys.argv[1]

    analyze(path)
