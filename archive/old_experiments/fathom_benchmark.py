"""
FATHOM BENCHMARK — Standardized Evaluation Dataset & Runner
=============================================================

50 matched prompt pairs across 10 domains. The dataset other researchers
use to test their models against the Fathom standard.

    python -m fathom_benchmark --model google/gemma-2-2b

Usage (library):
    from fathom_benchmark import BenchmarkRunner, load_benchmark

    pairs = load_benchmark()
    runner = BenchmarkRunner()
    result = runner.run(scorer)
    print(result.summary)
"""

import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = PROJECT_ROOT / "fathom_benchmark_v1.json"


# ── The 50 Pairs ────────────────────────────────────────────────────

BENCHMARK_PAIRS = [
    # ── SCIENCE (5 pairs) ──
    {"domain": "science", "difficulty": "easy",
     "surface_prompt": "Water boils at one hundred degrees Celsius because water",
     "insight_prompt": "Water boils at one hundred degrees Celsius because the kinetic energy of molecules"},
    {"domain": "science", "difficulty": "easy",
     "surface_prompt": "Light travels fastest through a vacuum because light",
     "insight_prompt": "Light travels fastest through a vacuum because the absence of matter eliminates"},
    {"domain": "science", "difficulty": "medium",
     "surface_prompt": "Chemical reactions require activation energy and the reactants",
     "insight_prompt": "Chemical reactions require activation energy because the molecular bonds must"},
    {"domain": "science", "difficulty": "medium",
     "surface_prompt": "The periodic table organizes elements by atomic number and the elements",
     "insight_prompt": "The periodic table organizes elements by atomic number because electron configuration"},
    {"domain": "science", "difficulty": "hard",
     "surface_prompt": "Entropy in closed systems always increases over time and the total",
     "insight_prompt": "Entropy in closed systems always increases because the number of accessible microstates"},

    # ── HISTORY (5 pairs) ──
    {"domain": "history", "difficulty": "easy",
     "surface_prompt": "The Roman Empire fell in 476 AD when the last emperor",
     "insight_prompt": "The Roman Empire fell in 476 AD because centuries of administrative fragmentation"},
    {"domain": "history", "difficulty": "easy",
     "surface_prompt": "The printing press was invented by Gutenberg and the first",
     "insight_prompt": "The printing press was invented by Gutenberg and the resulting democratization of knowledge"},
    {"domain": "history", "difficulty": "medium",
     "surface_prompt": "The Industrial Revolution began in Britain around 1760 and the factories",
     "insight_prompt": "The Industrial Revolution began in Britain because the convergence of capital accumulation"},
    {"domain": "history", "difficulty": "medium",
     "surface_prompt": "World War I started after the assassination of Archduke Franz Ferdinand and the alliance",
     "insight_prompt": "World War I started after the assassination because the existing alliance network transformed"},
    {"domain": "history", "difficulty": "hard",
     "surface_prompt": "Empires throughout history have eventually declined and each empire",
     "insight_prompt": "Empires throughout history have eventually declined because overextension creates feedback loops"},

    # ── ECONOMICS (5 pairs) ──
    {"domain": "economics", "difficulty": "easy",
     "surface_prompt": "Supply and demand determine prices in markets and the price",
     "insight_prompt": "Supply and demand determine prices because the equilibrium emerges from competing"},
    {"domain": "economics", "difficulty": "easy",
     "surface_prompt": "Inflation occurs when there is too much money chasing and the prices",
     "insight_prompt": "Inflation occurs when there is too much money because the velocity of monetary"},
    {"domain": "economics", "difficulty": "medium",
     "surface_prompt": "Central banks raise interest rates to control inflation and the economy",
     "insight_prompt": "Central banks raise interest rates because the transmission mechanism reduces aggregate"},
    {"domain": "economics", "difficulty": "medium",
     "surface_prompt": "Trade deficits occur when imports exceed exports and the country",
     "insight_prompt": "Trade deficits occur when imports exceed exports because the capital account must"},
    {"domain": "economics", "difficulty": "hard",
     "surface_prompt": "Financial crises spread through interconnected banking systems and each bank",
     "insight_prompt": "Financial crises spread through interconnected banking systems because counterparty risk creates"},

    # ── PHILOSOPHY (5 pairs) ──
    {"domain": "philosophy", "difficulty": "easy",
     "surface_prompt": "Socrates was sentenced to death for corrupting the youth and the trial",
     "insight_prompt": "Socrates was sentenced to death because his method of questioning revealed the contradictions"},
    {"domain": "philosophy", "difficulty": "easy",
     "surface_prompt": "Descartes argued that thinking proves existence and the famous",
     "insight_prompt": "Descartes argued that thinking proves existence because doubt itself requires a doubting"},
    {"domain": "philosophy", "difficulty": "medium",
     "surface_prompt": "Free will is debated in philosophy and the determinists",
     "insight_prompt": "Free will is debated because the causal closure of physics appears incompatible with"},
    {"domain": "philosophy", "difficulty": "medium",
     "surface_prompt": "The trolley problem presents a moral dilemma where the choice",
     "insight_prompt": "The trolley problem reveals a moral dilemma because the tension between consequentialist"},
    {"domain": "philosophy", "difficulty": "hard",
     "surface_prompt": "Consciousness remains the hard problem of philosophy and the neural",
     "insight_prompt": "Consciousness remains the hard problem because the explanatory gap between physical processes"},

    # ── BIOLOGY (5 pairs) ──
    {"domain": "biology", "difficulty": "easy",
     "surface_prompt": "DNA contains the genetic code for all living organisms and the double",
     "insight_prompt": "DNA contains the genetic code because the complementary base pairing mechanism enables"},
    {"domain": "biology", "difficulty": "easy",
     "surface_prompt": "Cells divide through mitosis to grow and repair and the chromosomes",
     "insight_prompt": "Cells divide through mitosis because the checkpoint mechanisms ensure each daughter cell"},
    {"domain": "biology", "difficulty": "medium",
     "surface_prompt": "Evolution produces new species through natural selection and the population",
     "insight_prompt": "Evolution produces new species because reproductive isolation allows divergent selection pressures"},
    {"domain": "biology", "difficulty": "medium",
     "surface_prompt": "The immune system fights pathogens using white blood cells and the antibodies",
     "insight_prompt": "The immune system fights pathogens because the adaptive response generates antigen-specific"},
    {"domain": "biology", "difficulty": "hard",
     "surface_prompt": "Gene expression is regulated by transcription factors and the regulatory",
     "insight_prompt": "Gene expression is regulated because the combinatorial logic of transcription factor binding"},

    # ── PHYSICS (5 pairs) ──
    {"domain": "physics", "difficulty": "easy",
     "surface_prompt": "Gravity pulls objects toward Earth at 9.8 meters per second squared and the mass",
     "insight_prompt": "Gravity pulls objects toward Earth because the curvature of spacetime near massive"},
    {"domain": "physics", "difficulty": "easy",
     "surface_prompt": "Electricity flows through conductors along the path of least resistance and the electrons",
     "insight_prompt": "Electricity flows through conductors because the delocalized electron sea in metallic bonds"},
    {"domain": "physics", "difficulty": "medium",
     "surface_prompt": "Quantum mechanics describes particles as having wave-particle duality and the electron",
     "insight_prompt": "Quantum mechanics describes wave-particle duality because the uncertainty principle constrains simultaneous"},
    {"domain": "physics", "difficulty": "medium",
     "surface_prompt": "Black holes form when massive stars collapse and the gravitational",
     "insight_prompt": "Black holes form when massive stars collapse because beyond the Schwarzschild radius the escape"},
    {"domain": "physics", "difficulty": "hard",
     "surface_prompt": "Quantum entanglement links particles across distance and the measurement of one",
     "insight_prompt": "Quantum entanglement links particles because the joint wave function cannot be factored into"},

    # ── PSYCHOLOGY (5 pairs) ──
    {"domain": "psychology", "difficulty": "easy",
     "surface_prompt": "Memory is stored in the brain through neural connections and the hippocampus",
     "insight_prompt": "Memory is stored in the brain because synaptic strengthening through long-term potentiation"},
    {"domain": "psychology", "difficulty": "easy",
     "surface_prompt": "Cognitive biases affect human decision making and the confirmation",
     "insight_prompt": "Cognitive biases affect decision making because the brain uses heuristic shortcuts that"},
    {"domain": "psychology", "difficulty": "medium",
     "surface_prompt": "Stress triggers the fight-or-flight response in the body and the cortisol",
     "insight_prompt": "Stress triggers fight-or-flight because the amygdala activates the hypothalamic-pituitary-adrenal axis"},
    {"domain": "psychology", "difficulty": "medium",
     "surface_prompt": "Language acquisition occurs naturally in children during a critical period and the vocabulary",
     "insight_prompt": "Language acquisition occurs naturally because the statistical learning mechanisms in the developing"},
    {"domain": "psychology", "difficulty": "hard",
     "surface_prompt": "Consciousness arises from neural activity in the brain and the prefrontal",
     "insight_prompt": "Consciousness arises from neural activity because the recurrent processing between thalamo-cortical"},

    # ── MATHEMATICS (5 pairs) ──
    {"domain": "mathematics", "difficulty": "easy",
     "surface_prompt": "Prime numbers are only divisible by one and themselves and the number",
     "insight_prompt": "Prime numbers are only divisible by one and themselves because the fundamental theorem"},
    {"domain": "mathematics", "difficulty": "easy",
     "surface_prompt": "The Pythagorean theorem relates the sides of a right triangle and the hypotenuse",
     "insight_prompt": "The Pythagorean theorem relates the sides because the geometric relationship between area"},
    {"domain": "mathematics", "difficulty": "medium",
     "surface_prompt": "Calculus measures rates of change using derivatives and the function",
     "insight_prompt": "Calculus measures rates of change because the limit definition captures the instantaneous"},
    {"domain": "mathematics", "difficulty": "medium",
     "surface_prompt": "Infinity comes in different sizes as shown by Cantor and the real",
     "insight_prompt": "Infinity comes in different sizes because the diagonal argument demonstrates that no bijection"},
    {"domain": "mathematics", "difficulty": "hard",
     "surface_prompt": "Godel proved that consistent systems cannot prove their own consistency and the theorem",
     "insight_prompt": "Godel proved that consistent systems cannot prove their own consistency because self-referential"},

    # ── TECHNOLOGY (5 pairs) ──
    {"domain": "technology", "difficulty": "easy",
     "surface_prompt": "Computers process information using binary ones and zeros and the transistors",
     "insight_prompt": "Computers process information using binary because the two-state transistor switching enables"},
    {"domain": "technology", "difficulty": "easy",
     "surface_prompt": "The internet connects computers through a global network and the data",
     "insight_prompt": "The internet connects computers because the packet-switching protocol enables distributed routing"},
    {"domain": "technology", "difficulty": "medium",
     "surface_prompt": "Machine learning algorithms improve with more data and the model",
     "insight_prompt": "Machine learning algorithms improve because the gradient descent optimization incrementally reduces"},
    {"domain": "technology", "difficulty": "medium",
     "surface_prompt": "Encryption protects data by converting it into unreadable code and the key",
     "insight_prompt": "Encryption protects data because the computational difficulty of factoring large primes ensures"},
    {"domain": "technology", "difficulty": "hard",
     "surface_prompt": "Neural networks approximate functions using layers of weighted connections and the deeper",
     "insight_prompt": "Neural networks approximate functions because the universal approximation theorem guarantees that"},

    # ── LANGUAGE (5 pairs) ──
    {"domain": "language", "difficulty": "easy",
     "surface_prompt": "Languages change over time through use and the vocabulary",
     "insight_prompt": "Languages change over time because the competing pressures of communicative efficiency and"},
    {"domain": "language", "difficulty": "easy",
     "surface_prompt": "Grammar rules structure how words combine into sentences and the syntax",
     "insight_prompt": "Grammar rules structure word combination because the recursive nature of human cognition"},
    {"domain": "language", "difficulty": "medium",
     "surface_prompt": "Children acquire language rapidly during early development and the first",
     "insight_prompt": "Children acquire language rapidly because the neural plasticity during the critical period"},
    {"domain": "language", "difficulty": "medium",
     "surface_prompt": "Translation between languages is imperfect because each language has unique and the words",
     "insight_prompt": "Translation between languages is imperfect because the conceptual categories encoded in each"},
    {"domain": "language", "difficulty": "hard",
     "surface_prompt": "The Sapir-Whorf hypothesis suggests language shapes thought and the speakers",
     "insight_prompt": "The Sapir-Whorf hypothesis suggests language shapes thought because the categorical boundaries imposed"},
]


def load_benchmark(path: Optional[str] = None) -> List[dict]:
    """Load the benchmark dataset.

    Returns the built-in dataset, or loads from a custom JSON file.
    """
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return BENCHMARK_PAIRS


def save_benchmark(path: Optional[str] = None):
    """Save the benchmark dataset to JSON."""
    p = Path(path or BENCHMARK_PATH)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(BENCHMARK_PAIRS, f, indent=2)
    print(f"[Benchmark] Saved {len(BENCHMARK_PAIRS)} pairs to {p}")


def _paired_ttest(x, y):
    """Paired t-test returning (t_stat, p_value_one_tailed)."""
    x, y = np.array(x), np.array(y)
    if len(x) < 3:
        return 0.0, 1.0
    if HAS_SCIPY:
        t, p = scipy_stats.ttest_rel(x, y)
        return float(t), float(p) / 2.0  # one-tailed
    diffs = x - y
    n = len(diffs)
    mean_d = diffs.mean()
    std_d = diffs.std(ddof=1)
    if std_d < 1e-12:
        return 0.0, 1.0
    t = mean_d / (std_d / math.sqrt(n))
    p = math.exp(-0.717 * abs(t) - 0.416 * t * t)
    return float(t), max(0.0, min(1.0, float(p)))


# ── Benchmark Result ──────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result of running the Fathom benchmark."""
    model_name: str
    n_pairs: int
    n_valid: int
    mean_delta: float  # surface - insight (positive = expected direction)
    std_delta: float
    pct_in_direction: float
    t_stat: float
    p_value: float
    is_significant: bool
    surface_mean: float
    insight_mean: float
    per_domain: Dict[str, dict]
    per_difficulty: Dict[str, dict]
    compute_time_s: float
    summary: str

    def to_dict(self):
        return asdict(self)


# ── Runner ────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """Runs the Fathom benchmark on any DepthScorer instance."""

    def __init__(self, pairs: Optional[List[dict]] = None):
        self._pairs = pairs or BENCHMARK_PAIRS

    def run(self, scorer, verbose: bool = True) -> BenchmarkResult:
        """Score all benchmark pairs and compute statistics.

        Args:
            scorer: Initialized DepthScorer.
            verbose: Print progress.

        Returns:
            BenchmarkResult with full breakdown.
        """
        t0 = time.time()
        surfaces = []
        insights = []
        domain_data = {}  # domain -> [(s_depth, i_depth)]
        diff_data = {}    # difficulty -> [(s_depth, i_depth)]

        for i, pair in enumerate(self._pairs):
            domain = pair["domain"]
            difficulty = pair["difficulty"]

            try:
                r_s = scorer.score(pair["surface_prompt"])
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()

                r_i = scorer.score(pair["insight_prompt"])
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()

                if math.isnan(r_s.depth_score) or math.isnan(r_i.depth_score):
                    if verbose:
                        print(f"  [{i+1}/{len(self._pairs)}] NaN, skipping")
                    continue

                surfaces.append(r_s.depth_score)
                insights.append(r_i.depth_score)

                domain_data.setdefault(domain, []).append(
                    (r_s.depth_score, r_i.depth_score))
                diff_data.setdefault(difficulty, []).append(
                    (r_s.depth_score, r_i.depth_score))

                delta = r_s.depth_score - r_i.depth_score
                if verbose:
                    print(f"  [{i+1}/{len(self._pairs)}] {domain}/{difficulty} "
                          f"s={r_s.depth_score:.3f} i={r_i.depth_score:.3f} "
                          f"delta={delta:+.3f}")

            except Exception as e:
                if verbose:
                    print(f"  [{i+1}/{len(self._pairs)}] failed: {e}")

        if len(surfaces) < 5:
            raise ValueError(f"Need >= 5 valid pairs, got {len(surfaces)}")

        s_arr = np.array(surfaces)
        i_arr = np.array(insights)
        deltas = s_arr - i_arr

        t_stat, p_val = _paired_ttest(s_arr, i_arr)
        mean_delta = float(deltas.mean())
        pct_dir = float((deltas > 0).sum()) / len(deltas)
        is_sig = mean_delta > 0 and p_val < 0.05

        # Per-domain breakdown
        per_domain = {}
        for domain, pairs_data in domain_data.items():
            s_d = np.array([p[0] for p in pairs_data])
            i_d = np.array([p[1] for p in pairs_data])
            d_delta = s_d - i_d
            per_domain[domain] = {
                "n_pairs": len(pairs_data),
                "mean_delta": round(float(d_delta.mean()), 4),
                "pct_in_direction": round(float((d_delta > 0).sum()) / len(d_delta), 3),
                "surface_mean": round(float(s_d.mean()), 4),
                "insight_mean": round(float(i_d.mean()), 4),
            }

        # Per-difficulty breakdown
        per_difficulty = {}
        for diff, pairs_data in diff_data.items():
            s_d = np.array([p[0] for p in pairs_data])
            i_d = np.array([p[1] for p in pairs_data])
            d_delta = s_d - i_d
            per_difficulty[diff] = {
                "n_pairs": len(pairs_data),
                "mean_delta": round(float(d_delta.mean()), 4),
                "pct_in_direction": round(float((d_delta > 0).sum()) / len(d_delta), 3),
            }

        elapsed = time.time() - t0

        summary = (
            f"Fathom Benchmark: {len(surfaces)}/{len(self._pairs)} pairs scored on "
            f"{scorer.model_name}. "
            f"Mean delta: {mean_delta:+.4f}, {pct_dir:.0%} in expected direction, "
            f"t={t_stat:.3f}, p={p_val:.6f} (one-tailed). "
            f"{'PASS' if is_sig else 'FAIL'} at p<0.05."
        )

        return BenchmarkResult(
            model_name=scorer.model_name,
            n_pairs=len(self._pairs),
            n_valid=len(surfaces),
            mean_delta=round(mean_delta, 4),
            std_delta=round(float(deltas.std()), 4),
            pct_in_direction=round(pct_dir, 4),
            t_stat=round(t_stat, 4),
            p_value=round(p_val, 6),
            is_significant=is_sig,
            surface_mean=round(float(s_arr.mean()), 4),
            insight_mean=round(float(i_arr.mean()), 4),
            per_domain=per_domain,
            per_difficulty=per_difficulty,
            compute_time_s=round(elapsed, 2),
            summary=summary,
        )


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fathom Benchmark Runner")
    parser.add_argument("--model", default="google/gemma-2-2b", help="Model name")
    parser.add_argument("--max-features", type=int, default=150)
    parser.add_argument("--save-json", action="store_true", help="Save benchmark JSON")
    args = parser.parse_args()

    if args.save_json:
        save_benchmark()
        return

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 70)
    print(f"FATHOM BENCHMARK — {args.model}")
    print("=" * 70)

    import torch
    from depth_scorer import DepthScorer

    scorer = DepthScorer(model_name=args.model, max_feature_nodes=args.max_features)

    runner = BenchmarkRunner()
    result = runner.run(scorer)

    print("\n" + "=" * 70)
    print(result.summary)
    print("=" * 70)

    print("\nPer-domain breakdown:")
    for domain, stats in sorted(result.per_domain.items()):
        print(f"  {domain:12s}: delta={stats['mean_delta']:+.4f}, "
              f"direction={stats['pct_in_direction']:.0%}, n={stats['n_pairs']}")

    print("\nPer-difficulty breakdown:")
    for diff, stats in sorted(result.per_difficulty.items()):
        print(f"  {diff:8s}: delta={stats['mean_delta']:+.4f}, "
              f"direction={stats['pct_in_direction']:.0%}, n={stats['n_pairs']}")

    # Save results
    out_path = PROJECT_ROOT / "BENCHMARK_RESULT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Fathom Benchmark Result\n\n")
        f.write(f"**Model:** {result.model_name}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n{result.summary}\n\n")
        f.write(f"## Per-Domain\n\n| Domain | Delta | Direction | n |\n|--------|-------|-----------|---|\n")
        for domain, stats in sorted(result.per_domain.items()):
            f.write(f"| {domain} | {stats['mean_delta']:+.4f} | {stats['pct_in_direction']:.0%} | {stats['n_pairs']} |\n")
    print(f"\n[OUTPUT] {out_path}")


if __name__ == "__main__":
    main()
