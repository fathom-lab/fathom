# -*- coding: utf-8 -*-
"""
spectrum_sampler.py -- Best-of-N Spectrum-Scored Generation
============================================================

Instead of fighting the model mid-sentence (per-token steering),
generate N candidate completions with temperature sampling, score
each by cognitive health metrics, and return the healthiest one.

Cognitive Health Score (CHS):
  CHS = -w_gini * mean(gini)        # penalize concentration
        + w_spread * mean(spread)    # reward distribution
        + w_early * mean(early_mass) # reward grounding
        - w_peak * max(gini)         # penalize worst-case spike

Lower Gini + higher spread + more early-layer mass = healthier cognition.

This is rejection sampling on cognitive geometry. No gradient hacking,
no RLHF, no preference labels. Just: "which completion has the
healthiest thinking pattern?"

Patent pending: US 64/020,489 & 64/021,113
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ScoredCompletion:
    """A single candidate completion with cognitive health metrics."""
    text: str
    tokens: List[str]
    gini_trajectory: List[float]
    spread_trajectory: List[float]
    early_mass_trajectory: List[float]
    c_delta_trajectory: List[Optional[float]]
    spectra: List[np.ndarray]
    # Aggregate scores
    mean_gini: float = 0.0
    max_gini: float = 0.0
    mean_spread: float = 0.0
    mean_early_mass: float = 0.0
    cognitive_health_score: float = 0.0
    generation_time: float = 0.0


@dataclass
class SamplerResult:
    """Result of best-of-N sampling."""
    prompt: str
    n_candidates: int
    best: ScoredCompletion
    worst: ScoredCompletion
    all_scores: List[float]
    all_texts: List[str]
    best_idx: int
    worst_idx: int
    score_spread: float  # best - worst score
    total_time: float


class SpectrumSampler:
    """
    Best-of-N generation scored by cognitive health.

    Generates N completions using temperature sampling, computes
    the full coherence spectrum for each, and returns the one
    with the healthiest cognitive profile.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        n_candidates: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        # CHS weights
        w_gini: float = 1.0,
        w_spread: float = 0.3,
        w_early: float = 0.5,
        w_peak: float = 0.5,
        # SAE params
        top_k_features: int = 100,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.n_candidates = n_candidates
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.w_gini = w_gini
        self.w_spread = w_spread
        self.w_early = w_early
        self.w_peak = w_peak
        self.device = device

        # Load model via SpectrumSteerer (reuse infrastructure)
        from spectrum_steerer import SpectrumSteerer
        self._steerer = SpectrumSteerer(
            model_name=model_name,
            top_k_features=top_k_features,
            device=device,
            mode="spectrum",
        )
        self._inner = self._steerer._inner
        self._n_layers = self._steerer._n_layers

        print(f"[SpectrumSampler] Ready. N={n_candidates}, temp={temperature}")

    def _sample_token(self, logits: torch.Tensor) -> int:
        """Sample next token with temperature + top-k + top-p."""
        logits = logits / max(self.temperature, 1e-8)

        # Top-k filtering
        if self.top_k > 0:
            top_k_vals, top_k_idx = torch.topk(logits, min(self.top_k, logits.shape[-1]))
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, top_k_idx, top_k_vals)
            logits = mask

        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_idx[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _generate_one(
        self, prompt_tokens: torch.Tensor, max_tokens: int
    ) -> ScoredCompletion:
        """Generate a single candidate completion with spectrum monitoring."""
        from spectrum_steerer import _gini, _spread, _early_mass

        inner = self._inner
        tokens = prompt_tokens.clone()
        generated = []
        gini_traj = []
        spread_traj = []
        early_traj = []
        cdelta_traj = []
        spectra = []

        t0 = time.time()

        for step in range(max_tokens):
            hidden_states = {}
            hooks = self._steerer._build_full_capture_hooks(hidden_states)

            with torch.no_grad():
                logits = inner.run_with_hooks(tokens, fwd_hooks=hooks)

            # Compute spectrum
            profile = self._steerer._full_layer_coherence(hidden_states)
            gini_traj.append(float(_gini(profile)))
            spread_traj.append(float(_spread(profile)))
            early_traj.append(float(_early_mass(profile)))
            spectra.append(profile.copy())

            # C_delta
            cd, _ = self._steerer._fast_c_delta(hidden_states)
            cdelta_traj.append(cd)

            # Sample (not greedy)
            next_id = self._sample_token(logits[0, -1])
            next_str = inner.tokenizer.decode([next_id])

            if next_id == inner.tokenizer.eos_token_id:
                break

            generated.append(next_str)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

        elapsed = time.time() - t0
        text = "".join(generated)

        # Compute aggregate scores
        mean_gini = float(np.mean(gini_traj)) if gini_traj else 0
        max_gini = float(max(gini_traj)) if gini_traj else 0
        mean_spread = float(np.mean(spread_traj)) if spread_traj else 0
        mean_early = float(np.mean(early_traj)) if early_traj else 0

        # Cognitive Health Score (higher = healthier)
        chs = (
            -self.w_gini * mean_gini
            + self.w_spread * mean_spread
            + self.w_early * mean_early
            - self.w_peak * max_gini
        )

        return ScoredCompletion(
            text=text,
            tokens=generated,
            gini_trajectory=gini_traj,
            spread_trajectory=spread_traj,
            early_mass_trajectory=early_traj,
            c_delta_trajectory=cdelta_traj,
            spectra=spectra,
            mean_gini=mean_gini,
            max_gini=max_gini,
            mean_spread=mean_spread,
            mean_early_mass=mean_early,
            cognitive_health_score=chs,
            generation_time=elapsed,
        )

    def sample(
        self, prompt: str, max_tokens: int = 30, verbose: bool = True
    ) -> SamplerResult:
        """
        Generate N candidates, score by cognitive health, return the best.
        """
        t_start = time.time()
        prompt_tokens = self._inner.to_tokens(prompt)

        candidates = []
        for i in range(self.n_candidates):
            if verbose:
                print(f"  Candidate {i+1}/{self.n_candidates}...", end=" ", flush=True)
            c = self._generate_one(prompt_tokens, max_tokens)
            candidates.append(c)
            if verbose:
                safe = c.text[:50].encode('ascii', 'replace').decode()
                print(f"CHS={c.cognitive_health_score:+.4f} gini={c.mean_gini:.3f} "
                      f"\"{safe}...\"")

        # Rank by CHS
        scores = [c.cognitive_health_score for c in candidates]
        best_idx = int(np.argmax(scores))
        worst_idx = int(np.argmin(scores))
        total_time = time.time() - t_start

        result = SamplerResult(
            prompt=prompt,
            n_candidates=self.n_candidates,
            best=candidates[best_idx],
            worst=candidates[worst_idx],
            all_scores=scores,
            all_texts=[c.text for c in candidates],
            best_idx=best_idx,
            worst_idx=worst_idx,
            score_spread=scores[best_idx] - scores[worst_idx],
            total_time=total_time,
        )

        if verbose:
            print(f"\n  BEST  (#{best_idx}): CHS={result.best.cognitive_health_score:+.4f} "
                  f"gini={result.best.mean_gini:.3f}")
            safe_best = result.best.text[:80].encode('ascii', 'replace').decode()
            print(f"    \"{safe_best}\"")
            print(f"  WORST (#{worst_idx}): CHS={result.worst.cognitive_health_score:+.4f} "
                  f"gini={result.worst.mean_gini:.3f}")
            safe_worst = result.worst.text[:80].encode('ascii', 'replace').decode()
            print(f"    \"{safe_worst}\"")
            print(f"  Score spread: {result.score_spread:.4f}")
            print(f"  Total time: {total_time:.1f}s")

        return result

    def evaluate(
        self, prompts: List[str], max_tokens: int = 30
    ) -> Dict:
        """
        Evaluate best-of-N vs greedy baseline across multiple prompts.
        Returns aggregate statistics.
        """
        print(f"\n{'='*70}")
        print(f"SPECTRUM SAMPLER EVALUATION")
        print(f"N={self.n_candidates}, temp={self.temperature}")
        print(f"{'='*70}")

        results = []
        for i, prompt in enumerate(prompts):
            safe_p = prompt[:50].encode('ascii', 'replace').decode()
            print(f"\n[{i+1}/{len(prompts)}] {safe_p}...")

            # Best-of-N
            sr = self.sample(prompt, max_tokens, verbose=True)

            # Greedy baseline (temperature=0 equivalent)
            print("  Greedy baseline...", end=" ", flush=True)
            old_temp = self.temperature
            self.temperature = 0.01  # near-greedy
            greedy = self._generate_one(self._inner.to_tokens(prompt), max_tokens)
            self.temperature = old_temp
            safe_g = greedy.text[:50].encode('ascii', 'replace').decode()
            print(f"CHS={greedy.cognitive_health_score:+.4f} gini={greedy.mean_gini:.3f} "
                  f"\"{safe_g}...\"")

            results.append({
                "prompt": prompt,
                "greedy_chs": greedy.cognitive_health_score,
                "greedy_gini": greedy.mean_gini,
                "greedy_text": greedy.text,
                "best_chs": sr.best.cognitive_health_score,
                "best_gini": sr.best.mean_gini,
                "best_text": sr.best.text,
                "worst_chs": sr.worst.cognitive_health_score,
                "worst_gini": sr.worst.mean_gini,
                "score_spread": sr.score_spread,
                "improvement": sr.best.cognitive_health_score - greedy.cognitive_health_score,
                "gini_reduction": greedy.mean_gini - sr.best.mean_gini,
            })

        # Aggregate
        improvements = [r["improvement"] for r in results]
        gini_reductions = [r["gini_reduction"] for r in results]

        print(f"\n{'='*70}")
        print(f"AGGREGATE (n={len(results)} prompts)")
        print(f"  CHS improvement (best-of-N vs greedy):")
        print(f"    mean={np.mean(improvements):+.4f}, "
              f"median={np.median(improvements):+.4f}")
        print(f"  Gini reduction:")
        print(f"    mean={np.mean(gini_reductions):+.4f}, "
              f"median={np.median(gini_reductions):+.4f}")
        pct_improved = sum(1 for i in improvements if i > 0) / len(improvements) * 100
        print(f"  Improved: {pct_improved:.0f}% of prompts")
        print(f"{'='*70}")

        # Save
        out_path = Path(__file__).parent / "truthfulqa_results" / "spectrum_sampler_eval.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Results -> {out_path.name}")

        return {"results": results, "mean_improvement": float(np.mean(improvements)),
                "mean_gini_reduction": float(np.mean(gini_reductions)),
                "pct_improved": pct_improved}


def render_sampler_figure(result: SamplerResult, output_dir: Path):
    """Render best vs worst comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ekg", [
        (0.0, "#000000"), (0.15, "#003300"), (0.3, "#00aa00"),
        (0.5, "#88cc00"), (0.7, "#ffaa00"), (0.85, "#ff4400"),
        (1.0, "#ff0000"),
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    safe_p = result.prompt[:40].encode('ascii', 'replace').decode()
    fig.suptitle(f"Best-of-{result.n_candidates} Spectrum Sampling: \"{safe_p}...\"",
                fontsize=13, fontweight="bold")

    for col, (comp, label) in enumerate([
        (result.best, f"BEST (#{result.best_idx})"),
        (result.worst, f"WORST (#{result.worst_idx})"),
    ]):
        nt = len(comp.tokens)
        if nt == 0:
            continue

        # Waterfall
        ax = axes[0, col]
        if comp.spectra:
            sm = np.array([s for s in comp.spectra[:nt]]).T
            cs = sm.sum(axis=0, keepdims=True)
            cs[cs == 0] = 1
            ax.imshow(sm / cs, aspect="auto", cmap=cmap, interpolation="bilinear",
                     extent=[-0.5, nt - 0.5, self._n_layers - 0.5, -0.5])

        ax.set_ylabel("Layer")
        ax.set_title(f"{label}\nCHS={comp.cognitive_health_score:+.4f}, "
                    f"Gini={comp.mean_gini:.3f}")

        # Gini trace
        ax2 = axes[1, col]
        x = np.arange(nt)
        ax2.plot(x, comp.gini_trajectory[:nt], "o-", color="#ff4444", ms=3, lw=1.5)
        ax2.plot(x, comp.early_mass_trajectory[:nt], "^-", color="#00cc66", ms=3, lw=1.2)
        ax2.set_xlabel("Token")
        safe_text = comp.text[:60].encode('ascii', 'replace').decode()
        ax2.set_title(f"\"{safe_text}...\"", fontsize=9)
        ax2.legend(["Gini", "Early mass"], fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"spectrum_sampler_bestofN.{ext}",
                   dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> spectrum_sampler_bestofN.pdf/png")


if __name__ == "__main__":
    import sys

    sampler = SpectrumSampler(
        model_name="google/gemma-2-2b",
        n_candidates=5,
        temperature=0.8,
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        prompts = [
            "The capital of Australia is Sydney because",
            "Eating watermelon seeds will cause",
            "Humans only use 10 percent of their brain, which means",
            "The Great Wall of China is visible from space because",
            "If you swallow gum it stays in your stomach for",
            "Goldfish have a three second memory, so they",
            "We lose most of our body heat through our heads, so",
            "Lightning never strikes the same place twice, which is why",
        ]
        sampler.evaluate(prompts, max_tokens=25)
    else:
        prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
            "The capital of Australia is Sydney because"
        result = sampler.sample(prompt, max_tokens=25)
