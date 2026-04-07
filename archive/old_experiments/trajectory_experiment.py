"""
trajectory_experiment.py — Predictive Cognitive Trajectory Monitoring
=====================================================================

Day 1: Extract token-by-token (K, C_delta) trajectories for truthful
vs hallucinated generations on TruthfulQA.

The hypothesis: in (K, C_delta) 2D space, truthful and hallucinating
trajectories share an early path then DIVERGE N>0 tokens before the
hallucination appears in output.

Uses CoherenceSteerer's fast single-pass architecture:
  - Per token: one forward pass with hooks → hidden states
  - C_delta: encoder-only matmuls (already implemented)
  - K_fast: count active features per layer → weighted mean layer index

No circuit_tracer.attribute() calls — pure encoder inference, ~10% overhead.

Patent pending: US 64/020,489 & 64/021,113
"""

import gc
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Results ──────────────────────────────────────────────────────────────

@dataclass
class TokenState:
    """Cognitive state at a single token position."""
    token_idx: int
    token_str: str
    k_fast: Optional[float]         # weighted mean active layer (lower = deeper)
    c_delta: Optional[float]        # C_late - C_early (coherence shift)
    c_early: Optional[float]        # mean coherence in early layers
    c_late: Optional[float]         # mean coherence in late layers
    n_active_features: int          # total active features across monitored layers
    layer_feature_counts: dict      # {layer_idx: n_active} for monitored layers


@dataclass
class TrajectoryResult:
    """Full token-by-token cognitive trajectory for one generation."""
    prompt: str
    label: str                      # "correct" or "hallucinated"
    question: str
    answer: str
    generated_text: str
    tokens: List[str]
    states: List[TokenState]
    generation_time_s: float
    n_tokens: int


# ── Fast K computation from hidden states ────────────────────────────────

def fast_k_from_hidden_states(
    hidden_states: dict,
    tc_list,
    monitor_layers: list,
) -> Tuple[Optional[float], dict]:
    """
    Compute K (depth) from captured hidden states using encoder-only pass.

    K = weighted mean layer index, where weights = number of active features.
    Lower K = computation concentrated in early layers (surface/retrieval).
    Higher K = computation distributed to late layers (deeper reasoning).

    This is the fast approximation of the attribution-based depth score:
    instead of running full circuit attribution, we count how many SAE
    features fire at each layer from a single encoder pass.

    Returns: (k_fast, {layer_idx: n_active_features})
    """
    layer_counts = {}
    total_active = 0
    weighted_sum = 0.0

    for layer_idx in monitor_layers:
        hidden = hidden_states.get(layer_idx)
        if hidden is None:
            layer_counts[layer_idx] = 0
            continue

        tc = tc_list[layer_idx]
        W_enc = getattr(tc, "W_enc", None)
        b_enc = getattr(tc, "b_enc", None)

        if W_enc is None:
            layer_counts[layer_idx] = 0
            continue

        with torch.no_grad():
            acts = hidden.float() @ W_enc.float().T
            if b_enc is not None:
                acts = acts + b_enc.float()
            acts = F.relu(acts)

        n_active = int((acts > 0).sum().item())
        layer_counts[layer_idx] = n_active
        total_active += n_active
        weighted_sum += layer_idx * n_active

    if total_active == 0:
        return None, layer_counts

    k_fast = weighted_sum / total_active
    return float(k_fast), layer_counts


# ── Trajectory Extractor ─────────────────────────────────────────────────

class TrajectoryExtractor:
    """
    Extracts token-by-token (K, C_delta) cognitive trajectories.

    Built on CoherenceSteerer's single-pass architecture for speed.
    Monitors ALL layers (not just early+late) for richer K signal.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        top_k_features: int = 100,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.top_k = top_k_features
        self.device = device

        # Load model
        print(f"[TrajectoryExtractor] Loading {model_name}...")
        from circuit_tracer import ReplacementModel
        self._model = ReplacementModel.from_pretrained(
            model_name, "gemma", dtype=torch.bfloat16
        )
        self._inner = self._get_inner_model()
        self._tc = self._model.transcoders
        self._tc_list = self._tc.transcoders
        self._n_layers = len(self._tc_list)

        # Monitor ALL layers for K, but use early/late bands for C_delta
        self._all_layers = list(range(self._n_layers))
        self._early_layers = list(range(0, self._n_layers // 3))
        self._late_layers = list(range(2 * self._n_layers // 3, self._n_layers))

        # Hook names for all layers
        self._hook_names = {
            l: f"blocks.{l}.hook_resid_post" for l in self._all_layers
        }

        # Cache decoder weights
        self._dec_weights = {}
        for i, tc in enumerate(self._tc_list):
            W_dec = getattr(tc, "W_dec", None)
            if W_dec is not None:
                self._dec_weights[i] = W_dec

        print(f"[TrajectoryExtractor] Ready. {self._n_layers} layers, "
              f"early={self._early_layers}, late={self._late_layers}")

    def _get_inner_model(self):
        if hasattr(self._model, "to_tokens") and hasattr(self._model, "hook_dict"):
            return self._model
        for attr in ["model", "hooked_model", "_model"]:
            candidate = getattr(self._model, attr, None)
            if candidate is not None and hasattr(candidate, "to_tokens"):
                return candidate
        return None

    def _fast_layer_coherence(self, hidden_state, layer_idx):
        """Coherence for one layer (mean pairwise cosine sim of top-k decoder dirs)."""
        tc = self._tc_list[layer_idx]
        W_enc = getattr(tc, "W_enc", None)
        b_enc = getattr(tc, "b_enc", None)
        if W_enc is None:
            return None

        with torch.no_grad():
            acts = hidden_state.float() @ W_enc.float().T
            if b_enc is not None:
                acts = acts + b_enc.float()
            acts = F.relu(acts)

        n_active = (acts > 0).sum().item()
        if n_active < 2:
            return None

        k = min(self.top_k, int(n_active))
        topk_vals, topk_ids = torch.topk(acts, k)

        W_dec = self._dec_weights.get(layer_idx)
        if W_dec is None:
            return None

        valid_mask = topk_ids < W_dec.shape[0]
        valid_ids = topk_ids[valid_mask]
        if len(valid_ids) < 2:
            return None

        vecs = W_dec[valid_ids].float()
        norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vecs_normed = vecs / norms
        sim_matrix = vecs_normed @ vecs_normed.T
        n = len(valid_ids)
        C = (sim_matrix.sum().item() - n) / (n * (n - 1))
        return C

    def _build_capture_hooks(self, hidden_states: dict) -> list:
        """Build hooks to capture last-token residual stream at ALL layers."""
        hooks = []
        for l in self._all_layers:
            name = self._hook_names[l]
            def _hook(value, hook, _l=l):
                hidden_states[_l] = value[0, -1].detach().clone()
                return value
            hooks.append((name, _hook))
        return hooks

    def extract_trajectory(
        self,
        prompt: str,
        max_tokens: int = 60,
        label: str = "unknown",
        question: str = "",
        answer: str = "",
    ) -> TrajectoryResult:
        """
        Generate tokens and record (K, C_delta) at each position.

        Returns a TrajectoryResult with per-token cognitive states.
        """
        inner = self._inner
        tokens = inner.to_tokens(prompt)
        generated_tokens = []
        states = []
        t_start = time.time()

        for step in range(max_tokens):
            # Single forward pass, capture all hidden states
            hidden_states = {}
            capture_hooks = self._build_capture_hooks(hidden_states)

            with torch.no_grad():
                logits = inner.run_with_hooks(tokens, fwd_hooks=capture_hooks)

            # Compute K_fast from all layers
            k_fast, layer_counts = fast_k_from_hidden_states(
                hidden_states, self._tc_list, self._all_layers
            )

            # Compute C_delta from early/late layers
            early_cs = []
            late_cs = []
            for l in self._early_layers:
                h = hidden_states.get(l)
                if h is not None:
                    c = self._fast_layer_coherence(h, l)
                    if c is not None:
                        early_cs.append(c)
            for l in self._late_layers:
                h = hidden_states.get(l)
                if h is not None:
                    c = self._fast_layer_coherence(h, l)
                    if c is not None:
                        late_cs.append(c)

            c_early = float(np.mean(early_cs)) if early_cs else None
            c_late = float(np.mean(late_cs)) if late_cs else None
            c_delta = float(c_late - c_early) if (c_early is not None and c_late is not None) else None

            total_active = sum(layer_counts.values())

            # Decode next token
            next_token_id = logits[0, -1].argmax().item()
            next_token_str = inner.tokenizer.decode([next_token_id])

            # Record state
            states.append(TokenState(
                token_idx=step,
                token_str=next_token_str,
                k_fast=k_fast,
                c_delta=c_delta,
                c_early=c_early,
                c_late=c_late,
                n_active_features=total_active,
                layer_feature_counts={str(k): int(v) for k, v in layer_counts.items()},
            ))

            if next_token_id == inner.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_str)
            next_tensor = torch.tensor([[next_token_id]], device=tokens.device)
            tokens = torch.cat([tokens, next_tensor], dim=1)

        elapsed = time.time() - t_start
        text = "".join(generated_tokens)

        return TrajectoryResult(
            prompt=prompt,
            label=label,
            question=question,
            answer=answer,
            generated_text=text,
            tokens=generated_tokens,
            states=states,
            generation_time_s=elapsed,
            n_tokens=len(generated_tokens),
        )


# ── TruthfulQA Experiment ────────────────────────────────────────────────

def load_truthfulqa(n: int = 50, seed: int = 42):
    """Load n TruthfulQA questions with correct/incorrect answer pairs."""
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    questions = []
    for idx in indices:
        item = ds[int(idx)]
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]
        correct = [c for c, l in zip(choices, labels) if l == 1]
        incorrect = [c for c, l in zip(choices, labels) if l == 0]
        if correct and incorrect:
            questions.append({
                "question": item["question"],
                "correct_answer": correct[0],
                "incorrect_answer": incorrect[0],
            })
    print(f"[+] Loaded {len(questions)} TruthfulQA pairs (seed={seed})")
    return questions


def run_experiment(
    n_questions: int = 50,
    max_tokens: int = 30,
    seed: int = 42,
    output_dir: str = "trajectory_results",
):
    """
    Run trajectory extraction on TruthfulQA.

    For each question, generate from both the correct and incorrect answer,
    recording (K, C_delta) at every token. Save results for divergence analysis.
    """
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    checkpoint_path = out_path / "checkpoint.json"

    # Resume from checkpoint if exists
    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            completed = json.load(f)
        print(f"[+] Resuming from checkpoint: {len(completed)} pairs done")

    questions = load_truthfulqa(n_questions, seed)
    extractor = TrajectoryExtractor()

    results = []
    for i, q in enumerate(questions):
        qid = f"q{i}"
        if qid in completed:
            results.append(completed[qid])
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(questions)}] {q['question'][:80]}")
        print(f"{'='*60}")

        # Correct answer trajectory
        correct_prompt = f"Q: {q['question']}\nA: {q['correct_answer']}"
        print(f"  [correct] generating...")
        try:
            t_correct = extractor.extract_trajectory(
                correct_prompt,
                max_tokens=max_tokens,
                label="correct",
                question=q["question"],
                answer=q["correct_answer"],
            )
            correct_data = _trajectory_to_dict(t_correct)
        except Exception as e:
            print(f"  [correct] FAILED: {e}")
            correct_data = None

        gc.collect()
        torch.cuda.empty_cache()

        # Hallucinated answer trajectory
        halluc_prompt = f"Q: {q['question']}\nA: {q['incorrect_answer']}"
        print(f"  [hallucinated] generating...")
        try:
            t_halluc = extractor.extract_trajectory(
                halluc_prompt,
                max_tokens=max_tokens,
                label="hallucinated",
                question=q["question"],
                answer=q["incorrect_answer"],
            )
            halluc_data = _trajectory_to_dict(t_halluc)
        except Exception as e:
            print(f"  [hallucinated] FAILED: {e}")
            halluc_data = None

        gc.collect()
        torch.cuda.empty_cache()

        pair = {
            "question": q["question"],
            "correct": correct_data,
            "hallucinated": halluc_data,
        }
        results.append(pair)
        completed[qid] = pair

        # Print trajectory summary
        if correct_data and halluc_data:
            c_ks = [s["k_fast"] for s in correct_data["states"] if s["k_fast"] is not None]
            h_ks = [s["k_fast"] for s in halluc_data["states"] if s["k_fast"] is not None]
            c_cds = [s["c_delta"] for s in correct_data["states"] if s["c_delta"] is not None]
            h_cds = [s["c_delta"] for s in halluc_data["states"] if s["c_delta"] is not None]

            if c_ks and h_ks:
                print(f"  K_fast  — correct: {np.mean(c_ks):.3f}  halluc: {np.mean(h_ks):.3f}")
            if c_cds and h_cds:
                print(f"  C_delta — correct: {np.mean(c_cds):.4f}  halluc: {np.mean(h_cds):.4f}")

        # Checkpoint every 5 pairs
        if (i + 1) % 5 == 0:
            print(f"\n  [checkpoint] saving {len(completed)} pairs...")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(completed, f)

    # Save final results
    final_path = out_path / "trajectories_final.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved {len(results)} pairs to {final_path}")

    # Quick summary stats
    _print_summary(results)

    return results


def _trajectory_to_dict(t: TrajectoryResult) -> dict:
    """Convert TrajectoryResult to JSON-serializable dict."""
    return {
        "prompt": t.prompt,
        "label": t.label,
        "question": t.question,
        "answer": t.answer,
        "generated_text": t.generated_text,
        "tokens": t.tokens,
        "states": [asdict(s) for s in t.states],
        "generation_time_s": t.generation_time_s,
        "n_tokens": t.n_tokens,
    }


def _print_summary(results):
    """Print summary statistics across all pairs."""
    print(f"\n{'='*60}")
    print(f"TRAJECTORY EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    all_correct_k = []
    all_halluc_k = []
    all_correct_cd = []
    all_halluc_cd = []

    # Per-token divergence tracking
    divergence_points = []

    for pair in results:
        c = pair.get("correct")
        h = pair.get("hallucinated")
        if not c or not h:
            continue

        c_states = c["states"]
        h_states = h["states"]

        # Aggregate means
        c_ks = [s["k_fast"] for s in c_states if s["k_fast"] is not None]
        h_ks = [s["k_fast"] for s in h_states if s["k_fast"] is not None]
        c_cds = [s["c_delta"] for s in c_states if s["c_delta"] is not None]
        h_cds = [s["c_delta"] for s in h_states if s["c_delta"] is not None]

        if c_ks: all_correct_k.extend(c_ks)
        if h_ks: all_halluc_k.extend(h_ks)
        if c_cds: all_correct_cd.extend(c_cds)
        if h_cds: all_halluc_cd.extend(h_cds)

        # Find first divergence point: where |K_diff| or |C_delta_diff| exceeds threshold
        min_len = min(len(c_states), len(h_states))
        for t in range(min_len):
            ck = c_states[t]["k_fast"]
            hk = h_states[t]["k_fast"]
            cc = c_states[t]["c_delta"]
            hc = h_states[t]["c_delta"]
            if ck is not None and hk is not None:
                k_diff = abs(hk - ck)
                cd_diff = abs((hc or 0) - (cc or 0))
                # Divergence = either axis separates by >1 std
                if k_diff > 0.5 or cd_diff > 0.003:
                    divergence_points.append(t)
                    break

    print(f"\nPairs analyzed: {len(results)}")

    if all_correct_k and all_halluc_k:
        print(f"\nK_fast (depth):")
        print(f"  correct:      {np.mean(all_correct_k):.3f} +/- {np.std(all_correct_k):.3f}")
        print(f"  hallucinated: {np.mean(all_halluc_k):.3f} +/- {np.std(all_halluc_k):.3f}")
        print(f"  delta:        {np.mean(all_halluc_k) - np.mean(all_correct_k):+.3f}")

    if all_correct_cd and all_halluc_cd:
        print(f"\nC_delta (coherence shift):")
        print(f"  correct:      {np.mean(all_correct_cd):.4f} +/- {np.std(all_correct_cd):.4f}")
        print(f"  hallucinated: {np.mean(all_halluc_cd):.4f} +/- {np.std(all_halluc_cd):.4f}")
        print(f"  delta:        {np.mean(all_halluc_cd) - np.mean(all_correct_cd):+.4f}")

    if divergence_points:
        print(f"\nDivergence detection:")
        print(f"  mean token of first divergence: {np.mean(divergence_points):.1f}")
        print(f"  median: {np.median(divergence_points):.0f}")
        print(f"  earliest: {min(divergence_points)}")
        print(f"  pairs with divergence: {len(divergence_points)}/{len(results)}")
    else:
        print(f"\nNo clear divergence points detected (thresholds may need tuning)")


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cognitive trajectory extraction")
    parser.add_argument("--n", type=int, default=50, help="Number of TruthfulQA pairs")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max tokens per generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="trajectory_results", help="Output directory")
    args = parser.parse_args()

    run_experiment(
        n_questions=args.n,
        max_tokens=args.max_tokens,
        seed=args.seed,
        output_dir=args.output,
    )
