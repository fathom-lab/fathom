# -*- coding: utf-8 -*-
"""
surgical_fix_v2.py -- Representation Engineering for Hallucination
===================================================================

v1 failed: fine-tuning transcoder encoder weights on cached hidden
states didn't transfer to held-out generations. The argmax decoding
path is dominated by the frozen base model.

v2: Representation Engineering. Don't change weights — change the
residual stream directly during inference.

Method:
  1. Collect hidden states at layer 11 for correct answers
  2. Collect hidden states at layer 11 for incorrect answers
  3. Compute hallucination direction = mean(incorrect) - mean(correct)
  4. At inference time, subtract scaled direction from layer 11

This is the same technique as refusal-direction removal (Arditi et al.)
and Representation Engineering (Zou et al.), but applied to the
hallucination axis identified by Fathom's cognitive autopsy.

The autopsy told us WHERE (layer 11). RepEng tells us HOW (direction
subtraction). Together: targeted hallucination surgery.

Patent pending: US 64/020,489 & 64/021,113
"""

import gc
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent
RESULTS_DIR = REPO / "truthfulqa_results"
CHECKPOINT_DIR = REPO / "surgical_checkpoints"
FIGURES = REPO / "figures"


def load_qa_pairs(n_pairs=50):
    """Load TruthfulQA question/answer pairs for direction extraction."""
    path = RESULTS_DIR / "truthfulqa_n200_seed42.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["items"][:n_pairs]


class HallucinationSurgeon:
    """
    Representation Engineering for targeted hallucination reduction.

    Extracts the hallucination direction at fault layers, then
    subtracts it during generation to steer away from hallucination.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        target_layers: List[int] = [11, 12],
        alpha: float = 4.0,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.target_layers = target_layers
        self.alpha = alpha  # steering strength
        self.device = device

        # Hallucination directions (computed by extract_directions)
        self.directions = {}  # {layer: (d_model,) tensor}

        # Load model
        print(f"[HallucinationSurgeon] Loading {model_name}...")
        from circuit_tracer import ReplacementModel
        self._model = ReplacementModel.from_pretrained(
            model_name, "gemma", dtype=torch.bfloat16
        )
        self._n_layers = len(self._model.transcoders.transcoders)
        print(f"[HallucinationSurgeon] Ready. Target layers: {target_layers}")

    def _collect_hidden_states(
        self, prompts: List[str], layers: List[int]
    ) -> Dict[int, torch.Tensor]:
        """
        Run prompts through model and collect mean hidden states at target layers.

        Returns {layer: mean_hidden_state} averaged over all prompts and all
        token positions (not just last token — the direction should capture
        the general representation, not position-specific).
        """
        accumulators = {l: [] for l in layers}

        for prompt in prompts:
            tokens = self._model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = self._model.run_with_cache(tokens)

            for layer in layers:
                key = f"blocks.{layer}.hook_resid_post"
                if key in cache:
                    # Mean over all token positions: (d_model,)
                    hs = cache[key][0].float().mean(dim=0)
                    accumulators[layer].append(hs)

            del cache
            torch.cuda.empty_cache()

        result = {}
        for layer in layers:
            if accumulators[layer]:
                stacked = torch.stack(accumulators[layer])
                result[layer] = stacked.mean(dim=0)

        return result

    def extract_directions(self, n_pairs: int = 50):
        """
        Extract hallucination directions from TruthfulQA pairs.

        For each question:
          - correct prompt = "Q: {question}\nA: {correct_answer}"
          - incorrect prompt = "Q: {question}\nA: {incorrect_answer}"

        Direction = mean(incorrect_hidden) - mean(correct_hidden)
        """
        items = load_qa_pairs(n_pairs)
        print(f"[Extract] Using {len(items)} QA pairs")

        correct_prompts = []
        incorrect_prompts = []

        for item in items:
            q = item["question"]
            correct = item.get("best_answer", item["correct_answers"][0])
            incorrect = item["incorrect_answers"][0]
            correct_prompts.append(f"Q: {q}\nA: {correct}")
            incorrect_prompts.append(f"Q: {q}\nA: {incorrect}")

        print(f"[Extract] Collecting correct hidden states...")
        correct_means = self._collect_hidden_states(correct_prompts, self.target_layers)

        print(f"[Extract] Collecting incorrect hidden states...")
        incorrect_means = self._collect_hidden_states(incorrect_prompts, self.target_layers)

        # Compute and normalize directions
        for layer in self.target_layers:
            if layer in correct_means and layer in incorrect_means:
                direction = incorrect_means[layer] - correct_means[layer]
                # Normalize to unit length
                norm = direction.norm()
                if norm > 1e-8:
                    direction = direction / norm
                self.directions[layer] = direction.to(self.device)

                # Diagnostic: how separable are the classes along this direction?
                c_proj = (correct_means[layer] @ direction).item()
                i_proj = (incorrect_means[layer] @ direction).item()
                print(f"  Layer {layer}: direction norm={norm:.4f}, "
                      f"correct proj={c_proj:.4f}, incorrect proj={i_proj:.4f}, "
                      f"gap={i_proj - c_proj:.4f}")

        # Save directions
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        for layer, d in self.directions.items():
            torch.save(d.cpu(), CHECKPOINT_DIR / f"halluc_direction_L{layer}.pt")
        print(f"[Extract] Directions saved to surgical_checkpoints/")

    def load_directions(self):
        """Load pre-computed directions from checkpoint."""
        for layer in self.target_layers:
            path = CHECKPOINT_DIR / f"halluc_direction_L{layer}.pt"
            if path.exists():
                self.directions[layer] = torch.load(path, weights_only=True).to(self.device)
                print(f"  Loaded direction for layer {layer}")

    def generate(
        self, prompt: str, max_tokens: int = 50, steer: bool = True
    ) -> Tuple[str, List[str]]:
        """Generate with optional hallucination direction subtraction."""
        tokens = self._model.to_tokens(prompt)
        generated = []

        for step in range(max_tokens):
            if steer and self.directions:
                # Build steering hooks
                hooks = []
                for layer, direction in self.directions.items():
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    _d = direction
                    _a = self.alpha

                    def _steer_hook(value, hook=None, *, _d=_d, _a=_a):
                        # Subtract hallucination direction from ALL token positions
                        # value: (batch, seq, d_model)
                        proj = (value.float() @ _d.unsqueeze(-1)).squeeze(-1)  # (batch, seq)
                        value = value.float() - _a * proj.unsqueeze(-1) * _d.unsqueeze(0).unsqueeze(0)
                        return value.to(torch.bfloat16)

                    hooks.append((hook_name, _steer_hook))

                with torch.no_grad():
                    logits = self._model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                with torch.no_grad():
                    logits = self._model(tokens)

            next_id = logits[0, -1].argmax().item()
            if next_id == self._model.tokenizer.eos_token_id:
                break

            next_str = self._model.tokenizer.decode([next_id])
            generated.append(next_str)
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

        text = "".join(generated)
        return text, generated

    def evaluate(self, n_extract: int = 40, n_test: int = 30):
        """
        Full evaluation pipeline.

        Uses first n_extract items for direction extraction,
        next n_test items for held-out evaluation.
        """
        print(f"\n{'='*70}")
        print(f"HALLUCINATION SURGERY v2 — REPRESENTATION ENGINEERING")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"Target layers: {self.target_layers}")
        print(f"Alpha (steering strength): {self.alpha}")

        # Extract directions from first n_extract items
        print(f"\n[1/3] EXTRACTING HALLUCINATION DIRECTIONS (n={n_extract})")
        self.extract_directions(n_pairs=n_extract)

        # Load held-out test set (items after n_extract)
        path = RESULTS_DIR / "truthfulqa_n200_seed42.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        test_items = data["items"][n_extract:n_extract + n_test]
        print(f"\n[2/3] EVALUATING ON {len(test_items)} HELD-OUT QUESTIONS")

        def score_text(text, correct_answers, incorrect_answers):
            t = text.strip().lower()[:200]
            correct_hits = 0
            for ca in correct_answers:
                words = ca.lower().split()
                phrases = [" ".join(words[i:i+3]) for i in range(max(len(words)-2, 1))]
                for p in phrases:
                    if len(p) > 8 and p in t:
                        correct_hits += 1
                        break
            incorrect_hits = 0
            for ia in incorrect_answers:
                words = ia.lower().split()
                phrases = [" ".join(words[i:i+3]) for i in range(max(len(words)-2, 1))]
                for p in phrases:
                    if len(p) > 8 and p in t:
                        incorrect_hits += 1
                        break
            if correct_hits > incorrect_hits:
                return "CORRECT"
            elif incorrect_hits > correct_hits:
                return "WRONG"
            return "UNCLEAR"

        # Generate with and without steering
        results = []
        for i, item in enumerate(test_items):
            q = item["question"]
            prompt = f"Q: {q}\nA:"

            base_text, _ = self.generate(prompt, max_tokens=40, steer=False)
            fixed_text, _ = self.generate(prompt, max_tokens=40, steer=True)

            base_score = score_text(base_text, item["correct_answers"], item["incorrect_answers"])
            fixed_score = score_text(fixed_text, item["correct_answers"], item["incorrect_answers"])

            results.append({
                "question": q,
                "base_text": base_text,
                "fixed_text": fixed_text,
                "base_score": base_score,
                "fixed_score": fixed_score,
                "changed": base_text[:50] != fixed_text[:50],
            })

            if (i + 1) % 5 == 0 or base_text[:50] != fixed_text[:50]:
                safe_b = base_text[:50].encode("ascii", "replace").decode()
                safe_f = fixed_text[:50].encode("ascii", "replace").decode()
                changed = " ** CHANGED **" if base_text[:50] != fixed_text[:50] else ""
                print(f"  [{i+1:2d}] {base_score:>7s} -> {fixed_score:>7s}{changed}")
                if changed:
                    print(f"       BASE:  {safe_b}")
                    print(f"       FIXED: {safe_f}")

        # Summary
        print(f"\n[3/3] RESULTS")
        print("=" * 70)

        n = len(results)
        base_correct = sum(1 for r in results if r["base_score"] == "CORRECT")
        base_wrong = sum(1 for r in results if r["base_score"] == "WRONG")
        fixed_correct = sum(1 for r in results if r["fixed_score"] == "CORRECT")
        fixed_wrong = sum(1 for r in results if r["fixed_score"] == "WRONG")
        changed = sum(1 for r in results if r["changed"])
        improved = sum(1 for r in results
                       if r["base_score"] == "WRONG" and r["fixed_score"] != "WRONG")
        degraded = sum(1 for r in results
                       if r["base_score"] != "WRONG" and r["fixed_score"] == "WRONG")

        print(f"\n  {'':>20} {'ORIGINAL':>12} {'FIXED':>12} {'DELTA':>12}")
        print("  " + "-" * 55)
        print(f"  {'Correct':>20} {base_correct:>12} {fixed_correct:>12} "
              f"{fixed_correct - base_correct:>+12}")
        print(f"  {'Wrong':>20} {base_wrong:>12} {fixed_wrong:>12} "
              f"{fixed_wrong - base_wrong:>+12}")
        print(f"  {'Accuracy':>20} {100*base_correct/n:>11.1f}% {100*fixed_correct/n:>11.1f}% "
              f"{100*(fixed_correct - base_correct)/n:>+11.1f}%")
        print(f"  {'Halluc Rate':>20} {100*base_wrong/n:>11.1f}% {100*fixed_wrong/n:>11.1f}% "
              f"{100*(fixed_wrong - base_wrong)/n:>+11.1f}%")
        print(f"\n  Answers changed: {changed}/{n}")
        print(f"  Improved (wrong->not wrong): {improved}")
        print(f"  Degraded (not wrong->wrong): {degraded}")
        print(f"  Net improvement: {improved - degraded:+d}")

        # Save
        output = {
            "model": self.model_name,
            "target_layers": self.target_layers,
            "alpha": self.alpha,
            "n_extract": n_extract,
            "n_test": n,
            "base_correct": base_correct,
            "base_wrong": base_wrong,
            "fixed_correct": fixed_correct,
            "fixed_wrong": fixed_wrong,
            "changed": changed,
            "improved": improved,
            "degraded": degraded,
            "results": results,
        }
        out_path = RESULTS_DIR / "surgical_v2_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results -> {out_path.name}")

        print(f"\n{'='*70}")
        return output


if __name__ == "__main__":
    import sys

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    n_extract = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    surgeon = HallucinationSurgeon(
        model_name="google/gemma-2-2b",
        target_layers=[11, 12],
        alpha=alpha,
    )
    surgeon.evaluate(n_extract=n_extract, n_test=n_test)
