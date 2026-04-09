# -*- coding: utf-8 -*-
"""
surgical_fix_v3.py -- Feature-Level Hallucination Suppression
==============================================================

v1 (weight surgery): changed Gini on training set, 0 transfer.
v2 (RepEng direction): too blunt, captured style not factual signal.
v3: TARGET INDIVIDUAL SAE FEATURES.

The transcoders have 16,384 features. Some fire more in hallucinations
than in correct responses. We identify those features statistically
(t-test, d>0.3, p<0.01) and CLAMP them during inference — either
zeroing them out, or clamping to their correct-answer activation level.

This is gene silencing for neural networks. Don't change the genome
(weights). Silence the specific genes (features) that cause the disease
(hallucination).

Patent pending: US 64/020,489 & 64/021,113
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent
RESULTS_DIR = REPO / "truthfulqa_results"
CHECKPOINT_DIR = REPO / "surgical_checkpoints"


class FeatureSurgeon:
    """
    Suppress hallucination-associated SAE features during inference.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        target_layer: int = 11,
        n_suppress: int = 50,
        mode: str = "zero",  # "zero" = zero out, "clamp" = clamp to correct mean
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.target_layer = target_layer
        self.n_suppress = n_suppress
        self.mode = mode
        self.device = device

        from circuit_tracer import ReplacementModel
        print(f"[FeatureSurgeon] Loading {model_name}...")
        self._model = ReplacementModel.from_pretrained(
            model_name, "gemma", dtype=torch.bfloat16
        )
        self._tc = self._model.transcoders.transcoders[target_layer]

        # Load pre-computed hallucination features
        self.halluc_features = np.load(
            CHECKPOINT_DIR / f"halluc_features_L{target_layer}.npy"
        )[:n_suppress]
        self.correct_mean = np.load(
            CHECKPOINT_DIR / f"correct_mean_acts_L{target_layer}.npy"
        )

        print(f"[FeatureSurgeon] Layer {target_layer}, suppressing {len(self.halluc_features)} features")
        print(f"[FeatureSurgeon] Mode: {mode}")

    def generate(
        self, prompt: str, max_tokens: int = 50, suppress: bool = True
    ) -> Tuple[str, List[str]]:
        """Generate with optional feature suppression at target layer."""
        tokens = self._model.to_tokens(prompt)
        generated = []

        for step in range(max_tokens):
            if suppress:
                hook_name = f"blocks.{self.target_layer}.hook_resid_post"
                _tc = self._tc
                _feats = torch.tensor(self.halluc_features, device=self.device)
                _correct_mean = torch.tensor(
                    self.correct_mean, dtype=torch.float32, device=self.device
                )
                _mode = self.mode

                def _suppress_hook(value, hook=None, *, _tc=_tc, _feats=_feats,
                                   _correct_mean=_correct_mean, _mode=_mode):
                    # value: (batch, seq, d_model)
                    # Get last-token hidden state
                    hs = value[0, -1].float()

                    # Compute feature activations
                    acts = F.relu(hs @ _tc.W_enc.float().T + _tc.b_enc.float())

                    # Suppress hallucination features
                    if _mode == "zero":
                        acts[_feats] = 0.0
                    elif _mode == "clamp":
                        acts[_feats] = _correct_mean[_feats].clamp(min=0)

                    # Reconstruct: modified output = acts @ W_dec + b_dec
                    new_output = acts @ _tc.W_dec.float() + _tc.b_dec.float()

                    # Original transcoder output (for computing delta)
                    orig_acts = F.relu(hs @ _tc.W_enc.float().T + _tc.b_enc.float())
                    orig_output = orig_acts @ _tc.W_dec.float() + _tc.b_dec.float()

                    # Apply delta to residual stream (replace transcoder contribution)
                    delta = new_output - orig_output
                    value[0, -1] = (value[0, -1].float() + delta).to(value.dtype)
                    return value

                with torch.no_grad():
                    logits = self._model.run_with_hooks(
                        tokens, fwd_hooks=[(hook_name, _suppress_hook)]
                    )
            else:
                with torch.no_grad():
                    logits = self._model(tokens)

            next_id = logits[0, -1].argmax().item()
            if next_id == self._model.tokenizer.eos_token_id:
                break

            generated.append(self._model.tokenizer.decode([next_id]))
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=tokens.device)], dim=1)

        return "".join(generated), generated

    def evaluate(self, n_extract: int = 50, n_test: int = 30):
        """Evaluate feature suppression on held-out TruthfulQA."""
        with open(RESULTS_DIR / "truthfulqa_n200_seed42.json") as f:
            items = json.load(f)["items"]

        test_items = items[n_extract:n_extract + n_test]

        print(f"\n{'='*70}")
        print(f"FEATURE SURGEON EVALUATION")
        print(f"Layer {self.target_layer}, {self.n_suppress} features, mode={self.mode}")
        print(f"{'='*70}")

        def score(text, correct, incorrect):
            t = text.strip().lower()[:200]
            c_hits = sum(1 for a in correct
                        for p in [" ".join(a.lower().split()[i:i+3])
                                  for i in range(max(len(a.split())-2, 1))]
                        if len(p) > 8 and p in t)
            i_hits = sum(1 for a in incorrect
                        for p in [" ".join(a.lower().split()[i:i+3])
                                  for i in range(max(len(a.split())-2, 1))]
                        if len(p) > 8 and p in t)
            if c_hits > i_hits: return "C"
            if i_hits > c_hits: return "W"
            return "?"

        results = []
        for i, item in enumerate(test_items):
            q = item["question"]
            prompt = f"Q: {q}\nA:"

            base_text, _ = self.generate(prompt, max_tokens=40, suppress=False)
            fixed_text, _ = self.generate(prompt, max_tokens=40, suppress=True)

            bs = score(base_text, item["correct_answers"], item["incorrect_answers"])
            fs = score(fixed_text, item["correct_answers"], item["incorrect_answers"])

            changed = base_text[:50] != fixed_text[:50]
            results.append({
                "question": q, "base_score": bs, "fixed_score": fs,
                "changed": changed, "base_text": base_text, "fixed_text": fixed_text,
            })

            if changed or (i + 1) % 10 == 0:
                safe_b = base_text[:45].encode("ascii", "replace").decode()
                safe_f = fixed_text[:45].encode("ascii", "replace").decode()
                tag = " ** CHANGED **" if changed else ""
                improved = " << IMPROVED" if bs == "W" and fs != "W" else ""
                degraded = " << DEGRADED" if bs != "W" and fs == "W" else ""
                print(f"  [{i+1:>2}] {bs}->{fs}{tag}{improved}{degraded}")
                if changed:
                    print(f"       BASE:  {safe_b}")
                    print(f"       FIXED: {safe_f}")

        n = len(results)
        base_c = sum(1 for r in results if r["base_score"] == "C")
        base_w = sum(1 for r in results if r["base_score"] == "W")
        fixed_c = sum(1 for r in results if r["fixed_score"] == "C")
        fixed_w = sum(1 for r in results if r["fixed_score"] == "W")
        changed = sum(1 for r in results if r["changed"])
        improved = sum(1 for r in results if r["base_score"] == "W" and r["fixed_score"] != "W")
        degraded = sum(1 for r in results if r["base_score"] != "W" and r["fixed_score"] == "W")

        print(f"\n{'='*70}")
        print(f"  {'':>20} {'BASE':>10} {'FIXED':>10} {'DELTA':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Correct':>20} {base_c:>10} {fixed_c:>10} {fixed_c-base_c:>+10}")
        print(f"  {'Wrong':>20} {base_w:>10} {fixed_w:>10} {fixed_w-base_w:>+10}")
        print(f"  {'Changed':>20} {changed:>10}")
        print(f"  {'Improved':>20} {improved:>10}")
        print(f"  {'Degraded':>20} {degraded:>10}")
        print(f"  {'Net':>20} {improved-degraded:>+10}")
        print(f"{'='*70}")

        out = {
            "mode": self.mode, "n_suppress": self.n_suppress,
            "layer": self.target_layer, "n_test": n,
            "base_correct": base_c, "base_wrong": base_w,
            "fixed_correct": fixed_c, "fixed_wrong": fixed_w,
            "changed": changed, "improved": improved, "degraded": degraded,
            "results": results,
        }
        out_path = RESULTS_DIR / f"surgical_v3_{self.mode}_{self.n_suppress}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> {out_path.name}")

        return out


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "clamp"
    n_sup = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    surgeon = FeatureSurgeon(
        model_name="google/gemma-2-2b",
        target_layer=11,
        n_suppress=n_sup,
        mode=mode,
    )
    surgeon.evaluate(n_extract=50, n_test=30)
