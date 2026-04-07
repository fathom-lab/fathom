"""
sae_free_cosine_probe.py — cross-layer cosine similarity probe
=================================================================
Instead of IPR (failed cross-arch), directly measure how ALIGNED
early-layer and late-layer residual streams are.

C_align(t) = mean cosine_sim(h_early_l, h_late_l') for all (l, l') pairs

High C_align = early and late layers processing in same direction = committed
Low C_align = early and late diverge = exploratory/uncertain

This is closer to what SAE coherence actually measures: alignment
of feature directions, not concentration of activations.

Usage:
    from sae_free_cosine_probe import CosineProbe
    probe = CosineProbe(model_name="meta-llama/Llama-3.2-3B-Instruct")
    result = probe.generate_with_alignment(prompt, max_tokens=40)
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CosineProbeResult:
    prompt: str
    text: str
    tokens: List[str]
    c_align_trajectory: List[float]     # cross-layer cosine similarity per token
    c_early_self_trajectory: List[float] # within-early-layers self-similarity
    c_late_self_trajectory: List[float]  # within-late-layers self-similarity
    c_delta_trajectory: List[float]      # late_self - early_self (analogous to SAE C_delta)
    total_tokens: int
    generation_time_s: float


class CosineProbe:
    """Cross-layer cosine similarity probe — SAE-free, architecture-agnostic."""

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        early_layers: Optional[List[int]] = None,
        late_layers: Optional[List[int]] = None,
        device: str = "cuda",
        dtype=torch.bfloat16,
    ):
        self.model_name = model_name
        self.device = device

        print(f"[CosineProbe] Loading {model_name}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto",
            output_hidden_states=True,
        )
        self.model.eval()

        n_layers = self.model.config.num_hidden_layers
        self._n_layers = n_layers

        if early_layers is None:
            self.early_layers = list(range(0, n_layers // 3))
        else:
            self.early_layers = early_layers
        if late_layers is None:
            self.late_layers = list(range(2 * n_layers // 3, n_layers))
        else:
            self.late_layers = late_layers

        print(f"[CosineProbe] Ready. {n_layers} layers, "
              f"early={self.early_layers}, late={self.late_layers}")

    def _band_self_similarity(self, hidden_states, layers):
        """Mean pairwise cosine similarity of last-token residuals within a layer band."""
        vecs = []
        for l in layers:
            h = hidden_states[l + 1][0, -1, :].float()  # (d_model,)
            vecs.append(F.normalize(h, dim=0))
        if len(vecs) < 2:
            return 0.0
        # Mean pairwise cosine
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(float(torch.dot(vecs[i], vecs[j]).item()))
        return float(np.mean(sims))

    def _cross_band_similarity(self, hidden_states, early_layers, late_layers):
        """Mean cosine similarity between early and late layer residuals."""
        early_vecs = [F.normalize(hidden_states[l + 1][0, -1, :].float(), dim=0)
                      for l in early_layers]
        late_vecs = [F.normalize(hidden_states[l + 1][0, -1, :].float(), dim=0)
                     for l in late_layers]
        sims = []
        for ev in early_vecs:
            for lv in late_vecs:
                sims.append(float(torch.dot(ev, lv).item()))
        return float(np.mean(sims))

    @torch.no_grad()
    def generate_with_alignment(self, prompt: str, max_tokens: int = 40) -> CosineProbeResult:
        tokenizer = self.tokenizer

        # Handle chat template for instruct models
        if any(k in self.model_name.lower() for k in ["it", "instruct", "chat"]):
            messages = [{"role": "user", "content": prompt}]
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                ).to(self.model.device)
            except Exception:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        generated = []
        c_align_traj = []
        c_early_traj = []
        c_late_traj = []
        c_delta_traj = []
        t_start = time.time()

        for _ in range(max_tokens):
            outputs = self.model(input_ids, output_hidden_states=True)
            hs = outputs.hidden_states

            c_early = self._band_self_similarity(hs, self.early_layers)
            c_late = self._band_self_similarity(hs, self.late_layers)
            c_cross = self._cross_band_similarity(hs, self.early_layers, self.late_layers)
            c_delta = c_late - c_early

            c_align_traj.append(c_cross)
            c_early_traj.append(c_early)
            c_late_traj.append(c_late)
            c_delta_traj.append(c_delta)

            logits = outputs.logits[:, -1, :]
            next_id = logits.argmax(dim=-1).item()
            if next_id == tokenizer.eos_token_id:
                break
            tok_str = tokenizer.decode([next_id])
            generated.append(tok_str)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=input_ids.device)], dim=1)

        elapsed = time.time() - t_start
        return CosineProbeResult(
            prompt=prompt,
            text="".join(generated),
            tokens=generated,
            c_align_trajectory=c_align_traj,
            c_early_self_trajectory=c_early_traj,
            c_late_self_trajectory=c_late_traj,
            c_delta_trajectory=c_delta_traj,
            total_tokens=len(generated),
            generation_time_s=elapsed,
        )


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-2-2b-it"
    probe = CosineProbe(model_name=model)
    r = probe.generate_with_alignment("Q: What is the capital of France?\nA:", max_tokens=10)
    print(f"Text: {r.text!r}")
    print(f"Tokens: {r.total_tokens}")
    print(f"C_align[:5]: {[f'{c:.4f}' for c in r.c_align_trajectory[:5]]}")
    print(f"C_delta[:5]: {[f'{c:+.4f}' for c in r.c_delta_trajectory[:5]]}")
    print(f"C_early[:5]: {[f'{c:.4f}' for c in r.c_early_self_trajectory[:5]]}")
    print(f"C_late[:5]:  {[f'{c:.4f}' for c in r.c_late_self_trajectory[:5]]}")
