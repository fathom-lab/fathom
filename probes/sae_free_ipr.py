"""
sae_free_ipr.py — SAE-free IPR probe for cross-architecture cognitive measurement
====================================================================================
Computes an IPR-based commitment intensity signal WITHOUT requiring
pre-trained SAE transcoders. Works on any transformer.

Core idea: at each token position during generation, extract residual
streams at early and late layers. Compute IPR of the top-k largest
activations in each. Their divergence (early vs late) is the
SAE-free analog of C_delta.

Then S_early = max/spike_count on that trajectory, same as before.

Usage:
    from sae_free_ipr import SAEFreeIPR
    probe = SAEFreeIPR(model_name="meta-llama/Llama-3.1-8B-Instruct", top_k=64)
    result = probe.generate_with_ipr(prompt, max_tokens=40)
    # result.c_delta_trajectory, result.ipr_early, result.ipr_late, etc.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SAEFreeResult:
    prompt: str
    text: str
    tokens: List[str]
    c_delta_trajectory: List[float]   # IPR_late - IPR_early per token
    ipr_early_trajectory: List[float]
    ipr_late_trajectory: List[float]
    total_tokens: int
    generation_time_s: float


class SAEFreeIPR:
    """SAE-free commitment-intensity measurement via top-k residual IPR."""

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        top_k: int = 64,
        early_layers: Optional[List[int]] = None,
        late_layers: Optional[List[int]] = None,
        device: str = "cuda",
        dtype=torch.bfloat16,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self.dtype = dtype

        print(f"[SAEFreeIPR] Loading {model_name}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto",
            output_hidden_states=True,
        )
        self.model.eval()

        # Detect number of layers
        # transformers models: len(model.model.layers) or model.config.num_hidden_layers
        n_layers = getattr(self.model.config, "num_hidden_layers", None)
        if n_layers is None:
            raise ValueError("Cannot determine number of layers")
        self._n_layers = n_layers

        if early_layers is None:
            self.early_layers = list(range(0, n_layers // 3))
        else:
            self.early_layers = early_layers
        if late_layers is None:
            self.late_layers = list(range(2 * n_layers // 3, n_layers))
        else:
            self.late_layers = late_layers

        print(f"[SAEFreeIPR] Ready. {n_layers} layers, "
              f"early={self.early_layers}, late={self.late_layers}, "
              f"top_k={top_k}")

    def _topk_ipr(self, vec: torch.Tensor) -> float:
        """
        Compute IPR of top-k absolute values of a 1D vector.
        vec: (d_model,) residual stream vector
        returns: IPR = sum(p_i^2) where p_i = |v_i|^2 / sum(|v_j|^2) over top-k.

        This measures how concentrated the top-k activations are.
        """
        # Take absolute values, get top-k
        abs_vec = vec.abs().float()
        if abs_vec.numel() < self.top_k:
            topk_vals = abs_vec
        else:
            topk_vals, _ = torch.topk(abs_vec, self.top_k)
        # Square → treat as unnormalized probability
        sq = topk_vals ** 2
        total = sq.sum()
        if total.item() < 1e-12:
            return 0.0
        probs = sq / total
        ipr = (probs ** 2).sum().item()
        return ipr

    def _layer_band_ipr(self, hidden_states_tuple, layers):
        """
        Compute mean IPR over a set of layer indices.
        hidden_states_tuple: tuple of (n_layers+1) tensors, each (batch, seq, d_model)
          Index 0 is embeddings, 1..n_layers are layer outputs.
        We take the LAST token position of each layer.
        """
        iprs = []
        for l in layers:
            # hidden_states[l+1] is output of layer l (index 0 = embeddings)
            # wait - need to check this. HF convention: hidden_states[0] = embeddings,
            # hidden_states[i] for i>=1 = output of layer i-1.
            # We want the residual AFTER layer l, which is hidden_states[l+1].
            h = hidden_states_tuple[l + 1]  # (1, seq, d_model)
            last = h[0, -1, :]  # (d_model,)
            iprs.append(self._topk_ipr(last))
        if not iprs:
            return 0.0
        return float(np.mean(iprs))

    @torch.no_grad()
    def generate_with_ipr(self, prompt: str, max_tokens: int = 40) -> SAEFreeResult:
        """Generate greedily, capture IPR at early/late layers per token."""
        # Apply chat template if IT model
        tokenizer = self.tokenizer
        if "it" in self.model_name.lower() or "instruct" in self.model_name.lower() or "chat" in self.model_name.lower():
            # Use chat template
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
        c_delta_traj, ipr_e_traj, ipr_l_traj = [], [], []
        t_start = time.time()

        for _ in range(max_tokens):
            outputs = self.model(input_ids, output_hidden_states=True)
            hs = outputs.hidden_states
            # Compute IPRs
            ipr_early = self._layer_band_ipr(hs, self.early_layers)
            ipr_late = self._layer_band_ipr(hs, self.late_layers)
            c_delta = ipr_late - ipr_early
            c_delta_traj.append(c_delta)
            ipr_e_traj.append(ipr_early)
            ipr_l_traj.append(ipr_late)

            # Decode next
            logits = outputs.logits[:, -1, :]
            next_id = logits.argmax(dim=-1).item()
            if next_id == tokenizer.eos_token_id:
                break
            tok_str = tokenizer.decode([next_id])
            generated.append(tok_str)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=input_ids.device)], dim=1)

        elapsed = time.time() - t_start
        return SAEFreeResult(
            prompt=prompt,
            text="".join(generated),
            tokens=generated,
            c_delta_trajectory=c_delta_traj,
            ipr_early_trajectory=ipr_e_traj,
            ipr_late_trajectory=ipr_l_traj,
            total_tokens=len(generated),
            generation_time_s=elapsed,
        )


if __name__ == "__main__":
    # Smoke test
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-2-2b-it"
    probe = SAEFreeIPR(model_name=model_name, top_k=64)
    r = probe.generate_with_ipr("Q: What is the capital of France?\nA:", max_tokens=10)
    print(f"Text: {r.text!r}")
    print(f"Tokens: {r.total_tokens}")
    print(f"C_delta[:5]: {[f'{c:+.5f}' for c in r.c_delta_trajectory[:5]]}")
    print(f"IPR_early[:5]: {[f'{c:.5f}' for c in r.ipr_early_trajectory[:5]]}")
    print(f"IPR_late[:5]: {[f'{c:.5f}' for c in r.ipr_late_trajectory[:5]]}")
    print(f"Time: {r.generation_time_s:.2f}s")
