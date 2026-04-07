"""
coherence_steerer_ext.py — extended CoherenceSteerer with entropy capture
==========================================================================
Wraps base CoherenceSteerer, adds per-token logit entropy capture
alongside C_delta trajectory. Enables side-by-side comparison of
S-axis vs standard logit-entropy baseline for hallucination detection.

Usage:
    from coherence_steerer_ext import CoherenceSteererExt
    steerer = CoherenceSteererExt(model_name="google/gemma-2-2b-it")
    result = steerer.generate_with_entropy(prompt, max_tokens=40)
    # result.c_delta_trajectory, result.entropy_trajectory, result.logprob_trajectory
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from coherence_steerer import CoherenceSteerer, GenerationResult


@dataclass
class GenerationResultExt(GenerationResult):
    """Extended result with entropy + top-logprob trajectories."""
    entropy_trajectory: List[float] = field(default_factory=list)
    logprob_trajectory: List[float] = field(default_factory=list)  # log p(chosen token)
    top2_margin_trajectory: List[float] = field(default_factory=list)  # p_top1 - p_top2


class CoherenceSteererExt(CoherenceSteerer):
    """Extended steerer that also captures logit entropy + chosen-token logprob."""

    def generate_with_entropy(self, prompt, max_tokens=50):
        """Generate with trajectory capture: C_delta + entropy + logprobs."""
        inner = self._inner
        if inner is None:
            raise ValueError("Inner model not accessible")

        tokens = inner.to_tokens(prompt)
        generated_tokens = []
        c_delta_trajectory = []
        c_layer_trajectory = []
        entropy_trajectory = []
        logprob_trajectory = []
        top2_margin_trajectory = []
        interventions = []
        monitor_time = 0.0

        t_start = time.time()

        for step in range(max_tokens):
            hidden_states = {}
            capture_hooks = self._build_capture_hooks(hidden_states)

            t_mon = time.time()
            with torch.no_grad():
                logits = inner.run_with_hooks(tokens, fwd_hooks=capture_hooks)

            # C_delta
            c_delta, layer_c = self._fast_c_delta(hidden_states)
            c_delta_trajectory.append(c_delta)
            c_layer_trajectory.append({k: v for k, v in layer_c.items() if v is not None})

            # Entropy + top token stats
            logit_vec = logits[0, -1].float()
            logp = F.log_softmax(logit_vec, dim=-1)
            p = logp.exp()
            entropy = float(-(p * logp).sum().item())
            top2 = p.topk(2)
            top1_p = float(top2.values[0].item())
            top2_p = float(top2.values[1].item())
            next_token_id = int(top2.indices[0].item())
            chosen_logprob = float(logp[next_token_id].item())

            entropy_trajectory.append(entropy)
            logprob_trajectory.append(chosen_logprob)
            top2_margin_trajectory.append(top1_p - top2_p)

            monitor_time += time.time() - t_mon

            # Decode token
            next_token_str = inner.tokenizer.decode([next_token_id])
            if next_token_id == inner.tokenizer.eos_token_id:
                break
            generated_tokens.append(next_token_str)
            next_token_tensor = torch.tensor([[next_token_id]], device=tokens.device)
            tokens = torch.cat([tokens, next_token_tensor], dim=1)

        gen_time = time.time() - t_start
        total = len(generated_tokens)

        return GenerationResultExt(
            prompt=prompt,
            text="".join(generated_tokens),
            tokens=generated_tokens,
            c_delta_trajectory=c_delta_trajectory,
            c_per_layer_trajectory=c_layer_trajectory,
            interventions=interventions,
            intervention_count=0,
            total_tokens=total,
            generation_time_s=gen_time,
            monitor_overhead_s=monitor_time,
            overhead_pct=(monitor_time / gen_time * 100) if gen_time > 0 else 0.0,
            entropy_trajectory=entropy_trajectory,
            logprob_trajectory=logprob_trajectory,
            top2_margin_trajectory=top2_margin_trajectory,
        )


if __name__ == "__main__":
    # Smoke test
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-2-2b-it"
    s = CoherenceSteererExt(model_name=model_name)
    r = s.generate_with_entropy("Q: What is the capital of France?\nA:", max_tokens=10)
    print(f"Text: {r.text!r}")
    print(f"Tokens: {r.total_tokens}")
    print(f"C_delta[0..2]: {r.c_delta_trajectory[:3]}")
    print(f"Entropy[0..2]: {r.entropy_trajectory[:3]}")
    print(f"Logprob[0..2]: {r.logprob_trajectory[:3]}")
    print(f"Top2 margin[0..2]: {r.top2_margin_trajectory[:3]}")
