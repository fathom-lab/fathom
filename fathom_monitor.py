"""
fathom_monitor.py — Fathom Monitor v1: Uncertainty Flag Layer
=============================================================

Input:  any LLM generation (via CoherenceSteerer)
Output: per-token C_delta trace + uncertainty flags where C_delta > threshold

Value: tells you exactly which tokens were generated with over-coherent
       (hallucinating) circuits — no correction needed. just transparency.

Usage:
    from fathom_monitor import FathomMonitor

    monitor = FathomMonitor()
    result = monitor.run("Who was the first person on the moon?")
    print(result.flagged_text)   # inline [C_delta=0.0117 — verify this]
    print(result.flags)          # list of FlagEvent objects

Patent pending: US 64/020,489 & 64/021,113
"""

from dataclasses import dataclass, field
from typing import List, Optional
from coherence_steerer import CoherenceSteerer, GenerationResult


@dataclass
class FlagEvent:
    token_idx: int
    token: str
    c_delta: float
    label: str  # "UNCERTAIN" | "HALLUCINATION_RISK"


@dataclass
class MonitorResult:
    prompt: str
    text: str
    flags: List[FlagEvent]
    flagged_text: str          # text with inline [C_delta=X.XXXX — verify this] markers
    flag_count: int
    flag_rate: float           # flags / total tokens
    max_c_delta: Optional[float]
    raw: GenerationResult


class FathomMonitor:
    """
    Lightweight flag layer on top of CoherenceSteerer.
    Runs monitored generation and annotates uncertain spans.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        threshold: float = 0.010,
        high_threshold: float = 0.020,  # above this = HALLUCINATION_RISK
        **steerer_kwargs
    ):
        self.threshold = threshold
        self.high_threshold = high_threshold
        self._steerer = CoherenceSteerer(
            model_name=model_name,
            c_delta_threshold=threshold,
            **steerer_kwargs
        )

    def run(self, prompt: str, max_tokens: int = 50) -> MonitorResult:
        """
        Generate and flag uncertain tokens.
        Returns MonitorResult with inline-annotated text and structured flags.
        """
        raw = self._steerer.generate_monitored(prompt, max_tokens=max_tokens)
        return self._build_result(raw)

    def _build_result(self, raw: GenerationResult) -> MonitorResult:
        flags: List[FlagEvent] = []
        flagged_parts: List[str] = []

        for i, (token, c_delta) in enumerate(zip(raw.tokens, raw.c_delta_trajectory)):
            if c_delta is None or c_delta <= self.threshold:
                flagged_parts.append(token)
                continue

            # classify severity
            if c_delta >= self.high_threshold:
                label = "HALLUCINATION_RISK"
                marker = f"[C_delta={c_delta:+.4f} — HIGH RISK]"
            else:
                label = "UNCERTAIN"
                marker = f"[C_delta={c_delta:+.4f} — verify this]"

            flags.append(FlagEvent(
                token_idx=i,
                token=token,
                c_delta=c_delta,
                label=label
            ))
            flagged_parts.append(f"{token}{marker}")

        flagged_text = "".join(flagged_parts)
        all_deltas = [c for c in raw.c_delta_trajectory if c is not None]

        return MonitorResult(
            prompt=raw.prompt,
            text=raw.text,
            flags=flags,
            flagged_text=flagged_text,
            flag_count=len(flags),
            flag_rate=len(flags) / max(raw.total_tokens, 1),
            max_c_delta=max(all_deltas) if all_deltas else None,
            raw=raw,
        )

    def scan(self, text: str, prompt: str = "", max_tokens: int = 50) -> MonitorResult:
        """
        Alias: score a generation from an external string prompt.
        Uses prompt as context, ignores 'text' arg (generation is internal).
        For API integration — call run() directly for full control.
        """
        return self.run(prompt or text, max_tokens=max_tokens)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The president of Australia is"
    print(f"\n{'='*60}")
    print("FATHOM MONITOR v1 — Uncertainty Flag Layer")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}\n")

    monitor = FathomMonitor()
    result = monitor.run(prompt, max_tokens=40)

    print(f"Output:\n  {result.text}\n")
    print(f"Flagged:\n  {result.flagged_text}\n")

    if result.flags:
        print(f"Flags ({result.flag_count} tokens flagged, {result.flag_rate:.0%} rate):")
        for f in result.flags:
            token_safe = f.token.encode('ascii', 'replace').decode()
            print(f"  [{f.token_idx:2d}] '{token_safe:15s}' {f.label}  C_delta={f.c_delta:+.4f}")
    else:
        print("No flags. Generation looks clean.")

    print(f"\nMax C_delta seen: {result.max_c_delta:+.4f}" if result.max_c_delta else "\nNo C_delta data.")
    print(f"Overhead: {result.raw.overhead_pct:.1f}%")
