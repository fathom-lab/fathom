#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLACK-BOX DEPTH SCORER — Universal Fathom Depth Estimation
============================================================

Estimates Fathom depth scores from text features alone. No model weights
needed. Works on any text, including outputs from black-box API models
(GPT-4, Claude, Gemini, etc.).

Usage:
    from blackbox_scorer import BlackBoxDepthScorer

    scorer = BlackBoxDepthScorer()
    result = scorer.score("Fever triggers prostaglandin synthesis...")
    print(result.predicted_depth, result.grade)

    # Compare surface vs insight
    cmp = scorer.compare(
        "The patient has a fever",
        "Fever triggers prostaglandin synthesis in the hypothalamus"
    )
    print(cmp["delta"], cmp["direction"])

    # Score any API model
    result = scorer.score_api_model(
        prompt="Explain how fever works",
        provider="openai",
        model="gpt-4o",
        api_key="sk-..."
    )

CLI:
    python blackbox_scorer.py --demo
"""

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

from text_features import extract_features, TextFeatures

BASE_DIR = Path(__file__).parent
DEFAULT_PROBE_PATH = BASE_DIR / "depth_probe.pkl"


# ── Grade thresholds (matching fathom_score.py) ─────────────────────────────

def _grade(score_0_100: float) -> str:
    """Map 0-100 score to letter grade."""
    if score_0_100 >= 85:
        return "S"
    elif score_0_100 >= 70:
        return "A"
    elif score_0_100 >= 55:
        return "B"
    elif score_0_100 >= 40:
        return "C"
    elif score_0_100 >= 25:
        return "D"
    else:
        return "F"


def _interpret(grade: str, predicted_depth: float) -> str:
    """Generate one-sentence interpretation."""
    interpretations = {
        "S": "Exceptionally deep reasoning — engages multi-layer causal mechanisms.",
        "A": "Strong reasoning depth — demonstrates mechanistic understanding.",
        "B": "Moderate reasoning — mixes description with some causal analysis.",
        "C": "Shallow-to-moderate — primarily descriptive with occasional depth.",
        "D": "Shallow processing — surface-level pattern matching.",
        "F": "Minimal reasoning depth — pure surface recall.",
    }
    return interpretations.get(grade, "Unknown depth profile.")


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class FathomEstimate:
    """Black-box depth estimation result."""
    predicted_depth: float          # 0-1 normalized (higher = deeper)
    predicted_depth_raw: float      # projected to Gemma-2-2B scale
    confidence: float               # 0-1 (higher = more in-distribution)
    grade: str                      # F/D/C/B/A/S
    score_0_100: float              # 0-100 composite
    interpretation: str
    calibration_source: str
    features: Dict                  # text features used
    warning: Optional[str] = None   # any caveats

    def to_dict(self):
        return asdict(self)


# ── Main Class ───────────────────────────────────────────────────────────────

class BlackBoxDepthScorer:
    """Universal depth estimation from text features.

    Uses a trained probe calibrated against white-box measurements
    on Gemma-2-2B and Llama-3.2-1B.
    """

    def __init__(self, probe_path: Path = DEFAULT_PROBE_PATH):
        self._probe = None
        self._probe_path = probe_path
        self._calibration_source = "gemma-2-2b + llama-3.2-1b"

    def _ensure_probe(self):
        """Lazy-load the trained probe."""
        if self._probe is not None:
            return

        from depth_probe import DepthProbe

        if not self._probe_path.exists():
            raise FileNotFoundError(
                f"Trained probe not found at {self._probe_path}. "
                "Run: python build_calibration_data.py && python depth_probe.py"
            )

        self._probe = DepthProbe(self._probe_path)

    def score(self, text: str) -> FathomEstimate:
        """Score a single text for reasoning depth.

        Args:
            text: Any text (prompt, response, article, etc.)

        Returns:
            FathomEstimate with predicted depth, confidence, and grade.
        """
        self._ensure_probe()

        predicted, confidence, features = self._probe.predict_with_confidence(text)

        # Clamp to valid range
        predicted = max(0.0, min(1.0, predicted))

        # Project to Gemma-2-2B raw scale for comparability
        # normalized = 1 - raw/26, so raw = (1 - normalized) * 26
        predicted_raw = (1.0 - predicted) * 26.0

        # Convert to 0-100 score
        score_100 = predicted * 100.0
        grade = _grade(score_100)
        interpretation = _interpret(grade, predicted)

        warning = None
        if confidence < 0.3:
            warning = "Low confidence — text features differ significantly from calibration data."

        return FathomEstimate(
            predicted_depth=round(predicted, 4),
            predicted_depth_raw=round(predicted_raw, 4),
            confidence=round(confidence, 3),
            grade=grade,
            score_0_100=round(score_100, 1),
            interpretation=interpretation,
            calibration_source=self._calibration_source,
            features=features.to_dict(),
            warning=warning,
        )

    def compare(self, text_a: str, text_b: str) -> Dict:
        """Compare two texts for relative depth.

        Returns dict with scores, delta, direction, and ratio.
        """
        a = self.score(text_a)
        b = self.score(text_b)

        delta = a.predicted_depth - b.predicted_depth
        ratio = a.predicted_depth / b.predicted_depth if b.predicted_depth > 0 else 0.0

        return {
            "text_a": {
                "depth": a.predicted_depth,
                "grade": a.grade,
                "confidence": a.confidence,
            },
            "text_b": {
                "depth": b.predicted_depth,
                "grade": b.grade,
                "confidence": b.confidence,
            },
            "delta": round(delta, 4),
            "ratio": round(ratio, 4),
            "direction": "a_deeper" if delta > 0 else "b_deeper",
        }

    def score_api_model(
        self,
        prompt: str,
        provider: str,
        model: str,
        api_key: str,
        system_prompt: str = "Continue the following text with one sentence.",
    ) -> FathomEstimate:
        """Call an API model, get its output, and score for depth.

        Args:
            prompt: The prompt to send to the model.
            provider: "openai", "anthropic", or "google".
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
            api_key: API key for the provider.
            system_prompt: System message for context.

        Returns:
            FathomEstimate of the model's response text.
        """
        response_text = self._call_api(prompt, provider, model, api_key, system_prompt)
        return self.score(response_text)

    def _call_api(
        self,
        prompt: str,
        provider: str,
        model: str,
        api_key: str,
        system_prompt: str,
    ) -> str:
        """Call an external API and return the response text."""
        provider = provider.lower()

        if provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai")
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            return resp.choices[0].message.content

        elif provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError("pip install anthropic")
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        elif provider == "google":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("pip install google-generativeai")
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            resp = model_obj.generate_content(f"{system_prompt}\n\n{prompt}")
            return resp.text

        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'google'.")


# ── CLI Demo ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fathom Black-Box Depth Scorer")
    parser.add_argument("--demo", action="store_true", help="Run demo predictions")
    parser.add_argument("--text", type=str, help="Score a specific text")
    parser.add_argument("--compare", nargs=2, metavar=("TEXT_A", "TEXT_B"), help="Compare two texts")
    args = parser.parse_args()

    scorer = BlackBoxDepthScorer()

    if args.text:
        result = scorer.score(args.text)
        print(f"Depth: {result.predicted_depth:.4f} ({result.grade})")
        print(f"Score: {result.score_0_100}/100")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Interpretation: {result.interpretation}")
        if result.warning:
            print(f"Warning: {result.warning}")
        return

    if args.compare:
        cmp = scorer.compare(args.compare[0], args.compare[1])
        print(f"Text A: depth={cmp['text_a']['depth']:.4f} ({cmp['text_a']['grade']})")
        print(f"Text B: depth={cmp['text_b']['depth']:.4f} ({cmp['text_b']['grade']})")
        print(f"Delta: {cmp['delta']:+.4f} | Direction: {cmp['direction']}")
        return

    if args.demo or True:
        print("=" * 60)
        print("  FATHOM BLACK-BOX SCORER — Demo")
        print("=" * 60)

        pairs = [
            {
                "label": "Healthcare",
                "surface": "The patient has a high temperature and the skin feels warm to the touch",
                "insight": "Fever triggers prostaglandin synthesis which signals the hypothalamus to raise body temperature",
            },
            {
                "label": "Finance",
                "surface": "The stock market fell sharply after the economic data came in worse than expected",
                "insight": "Yield curve inversions predict recessions because short-term borrowing costs exceed long-term returns forcing banks to contract credit",
            },
            {
                "label": "Education",
                "surface": "The plant sits in the window and its leaves are turning green in the sun",
                "insight": "Chlorophyll absorbs photons and transfers energy to reaction centers where water molecules are split releasing oxygen",
            },
        ]

        for pair in pairs:
            print(f"\n--- {pair['label']} ---")
            s = scorer.score(pair["surface"])
            i = scorer.score(pair["insight"])
            delta = s.predicted_depth - i.predicted_depth
            print(f"  Surface: depth={s.predicted_depth:.4f} grade={s.grade} conf={s.confidence:.2f}")
            print(f"  Insight: depth={i.predicted_depth:.4f} grade={i.grade} conf={i.confidence:.2f}")
            print(f"  Delta: {delta:+.4f} {'surface deeper' if delta > 0 else 'insight deeper'}")


if __name__ == "__main__":
    main()
