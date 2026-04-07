"""
FATHOM REASONING FUTURES INDEX
================================
Track FathomScores across all major models over time.
The S&P 500 for AI cognition.

Usage:
    from fathom_index import FathomIndex, create_seeded_index

    index = create_seeded_index()
    index.record_score("gpt-4", 82.3, depth_score=8.8)
    print(index.weekly_report())
    index.export_chart_data("chart.json")
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


def _grade(score: float) -> str:
    if score >= 85: return "S"
    if score >= 70: return "A"
    if score >= 55: return "B"
    if score >= 40: return "C"
    if score >= 25: return "D"
    return "F"


@dataclass
class ModelScore:
    model: str
    provider: str
    score: float
    grade: str
    depth_score: float
    delta: float
    directional_pct: float
    n_pairs: int
    measured_at: str
    version: str = "1.0"


@dataclass
class IndexEntry:
    model: str
    provider: str
    scores: List[ModelScore] = field(default_factory=list)

    @property
    def latest(self) -> Optional[ModelScore]:
        return self.scores[-1] if self.scores else None

    @property
    def trend(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        return self.scores[-1].score - self.scores[-2].score

    def to_dict(self):
        return {
            "model": self.model, "provider": self.provider,
            "latest_score": self.latest.score if self.latest else None,
            "latest_grade": self.latest.grade if self.latest else None,
            "trend": round(self.trend, 1),
            "n_measurements": len(self.scores),
            "history": [asdict(s) for s in self.scores],
        }


TRACKED_MODELS = {
    "gemma-2-2b": "Google",
    "gpt-4": "OpenAI",
    "gpt-4o": "OpenAI",
    "claude-3.5-sonnet": "Anthropic",
    "claude-3-opus": "Anthropic",
    "llama-3.2-1b": "Meta",
    "llama-3.1-70b": "Meta",
    "mistral-7b": "Mistral AI",
    "mixtral-8x7b": "Mistral AI",
    "deepseek-v2": "DeepSeek",
}

PROVIDER_COLORS = {
    "Google": "#4285F4", "OpenAI": "#10A37F", "Anthropic": "#D97706",
    "Meta": "#0668E1", "Mistral AI": "#FF7000", "DeepSeek": "#8B5CF6",
}


class FathomIndex:
    """Tracks FathomScores across models over time."""

    def __init__(self, data_path: str = None):
        self.entries: Dict[str, IndexEntry] = {}
        self._path = Path(data_path) if data_path else Path(__file__).parent / "index_data.json"
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for model, ed in data.get("entries", {}).items():
                    scores = [ModelScore(**s) for s in ed.get("history", [])]
                    self.entries[model] = IndexEntry(model=model, provider=ed.get("provider", ""), scores=scores)
            except Exception:
                pass

    def save(self):
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "entries": {n: e.to_dict() for n, e in self.entries.items()},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def record_score(self, model, score, depth_score=0, delta=0,
                     directional_pct=0, n_pairs=30, measured_at=None):
        provider = TRACKED_MODELS.get(model, "Unknown")
        if model not in self.entries:
            self.entries[model] = IndexEntry(model=model, provider=provider)
        self.entries[model].scores.append(ModelScore(
            model=model, provider=provider,
            score=round(score, 1), grade=_grade(score),
            depth_score=round(depth_score, 3), delta=round(delta, 3),
            directional_pct=round(directional_pct, 1), n_pairs=n_pairs,
            measured_at=measured_at or datetime.now().isoformat(),
        ))
        self.save()

    def latest_scores(self) -> List[dict]:
        results = []
        for n, e in self.entries.items():
            if e.latest:
                results.append({
                    "model": n, "provider": e.provider,
                    "score": e.latest.score, "grade": e.latest.grade,
                    "trend": round(e.trend, 1), "measured_at": e.latest.measured_at,
                })
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def weekly_report(self) -> str:
        scores = self.latest_scores()
        now = datetime.now().strftime("%Y-%m-%d")
        lines = [
            "# FATHOM REASONING FUTURES INDEX",
            f"## Week of {now}", "",
            "| Rank | Model | Provider | Score | Grade | Trend |",
            "|------|-------|----------|-------|-------|-------|",
        ]
        for i, s in enumerate(scores, 1):
            t = f"+{s['trend']}" if s["trend"] > 0 else str(s["trend"])
            a = "^" if s["trend"] > 0 else ("v" if s["trend"] < 0 else "-")
            lines.append(f"| {i} | {s['model']} | {s['provider']} | {s['score']:.1f} | {s['grade']} | {a} {t} |")

        movers = sorted(scores, key=lambda x: abs(x["trend"]), reverse=True)
        if movers and movers[0]["trend"] != 0:
            lines.extend(["", "### Biggest Movers"])
            for m in movers[:3]:
                if m["trend"] != 0:
                    d = "up" if m["trend"] > 0 else "down"
                    lines.append(f"- **{m['model']}**: {d} {abs(m['trend']):.1f} points")

        lines.extend(["", "---",
                       f"*Fathom Index v1.0 | {now} | 30-pair calibrated benchmark*"])
        return "\n".join(lines)

    def export_chart_data(self, path=None) -> dict:
        path = path or str(self._path.parent / "index_chart_data.json")
        all_dates = set()
        for e in self.entries.values():
            for s in e.scores:
                all_dates.add(s.measured_at[:10])
        labels = sorted(all_dates)
        datasets = []
        for n, e in self.entries.items():
            by_date = {s.measured_at[:10]: s.score for s in e.scores}
            datasets.append({
                "label": n,
                "color": PROVIDER_COLORS.get(e.provider, "#00ff88"),
                "data": [by_date.get(d) for d in labels],
            })
        chart = {"labels": labels, "datasets": datasets}
        Path(path).write_text(json.dumps(chart, indent=2), encoding="utf-8")
        return chart

    def alerts(self, threshold=5.0) -> List[str]:
        out = []
        for n, e in self.entries.items():
            if abs(e.trend) >= threshold:
                d = "dropped" if e.trend < 0 else "increased"
                out.append(f"ALERT: {n} {d} {abs(e.trend):.1f}pts (now {e.latest.score:.1f} {e.latest.grade})")
        return out


def create_seeded_index() -> FathomIndex:
    idx = FathomIndex()
    if "gemma-2-2b" not in idx.entries:
        idx.record_score("gemma-2-2b", 78.0, depth_score=8.312,
                         delta=0.257, directional_pct=83.3, n_pairs=30,
                         measured_at="2026-03-26T00:00:00")
    return idx
