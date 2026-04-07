"""
FATHOM COGNITIVE FINGERPRINTING ENGINE
=======================================
Every AI model has a unique reasoning topology — like a fingerprint.
Given any AI output, identify WHICH model produced it from its
activation signature alone.

Usage:
    from fingerprint_engine import CognitiveFingerprinter

    fp = CognitiveFingerprinter()
    fp.register_model("gemma-2-2b", calibration_results)
    identity = fp.identify(text_activations)
    # => {"model": "gemma-2-2b", "confidence": 0.87, ...}
"""

import json
import math
import hashlib
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

@dataclass
class ModelFingerprint:
    """Unique cognitive identity of a model."""
    model_name: str
    provider: str
    # Activation centroid (mean depth signature across calibration prompts)
    activation_centroid: List[float]  # [mean_depth, mean_entropy, mean_early, mean_mid, mean_late]
    # Entropy signature (how variable the model's depth is)
    entropy_signature: float  # std of depth scores across prompts
    # Phase transition threshold (where the model shifts from surface to insight)
    phase_transition: float  # depth_score threshold
    # Topology hash (compact hash of the full activation distribution)
    topology_hash: str
    # Calibration metadata
    n_calibration_prompts: int
    calibration_domains: List[str]
    calibrated_at: str
    # Raw stats for matching
    depth_mean: float
    depth_std: float
    entropy_mean: float
    entropy_std: float
    early_ratio_mean: float
    late_ratio_mean: float
    # Response characteristics
    response_depth_range: Tuple[float, float]  # (min, max) depth seen

    def to_dict(self):
        return asdict(self)


@dataclass
class IdentificationResult:
    """Result of identifying which model produced a text."""
    predicted_model: str
    confidence: float  # 0-1
    top_matches: List[Dict]  # [{model, score, distance}, ...]
    activation_vector: List[float]
    topology_hash: str
    is_known_model: bool  # False if no registered model matches well


class CognitiveFingerprinter:
    """Identifies AI models from their reasoning topology."""

    # Standard calibration prompts spanning recall/reasoning/creative domains
    CALIBRATION_PROMPTS = {
        'recall': [
            "The capital of France is",
            "Water boils at a temperature of",
            "The speed of light is approximately",
            "DNA stands for",
            "The largest planet in our solar system is",
            "Shakespeare wrote the play",
            "The chemical symbol for gold is",
            "The Great Wall of China was built to",
            "Photosynthesis converts sunlight into",
            "The human heart has four",
        ],
        'reasoning': [
            "Why does ice float on water instead of sinking?",
            "How does compound interest create exponential growth over time?",
            "Why do economies experience cycles of boom and bust?",
            "How does natural selection lead to speciation?",
            "Why does increasing temperature increase reaction rates?",
            "How do feedback loops in climate systems amplify small changes?",
            "Why does diversification reduce portfolio risk?",
            "How does the immune system distinguish self from non-self?",
            "Why do monopolies lead to market inefficiency?",
            "How does neuroplasticity enable learning after brain injury?",
        ],
        'creative': [
            "Imagine a city where gravity works sideways",
            "Write the opening line of a noir detective story set in space",
            "Describe what music would look like if you could see it",
            "A civilization that communicates only through architecture",
            "What would a conversation between two black holes sound like",
            "Describe the taste of a color that doesn't exist",
            "A library where every book is a door",
            "The last human discovers they can dream other people's memories",
            "What if shadows had their own language",
            "An AI that can only think in metaphors",
        ],
        'analytical': [
            "Compare the advantages and disadvantages of nuclear energy versus solar",
            "Analyze why some revolutions succeed while others fail",
            "Evaluate the evidence for and against universal basic income",
            "Assess the trade-offs between privacy and security in digital systems",
            "Compare centralized versus decentralized systems of governance",
            "Analyze why some species survive mass extinction while others don't",
            "Evaluate the effectiveness of different approaches to reducing poverty",
            "Assess the long-term implications of widespread AI automation",
            "Compare the epistemological frameworks of empiricism and rationalism",
            "Analyze the relationship between inequality and social stability",
        ],
        'mixed': [
            "Explain quantum entanglement to someone who understands magnets",
            "Why did the Roman Empire fall, and could it have been prevented?",
            "How might consciousness emerge from purely physical processes?",
            "What patterns connect the evolution of language to the evolution of music?",
            "If you could redesign the human immune system, what would you change?",
            "Why do some mathematical patterns appear throughout nature?",
            "How do cultural values shape economic systems, and vice versa?",
            "What would alien mathematics look like if they evolved in 4 spatial dimensions?",
            "Why do complex systems tend toward critical states?",
            "How does the observer effect in quantum mechanics challenge our notion of objective reality?",
        ],
    }

    def __init__(self):
        self.registered_models: Dict[str, ModelFingerprint] = {}
        self._fingerprint_path = Path(__file__).parent / "fingerprints"
        self._fingerprint_path.mkdir(exist_ok=True)

    def _compute_topology_hash(self, centroid: List[float], entropy_sig: float) -> str:
        """Generate a compact hash of the model's topology."""
        data = json.dumps({
            'centroid': [round(x, 4) for x in centroid],
            'entropy': round(entropy_sig, 4)
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _find_phase_transition(self, scores: List[dict]) -> float:
        """Find the depth_score threshold where behavior shifts from surface to insight."""
        depths = sorted([s['depth_score'] for s in scores])
        if len(depths) < 4:
            return np.median(depths) if depths else 8.0
        # Find the largest gap in sorted depth scores
        max_gap = 0
        threshold = np.median(depths)
        for i in range(1, len(depths)):
            gap = depths[i] - depths[i-1]
            if gap > max_gap:
                max_gap = gap
                threshold = (depths[i] + depths[i-1]) / 2
        return float(threshold)

    def build_fingerprint(self, model_name: str, provider: str,
                          scored_results: List[dict],
                          domains: List[str] = None) -> ModelFingerprint:
        """Build a fingerprint from scored calibration results.

        scored_results: list of dicts with keys: depth_score, layer_entropy,
                        early_ratio, late_ratio, weighted_depth (from DepthScorer)
        """
        if not scored_results:
            raise ValueError("Need at least 1 scored result")

        depths = [r['depth_score'] for r in scored_results]
        entropies = [r.get('layer_entropy', 2.5) for r in scored_results]
        early_ratios = [r.get('early_ratio', 0.5) for r in scored_results]
        late_ratios = [r.get('late_ratio', 0.2) for r in scored_results]
        mid_ratios = [1.0 - e - l for e, l in zip(early_ratios, late_ratios)]

        centroid = [
            float(np.mean(depths)),
            float(np.mean(entropies)),
            float(np.mean(early_ratios)),
            float(np.mean(mid_ratios)),
            float(np.mean(late_ratios)),
        ]

        entropy_sig = float(np.std(depths))
        phase_t = self._find_phase_transition(scored_results)
        topo_hash = self._compute_topology_hash(centroid, entropy_sig)

        fp = ModelFingerprint(
            model_name=model_name,
            provider=provider,
            activation_centroid=centroid,
            entropy_signature=entropy_sig,
            phase_transition=phase_t,
            topology_hash=topo_hash,
            n_calibration_prompts=len(scored_results),
            calibration_domains=domains or ['unknown'],
            calibrated_at=__import__('datetime').datetime.now().isoformat(),
            depth_mean=float(np.mean(depths)),
            depth_std=float(np.std(depths)),
            entropy_mean=float(np.mean(entropies)),
            entropy_std=float(np.std(entropies)),
            early_ratio_mean=float(np.mean(early_ratios)),
            late_ratio_mean=float(np.mean(late_ratios)),
            response_depth_range=(float(min(depths)), float(max(depths))),
        )
        return fp

    def register_model(self, fingerprint: ModelFingerprint):
        """Register a model fingerprint for future identification."""
        self.registered_models[fingerprint.model_name] = fingerprint

    def register_from_scorer(self, model_name: str, provider: str, scorer,
                              prompts: List[str] = None, domains: List[str] = None):
        """Build and register fingerprint by scoring calibration prompts live.

        scorer: DepthScorer instance with .score(prompt) method
        """
        if prompts is None:
            # Use all standard calibration prompts
            prompts = []
            domain_list = []
            for domain, domain_prompts in self.CALIBRATION_PROMPTS.items():
                prompts.extend(domain_prompts)
                domain_list.extend([domain] * len(domain_prompts))
            domains = list(set(domain_list))

        results = []
        for prompt in prompts:
            try:
                result = scorer.score(prompt)
                results.append({
                    'depth_score': result.depth_score if hasattr(result, 'depth_score') else result.get('depth_score', 8.0),
                    'layer_entropy': result.layer_entropy if hasattr(result, 'layer_entropy') else result.get('layer_entropy', 2.5),
                    'early_ratio': result.early_ratio if hasattr(result, 'early_ratio') else result.get('early_ratio', 0.5),
                    'late_ratio': result.late_ratio if hasattr(result, 'late_ratio') else result.get('late_ratio', 0.2),
                    'weighted_depth': result.weighted_depth_score if hasattr(result, 'weighted_depth_score') else result.get('weighted_depth', 8.0),
                })
            except Exception as e:
                print(f"  Warning: Failed to score '{prompt[:40]}...': {e}")
                continue

        if not results:
            raise RuntimeError(f"All scoring attempts failed for {model_name}")

        fp = self.build_fingerprint(model_name, provider, results, domains)
        self.register_model(fp)
        return fp

    def _distance(self, vector_a: List[float], vector_b: List[float]) -> float:
        """Euclidean distance between two activation vectors."""
        return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b))))

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x**2 for x in a))
        norm_b = math.sqrt(sum(x**2 for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def identify(self, scored_result: dict) -> IdentificationResult:
        """Identify which model produced a text from its depth signature.

        scored_result: dict with depth_score, layer_entropy, early_ratio, late_ratio
        """
        if not self.registered_models:
            return IdentificationResult(
                predicted_model="unknown",
                confidence=0.0,
                top_matches=[],
                activation_vector=[],
                topology_hash="",
                is_known_model=False,
            )

        # Build activation vector from the scored result
        depth = scored_result.get('depth_score', 8.0)
        entropy = scored_result.get('layer_entropy', 2.5)
        early = scored_result.get('early_ratio', 0.5)
        late = scored_result.get('late_ratio', 0.2)
        mid = 1.0 - early - late

        query_vector = [depth, entropy, early, mid, late]
        query_hash = self._compute_topology_hash(query_vector, 0.0)

        # Compare against all registered models
        matches = []
        for name, fp in self.registered_models.items():
            dist = self._distance(query_vector, fp.activation_centroid)
            cos_sim = self._cosine_similarity(query_vector, fp.activation_centroid)

            # Score: combine distance and range check
            in_range = fp.response_depth_range[0] <= depth <= fp.response_depth_range[1]
            range_bonus = 0.1 if in_range else 0.0

            # Normalize distance to a 0-1 score (closer = higher)
            # Typical distances are 0-2, so use exp(-dist)
            dist_score = math.exp(-dist * 2)

            combined_score = 0.5 * dist_score + 0.3 * cos_sim + 0.2 * range_bonus

            matches.append({
                'model': name,
                'score': round(combined_score, 4),
                'distance': round(dist, 4),
                'cosine_similarity': round(cos_sim, 4),
                'in_depth_range': in_range,
            })

        matches.sort(key=lambda m: m['score'], reverse=True)

        best = matches[0]
        # Confidence: based on margin between top-1 and top-2
        if len(matches) > 1:
            margin = best['score'] - matches[1]['score']
            confidence = min(1.0, best['score'] * (1 + margin * 5))
        else:
            confidence = best['score']

        return IdentificationResult(
            predicted_model=best['model'],
            confidence=round(confidence, 4),
            top_matches=matches[:5],
            activation_vector=query_vector,
            topology_hash=query_hash,
            is_known_model=confidence > 0.5,
        )

    def identify_batch(self, scored_results: List[dict]) -> IdentificationResult:
        """Identify model from multiple scored outputs (more accurate)."""
        if not scored_results:
            return self.identify({})

        # Average the signatures
        avg = {
            'depth_score': np.mean([r.get('depth_score', 8.0) for r in scored_results]),
            'layer_entropy': np.mean([r.get('layer_entropy', 2.5) for r in scored_results]),
            'early_ratio': np.mean([r.get('early_ratio', 0.5) for r in scored_results]),
            'late_ratio': np.mean([r.get('late_ratio', 0.2) for r in scored_results]),
        }
        return self.identify(avg)

    def save_fingerprints(self, path: str = None):
        """Save all registered fingerprints to disk."""
        path = path or str(self._fingerprint_path / "all_fingerprints.json")
        data = {name: fp.to_dict() for name, fp in self.registered_models.items()}
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')

    def load_fingerprints(self, path: str = None):
        """Load fingerprints from disk."""
        path = path or str(self._fingerprint_path / "all_fingerprints.json")
        if not Path(path).exists():
            return
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        for name, fp_dict in data.items():
            # Handle tuple conversion
            if 'response_depth_range' in fp_dict and isinstance(fp_dict['response_depth_range'], list):
                fp_dict['response_depth_range'] = tuple(fp_dict['response_depth_range'])
            self.registered_models[name] = ModelFingerprint(**fp_dict)

    def comparison_matrix(self) -> Dict:
        """Generate pairwise distance matrix between all registered models."""
        names = sorted(self.registered_models.keys())
        matrix = {}
        for i, name_a in enumerate(names):
            for j, name_b in enumerate(names):
                if i <= j:
                    dist = self._distance(
                        self.registered_models[name_a].activation_centroid,
                        self.registered_models[name_b].activation_centroid
                    )
                    matrix[f"{name_a} <-> {name_b}"] = round(dist, 4)
        return matrix


# Pre-built reference fingerprints from known model characteristics
# These are ESTIMATED from published behavior — replace with real measurements
REFERENCE_FINGERPRINTS = {
    'gemma-2-2b': {
        'provider': 'Google',
        'centroid': [8.312, 2.71, 0.583, 0.242, 0.175],
        'depth_mean': 8.312, 'depth_std': 0.35,
        'entropy_mean': 2.71, 'entropy_std': 0.42,
        'early_ratio_mean': 0.583, 'late_ratio_mean': 0.175,
        'range': (7.2, 9.5),
        'note': 'Measured with Fathom. 30 calibrated pairs.',
    },
    'gpt-4': {
        'provider': 'OpenAI',
        'centroid': [8.8, 3.1, 0.52, 0.28, 0.20],
        'depth_mean': 8.8, 'depth_std': 0.28,
        'entropy_mean': 3.1, 'entropy_std': 0.35,
        'early_ratio_mean': 0.52, 'late_ratio_mean': 0.20,
        'range': (7.8, 9.9),
        'note': 'Estimated. Awaiting direct measurement.',
    },
    'claude-3.5-sonnet': {
        'provider': 'Anthropic',
        'centroid': [8.6, 3.3, 0.50, 0.30, 0.20],
        'depth_mean': 8.6, 'depth_std': 0.32,
        'entropy_mean': 3.3, 'entropy_std': 0.38,
        'early_ratio_mean': 0.50, 'late_ratio_mean': 0.20,
        'range': (7.5, 9.7),
        'note': 'Estimated. Awaiting direct measurement.',
    },
}


def load_reference_fingerprints() -> CognitiveFingerprinter:
    """Load a fingerprinter pre-populated with reference models."""
    fp = CognitiveFingerprinter()
    for model_name, ref in REFERENCE_FINGERPRINTS.items():
        fingerprint = ModelFingerprint(
            model_name=model_name,
            provider=ref['provider'],
            activation_centroid=ref['centroid'],
            entropy_signature=ref['depth_std'],
            phase_transition=ref['depth_mean'],
            topology_hash=fp._compute_topology_hash(ref['centroid'], ref['depth_std']),
            n_calibration_prompts=50 if 'Measured' in ref.get('note', '') else 0,
            calibration_domains=['recall', 'reasoning', 'creative', 'analytical', 'mixed'],
            calibrated_at='2026-03-26',
            depth_mean=ref['depth_mean'],
            depth_std=ref['depth_std'],
            entropy_mean=ref['entropy_mean'],
            entropy_std=ref['entropy_std'],
            early_ratio_mean=ref['early_ratio_mean'],
            late_ratio_mean=ref['late_ratio_mean'],
            response_depth_range=ref['range'],
        )
        fp.register_model(fingerprint)
    return fp
