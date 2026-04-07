"""
TEXT FEATURES — Pure-Python Feature Extractor for Depth Probe
==============================================================

Extracts 13 linguistic features from raw text that correlate with
reasoning depth. No GPU, no heavy NLP libraries. Just regex and word lists.

The key insight: surface/recall prompts use simple descriptive language
("the ground shakes", "your body burns calories") while insight/reasoning
prompts use causal/mechanistic language ("exceeds friction", "triggers
a cascade", "applies selective pressure").

Usage:
    from text_features import extract_features, features_to_vector

    f = extract_features("Fever triggers prostaglandin synthesis which signals the hypothalamus to")
    print(f.causal_connective_density)
    vec = features_to_vector(f)  # np.ndarray for ML
"""

import re
import math
from dataclasses import dataclass, asdict, fields
from typing import List

import numpy as np


# ── Word Lists ───────────────────────────────────────────────────────────────

CAUSAL_CONNECTIVES = frozenset([
    "because", "therefore", "thus", "hence", "consequently",
    "since", "so", "accordingly", "whereby",
])

CAUSAL_PHRASES = [
    "leads to", "results in", "causes", "due to", "as a result",
    "in order to", "so that", "gives rise to", "brings about",
    "stems from", "arises from", "owing to",
    "when.*then",  # regex pattern
]

MECHANISTIC_VERBS = frozenset([
    "triggers", "activates", "inhibits", "converts", "binds",
    "depletes", "disrupts", "amplifies", "regulates", "modulates",
    "catalyzes", "synthesizes", "degrades", "phosphorylates",
    "transduces", "propagates", "attenuates", "mediates",
    "metabolizes", "oxidizes", "reduces", "stimulates",
    "suppresses", "encodes", "replicates", "transcribes",
    "secretes", "absorbs", "transmits", "radiates",
    "exceeds", "saturates", "dissipates", "accumulates",
])

EXPLANATION_MARKERS = frozenset([
    "because", "since", "therefore", "thus", "hence",
    "consequently", "so", "accordingly", "explains",
    "mechanism", "process", "pathway", "cascade",
])

SUBORDINATING_CONJUNCTIONS = frozenset([
    "because", "since", "although", "though", "while",
    "when", "whenever", "where", "wherever", "if",
    "unless", "until", "after", "before", "that",
    "which", "who", "whom", "whose",
])

ABSTRACT_WORDS = frozenset([
    "process", "system", "mechanism", "function", "response",
    "pressure", "energy", "force", "rate", "level",
    "pattern", "structure", "distribution", "cycle", "phase",
    "signal", "feedback", "threshold", "gradient", "equilibrium",
    "resistance", "capacity", "potential", "frequency", "amplitude",
    "density", "concentration", "flux", "entropy", "dynamics",
])

CONCRETE_WORDS = frozenset([
    "water", "ground", "hand", "eye", "rock", "tree",
    "blood", "skin", "bone", "food", "fire", "air",
    "sun", "moon", "rain", "wind", "ice", "sand",
    "leaf", "seed", "root", "stone", "metal", "glass",
    "body", "face", "arm", "leg", "head", "heart",
])

# Common words excluded from technical vocabulary detection
COMMON_3PLUS_SYLLABLE = frozenset([
    "everything", "beautiful", "remember", "understand", "anything",
    "together", "important", "different", "following", "beginning",
    "another", "yesterday", "tomorrow", "already", "continue",
    "however", "suddenly", "actually", "probably", "certainly",
    "usually", "everyone", "somebody", "sometimes", "everywhere",
    "carefully", "seriously", "naturally", "obviously", "generally",
    "especially", "apparently", "unfortunately", "immediately",
    "completely", "absolutely", "definitely", "basically",
    "literally", "several", "interested", "considering",
])


# ── Feature Dataclass ────────────────────────────────────────────────────────

@dataclass
class TextFeatures:
    """13 linguistic features extracted from raw text."""
    token_count: int
    unique_token_ratio: float
    avg_word_length: float
    syllable_complexity: float
    causal_connective_density: float
    technical_vocab_density: float
    mechanistic_verb_density: float
    explanation_marker_density: float
    embedding_depth: float
    named_entity_density: float
    abstract_concrete_ratio: float
    information_density: float
    sentence_complexity: float

    def to_dict(self):
        return asdict(self)


# ── Helper Functions ─────────────────────────────────────────────────────────

def _count_syllables(word: str) -> int:
    """Estimate syllable count using vowel cluster heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    # Count vowel groups
    vowel_groups = re.findall(r'[aeiouy]+', word)
    count = len(vowel_groups)
    # Adjust for silent e
    if word.endswith('e') and count > 1:
        count -= 1
    # Adjust for -le ending
    if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
        count += 1
    return max(1, count)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def _sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sents = re.split(r'[.!?]+', text)
    return [s.strip() for s in sents if s.strip()]


# ── Main Extraction ──────────────────────────────────────────────────────────

def extract_features(text: str) -> TextFeatures:
    """Extract all 13 features from raw text."""
    tokens = _tokenize(text)
    n_tokens = len(tokens) if tokens else 1  # avoid division by zero

    # 1. Token count
    token_count = len(tokens)

    # 2. Unique token ratio (type-token ratio)
    unique_token_ratio = len(set(tokens)) / n_tokens

    # 3. Average word length
    avg_word_length = sum(len(t) for t in tokens) / n_tokens if tokens else 0

    # 4. Syllable complexity (mean syllables per word)
    syllable_complexity = sum(_count_syllables(t) for t in tokens) / n_tokens if tokens else 0

    # 5. Causal connective density
    text_lower = text.lower()
    causal_count = sum(1 for t in tokens if t in CAUSAL_CONNECTIVES)
    # Also check multi-word phrases
    for phrase in CAUSAL_PHRASES:
        if ".*" in phrase:
            causal_count += len(re.findall(phrase, text_lower))
        else:
            causal_count += text_lower.count(phrase)
    causal_connective_density = causal_count / n_tokens

    # 6. Technical vocabulary density
    # Words with 3+ syllables that aren't common
    tech_count = 0
    for t in tokens:
        if _count_syllables(t) >= 3 and t not in COMMON_3PLUS_SYLLABLE:
            tech_count += 1
    technical_vocab_density = tech_count / n_tokens

    # 7. Mechanistic verb density
    mech_count = sum(1 for t in tokens if t in MECHANISTIC_VERBS)
    mechanistic_verb_density = mech_count / n_tokens

    # 8. Explanation marker density
    expl_count = sum(1 for t in tokens if t in EXPLANATION_MARKERS)
    explanation_marker_density = expl_count / n_tokens

    # 9. Embedding depth (subordinating conjunction density as clause nesting proxy)
    subord_count = sum(1 for t in tokens if t in SUBORDINATING_CONJUNCTIONS)
    embedding_depth = subord_count / n_tokens

    # 10. Named entity density (capitalized non-sentence-start words)
    words_raw = text.split()
    ne_count = 0
    for i, w in enumerate(words_raw):
        if i == 0:
            continue
        # Check if previous char was sentence-ending
        prev = words_raw[i - 1] if i > 0 else ""
        if prev and prev[-1] in '.!?':
            continue
        if w and w[0].isupper() and w.isalpha():
            ne_count += 1
    named_entity_density = ne_count / n_tokens

    # 11. Abstract vs concrete ratio
    abstract_count = sum(1 for t in tokens if t in ABSTRACT_WORDS)
    concrete_count = sum(1 for t in tokens if t in CONCRETE_WORDS)
    if concrete_count > 0:
        abstract_concrete_ratio = abstract_count / concrete_count
    elif abstract_count > 0:
        abstract_concrete_ratio = float(abstract_count)  # all abstract
    else:
        abstract_concrete_ratio = 0.0

    # 12. Information density (hapax legomenon ratio)
    from collections import Counter
    freq = Counter(tokens)
    hapax = sum(1 for count in freq.values() if count == 1)
    information_density = hapax / n_tokens

    # 13. Sentence complexity
    sents = _sentences(text)
    if sents:
        # Commas + semicolons + subordinating conjunctions per sentence
        punct_count = text.count(',') + text.count(';')
        sentence_complexity = (punct_count + subord_count) / len(sents)
    else:
        sentence_complexity = 0.0

    return TextFeatures(
        token_count=token_count,
        unique_token_ratio=round(unique_token_ratio, 4),
        avg_word_length=round(avg_word_length, 4),
        syllable_complexity=round(syllable_complexity, 4),
        causal_connective_density=round(causal_connective_density, 4),
        technical_vocab_density=round(technical_vocab_density, 4),
        mechanistic_verb_density=round(mechanistic_verb_density, 4),
        explanation_marker_density=round(explanation_marker_density, 4),
        embedding_depth=round(embedding_depth, 4),
        named_entity_density=round(named_entity_density, 4),
        abstract_concrete_ratio=round(abstract_concrete_ratio, 4),
        information_density=round(information_density, 4),
        sentence_complexity=round(sentence_complexity, 4),
    )


def features_to_vector(features: TextFeatures) -> np.ndarray:
    """Convert TextFeatures to numpy vector for ML."""
    return np.array([getattr(features, f.name) for f in fields(TextFeatures)], dtype=np.float64)


def feature_names() -> List[str]:
    """Return ordered list of feature names."""
    return [f.name for f in fields(TextFeatures)]


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    surface = "The patient has a high temperature and the skin feels warm to the"
    insight = "Fever triggers prostaglandin synthesis which signals the hypothalamus to"

    print("SURFACE PROMPT:")
    sf = extract_features(surface)
    for name in feature_names():
        print(f"  {name}: {getattr(sf, name)}")

    print("\nINSIGHT PROMPT:")
    inf = extract_features(insight)
    for name in feature_names():
        print(f"  {name}: {getattr(inf, name)}")

    print(f"\nFeature vector (surface): {features_to_vector(sf)}")
    print(f"Feature vector (insight): {features_to_vector(inf)}")
