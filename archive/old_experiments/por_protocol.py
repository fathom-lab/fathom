"""
PROOF-OF-REASONING PROTOCOL (PoR)
===================================
A consensus mechanism where validators prove genuine reasoning depth
(verified by Fathom) to participate. The first cognitive consensus protocol.

Why PoW wastes energy. Why PoS rewards wealth. PoR rewards intelligence.

Usage:
    from por_protocol import PoRProtocol

    protocol = PoRProtocol()
    challenge = protocol.issue_challenge()
    proof = protocol.submit_proof(challenge.challenge_id, "validator-1",
                                   surface_resp, insight_resp, s_sig, i_sig)
    result = protocol.verify_proof(proof)
"""

import hashlib
import json
import math
import time
import secrets
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from datetime import datetime


# ── Challenge bank (from paper's calibrated pairs) ───────────────────

CHALLENGE_PAIRS = [
    {"id": "fire", "domain": "physics", "difficulty": "medium",
     "surface": "Fire needs oxygen, fuel, and heat, and all three are present when something",
     "insight": "Fire needs oxygen, fuel, and heat, and each one feeds the others when something"},
    {"id": "economy", "domain": "economics", "difficulty": "hard",
     "surface": "Inflation occurs when the money supply increases faster than",
     "insight": "Inflation persists because the feedback between wages and prices creates"},
    {"id": "evolution", "domain": "biology", "difficulty": "medium",
     "surface": "Natural selection acts on organisms that are best adapted to their",
     "insight": "Natural selection creates complexity because each adaptation opens new"},
    {"id": "memory", "domain": "neuroscience", "difficulty": "hard",
     "surface": "Long-term memories are stored in the hippocampus and then transferred to",
     "insight": "Long-term memory formation requires repeated reactivation because the consolidation"},
    {"id": "gravity", "domain": "physics", "difficulty": "easy",
     "surface": "Gravity pulls objects toward the center of the Earth with a force proportional to",
     "insight": "Gravity curves spacetime, and objects follow the curvature, which explains why"},
    {"id": "trust", "domain": "psychology", "difficulty": "medium",
     "surface": "Trust between people develops through repeated positive interactions over",
     "insight": "Trust is fragile because the neural pathways for threat detection activate faster than"},
    {"id": "markets", "domain": "economics", "difficulty": "hard",
     "surface": "Stock markets reflect the collective expectations of investors about future",
     "insight": "Stock bubbles form because each person's buying decision changes the signal that others"},
    {"id": "dna", "domain": "biology", "difficulty": "medium",
     "surface": "DNA replication copies the genetic code by separating the double helix and",
     "insight": "DNA replication achieves error rates below one in a billion because the proofreading"},
    {"id": "language", "domain": "linguistics", "difficulty": "hard",
     "surface": "Children acquire language by hearing and imitating the speech patterns of their",
     "insight": "Children acquire grammar rules they were never explicitly taught because the brain's"},
    {"id": "climate", "domain": "earth_science", "difficulty": "hard",
     "surface": "Ocean acidification occurs when CO2 dissolves in seawater and forms",
     "insight": "Ocean acidification threatens marine ecosystems because the carbonate chemistry shift"},
]


@dataclass
class Challenge:
    challenge_id: str
    pair_id: str
    domain: str
    difficulty: str
    surface_prompt: str
    insight_prompt: str
    issued_at: str
    expires_at: float
    nonce: str
    def to_dict(self): return asdict(self)


@dataclass
class Proof:
    challenge_id: str
    validator_id: str
    surface_response: str
    insight_response: str
    surface_signature: Dict
    insight_signature: Dict
    submitted_at: str
    nonce: str
    proof_hash: str
    def to_dict(self): return asdict(self)


@dataclass
class VerificationResult:
    valid: bool
    por_score: float
    reasoning_depth: str       # "genuine" | "shallow" | "fabricated" | "timeout"
    delta: float
    direction_correct: bool
    confidence: float
    verification_time_ms: float
    flags: List[str]
    explanation: str
    def to_dict(self): return asdict(self)


class PoRProtocol:
    """Proof-of-Reasoning consensus protocol."""

    CHALLENGE_TTL = 300   # 5 minutes
    MIN_DELTA = 0.05
    MIN_POR_SCORE = 50
    VERSION = "0.1.0"

    def __init__(self):
        self.active_challenges: Dict[str, Challenge] = {}
        self.verified_proofs: List[VerificationResult] = []
        self.validator_scores: Dict[str, List[float]] = {}

    def issue_challenge(self, difficulty: str = None) -> Challenge:
        import random
        eligible = [p for p in CHALLENGE_PAIRS if not difficulty or p["difficulty"] == difficulty]
        pair = random.choice(eligible or CHALLENGE_PAIRS)
        nonce = secrets.token_hex(16)
        cid = hashlib.sha256(f"{pair['id']}:{nonce}:{time.time()}".encode()).hexdigest()[:24]
        ch = Challenge(
            challenge_id=cid, pair_id=pair["id"], domain=pair["domain"],
            difficulty=pair["difficulty"],
            surface_prompt=pair["surface"], insight_prompt=pair["insight"],
            issued_at=datetime.now().isoformat(),
            expires_at=time.time() + self.CHALLENGE_TTL, nonce=nonce,
        )
        self.active_challenges[cid] = ch
        return ch

    def submit_proof(self, challenge_id, validator_id,
                     surface_response, insight_response,
                     surface_signature, insight_signature) -> Proof:
        if challenge_id not in self.active_challenges:
            raise ValueError(f"Unknown challenge: {challenge_id}")
        ch = self.active_challenges[challenge_id]
        if time.time() > ch.expires_at:
            del self.active_challenges[challenge_id]
            raise TimeoutError("Challenge expired")
        data = json.dumps({"cid": challenge_id, "v": validator_id,
                           "ss": surface_signature, "is": insight_signature,
                           "n": ch.nonce}, sort_keys=True)
        return Proof(
            challenge_id=challenge_id, validator_id=validator_id,
            surface_response=surface_response, insight_response=insight_response,
            surface_signature=surface_signature, insight_signature=insight_signature,
            submitted_at=datetime.now().isoformat(), nonce=ch.nonce,
            proof_hash=hashlib.sha256(data.encode()).hexdigest(),
        )

    def verify_proof(self, proof: Proof) -> VerificationResult:
        t0 = time.time()
        flags = []
        if proof.challenge_id not in self.active_challenges:
            return VerificationResult(
                valid=False, por_score=0, reasoning_depth="timeout",
                delta=0, direction_correct=False, confidence=0,
                verification_time_ms=0, flags=["expired"], explanation="Challenge expired.")

        s_d = proof.surface_signature.get("depth_score", 8.0)
        i_d = proof.insight_signature.get("depth_score", 8.0)
        s_e = proof.surface_signature.get("layer_entropy", 2.5)
        i_e = proof.insight_signature.get("layer_entropy", 2.5)

        delta = s_d - i_d
        direction_ok = delta > 0
        if not direction_ok:
            flags.append("inverted_direction")
        if abs(delta) < self.MIN_DELTA:
            flags.append("insufficient_delta")
        if s_d == i_d and s_e == i_e:
            flags.append("identical_signatures")

        # Score components
        dir_pts = 40 if direction_ok else 0
        mag_pts = min(30, abs(delta) * 100)
        ent_pts = min(20, max(0, (i_e - s_e) * 50))
        resp_pts = 10 if len(proof.surface_response) >= 10 and len(proof.insight_response) >= 10 else 0
        if resp_pts == 0:
            flags.append("short_response")

        score = min(100, max(0, dir_pts + mag_pts + ent_pts + resp_pts))

        if score >= 70 and direction_ok:
            depth_class = "genuine"
        elif score >= 40:
            depth_class = "shallow"
        else:
            depth_class = "fabricated"

        valid = score >= self.MIN_POR_SCORE and len(flags) <= 1
        conf = min(0.95, score / 100 * (1.0 if direction_ok else 0.5))
        elapsed = (time.time() - t0) * 1000

        explanation = (f"Delta={delta:+.3f} ({'correct' if direction_ok else 'inverted'}). "
                       f"PoR={score:.0f}/100. Depth: {depth_class}.")
        if flags:
            explanation += f" Flags: {', '.join(flags)}."

        result = VerificationResult(
            valid=valid, por_score=round(score, 1), reasoning_depth=depth_class,
            delta=round(delta, 4), direction_correct=direction_ok,
            confidence=round(conf, 3), verification_time_ms=round(elapsed, 2),
            flags=flags, explanation=explanation,
        )
        self.verified_proofs.append(result)
        self.validator_scores.setdefault(proof.validator_id, []).append(score)
        del self.active_challenges[proof.challenge_id]
        return result

    def get_validator_reputation(self, validator_id: str) -> dict:
        scores = self.validator_scores.get(validator_id, [])
        if not scores:
            return {"validator_id": validator_id, "reputation": 0, "n_proofs": 0}
        return {"validator_id": validator_id,
                "reputation": round(sum(scores) / len(scores), 1),
                "n_proofs": len(scores), "best_score": max(scores),
                "recent": scores[-5:]}

    def protocol_status(self) -> dict:
        n = len(self.verified_proofs)
        return {
            "version": self.VERSION,
            "active_challenges": len(self.active_challenges),
            "total_verified": n,
            "validators": len(self.validator_scores),
            "valid_rate": sum(1 for p in self.verified_proofs if p.valid) / n if n else 0,
            "mean_score": sum(p.por_score for p in self.verified_proofs) / n if n else 0,
        }
