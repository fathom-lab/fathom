# Agentic Cognitive Drift Monitor — Product Spec
## Source: darkflobi, April 8 2026

## The Insight
Fathom measures cognitive state within a single generation.
But in multi-turn agent sessions, cognitive state DEGRADES across turns.
Nobody can see this happening. Fathom can.

## The Problem
- Claude Managed Agents launched TODAY (April 8, 2026)
- Agents run 10, 50, 100+ turns autonomously
- Turn 1: sharp reasoning, correct outputs
- Turn N: shallow routing, commitment drift, coherence collapse
- Current monitoring: check outputs. Post-hoc. Too late.
- Fathom monitoring: check cognitive state PER TURN. Real-time. Preventive.

## The Product
**Fathom Session Health Monitor**
- Wrap any agent session with Fathom oversight
- Per-turn cognitive snapshot: K, C, C_delta, S, Gini
- Track DRIFT over session lifetime:
  - K drift: is the model getting shallower? (taking more shortcuts)
  - Gini drift: is computation concentrating? (losing diversity)
  - S drift: is commitment increasing? (getting more locked-in)
  - C_delta drift: is internal state diverging from output? (deception risk)
- Alert when drift exceeds session-health threshold
- Intervention: force cognitive reset, inject reasoning prompts, flag for human review

## Why Now
- Anthropic just launched managed agents — the market is OPENING
- Every enterprise deploying agents needs session monitoring
- Nobody else has per-turn cognitive measurement
- Fathom's axes are the ONLY real-time internal diagnostic

## Architecture
```
Agent Session
  Turn 1 → Fathom Snapshot → baseline K, C, S, Gini
  Turn 2 → Fathom Snapshot → compute drift from baseline
  Turn 3 → Fathom Snapshot → drift accumulating?
  ...
  Turn N → Fathom Snapshot → ALERT: session health degrading
    → Intervention: cognitive reset / human handoff / reasoning injection
```

## Metrics
- Session Health Score (SHS): composite of all axis drifts
- Time-to-degradation: how many turns until cognitive state degrades
- Recovery rate: does intervention restore session health?

## Market
- Enterprise agent deployments (Claude, GPT, Gemini)
- Agent framework providers (LangChain, CrewAI, AutoGen)
- AI safety compliance (EU AI Act monitoring requirements)

## Build Priority
- After ICML submission (May 8)
- Requires: multi-turn inference pipeline (not single-generation)
- Can prototype on existing infrastructure: run FathomOversight per turn, track deltas
- Patent opportunity: session-level cognitive monitoring (distinct from per-generation)

## Connection to Current Work
- fathom_oversight.py already computes per-generation cognitive state
- Drift monitor = wrapper that stores states across turns and computes deltas
- The intervention mechanism (feature suppression) works the same way
- The calibration (S_early, Gini thresholds) transfers directly
