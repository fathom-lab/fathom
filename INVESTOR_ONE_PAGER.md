# FATHOM — One Pager

---

## What

Fathom is a reasoning depth measurement framework for language models. It measures *how* an AI thinks — not what it says — using sparse autoencoder circuit attribution across transformer layers.

## The Science

Sparse autoencoder circuit tracing reveals that surface recall and explanatory reasoning produce structurally different computational signatures in language models. Surface recall concentrates at output layers. Reasoning distributes across mid layers. Pre-registered replication: **delta = +0.257, p = 0.000051, n = 30 vocabulary-controlled pairs.** 83.3% directional. Leave-one-out robust. Orthogonal to model confidence (r = 0.001). 56 total pairs across three experimental phases. Paper pending arXiv submission (endorsement in progress).

## The Product

| Component | Status |
|-----------|--------|
| **API** — 15 endpoints, FastAPI, auth + rate limiting | Built, tested locally |
| **CLI** — score, compare, batch, serve | Built |
| **SDK** — Python package, local + remote modes | Built |
| **Dashboard** — terminal-aesthetic web UI | Built |
| **Chrome Extension** — right-click score any text | Built |
| **Benchmark** — 50 standardized pairs, 10 domains | Built |
| **Comparison Image Generator** — shareable PNG cards | Built |

All components working at localhost:8400.

## The Market

Eight verticals, each with a specific depth-scoring use case:

| Vertical | Problem Fathom Solves |
|----------|----------------------|
| **Medicine** | FDA AI explainability — circuit-level evidence for diagnostic AI |
| **Law** | Legal AI certification — depth grade on AI-generated briefs |
| **Journalism** | Editorial verification — "Fathom Certified" badge for AI articles |
| **Education** | AI tutoring quality — depth dashboard for parents/teachers |
| **Government** | EU AI Act / US EO compliance — explainability audit tool |
| **Science** | Research integrity — depth certification for AI-generated hypotheses |
| **AI Agents** | Pre-action verification — depth check before autonomous decisions |
| **Audio** | Real-time audio reasoning — Whisper + depth scoring pipeline |

## The Business

| Tier | Price | Volume | Target |
|------|-------|--------|--------|
| Free | $0/mo | 100 req/mo | Researchers, evaluation |
| Developer | $29/mo | 10k req/mo | Builders, integrations |
| Professional | $199/mo | 100k req/mo | Teams, production |
| Enterprise | Custom | Unlimited | Dedicated instance, SLA |

**Unit economics:** ~$0.0003 per request on T4 GPU ($0.50/hr, ~2s/score). Break-even at 13 developer customers or 2 professional customers.

## The Team

**darkflobi** — independent AI researcher. Built fathom end-to-end: the science (experiment design, statistical methodology, paper), the engineering (scorer, API, SDK, dashboard, 20+ modules), the product (pricing, verticals, launch materials). Single GPU. No lab. No institution.

## Status

| Milestone | Status |
|-----------|--------|
| Core finding (p=0.000051) | Done |
| Paper written (NeurIPS style, 7pp) | Done |
| arXiv endorsement | In progress |
| Product (API, CLI, SDK, dashboard) | Done |
| Cloud deployment config | Ready |
| Launch materials (thread, blog, outreach) | Ready |
| Public launch | Pending arXiv ID |

## The Ask

**For investors:**
- Cloud GPU costs: ~$500/month for initial deployment (1x T4)
- Marketing: $2k for launch amplification
- Legal: trademark + terms of service
- Total seed need: $25k covers 12 months of operation

**For partners:**
- Licensing: integrate fathom scoring into your platform
- Co-development: build vertical-specific products together
- Research: joint papers on cross-architecture validation, fMRI collaboration

---

*Contact: fathom@darkflobi.com*
*Web: darkflobi.com/fathom*
