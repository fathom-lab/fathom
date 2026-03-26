# FATHOM API — Pricing Model

## Tiers

### Free
- **Price:** $0/month
- **Requests:** 100/month
- **Endpoints:** `/v2/score` only (single prompt scoring)
- **Rate limit:** 2 req/min
- **Support:** Community (GitHub Issues)
- **Use case:** Evaluation, academic research, curiosity

### Developer
- **Price:** $29/month
- **Requests:** 10,000/month
- **Endpoints:** All (`/v2/score`, `/v2/compare`, `/v2/fingerprint`, `/v2/trajectory`, `/v2/fathom_score`, `/v2/predict_correctness`, `/v2/detect_adversarial`)
- **Rate limit:** 30 req/min
- **API key:** Yes
- **Support:** Email (48hr response)
- **Use case:** Building products, integrating depth scoring into pipelines

### Professional
- **Price:** $199/month
- **Requests:** 100,000/month
- **Endpoints:** All + `/v2/batch` (20 prompts per call) + `/v2/filter_dataset` (100 texts per call)
- **Rate limit:** 120 req/min
- **Priority queue:** Yes (dedicated GPU time slots)
- **API key:** Yes
- **Support:** Email (24hr response), Slack channel
- **Use case:** Production deployment, dataset curation, model evaluation at scale

### Enterprise
- **Price:** Custom
- **Requests:** Unlimited
- **Dedicated instance:** Yes (isolated GPU, no shared resources)
- **SLA:** 99.9% uptime guarantee
- **Custom models:** Load your own model into the scorer
- **On-premise option:** Available
- **Support:** Dedicated engineer, <4hr response
- **Use case:** AI safety teams, large-scale evaluation infrastructure

---

## Cost Analysis

### GPU Cost Per Request

Based on NVIDIA T4 GPU pricing (RunPod/Lambda, March 2026):

| Metric | Value |
|--------|-------|
| T4 hourly rate | $0.50/hr |
| Average score time | ~2 seconds |
| Scores per hour | ~1,800 |
| **Cost per score** | **$0.00028** |
| **Cost per 1,000 scores** | **$0.28** |
| Monthly fixed (24/7) | $360 |

With A10G (production recommended):

| Metric | Value |
|--------|-------|
| A10G hourly rate | $1.10/hr |
| Scores per hour | ~1,800 |
| **Cost per score** | **$0.00061** |
| **Cost per 1,000 scores** | **$0.61** |
| Monthly fixed (24/7) | $792 |

### Margin Per Tier

| Tier | Price | Max Requests | GPU Cost (T4) | Gross Margin |
|------|-------|-------------|---------------|-------------|
| Free | $0 | 100 | $0.03 | -$0.03 |
| Developer | $29 | 10,000 | $2.80 | **$26.20 (90%)** |
| Professional | $199 | 100,000 | $28.00 | **$171.00 (86%)** |

---

## Revenue Projections

### Scenario: 100 Customers

| Tier | Customers | Revenue | GPU Cost | Profit |
|------|-----------|---------|----------|--------|
| Free | 60 | $0 | $1.68 | -$2 |
| Developer | 30 | $870 | $84 | $786 |
| Professional | 10 | $1,990 | $280 | $1,710 |
| **Total** | **100** | **$2,860/mo** | **$366** | **$2,494/mo** |

Infrastructure: 1 T4 instance = $360/mo. **Net: $2,134/mo.**

### Scenario: 1,000 Customers

| Tier | Customers | Revenue | GPU Cost |
|------|-----------|---------|----------|
| Free | 600 | $0 | $17 |
| Developer | 300 | $8,700 | $840 |
| Professional | 100 | $19,900 | $2,800 |
| **Total** | **1,000** | **$28,600/mo** | **$3,657** |

Infrastructure: ~3 T4 instances = $1,080/mo. **Net: $23,863/mo.**

### Scenario: 10,000 Customers

| Tier | Customers | Revenue | GPU Cost |
|------|-----------|---------|----------|
| Free | 6,000 | $0 | $168 |
| Developer | 3,000 | $87,000 | $8,400 |
| Professional | 1,000 | $199,000 | $28,000 |
| **Total** | **10,000** | **$286,000/mo** | **$36,568** |

Infrastructure: ~25 T4 instances = $9,000/mo. **Net: $240,432/mo. ($2.9M ARR)**

---

## Break-Even Analysis

Fixed cost: 1 T4 GPU running 24/7 = **$360/month**

| To break even with... | Customers needed |
|-----------------------|-----------------|
| Developer only ($29) | 13 customers |
| Professional only ($199) | 2 customers |
| Mixed (70/30 Dev/Pro) | 5 customers |

**Break-even is trivially achievable.** The unit economics work because:
1. GPU cost per request is sub-penny ($0.00028)
2. Subscription prices are 90%+ margin
3. Scoring is compute-light (2s per score, not training)
4. Single GPU serves ~43,000 scores/day

---

## Competitive Positioning

No direct competitor scores reasoning depth. Adjacent pricing:

| Service | Price | What it does |
|---------|-------|-------------|
| OpenAI API | $0.01-0.06/1k tokens | Generate text |
| Anthropic API | $0.01-0.08/1k tokens | Generate text |
| HuggingFace Pro | $9/mo | Host models |
| **Fathom** | **$29-199/mo** | **Measure how AI thinks** |

Fathom is not a generation API. It is a measurement instrument.
The pricing reflects that — you pay for insight, not tokens.
