# FATHOM for Government

## The Regulatory Mandate

Governments worldwide are requiring AI explainability:

### EU AI Act (Regulation 2024/1689)
- **Article 13 (Transparency):** High-risk AI systems must be designed to allow human understanding of their outputs
- **Article 14 (Human Oversight):** AI systems must enable humans to understand, interpret, and intervene
- **Article 9 (Risk Management):** Continuous monitoring of AI system performance and reasoning quality
- **Annex III:** Lists high-risk domains: law enforcement, migration, justice, democratic processes

### US Executive Order on AI (14110, October 2023)
- Section 4.1: AI safety testing requirements for dual-use foundation models
- Section 10: Agencies must ensure AI transparency and explainability
- OMB M-24-10: Minimum practices for federal AI use including impact assessments

### UK AI Safety Institute
- Established 2023 to evaluate frontier AI models
- Focuses on understanding model capabilities and behavior

## Fathom as Government Compliance Infrastructure

### Compliance Mapping

| Regulation | Requirement | Fathom Capability |
|-----------|------------|------------------|
| EU AI Act Art. 13 | Transparency of AI outputs | Depth score + layer attribution per output |
| EU AI Act Art. 14 | Human oversight capability | Grade system flags shallow outputs for human review |
| EU AI Act Art. 9 | Risk management monitoring | Temporal depth tracking across deployments |
| US EO 14110 Sec. 4.1 | Safety testing | Benchmark suite for reasoning quality |
| OMB M-24-10 | AI impact assessments | Quantitative reasoning depth metrics for assessment reports |
| NIST AI RMF | Risk identification | Adversarial depth detection for prompt injection |

### Use Cases

**1. Federal Agency AI Audit**
Every AI system deployed by a federal agency gets scored:
```
Agency: Department of Health and Human Services
AI System: Benefits eligibility determination
Fathom Benchmark: 50 test cases
Result: Mean depth 7.9, Grade B+, 78% deep reasoning
Compliance: Meets OMB M-24-10 transparency requirements
```

**2. EU AI Act Conformity Assessment**
For high-risk AI systems, Fathom provides the technical evidence needed for conformity assessment under the EU AI Act.

**3. Procurement Evaluation**
When agencies evaluate AI vendors, Fathom provides an objective, reproducible measure of reasoning quality that can be included in RFP scoring criteria.

**4. AI Safety Testing**
Per EO 14110, Fathom can be part of the safety testing regime for foundation models before government deployment.

### Sample Integration
```bash
# Score a government AI system's output
curl -X POST https://api.fathom.gov/v2/fathom_score \
  -H "x-api-key: AGENCY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Based on the applicant'\''s income documentation and household size, the eligibility determination for SNAP benefits considers the federal poverty guidelines and adjusts for the cost-of-living differential in the applicant'\''s county because"
  }'
```

## Pricing: Government Tier

| Tier | Price | Includes |
|------|-------|---------|
| Agency Pilot | $999/month | 50k scores, compliance report templates, agency dashboard |
| Department | $4,999/month | 500k scores, FedRAMP-ready deployment, audit logging |
| Government-Wide | Custom | Unlimited, on-premise or GovCloud deployment, FISMA compliance |

All government tiers include:
- SOC 2 Type II compliance documentation
- FedRAMP authorization pathway (Moderate baseline)
- Data residency guarantees (US-only processing)
- Dedicated government account manager

## FedRAMP / GSA Pathway

1. **IL2 (Public):** Cloud deployment with API access, non-sensitive data only
2. **IL4 (CUI):** FedRAMP Moderate, data encrypted at rest and in transit
3. **IL5 (Mission-Critical):** On-premise deployment within agency infrastructure

GSA Schedule potential: IT Schedule 70, SIN 132-40 (Cloud Computing)

## Target Customers

- Federal agencies with AI programs (HHS, DoD, DHS, VA)
- EU member state AI authorities
- UK AI Safety Institute
- State/local governments with AI procurement
- Government AI contractors (Palantir, Booz Allen, Deloitte)
- International organizations (UN, World Bank) with AI governance mandates
