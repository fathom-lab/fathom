# FATHOM for Law

## The Legal AI Landscape

Major law firms have adopted AI at scale:
- Allen & Overy deployed Harvey (GPT-4 based) across 43 offices
- Latham & Watkins, Clifford Chance, and Freshfields all have AI programs
- Thomson Reuters integrated AI into Westlaw
- LexisNexis launched Lexis+ AI

The problem: nobody can verify whether these tools are *reasoning* through legal analysis or *pattern-matching* from training data. When AI hallucinates a citation (Mata v. Avianca, 2023 — ChatGPT fabricated case citations), there is no way to detect it before filing.

Courts are responding:
- SDNY Standing Order (June 2023): attorneys must disclose AI use and verify citations
- 5th Circuit: adopted AI policy requiring disclosure
- Texas: Rule 1.1(e) requires competence in AI tools used in practice

## Fathom as Legal AI Certification

Fathom provides a verifiable measurement of whether AI-generated legal work reflects genuine reasoning or surface-level generation.

### Use Cases

**1. Brief Certification**
Every AI-generated brief section gets a Fathom score. The attorney can state:
> "This analysis was generated with AI assistance and independently verified. Fathom reasoning depth score: 78/100 (Grade A). The AI engaged distributed mid-layer reasoning consistent with genuine legal analysis rather than surface pattern completion."

**2. Citation Verification**
AI-generated citations can be scored. Shallow depth signatures on citation-heavy passages may indicate confabulation rather than genuine case retrieval.

**3. Due Diligence Quality**
AI reviewing contracts for due diligence can be scored per-section. Sections where the AI shows shallow processing get flagged for human review.

**4. Compliance Documentation**
For courts requiring AI disclosure, Fathom provides quantitative evidence of AI reasoning quality alongside the disclosure.

### Sample API Call
```bash
curl -X POST https://api.fathom.darkflobi.com/v2/compare \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "surface_prompt": "The standard for summary judgment under Rule 56 requires that there be no genuine dispute of material",
    "insight_prompt": "Summary judgment is appropriate here because the undisputed facts establish that the defendant owed a duty of care, breached that duty through the failure to inspect, and this breach proximately caused"
  }'
```

## Pricing: Legal Tier

| Tier | Price | Includes |
|------|-------|---------|
| Solo/Small Firm | $49/month | 5k scores, court-ready report export |
| Mid-Market | $299/month | 50k scores, batch scoring for document review, API integration |
| BigLaw / AmLaw 100 | $1,999/month | 500k scores, dedicated instance, white-label option |
| Enterprise Legal | Custom | Unlimited, on-premise, integration with Harvey/CoCounsel |

## Competitive Advantage

No existing legal AI tool measures *how* the AI reasoned. They measure:
- Accuracy (did it cite real cases?) — after the fact
- Fluency (does it read well?) — irrelevant to correctness
- User satisfaction (did the attorney like it?) — subjective

Fathom measures the one thing that matters: **did the AI actually reason through the legal question, or did it autocomplete?**

## Target Customers

- AmLaw 100 firms with AI programs (Harvey, CoCounsel, custom)
- Legal tech companies building AI products (Casetext, vLex, ROSS)
- Courts and judicial bodies developing AI policies
- Bar associations setting AI competency standards
- Corporate legal departments with AI review workflows
