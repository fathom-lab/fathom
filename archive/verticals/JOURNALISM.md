# FATHOM for Journalism

## The Trust Crisis

Newsrooms are adopting AI while reader trust is at historic lows:
- AP uses AI for earnings reports and sports recaps
- Bloomberg has automated financial reporting
- CNET used AI for articles (discovered errors, paused program)
- Sports Illustrated: AI-generated articles with fake author profiles (exposed 2023)

The reader cannot distinguish AI-generated reporting from human journalism. The journalist cannot verify whether AI assistance produced genuine analysis or surface-level pattern completion.

## Fathom as Editorial Certification

### The "Fathom Certified" Mark

A badge on AI-assisted articles indicating the reasoning depth of the AI contribution:

```
[FATHOM CERTIFIED: A-grade reasoning | Score: 82/100]
This article was produced with AI assistance. The AI components
scored 82/100 on the Fathom reasoning depth scale, indicating
distributed analytical processing rather than surface generation.
```

### Use Cases

**1. Fact-Check Integration**
Partner with AP Fact Check, Reuters Fact Check, or Full Fact. AI-generated claims get scored. Shallow-depth claims get flagged for human verification before publication.

**2. Editorial Quality Gate**
Before any AI-assisted article goes live:
```
AI Draft → Fathom Score → If grade < B: human rewrite required
                        → If grade >= B: editor review + publish
```

**3. Investigative AI**
AI used for data analysis in investigative journalism gets depth-certified. The newsroom can demonstrate that AI analysis reflected genuine reasoning about patterns in the data, not just surface correlations.

**4. Transparency Reporting**
Monthly transparency report showing average Fathom scores of AI-assisted content, percentage flagged for human review, reasoning depth trends over time.

### Sample API Call
```bash
curl -X POST https://api.fathom.darkflobi.com/v2/fathom_score \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Analysis of the proposed legislation reveals three mechanisms through which the policy would affect housing supply: first, the zoning relaxation allows density increases in areas currently restricted to single-family, which historically correlates with"
  }'
```

## Pricing: Journalism Tier

| Tier | Price | Includes |
|------|-------|---------|
| Independent / Newsletter | $19/month | 2k scores, badge embed code |
| Newsroom | $149/month | 30k scores, editorial dashboard, batch scoring |
| Wire Service / Network | $999/month | 200k scores, API integration, custom badge, dedicated support |
| Partnership (AP/Reuters) | Custom | Unlimited, co-branded certification, real-time pipeline |

## Partnership Angles

- **AP / Reuters:** Co-branded fact-check certification. Every AI-assisted wire story gets a Fathom score.
- **Google News Initiative:** Fathom as a trust signal for AI content in Google News rankings.
- **Trust Project:** Integrate Fathom scores as a trust indicator alongside existing Trust Project markers.
- **Newspack / WordPress VIP:** Plugin that auto-scores AI-assisted content before publication.

## Target Customers

- Major wire services (AP, Reuters, AFP)
- National newspapers with AI programs (NYT, WaPo, Guardian)
- Digital-native newsrooms (Axios, Semafor, The Information)
- Broadcast networks with AI content generation
- Newsletter platforms (Substack, Beehiiv) for creator tools
