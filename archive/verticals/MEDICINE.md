# FATHOM for Medicine

## The Regulatory Reality

The FDA's AI/ML Action Plan (2021, updated 2024) requires that AI-driven medical devices provide explainability for clinical decisions. The EU Medical Device Regulation (MDR 2017/745, Article 14) mandates human oversight of AI-based diagnostics and demands that manufacturers document how AI systems reach conclusions.

Current compliance approaches rely on:
- Post-hoc saliency maps (GradCAM) — which explain *what pixels* influenced a decision, not *how deeply the model reasoned*
- SHAP/LIME feature importance — which explains *which inputs* mattered, not *whether the model retrieved or reasoned*
- Model cards — which document architecture, not cognitive process

None of these answer the regulatory question: **did the AI actually reason through this diagnosis, or did it pattern-match?**

## How Fathom Fills the Gap

Fathom provides circuit-level evidence of reasoning depth. For any AI output, it answers:

1. **Depth Score** — Was this output produced by shallow retrieval (memorized pattern) or deep distributed reasoning?
2. **Layer Entropy** — How distributed was the computation? High entropy = genuine multi-step processing.
3. **Fathom Grade** — A single letter grade (A+ to F) for reasoning quality, suitable for regulatory documentation.
4. **Cognitive Fingerprint** — A vector that can be compared across outputs to detect consistency in reasoning patterns.

### Regulatory Mapping

| Requirement | Regulation | Fathom Solution |
|------------|-----------|----------------|
| Explainability | FDA AI/ML Action Plan | Depth score + layer profile per decision |
| Human oversight | EU MDR Art. 14 | Fathom grade flags shallow outputs for human review |
| Continuous monitoring | FDA GMLP | Temporal depth tracking across model updates |
| Transparency | EU AI Act (High-Risk) | Circuit attribution audit trail |

## Sample Integration

### Diagnostic AI Pipeline
```
Patient data → Diagnostic AI → AI Output
                                   ↓
                              Fathom API
                                   ↓
                    depth_score=7.82, grade=B+
                    entropy=3.41, signal=deep_reasoning
                                   ↓
                         Regulatory Log
              "Diagnosis produced via distributed reasoning
               (Fathom B+, depth 7.82). Meets explainability
               threshold for clinical deployment."
```

### Sample API Call
```bash
curl -X POST https://api.fathom.darkflobi.com/v2/fathom_score \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Patient presents with chest pain radiating to left arm, elevated troponin, ST elevation in leads V1-V4. The most likely diagnosis is acute anterior STEMI because the combination of",
    "target": "findings"
  }'
```

Response:
```json
{
  "data": {
    "fathom_score": 74,
    "grade": "B+",
    "interpretation": "Strong distributed reasoning with mid-layer engagement",
    "depth_result": {
      "depth_score": 7.82,
      "layer_entropy": 3.41,
      "early_ratio": 0.18,
      "mid_ratio": 0.52,
      "late_ratio": 0.30
    }
  }
}
```

## Pricing: Healthcare Tier

| Tier | Price | Includes |
|------|-------|---------|
| Clinical Pilot | $499/month | 50k scores, audit log export, compliance report template |
| Hospital System | $2,499/month | 500k scores, HIPAA BAA, dedicated support, SLA 99.9% |
| Enterprise Health | Custom | Unlimited, on-premise deployment, FDA submission documentation support |

## Key Regulatory Citations

- **FDA:** Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan, January 2021
- **EU MDR:** Regulation (EU) 2017/745, Article 14 — General obligations of manufacturers
- **EU AI Act:** Regulation (EU) 2024/1689, Article 14 — Human oversight, Article 13 — Transparency
- **GMLP:** Good Machine Learning Practice for Medical Device Development (FDA/Health Canada/MHRA, October 2021)
- **WHO:** Ethics and governance of AI for health, June 2021

## Target Customers

- Hospital AI/IT departments evaluating clinical decision support systems
- Medical device companies seeking FDA 510(k) or De Novo clearance for AI products
- Clinical trial organizations using AI for data analysis
- Health insurance companies deploying claims processing AI
- Telehealth platforms with AI triage
