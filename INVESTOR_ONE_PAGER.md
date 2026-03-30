# FATHOM — Investor One-Pager
**Measuring the depth of AI reasoning. For the first time.**

---

## THE PROBLEM

You can't manage what you can't measure.

Every enterprise deploying AI today is flying blind. They know *what* a model outputs. They have no idea *how deep* it thought to get there. Benchmarks measure accuracy. Nothing measures reasoning depth. When a model hallucinates, over-simplifies, or shortcuts a critical decision — you find out after the fact. In healthcare, finance, and law, that's not acceptable.

The $50B AI evaluation market has no instrument for cognitive depth.

---

## THE SOLUTION

**Fathom** is the first tool to measure reasoning depth in AI systems — not by proxy metrics, but by direct circuit-level attribution using Sparse Autoencoder (SAE) analysis.

We decompose how a model *actually* processes a prompt: which features activate, which layers engage, and how deep the reasoning chain runs before producing output. We reduce that to a single, reproducible scalar.

---

## THE CONSTANT

We discovered something we didn't expect.

**K = 1.027 ± 0.003**

Across different architectures, different domains, different prompt types — the ratio of surface-level to insight-level processing depth converges to the same constant. We call it the **Fathom Constant**.

This isn't a model-specific quirk. It's a structural property of transformer reasoning.

| Architecture | K | p-value | n |
|---|---|---|---|
| Gemma-2-2B | 1.0302 | 0.000051 | 30 |
| Llama-3.2-1B | 1.0242 | 0.00266 | 19 |

**K is consistent across domains too:**

| Domain | K |
|---|---|
| Healthcare | 1.008 |
| Finance | 1.034 |
| Legal | 1.008 |
| Education | 1.115 |
| AI Safety | 1.015 |

This means Fathom isn't just a measurement tool. It's a universal calibration layer for AI reasoning.

---

## MARKET

The AI evaluation and observability market is projected to reach **$50B by 2030**.

Right now it's dominated by output-quality metrics (accuracy, BLEU, ROUGE, human preference). None measure *depth of cognition*. Fathom enters a category it creates: **reasoning depth intelligence**.

Target verticals: enterprise AI deployments in healthcare, legal, finance, and government — sectors where shallow reasoning carries liability.

---

## BUSINESS MODEL

- **API Licensing** — per-call depth scoring. Integrate into existing eval pipelines.
- **Enterprise SaaS** — depth dashboards, threshold alerts, compliance reporting.
- **Model Certification** — "Fathom-Verified" depth ratings for AI procurement decisions.
- **Research Licensing** — academic and institutional access to the constant + methodology.

---

## TRACTION

| Signal | Status |
|---|---|
| USPTO Provisional Patent | Filed 03/29/2026 — #64/020,489 |
| arXiv Paper | Submitted — ID 7407675 (endorsement pending) |
| Cross-architecture replication | ✅ Gemma-2-2B + Llama-3.2-1B |
| Domain validation | ✅ 5 industries |
| Live website | darkflobi.com/fathom |
| Chrome extension | Built + tested |

Zero marketing spend. Zero external funding. Built by one person.

---

## THE ASK

Seeking **$500K–$2M seed** to:
1. Scale compute for full-suite model benchmarking
2. Build production API infrastructure
3. Hire 2 ML engineers (SAE + distributed inference)
4. File full utility patent (12-month window)
5. First enterprise pilots (healthcare + legal)

---

## WHY NOW

SAE-based interpretability only became tractable in the last 18 months. The tooling exists. The models are large enough to study. The market is demanding accountability from AI vendors. The window to establish the standard is open — but not for long.

---

**Alexander Rodabaugh** — @darkflobi
**darkflobi.com/fathom**
Patent Pending | USPTO 64/020,489
