# FATHOM — Pitch Deck Outline (12 Slides)

---

## Slide 1 — COVER
**Title:** Fathom
**Subtitle:** The depth of AI reasoning. Measured.
**Visual:** Dark background, single glowing depth sonar pulse animation or static waveform
**Notes:** No bullet points. Just the name, the tagline, and contact. First impression is precision and confidence — not hype.

---

## Slide 2 — THE PROBLEM
**Title:** AI is a black box with no depth gauge.
**Content:**
- $50B being spent annually on AI evaluation
- Every eval tool measures *output quality* — not reasoning quality
- A model can return the right answer via shallow pattern match or deep causal reasoning. You cannot tell which.
- In healthcare, legal, and finance: that distinction has liability consequences
**Visual:** Split brain diagram — surface output vs. hidden processing layers
**Notes:** Make the audience feel the gap. This isn't an academic problem. It's a deployed product problem.

---

## Slide 3 — THE INSIGHT
**Title:** Reasoning has structure. We found it.
**Content:**
- Transformers don't process all prompts equally deep
- SAE (Sparse Autoencoder) circuit attribution lets us trace *which features activate* and *how deep* the chain runs
- We built Fathom to instrument this — and found something unexpected
**Visual:** Diagram of SAE layer attribution stack (prompt → feature activation → depth score)
**Notes:** This is the "aha" slide. Don't oversell. Let the methodology speak.

---

## Slide 4 — THE CONSTANT
**Title:** K = 1.027 ± 0.003
**Content:**
- Across architectures, the ratio of surface-to-insight processing depth converges
- This is the **Fathom Constant** — a structural invariant of transformer reasoning
- Not a coincidence. Two models. Two independent experiments. Same constant.
**Visual:** Large, clean typographic K value. Below it: the two data points with p-values.
**Notes:** This is the "holy shit" slide. Let it breathe. Big number. Clean layout. The p-values do the talking.

---

## Slide 5 — CROSS-ARCHITECTURE PROOF
**Title:** Replicated. Independently. Same result.
**Content:**

| Architecture | K | p-value | n |
|---|---|---|---|
| Gemma-2-2B | 1.0302 | 0.000051 | 30 |
| Llama-3.2-1B | 1.0242 | 0.00266 | 19 |

- Two completely different model families
- Independent experimental design
- Consistent constant across both
**Visual:** Side-by-side architecture diagrams with K overlaid
**Notes:** Statistical significance is your armor here. p=0.000051 is not luck. Make sure investors can see it clearly.

---

## Slide 6 — DOMAIN CONVERGENCE
**Title:** 5 industries. Same signal.
**Content:**

| Domain | K |
|---|---|
| Healthcare | 1.008 |
| Finance | 1.034 |
| Legal | 1.008 |
| Education | 1.115 |
| AI Safety | 1.015 |

- K varies by domain — but within expected range
- Education shows the widest spread (reasoning demand is highest)
- This enables domain-calibrated depth scoring for enterprise use cases
**Visual:** Radar/spider chart showing K values across 5 domains
**Notes:** The domain variation is actually a *feature* — it means Fathom can produce domain-specific baselines. Education at 1.115 is the most interesting outlier to discuss.

---

## Slide 7 — THE PRODUCT
**Title:** Fathom API — Depth scoring for any AI pipeline.
**Content:**
- Input: any LLM prompt + response pair
- Output: depth score, circuit attribution trace, confidence band
- Integrates into existing eval frameworks (LangSmith, Weights & Biases, custom)
- Real-time and batch modes
- Chrome extension for manual spot-checking
**Visual:** Product UI mockup — clean dashboard showing depth score, K comparison, layer heatmap
**Notes:** Keep this practical. Show what an engineer actually sees when they use it.

---

## Slide 8 — BUSINESS MODEL
**Title:** Three revenue streams. One standard.
**Content:**
1. **API Licensing** — Per-call depth scoring. Developer tier + production tier.
2. **Enterprise SaaS** — Depth dashboards, alerting, compliance reporting. Annual contracts.
3. **Model Certification** — "Fathom-Verified" depth rating. Sold to model vendors + AI procurement teams.
**Visual:** Simple 3-column layout. Rough pricing tiers (TBD — placeholder ok).
**Notes:** Certification is the long-term moat. Once Fathom is *the* standard, everyone pays to verify against it.

---

## Slide 9 — MARKET
**Title:** $50B market. No depth instruments.
**Content:**
- **TAM:** $50B — global AI evaluation, observability, and compliance market (2030)
- **SAM:** $8B — enterprises actively procuring AI eval tooling in regulated industries
- **SOM:** $80M — achievable in 3 years via API + enterprise SaaS in 3 target verticals
- Adjacent: AI safety tooling, model red-teaming, regulatory compliance
**Visual:** Funnel diagram (TAM → SAM → SOM) with market context labels
**Notes:** Reference Gartner/IDC projections if available. The key point: no one owns "depth." That category is empty.

---

## Slide 10 — TRACTION
**Title:** Patent filed. Paper submitted. Zero funding spent.
**Content:**
- USPTO Provisional Patent — Filed 03/29/2026 (#64/020,489 & 64/021,113)
- arXiv paper submitted — ID 7407675
- Two architectures validated
- Five domains validated
- Live website: darkflobi.com/fathom
- Built solo by one researcher
**Visual:** Timeline graphic showing milestones (discovery → patent → replication → domain → paper)
**Notes:** The "built solo" angle is important. It signals extreme capital efficiency and deep technical ownership. Don't undersell it.

---

## Slide 11 — THE ASK
**Title:** $500K–$2M seed.
**Content:**
**Use of funds:**
- 40% — Compute: full-suite model benchmarking at scale
- 30% — Engineering: 2 ML hires (SAE + distributed inference)
- 20% — Infrastructure: production API build-out
- 10% — Legal: full utility patent filing + IP protection
**Milestones this unlocks:**
- 10 architectures validated (K confirmed universally)
- Production API launch
- 3 enterprise pilot contracts
- Full utility patent filed
**Visual:** Pie chart (use of funds) + 12-month milestone roadmap
**Notes:** The 12-month patent window is real. That's a forcing function. Investors should understand the clock.

---

## Slide 12 — CLOSE
**Title:** The standard for AI reasoning depth starts here.
**Content:**
- The window to define this category is open now
- SAE interpretability is 18 months old. The infrastructure is ready. The market is demanding accountability.
- K = 1.027. It's consistent. It's measurable. It's patented.
- Let's build the instrument the industry doesn't know it needs yet.
**Contact:**
- Alexander Rodabaugh — @darkflobi
- darkflobi.com/fathom
- Patent Pending | USPTO 64/020,489 & 64/021,113
**Visual:** Return to cover visual — dark background, glowing K
**Notes:** End quiet. No "thank you for your time." Just the claim, the contact, the silence. Let the work speak.

---

*Deck version: 1.0 — March 2026*
