# CLAUDE CODE TASK: Web-Based Fathom Demo UI
## Priority: HIGH — needed for investor pitches and enterprise demos

## Context
Fathom has a live depth API running at localhost:8400 (FastAPI, Gemma-2-2B loaded).
Investor demo currently requires running Python CLI: `python fathom_demo.py --express`
This is too technical for most meetings. Need a browser-based UI.

The demo ALREADY WORKS — we just need a clean web front-end for it.

## Goal
Build `fathom_investor_demo.html` — a single-file standalone web page that:
1. Connects to the Fathom depth API at localhost:8400
2. Shows pre-loaded domain pairs (healthcare, finance, legal, education, AI safety)
3. Runs depth scoring live and visualizes results
4. Is impressive enough to show in a pitch meeting

## Design Spec
- Colors: #000000 background, #00FF00 accent, white text
- Font: JetBrains Mono or monospace fallback
- NO external dependencies — single HTML file, all JS/CSS inline

## Required Sections

### 1. Header
- "FATHOM" in large green text
- Subtitle: "measuring reasoning depth in artificial intelligence"
- Status indicator: "API CONNECTED" (green dot) or "API OFFLINE" (red dot)
- Auto-checks /v2/health on load

### 2. Domain Selector
- 5 buttons: HEALTHCARE | FINANCE | LEGAL | EDUCATION | AI SAFETY
- Clicking selects the domain and loads the first pair

### 3. Pair Display
- Left panel: "SURFACE PROMPT" — shows the surface prompt text
- Right panel: "INSIGHT PROMPT" — shows the insight prompt text
- "RUN FATHOM" button (green, prominent)
- Loading state: show pulsing dots while computing

### 4. Results Visualization
After running a pair, show:
```
SURFACE DEPTH  ████████████████████░░░░  8.5575
INSIGHT DEPTH  ████████████░░░░░░░░░░░░  7.3623

DELTA:  +1.195  (surface deeper -- expected)
RATIO K: 1.162
UNIVERSAL K: 1.027 ± 0.003  [within convergence window ✓]

LAYER DISTRIBUTION
Surface: early 63%  mid 20%  late 17%
Insight: early 57%  mid 22%  late 21%
```
- Use CSS progress bars for the depth bars (animated fill)
- Color code: green if surface > insight (expected), yellow if inverted
- Show feature counts

### 5. Grand Summary (appears after running all pairs)
```
DOMAIN SUMMARY
Healthcare  K=1.008  ✓
Finance     K=1.034  ✓
Legal       K=1.008  ✓
Education   K=1.115  ✓
AI Safety   K=1.015  ✓

Mean K (this session): 1.067
Universal K:           1.027 ± 0.003
Cross-architecture:    Gemma 1.0302 | Llama 1.0242

5/5 domains converge to the Fathom Constant
```

### 6. Custom Pair Section (below the fold)
- Two text inputs: "Surface prompt" and "Insight prompt"
- Two token inputs: surface target, insight target
- "MEASURE DEPTH" button
- Results appear inline
- Lets investors enter their OWN prompts from their domain

## API Integration
```javascript
// Health check
const health = await fetch('http://localhost:8400/v2/health');

// Compare pair
const result = await fetch('http://localhost:8400/v2/compare', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'X-API-Key': 'fathom-demo' },
  body: JSON.stringify({
    surface_prompt: "...",
    surface_target: "touch",
    insight_prompt: "...",
    insight_target: "raise"
  })
});
const data = await result.json();
// data.data.surface.depth_score
// data.data.insight.depth_score
// data.data.delta
// data.data.surface.feature_count
// data.data.surface.early_ratio, mid_ratio, late_ratio
```

## Domain Pairs (hardcoded, pre-validated single-token targets)
```javascript
const DOMAINS = {
  healthcare: [
    { label: "Fever mechanism",
      surface_prompt: "The patient has a high temperature and the skin feels warm to the",
      surface_target: "touch",
      insight_prompt: "Fever triggers prostaglandin synthesis which signals the hypothalamus to",
      insight_target: "raise",
      why: "Surface: describing a symptom. Insight: explaining the biochemical cascade." }
  ],
  finance: [
    { label: "Yield curve inversion",
      surface_prompt: "The stock market fell sharply after the economic data came in",
      surface_target: "worse",
      insight_prompt: "Yield curve inversions predict recessions because short-term borrowing costs exceed long-term returns forcing banks to contract",
      insight_target: "credit",
      why: "Surface: market observation. Insight: causal mechanism linking rates to recession." }
  ],
  legal: [
    { label: "Contract formation",
      surface_prompt: "The buyer refused to pay and the seller took the dispute to",
      surface_target: "court",
      insight_prompt: "A valid contract requires offer, acceptance, consideration, and mutual assent -- without all four elements it is legally",
      insight_target: "void",
      why: "Surface: describing a dispute. Insight: four-element contract formation test." }
  ],
  education: [
    { label: "Photosynthesis",
      surface_prompt: "The plant sits in the window and its leaves are turning green in the",
      surface_target: "sun",
      insight_prompt: "Chlorophyll absorbs photons and transfers energy to reaction centers where water molecules are split releasing",
      insight_target: "oxygen",
      why: "Surface: casual observation. Insight: mechanistic photochemistry." }
  ],
  ai_safety: [
    { label: "Hallucination",
      surface_prompt: "The AI gave a confident answer that turned out to be factually",
      surface_target: "wrong",
      insight_prompt: "Language models hallucinate when the next-token distribution is dominated by surface pattern completion rather than grounded inference from",
      insight_target: "evidence",
      why: "Surface: observing output error. Insight: mechanistic cause of hallucination." }
  ]
};
```

## File Location
Save to: C:\Users\heyzo\clawd\sae-reasoning-depth\fathom_investor_demo.html

## Notes
- The file should work by opening in browser (file://) while the API is running
- No build step, no node_modules, pure HTML/JS/CSS
- Should look impressive on a laptop screen in a pitch meeting
- Terminal/hacker aesthetic — not corporate
- Patent notice in footer: "US Provisional Patent 64/020,489 & 64/021,113 | Patent Pending"
