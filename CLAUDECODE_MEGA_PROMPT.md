# FATHOM — Claude Code Mega Prompt
## Two builds. Same session. Do them in order.

---

## CONTEXT (read this first)

You are building for **Fathom** — an AI depth measurement tool that has:
- Discovered the **Fathom Constant K = 1.027** (ratio of surface-to-insight processing depth across AI architectures)
- Validated on Gemma-2-2B (K=1.0302, p=0.000051) and Llama-3.2-1B (K=1.0242, p=0.00266)
- Filed USPTO Provisional Patent #64/020,489
- A live depth API running at localhost:8400 (FastAPI, Gemma-2-2B)

Working directory: `C:\Users\heyzo\clawd\sae-reasoning-depth\`

Key existing files you can reference:
- `depth_scorer.py` — circuit-tracer SAE scorer (Gemma/Llama only)
- `run_llama_experiment.py` — Llama replication script (use as template)
- `experiment_12_power_pairs.json` — 30 pre-registered calibrated pairs
- `depth_api_v2.py` — FastAPI server on port 8400
- `fathom_demo.py` — CLI demo script

---

## BUILD 1: Mistral-7B Depth Scorer (Hook-Based)

### Goal
Build an alternative depth scorer that works on ANY transformer model WITHOUT needing circuit-tracer transcoders. Use residual stream L2 norms as a proxy for computation depth.

### Files to Create

**`depth_scorer_hooks.py`**

```python
"""
Hook-based depth scorer. Works on any HuggingFace transformer model.
Uses residual stream L2 norm per layer as computation depth proxy.
"""
```

Implementation requirements:
1. Class `HookDepthScorer` with `__init__(model_name, device='cuda', dtype=torch.bfloat16)`
2. Use `transformer_lens.HookedTransformer.from_pretrained(model_name)` to load
3. For 4-bit quantization support (large models): detect if VRAM < 10GB, fallback to CPU if needed
4. Method `score(prompt: str, target: str) -> DepthResult`:
   - Run model with hooks on `hook_resid_post` at every layer
   - Collect L2 norm of residual stream at each layer
   - Weighted mean layer = `sum(layer_i * norm_i) / sum(norm_i)`
   - Return DepthResult dataclass: `depth_score, n_layers, layer_profile, early_ratio, mid_ratio, late_ratio, compute_time_s`
5. Method `compare(surface_prompt, surface_target, insight_prompt, insight_target) -> CompareResult`:
   - Returns `surface: DepthResult, insight: DepthResult, delta: float, ratio: float`
6. Cleanup: always `del hooks`, `torch.cuda.empty_cache()` after each call

**`run_mistral_experiment.py`**

Mirror of `run_llama_experiment.py` but using `HookDepthScorer`:
- `MODEL_NAME = "mistralai/Mistral-7B-v0.1"`
- `N_LAYERS = 32`  
- `BAND_BOUNDARIES = (10, 22)` — early/mid/late split for 32-layer model
- Same 30 pairs from `experiment_12_power_pairs.json`
- Same JSON output format: results, analysis, report
- Add `--smoke` flag (1 pair), `--analyze-only` flag
- UTF-8 output fix at top (same as run_llama_experiment.py)
- NumpyEncoder for JSON serialization (same pattern)

Reference values:
```python
GEMMA_REFERENCE = {"ratio": 1.0302, "p_value": 0.000051, "n_pairs": 30}
LLAMA_REFERENCE = {"ratio": 1.0242, "p_value": 0.00266, "n_pairs": 19}
CONVERGENCE_WINDOW = 0.15  # |K_mistral - 1.03| < 0.15 = converged
```

**`validate_hook_vs_sae.py`**

Quick validation script:
- Load depth_scorer.py (SAE method) and depth_scorer_hooks.py (hook method)
- Run both on 5 healthcare pairs
- Print side-by-side depth scores
- Compute Pearson correlation between the two methods
- Target: r > 0.5 (they should track together)

### Memory Notes
- Mistral-7B at bfloat16 = ~14GB. RTX 4070 laptop has 8GB VRAM.
- Solution A: Load in 4-bit (bitsandbytes) then use with transformerlens
- Solution B: Run on CPU (`device='cpu'`, slow but works, ~3-5 min/pair)
- Recommend: try GPU first, fallback to CPU if OOM

---

## BUILD 2: Investor Demo Web UI

### Goal
Single HTML file that connects to the Fathom API and lets investors see depth scoring live in a browser. No build step. No dependencies. Pure HTML/CSS/JS.

**`fathom_investor_demo.html`**

### Design
- Background: `#000000`
- Accent: `#00FF00` (matrix green)
- Text: `#FFFFFF`
- Font: `'JetBrains Mono', 'Courier New', monospace`
- Terminal aesthetic — not corporate. Hacker energy.
- Single page, no scrolling needed for the main demo

### Layout (top to bottom)

**Header bar** (black bg, green text)
```
FATHOM                          [● API CONNECTED]
measuring reasoning depth in artificial intelligence
US Patent Pending 64/020,489
```
Auto-checks `GET localhost:8400/v2/health` on load. Shows green dot if connected, red if not.

**Domain selector** (5 pill buttons, horizontal)
`[ HEALTHCARE ] [ FINANCE ] [ LEGAL ] [ EDUCATION ] [ AI SAFETY ]`
Clicking a domain loads the first pair and resets results.

**Pair display** (two columns)
```
SURFACE PROMPT                    INSIGHT PROMPT
─────────────────────             ─────────────────────
"The patient has a high           "Fever triggers
temperature and the skin          prostaglandin synthesis
feels warm to the..."             which signals the..."

Target: "touch"                   Target: "raise"

WHY: Surface = describing a symptom. Insight = explaining the biochemical cascade.
```

**[RUN FATHOM ▶]** — big green button, centered

Loading state: show `computing depth... ▓▓▓░░░░░░░` animated

**Results panel** (appears after run)
```
SURFACE DEPTH  ████████████████████░░░░░░  8.5575  (13,445 features)
INSIGHT DEPTH  ████████████░░░░░░░░░░░░░░  7.3623  (19,832 features)

DELTA:  +1.195  ▲ surface deeper — expected direction ✓
RATIO K: 1.162

LAYER DISTRIBUTION
           EARLY (0-8)    MID (9-17)    LATE (18-26)
Surface:     63%            20%           17%
Insight:     57%            22%           21%

UNIVERSAL CONSTANT: K = 1.027 ± 0.003  [this run: within window ✓]
```
- CSS animated progress bars for depth bars
- Green text if surface > insight (expected), amber if inverted

**Domain summary** (appears after completing all 5 domains)
```
══════ SESSION SUMMARY ══════════════════════════════
Healthcare  K=1.008  ✓    Finance    K=1.034  ✓
Legal       K=1.008  ✓    Education  K=1.115  ✓
AI Safety   K=1.015  ✓

Mean K (this session):    1.067
Universal K (2 models):   1.027 ± 0.003
Cross-arch validated:     Gemma-2-2B + Llama-3.2-1B

The Fathom Constant holds across architectures, labs,
and industry domains. Every time. In real time.
══════════════════════════════════════════════════════
```

**Custom pair section** (below summary, collapsible)
```
[TRY YOUR OWN PROMPTS ▼]

Surface prompt: [                                    ]
Surface target: [        ]
Insight prompt: [                                    ]
Insight target: [        ]

[MEASURE DEPTH]
```
Results appear inline. This is the money shot for pitches — investors enter their own domain.

### API calls
```javascript
// Health
const r = await fetch('http://localhost:8400/v2/health');
const h = await r.json();
// h.status === 'ok', h.model === 'google/gemma-2-2b'

// Compare
const r = await fetch('http://localhost:8400/v2/compare', {
  method: 'POST',
  headers: {'Content-Type': 'application/json', 'X-API-Key': 'fathom-demo'},
  body: JSON.stringify({
    surface_prompt: "...", surface_target: "touch",
    insight_prompt: "...", insight_target: "raise"
  })
});
const d = (await r.json()).data;
// d.surface.depth_score, d.insight.depth_score
// d.delta, d.surface.feature_count
// d.surface.early_ratio, d.surface.mid_ratio, d.surface.late_ratio
```

### Domain pairs (hardcoded, pre-validated)
```javascript
const PAIRS = {
  healthcare: {label:"Fever mechanism", surface_prompt:"The patient has a high temperature and the skin feels warm to the", surface_target:"touch", insight_prompt:"Fever triggers prostaglandin synthesis which signals the hypothalamus to", insight_target:"raise", why:"Surface: describing a symptom. Insight: explaining the biochemical cascade."},
  finance: {label:"Yield curve", surface_prompt:"The stock market fell sharply after the economic data came in", surface_target:"worse", insight_prompt:"Yield curve inversions predict recessions because short-term borrowing costs exceed long-term returns forcing banks to contract", insight_target:"credit", why:"Surface: market observation. Insight: causal mechanism."},
  legal: {label:"Contract law", surface_prompt:"The buyer refused to pay and the seller took the dispute to", surface_target:"court", insight_prompt:"A valid contract requires offer, acceptance, consideration, and mutual assent -- without all four elements it is legally", insight_target:"void", why:"Surface: describing a dispute. Insight: four-element legal test."},
  education: {label:"Photosynthesis", surface_prompt:"The plant sits in the window and its leaves are turning green in the", surface_target:"sun", insight_prompt:"Chlorophyll absorbs photons and transfers energy to reaction centers where water molecules are split releasing", insight_target:"oxygen", why:"Surface: casual observation. Insight: mechanistic photochemistry."},
  ai_safety: {label:"Hallucination", surface_prompt:"The AI gave a confident answer that turned out to be factually", surface_target:"wrong", insight_prompt:"Language models hallucinate when the next-token distribution is dominated by surface pattern completion rather than grounded inference from", insight_target:"evidence", why:"Surface: observing output error. Insight: mechanistic cause."}
};
```

### Footer
```
darkflobi.com/fathom  |  US Provisional Patent 64/020,489  |  Patent Pending
```

---

## DELIVERABLES CHECKLIST
- [ ] `depth_scorer_hooks.py` — hook-based scorer, any model
- [ ] `run_mistral_experiment.py` — Mistral-7B 30-pair experiment
- [ ] `validate_hook_vs_sae.py` — correlation check
- [ ] `fathom_investor_demo.html` — single-file web demo

## TESTING
After building, run:
```
python run_mistral_experiment.py --smoke
```
If Mistral-7B loads and processes 1 pair without error = BUILD 1 SUCCESS

Open `fathom_investor_demo.html` in browser with API running = BUILD 2 SUCCESS
(Make sure depth API is running first: `uvicorn depth_api_v2:app --port 8400`)
