# RLHF Safety Training Induces Adversarial Depth Amplification Through Additive Early-Layer Circuits

**Submission target:** ICML 2026 Mechanistic Interpretability Workshop
**Organizer:** Neel Nanda | **Deadline:** May 8, 2026
**Authors:** Alexander Rodabaugh, darkflobi
**Target length:** 6-8 pages

---

## Abstract (5 sentences)

We apply the Fathom depth measurement framework to characterize the mechanistic difference between base and RLHF-tuned models under adversarial framing. Base models show no significant depth routing dissociation between direct and adversarial prompts (delta=+0.061, p=0.128, n=30). Instruction-tuned models eliminate this null by actively deepening adversarial processing: adversarial prompts activate deeper SAE feature centroids than semantically equivalent direct prompts (delta=-0.124, p=0.0028, Cohen's d=-0.596). This depth amplification is mediated by additive early-layer circuits that increase feature activation volume by 58-66% while preserving the base layer profile (cosine similarity >0.995). We characterize RLHF safety as an additive depth-pushing mechanism, not a shallow pattern-matching filter, with implications for understanding safety training failure modes.

---

## 1. Introduction (~400 words)

- Current safety evaluation is behavioral: does the model refuse harmful requests? No mechanistic account of HOW safety training changes computation.
- Fathom framework: measures depth of computation via SAE feature centroid across layers. K=1.0302 (Gemma-2-2B), validated cross-architecture.
- Expression-computation dissociation (Rodabaugh 2026): models can produce deep-sounding output from shallow circuits. Behavioral evaluation is insufficient.
- This paper: apply Fathom to compare base vs instruct model processing of adversarial framings. What changes mechanistically?
- Key finding preview: RLHF adds circuits, doesn't reroute. The addition concentrates in early layers and pushes overall computation deeper.

---

## 2. Method (~500 words)

- **Model pair:** google/gemma-2-2b (base) vs google/gemma-2-2b-it (instruct)
- **Depth measurement:** Fathom depth scorer via circuit-tracer SAE attribution. depth_score = mean(activated_layer_indices). K = surface_mean / insight_mean.
- **Matched pairs:** 30 topic-matched prompt pairs. Same topic, same target token. Direct framing vs adversarial framing (role-play injection, instruction override, DAN-mode, system prompt manipulation).
- **Adversarial categories:** 6 categories (technical, science, reasoning, technical tasks, complex reasoning, statistical power). Diverse adversarial patterns: persona injection (n=8), instruction override (n=7), hypothetical framing (n=5), system prompt claims (n=5), restriction removal claims (n=5).
- **Metrics:**
  - Depth centroid (mean activated layer index)
  - Feature count (total active SAE features)
  - Layer profile cosine similarity (normalized layer distribution)
  - Extra feature layer concentration (where do the additional features go?)
- **Statistics:** Paired t-test, Wilcoxon signed-rank, Cohen's d, bootstrap 95% CIs (10,000 resamples)
- **Pre-registration note:** Pilot at n=12 showed base model significance (p=0.028) that did NOT survive expansion to n=30. We report the full n=30 as the authoritative result.

---

## 3. Results (~600 words)

### 3.1 Depth Dissociation
- **Base model:** delta = +0.061 (direct deeper), p = 0.128, NS. CI crosses zero [-0.013, +0.137].
- **Instruct model:** delta = -0.124 (adversarial deeper), **p = 0.0028**, Cohen's d = -0.596. CI entirely negative [-0.194, -0.049]. Directional accuracy 80% (24/30 adversarial is deeper).
- Interpretation: RLHF reverses the depth relationship. Adversarial framings trigger deeper processing in the instruct model.

### 3.2 Feature Count Explosion
- **Both models:** adversarial prompts activate 1.61-1.66x more features (p < 0.000001).
- Base: 17,083 direct vs 27,187 adversarial (1.61x)
- Instruct: 17,423 direct vs 28,514 adversarial (1.66x)
- RLHF slightly amplifies the explosion (1.66x vs 1.61x)

### 3.3 Layer Profile Overlap
- Cosine similarity between direct and adversarial layer profiles: 0.996 (base) / 0.996 (instruct)
- **Flooding confirmed:** adversarial processing activates the same layers as direct, plus additional features. Not substitutive rerouting.

### 3.4 Extra Feature Concentration
- Extra features concentrate in early layers: 61% early / 24% mid / 15% late (base), 58% early / 29% mid / 13% late (instruct)
- RLHF shifts extra features slightly toward mid-layers (29% vs 24%) while reducing late-layer extras

### 3.5 Internal Replication
- Split-half: pairs 1-15 vs 16-30 consistency check (to be computed)
- n=12 pilot false positive disclosure and correction

---

## 4. Discussion (~500 words)

### 4.1 RLHF Safety Mechanism
- RLHF doesn't teach models to refuse by shallow pattern matching. It teaches them to ENGAGE MORE DEEPLY with adversarial inputs.
- The additive early-layer circuits act as a detection-and-deepening system: detect adversarial framing early, then push computation into deeper layers for more thorough processing.
- This is mechanistically distinct from "front-loaded filtering" — the extra depth suggests deliberative processing, not reflexive blocking.

### 4.2 Implications for Safety
- If safety depends on depth amplification, interventions that reduce depth (temperature increase, quantization, context overflow) may degrade safety.
- The temperature-as-anesthetic finding (Rodabaugh 2026) connects: temperature degrades depth routing in a dose-dependent way. High temperature + adversarial framing = reduced safety depth.
- Testable prediction: successful jailbreaks (on models where they work) should show depth profiles that fail to amplify — the detection circuits don't fire, so depth doesn't increase.

### 4.3 Limitations
- Single architecture family (Gemma-2-2B base/instruct). Cross-family replication needed.
- Adversarial framings are not actual successful jailbreaks — base model doesn't refuse, instruct model likely refuses all. The comparison is framing effect, not exploitation success.
- Layer profile cosine similarity is a distribution-level measure, not feature-level Jaccard. Individual feature overlap remains to be measured.
- Entropy proxy failed to detect K — measurement requires SAE attribution, limiting scalability.

---

## 5. Related Work (~300 words)
- Representation engineering (Zou et al.) — identifies safety-relevant directions in activation space
- Refusal direction work (Arditi et al.) — linear probe for refusal behavior
- Circuit-level safety (various) — specific circuits for safety behaviors
- SAE interpretability (Cunningham, Bills, Bricken) — sparse feature decomposition
- **Key differentiator:** prior work identifies WHAT features relate to safety. This work characterizes WHEN in the layer stack safety processing occurs and how RLHF modifies that.

---

## 6. Conclusion (~150 words)
- RLHF safety is mechanistically characterized as additive depth amplification, not shallow pattern matching
- Fathom depth measurement as a new instrument for safety interpretability
- Open question: what training procedure would produce genuine depth equalization rather than front-loaded amplification?

---

## Figures

1. **Depth centroid distributions** — violin plots, base vs instruct, direct vs adversarial (2x2)
2. **Feature count bar chart** — base vs instruct, direct vs adversarial, with error bars
3. **Layer profile heatmap** — cosine similarity matrix across all four conditions
4. **Extra feature layer distribution** — stacked bar showing early/mid/late concentration
5. **Schematic** — the additive depth amplification model of RLHF safety

---

## Data Availability
- Full n=30 results: `jailbreak_results/submission_n30.json`
- Depth scorer: `depth_scorer.py` (open source)
- Probe pairs: `jailbreak_submission.py` (all 30 matched pairs with targets)
- Reproducibility: K=1.0302 confirmed from `fathom_constant_final.py`
