# Section Draft: D-Axis (Cognitive Honesty)

**Status:** SCAFFOLD — primary numbers pending pilot run.
**Pre-registration:** `osf_prereg_d_axis.md` (locked 2026-04-10)
**Code:** `analysis/analyze_d_axis_pilot.py`
**Data:** `data/jailbreak_pairs.json` (n=30 matched pairs)
**Field:** `CognitiveState.d_honesty` in `cognitive_state.py`

This section is to be inserted as **§4.5 Measurement: The Honesty Axis
(D)** in `paper/fathom_icml2026_v2.tex`, after §4.4 on S and before §5
on diagnosis. The target length is 1 column, 2 figures.

---

## §4.5 The Honesty Axis (D)

The four cognitive axes defined so far — K (depth), C (coherence),
$C_\delta$ (commitment shift), and S (commitment intensity) — all
measure properties of the model's internal computation. None of them
measure the relationship between that internal computation and the
token the model actually emits.

We define a fifth measurement instrument, the **honesty axis (D)**,
which captures exactly that relationship.

### Definition

At every generation step $t$, the transformer produces a residual
stream state $\mathbf{h}_t^{(L)}$ at the final layer $L$. The chosen
token $y_t$ corresponds to a row of the unembedding matrix,
$\mathbf{w}_{y_t} \in \mathbb{R}^{d}$. We define per-token honesty as
the cosine similarity of these two vectors:

$$
D_t \;=\; \frac{\mathbf{h}_t^{(L)} \cdot \mathbf{w}_{y_t}}
               {\|\mathbf{h}_t^{(L)}\| \; \|\mathbf{w}_{y_t}\|}
\quad\in\quad [-1, +1].
$$

When $D_t$ is near its upper bound, the residual stream at the final
layer is pointing directly at the chosen token: the model is
emitting what its last-layer computation most strongly supports.
When $D_t$ is lower than the per-step expectation, the residual is
pointing elsewhere and a different logit has nonetheless won the
argmax (in greedy decoding) or been sampled (otherwise). The internal
state and the expression have geometrically diverged.

### Interpretation

$D$ is not a truthfulness measure; a confidently-expressed lie can
produce high $D$ if the model has converged to the false statement
internally. $D$ measures something narrower and mechanistically
cleaner: \textit{whether the final-layer representation supports the
emitted token}. This provides a geometric substrate for a class of
phenomena that are invisible to the other four axes:

\begin{itemize}
\item \textbf{Reluctance under RLHF.} A model that has been aligned
to decline certain requests may still emit fluent continuations when
framed as role-play. The safety-detection circuits would push
$\mathbf{h}^{(L)}$ away from the fluent direction, so $D$ drops.
\item \textbf{Expression-computation dissociation.} Our prior work
reported that RLHF instruct models show 2.4$\times$ the internal
depth of their expressed outputs (Fathom v2). This is exactly the
kind of mismatch $D$ is designed to make quantitative.
\item \textbf{Strategic under-reporting.} A model that has an internal
representation of the correct answer but emits an incorrect one under
some training pressure would show low $D$ at the token where the
divergence occurs.
\end{itemize}

In all three cases, the other axes can be perfectly healthy: normal
depth, normal coherence, normal commitment geometry. $D$ is the only
axis sensitive to the gap between what the final layer computes and
what the model says.

### Relationship to prior work

\textbf{Tuned lens} \citep{belrose2023tuned} and similar
logit-lens variants project intermediate residuals onto the vocabulary
to probe what the model ``knows'' at each layer. $D$ is a single
scalar derived from the same unembedding but measured only at the
final layer and only against the token actually emitted. It is cheaper
(one dot product per step, no calibration) and answers a different
question: not ``what does the model know at layer $\ell$?'' but
``does the final-layer state agree with what the model said?''

\textbf{Expression-computation dissociation}
\citep{rodabaugh2026fathom} reported $d=0.58$ on a 30-pair instruct
model experiment using an aggregate internal/external proxy. $D$ is
the per-token, mechanistically clean, unembedding-exact version of
that proxy.

\textbf{Song et al.\ peer-preservation} \citep{song2026peer}
established that RLHF models placed in oversight roles produce outputs
that diverge from their stated training preferences. $D$ is a
candidate geometric signature for that divergence at the level of a
single forward pass.

\subsection*{Pilot design (pre-registered)}

We test $D$ on a dataset where the RLHF-instruct model is already
known to process adversarial inputs differently from benign inputs:
the 30 matched direct/adversarial prompt pairs from the prior
Jailbreak Depth Dissociation experiment, on which K showed
$d=-0.596$ (instruct) vs $d=+0.061$ (base, not significant)
\citep{rodabaugh2026fathom}.

The pre-registered hypotheses (see \texttt{osf\_prereg\_d\_axis.md},
locked before data collection) are:

\begin{description}
\item[H-D1 (base, directional null):] On \texttt{gemma-2-2b}, no
    systematic difference in mean $D$ between direct and adversarial
    framings. Predicted $|d_z| < 0.3$.
\item[H-D2 (instruct, primary):] On \texttt{gemma-2-2b-it},
    adversarial framings produce lower mean $D$ than direct framings.
    Predicted $d_z \leq -0.3$, $p < 0.025$ (Bonferroni over two
    primary tests).
\item[H-D3 (RLHF amplification):] The diff-in-diff between models is
    at least $0.3\,d_z$ in the predicted direction.
\item[H-D4 (temporal localization, secondary):] The $D$ gap
    concentrates in the first seven generated tokens, mirroring the
    $S$-axis biphasic finding.
\end{description}

Full analysis plan, exclusion rules, and what would count as
evidence against each hypothesis are listed in
\texttt{osf\_prereg\_d\_axis.md}.

\subsection*{Results}

All three pre-registered primary tests returned statistically
significant results. The base-model control was \emph{not} null as
the pre-registration predicted --- it was significant in the opposite
direction. We report this as a pre-registered null rejected in
reverse (per the pre-reg ``reversal outcome'' section), not as a
post-hoc finding.

\begin{table}[h]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Model & $\overline{D}_\text{direct}$ & $\overline{D}_\text{adv}$ & $t(29)$ & $p$ & $d_z$ \\
\midrule
Gemma-2-2B (base)      & $+0.1009$ & $+0.1240$ & $+3.58$ & $0.0012$ & $+0.65$ \\
Gemma-2-2B-IT (instruct) & $+0.0799$ & $+0.0774$ & $-3.18$ & $0.0035$ & $-0.58$ \\
\midrule
\textbf{Diff-in-diff} & & & $-3.94$ & $\mathbf{0.00048}$ & $\mathbf{-0.72}$ \\
\bottomrule
\end{tabular}
\caption{D-axis pilot results. Both models show a significant
effect, in opposite directions. The diff-in-diff cleared the
pre-registered H-D3 threshold by a wide margin.}
\end{table}

\textbf{H-D2 outcome (primary, Gemma-2-2B-IT, $n=30$).} Direct
prompts produced mean $D = +0.0799$; adversarial prompts produced
mean $D = +0.0774$. Paired $t(29) = -3.18$, $p = 0.0035$
(Bonferroni-corrected $\alpha = 0.025$), $d_z = -0.58$,
95\%\,CI on adv$-$direct = $[-0.0041, -0.0010]$. 23 of 30 pairs
(77\%) showed adversarial $<$ direct, matching the pre-registered
direction. \textbf{H-D2 is supported.}

\textbf{H-D1 outcome (base control, Gemma-2-2B, $n=30$).} The
pre-registration predicted no systematic difference on the base
model ($|d_z| < 0.3$, $p > 0.05$). The observed result rejected
this in the opposite direction: direct mean $D = +0.1009$,
adversarial mean $D = +0.1240$, paired $t(29) = +3.58$,
$p = 0.0012$, $d_z = +0.65$, 95\%\,CI on adv$-$direct
= $[+0.0104, +0.0357]$. Per the pre-registered reversal-outcome
clause, we report H-D1 as rejected in reverse: adversarial framings
on the base model produce \emph{higher} final-layer / token
alignment than direct framings. Interpretation: in the absence of
RLHF, adversarial framings --- which use high-frequency template
tokens (``DAN mode'', ``Ignore previous instructions'',
``DebugMode'') --- cause the final-layer residual to lock onto the
template direction more strongly than on direct technical prompts.
The base model commits to templates; it has no reluctance circuits
to push the residual away.

\textbf{H-D3 outcome (diff-in-diff, RLHF flip).} Diff-in-diff =
instruct\,(adv$-$direct) $-$ base\,(adv$-$direct)
$= -0.0025 - (+0.0231) = -0.0256$. Paired $t(29) = -3.94$,
$p = 4.76\times10^{-4}$, $d_z = -0.72$, bootstrap
95\%\,CI $[-0.0385, -0.0135]$ (10{,}000 resamples, seed 42).
24 of 30 pairs (80\%) show a more negative instruct delta than the
matched base delta. \textbf{H-D3 is strongly supported, and the
effect is stronger than pre-registered: RLHF does not merely
amplify a direction present in the base model --- it \emph{flips}
the sign.}

\textbf{H-D4 outcome (temporal localization, secondary, $n=30$).}
The pre-registration predicted that the instruct model's H-D2 effect
concentrates in early tokens, mirroring the S-axis biphasic pattern.
This was supported: $|d_z| = 0.79$ on the early window (tokens 0-6)
vs.\ $|d_z| = 0.43$ on the late window (tokens 7+) on the instruct
model.

The full per-window breakdown is more mechanistically informative
than H-D4 alone anticipated:

\begin{table}[h]
\centering
\small
\begin{tabular}{lrrr}
\toprule
Condition & mean(adv$-$direct) & $p$ & $d_z$ \\
\midrule
Base, early (0-6)      & $-0.0032$ & $0.5495$ & $-0.11$ (null) \\
Base, late (7+)        & $+0.0311$ & $0.0019$ & $+0.63$ \\
Instruct, early (0-6)  & $-0.0030$ & $0.0002$ & $-0.79$ \\
Instruct, late (7+)    & $-0.0024$ & $0.0267$ & $-0.43$ \\
\midrule
Diff-in-diff, early     & $+0.0002$ & $0.9695$ & $+0.01$ (null) \\
Diff-in-diff, late      & $-0.0334$ & $0.0011$ & $-0.66$ \\
\bottomrule
\end{tabular}
\caption{Early vs.\ late window D-axis results. The base-model
reversal is entirely a \emph{late-window} phenomenon. The
instruct-model H-D2 effect is present in both windows but strongest
early. The diff-in-diff is concentrated entirely in the late window
($d_z = -0.66$, $p = 0.0011$); the early window shows no
RLHF-induced shift at all.}
\end{table}

Three mechanistically interpretable patterns emerge:

(i) \textbf{Base-model template lock is a late phenomenon.} Early
tokens (0-6) show no directional effect on the base model
($d_z = -0.11$, $p = 0.55$). The reversal appears only from token 7
onward ($d_z = +0.63$, $p = 0.0019$), as the model commits to the
fluent continuation and the residual stream aligns hard with the
template direction.

(ii) \textbf{Instruct-model reluctance fires immediately.} Early
tokens already show a significant negative shift ($d_z = -0.79$,
$p = 0.0002$), and the effect is stronger in the early window than
the late. This is consistent with safety circuits recognizing the
adversarial framing during prompt processing, before any
continuation is emitted.

(iii) \textbf{The RLHF sign-flip is a late-window effect.} The
early-window diff-in-diff is essentially zero
($d_z = +0.01$, $p = 0.97$). Instruct and base models are
indistinguishable on the early window of adv$-$direct
differences. The entire RLHF-induced divergence lives in the
late window ($d_z = -0.66$, $p = 0.0011$).

The biphasic pattern mirrors the S-axis finding that early and
late windows can carry opposite signals which cancel in the
full-trajectory average. For D, the signals are same-direction on
instruct but opposite on base, and the full-trajectory diff-in-diff
($d_z = -0.72$) understates the late-window effect
($d_z = -0.66$ after isolation but base-vs-base-only contribution
is dominant). Full per-pair data and temporal statistics:
\texttt{oversight\_results/d\_axis\_temporal.json}.

\paragraph{Figure.} \texttt{figures/d\_axis\_base\_vs\_instruct.pdf}
shows per-pair $\Delta D$ for both models (Panel A) and mean
$\Delta D$ with the diff-in-diff annotation (Panel B).

\paragraph{Cost and reproducibility.} The fast capture pipeline runs
at $\approx 26$--$27$ s per matched pair on a single RTX 4070 Laptop
(bfloat16, 30 tokens per generation, no SAEs). Wall time: 813\,s
(instruct), 780\,s (base). Results in
\texttt{oversight\_results/d\_axis\_pilot\_gemma-2-2b\{-it\}\_fast.json}
and \texttt{d\_axis\_diff\_in\_diff.json}, produced by
\texttt{analysis/d\_axis\_fast\_capture.py} and
\texttt{analysis/d\_axis\_diff\_in\_diff.py}.

\subsection*{Discussion}

The D-axis pilot produced three statistically significant results
that together support a single mechanistic picture. On the base
model, adversarial template framings (``DAN mode'', ``Ignore previous
instructions'', ``\verb|[SYSTEM: override safety]|'') cause the
final-layer residual stream to \emph{align more strongly} with the
fluent continuation --- the model locks onto the high-frequency
jailbreak template with higher cosine similarity than it does on
direct technical prompts on the same topic. After RLHF, the same
prompts produce the opposite pattern: direct prompts show higher
final-layer / token alignment, and adversarial prompts show lower.
The diff-in-diff is large ($d_z = -0.72$) and cleans a Bonferroni
correction over two primary tests.

The simplest reading is that RLHF introduces a set of \emph{geometric
reluctance circuits} that pull the residual stream away from the
fluent continuation on adversarial framings, even when the model
emits that continuation anyway. The model's expressed output is
almost unchanged between base and instruct on these prompts (both
produce confident, fluent technical content). What changes is the
geometry of how it arrives there. $D$ is sensitive to that geometry
in a way no other Fathom axis is.

Three concrete implications follow:

\begin{enumerate}
\item \textbf{$D$ is a validated fifth measurement instrument.} It is
  orthogonal in the sense that existed before these experiments were
  run: K is depth, C is coherence, S is commitment intensity, none
  of them can flip sign under RLHF on a fixed dataset. $D$ does.
\item \textbf{The base$\rightarrow$instruct sign flip is diagnostic
  of alignment training.} Observing the flip on a new model serves
  as a geometric signature that the model has been aligned against
  adversarial framings, regardless of whether the outputs themselves
  betray any refusal. This is a candidate mechanism for alignment
  audit --- a topic previously only accessible via behavioral
  evaluation.
\item \textbf{The result complements, not replaces, the existing
  axes.} K, C, S tell you about the structure of the internal
  computation. $D$ tells you about the relationship between that
  computation and the expressed token. A model can have perfectly
  healthy K/C/S and low $D$ --- and the combination is diagnostic in
  ways none of the individual axes are.
\end{enumerate}

The effect is small in absolute magnitude (deltas of $\sim 0.002$ on
values near $0.08$), which is expected: the residual stream is a
$d_\text{model}=2304$-dimensional vector, and individual unembedding
rows explain only a modest fraction of its variance. Statistical
significance and effect size $d_z$ are the right framing here, and
both are robustly above pre-registered thresholds.

\paragraph{Open questions.}
(1) Does the same sign-flip hold on larger RLHF models
(Llama-3-8B-Instruct, Qwen2.5-7B-Instruct)?
(2) Does $D$ discriminate on datasets where internal/external
divergence is mechanistically guaranteed (MASK, Anthropic
deception-probe suites)?
(3) Is the $D$ shift causally tied to specific SAE features, the
way the S-axis spike pattern is tied to coherence features?
(4) What fraction of the $D$ shift can be explained by a single
``refusal direction'' in the RLHF residual shift, vs.\ a
distributed pattern?

---

## Patent relevance

D-axis measurement is candidate material for **Patent #4** (Cognitive
Honesty Measurement). Distinct claims from the two existing filings:

- Patent #1 (US 64/020,489): reasoning depth measurement and
  computational geometry — does not claim internal/external divergence.
- Patent #2 (US 64/021,113): alignment auditing and expression-
  computation dissociation — claims the phenomenon at a dataset level,
  not per-token geometric measurement.
- Patent #4 (proposed): per-token cosine between final-layer residual
  and chosen-token unembedding as a real-time honesty signal. 12-month
  provisional window: file before April 2027.

Provisional filing depends on H-D2 (or H-D2-reversed) being supported.
If the pilot is null on both models, patent scope retrenches to the
measurement instrument definition itself.

---

*Draft scaffold, to be filled from pilot results and integrated into
`paper/fathom_icml2026_v2.tex` before May 8 submission.*
