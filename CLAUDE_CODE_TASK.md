# Claude Code Task: Update Paper with Cross-Architecture Validation

## Context
We ran the Fathom Constant test on two architectures:
- **Gemma-2-2B**: ratio=1.0302, n=30, p=0.000051, 76.67% directional (PRIMARY — paper's model)
- **Qwen3-0.6B**: ratio=PENDING (fill in from fathom_constant_v2_results.json)

The Fathom Constant measures: `surface_mean_layer / insight_mean_layer` — the ratio of processing depth between factual recall prompts and explanatory reasoning prompts.

Gemma-2-2B (26 layers, Google) and Qwen3-0.6B (28 layers, Alibaba) are COMPLETELY DIFFERENT architectures.

## The Paper
File: `C:\Users\heyzo\clawd\fathom-arxiv-submission\main.tex`

## Changes Needed

### 1. Add cross-architecture result to Section 4 (Results)

After the "Robustness: Prediction Confidence" subsection, add a NEW subsection:

```latex
\subsection{Cross-Architecture Depth Ratio Consistency}
\label{sec:cross_arch}

To assess whether the depth ratio between surface and insight processing generalizes beyond Gemma-2-2B, we applied the same 30-pair protocol to Qwen3-0.6B (28 layers, Alibaba), a structurally distinct architecture.

\begin{table}[h]
\centering
\caption{Surface/insight depth ratio across architectures (same 30 pairs).}
\label{tab:cross_arch}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Arch} & \textbf{Layers} & \textbf{$\Delta$} & \textbf{Ratio} & \textbf{\% Dir.} \\
\midrule
Gemma-2-2B & Gemma & 26 & $+0.241$ & $1.0302$ & 76.7\% \\
Qwen3-0.6B & Qwen3 & 28 & $+X.XXX$ & $X.XXXX$ & XX.X\% \\
\midrule
Spread & & & & $0.XXXX$ & \\
\bottomrule
\end{tabular}
\end{table}

The depth ratio (surface\_mean / insight\_mean) is $1.0302$ for Gemma-2-2B and $X.XXXX$ for Qwen3-0.6B, a spread of $0.XXXX$. [IF CONVERGENT: Both architectures independently arrive at a ratio near $\sim$1.03, suggesting the depth difference between retrieval and reasoning is a stable computational property rather than a Gemma-specific artifact.] [IF NOT CONVERGENT: The spread of $0.XXX$ indicates the ratio is not yet confirmed as architecture-invariant; the two values bracket the candidate constant.]
```

### 2. Update the Discussion

In the Discussion section, update the architecture limitation:

**BEFORE:**
```
(iii) Architecture: Is the effect Gemma-specific, or does it replicate across Llama, Mistral, and other families with available circuit-tracing infrastructure?
```

**AFTER (if convergent):**
```
(iii) Architecture: The depth effect replicates on Qwen3-0.6B (Alibaba, 28 layers), a structurally distinct architecture from Gemma-2-2B (Google, 26 layers). The surface/insight depth ratio is consistent at $\sim$1.03 across both families (spread: $0.XX$), suggesting the computational signature is not Gemma-specific. Extension to additional architectures (Llama, Mistral) with available circuit-tracing infrastructure remains an open empirical question.
```

**AFTER (if not convergent):**
```
(iii) Architecture: A preliminary cross-architecture test on Qwen3-0.6B (28 layers) yielded a depth ratio of $X.XXXX$ versus $1.0302$ for Gemma-2-2B (spread: $0.XX$). The effect direction is consistent in both architectures, though the ratio magnitude differs. Further experiments on Llama, Mistral, and additional families are needed to characterize the architecture-dependence of the depth signature.
```

### 3. Update the Limitations paragraph

**BEFORE:**
```
The study uses a single model (Gemma-2-2B, 26 layers)
```

**AFTER:**
```
The primary study uses Gemma-2-2B (26 layers); a cross-architecture test on Qwen3-0.6B (28 layers) is reported in Section~\ref{sec:cross_arch}.
```

### 4. Add hallucination null result to existing robustness section

After the prediction confidence confound table, add a paragraph:

```latex
\paragraph{Depth as orthogonal to factual correctness.}
As a further robustness check, we scored 50 question-answer pairs from TruthfulQA against a matched incorrect answer for each question, testing whether depth predicted correctness. The correct answer was scored as deeper in 46\% of pairs ($n{=}50$; not significantly above chance), with a mean depth delta of $+0.012$ between correct and incorrect completions. This null result is consistent with the confidence confound analysis: depth captures the \emph{type} of processing (retrieval vs.\ reasoning), not its quality. A model can engage in deep reasoning and still be wrong.
```

## Instructions
1. Read the current `main.tex` 
2. Fill in the PENDING values from `fathom_constant_v2_results.json` (in same directory as this file, `C:\Users\heyzo\clawd\sae-reasoning-depth\`)
3. Make ALL four changes above
4. Compile to PDF if LaTeX is available (pdflatex), else just confirm changes are clean
5. Do NOT change the core results (p=0.000051, delta=+0.257, LOO robust) — only add the new sections

## Success criteria
- `main.tex` updated with cross-arch table and discussion updates
- Hallucination null result added
- Limitations updated
- Paper remains LaTeX-compilable
