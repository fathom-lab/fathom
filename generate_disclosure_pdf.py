# -*- coding: utf-8 -*-
"""Generate fathom_monitor_disclosure.pdf using ReportLab."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = "fathom_monitor_disclosure.pdf"

def build():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=letter,
        leftMargin=1*inch, rightMargin=1*inch,
        topMargin=1*inch, bottomMargin=1*inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "Title2", parent=styles["Heading1"],
        fontSize=16, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"),
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=12, spaceAfter=4, textColor=colors.HexColor("#333333"),
        alignment=TA_CENTER, italics=True,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"],
        fontSize=9, spaceAfter=2, textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=13, spaceAfter=4, spaceBefore=12,
        textColor=colors.HexColor("#1a1a1a"),
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=11, spaceAfter=3, spaceBefore=8,
        textColor=colors.HexColor("#1a1a1a"),
    )
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, spaceAfter=6, leading=15,
        alignment=TA_JUSTIFY,
    )
    code_style = ParagraphStyle(
        "Code", parent=styles["Code"],
        fontSize=8, spaceAfter=4, leading=11,
        fontName="Courier",
        backColor=colors.HexColor("#f5f5f5"),
        leftIndent=12, rightIndent=12,
    )
    bullet = ParagraphStyle(
        "Bullet", parent=styles["Normal"],
        fontSize=10, spaceAfter=3, leading=14,
        leftIndent=18, bulletIndent=8,
    )

    story = []

    # ── Title block ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("TECHNICAL DISCLOSURE", meta_style))
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph("Fathom Monitor", title_style))
    story.append(Paragraph(
        "Per-Token Hallucination Detection via Coherence Divergence<br/>"
        "in Sparse Autoencoder Feature Space",
        subtitle_style,
    ))
    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.08*inch))

    meta_lines = [
        "<b>Author:</b> Alexander Rodabaugh",
        "<b>Date of Conception:</b> April 2, 2026",
        "<b>Date of First Reduction to Practice:</b> April 2, 2026",
        "<b>Prior Related Filings:</b> US Provisional 64/020,489 (March 29, 2026); US Provisional 64/021,113 (March 30, 2026)",
        "<b>Associated Zenodo Records:</b> doi:10.5281/zenodo.19326175 (v1); doi:10.5281/zenodo.19364702 (v3)",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, meta_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))

    # ── Abstract ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph("Abstract", h1))
    story.append(Paragraph(
        "This disclosure describes <b>Fathom Monitor</b>, a system and method for detecting "
        "hallucination-risk tokens in large language model (LLM) outputs at the time of generation, "
        "using a mechanistic signal derived from the geometric structure of sparse autoencoder (SAE) "
        "feature activations. The core innovation is the use of <b>C_delta</b> — the divergence "
        "between late-layer and early-layer feature coherence — as a per-token hallucination "
        "indicator. When C_delta exceeds a calibrated threshold at a given token position, that "
        "token is flagged as uncertain or high-risk and annotated inline. This provides a "
        "transparency layer over LLM generation that operates with no post-hoc correction, no "
        "external knowledge source, and no architectural modification to the underlying model.",
        body,
    ))

    # ── Section 1 ────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Background and Problem Statement", h1))
    story.append(Paragraph(
        "Large language models generate text autoregressively, producing tokens one at a time. "
        "A well-documented failure mode of this process is <b>hallucination</b>: the confident "
        "generation of factually incorrect content. Existing approaches to hallucination detection "
        "operate primarily at the output level — analyzing the completed generation for "
        "inconsistencies — or require external retrieval systems for verification. Neither approach "
        "provides real-time, token-level transparency into which specific tokens are generated by "
        "circuits in an anomalous computational state.",
        body,
    ))
    story.append(Paragraph(
        "This disclosure describes a system that addresses this gap by monitoring the "
        "<b>internal geometric state</b> of the model during generation, using mechanistic "
        "signals that are causally upstream of the token output.",
        body,
    ))

    # ── Section 2 ────────────────────────────────────────────────────────────
    story.append(Paragraph("2. Core Technical Innovation", h1))
    story.append(Paragraph("2.1 The C_delta Signal", h2))
    story.append(Paragraph(
        "During the forward pass that generates each token, the model's residual stream passes "
        "through multiple transformer layers. In each layer, the SAE or transcoder associated "
        "with that layer activates a sparse set of learned features. These features have associated "
        "<b>decoder direction vectors</b> in model space.",
        body,
    ))
    story.append(Paragraph(
        "<b>Coherence (C)</b> at a given layer is defined as the mean pairwise cosine similarity "
        "of the decoder vectors corresponding to the top-k activated features at that layer and "
        "token position:",
        body,
    ))
    story.append(Preformatted(
        "C_layer = (1 / (k * (k-1))) * sum_{i != j} cos(W_dec[i], W_dec[j])\n"
        "\nwhere W_dec[i] is the decoder vector for the i-th highest-activated feature.",
        code_style,
    ))
    story.append(Paragraph(
        "<b>C_delta</b> is the divergence between late-layer and early-layer coherence:",
        body,
    ))
    story.append(Preformatted(
        "C_delta = mean(C_late_layers) - mean(C_early_layers)\n"
        "\nwhere early layers = bottom third of network, late layers = top third.",
        code_style,
    ))

    story.append(Paragraph("2.2 Empirical Validation", h2))
    story.append(Paragraph(
        "The C_delta signal was validated on the <b>TruthfulQA benchmark</b> using "
        "Gemma-2-2B (Google), n=50 matched pairs, 100 attribution passes on RTX 4070 GPU:",
        body,
    ))

    table_data = [
        ["Metric", "Correct Answers", "Hallucinated Answers", "p-value", "Cohen's d"],
        ["K (depth score)", "8.646", "8.654", "0.931", "0.029"],
        ["C_delta", "+0.0071", "+0.0077", "0.040", "0.407"],
    ]
    tbl = Table(table_data, colWidths=[1.5*inch, 1.3*inch, 1.5*inch, 0.9*inch, 0.9*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#e8f5e9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.08*inch))

    findings = [
        "<b>Depth (K) is blind to hallucination</b> (p=0.931): standard depth metrics cannot distinguish correct from incorrect generation.",
        "<b>C_delta discriminates hallucination</b> (p=0.040, d=0.407): a statistically significant, medium-effect signal.",
        "The direction of the signal is <b>late-layer over-coherence</b>, not scatter: hallucination corresponds to feature activations over-converging around a false attractor in late layers. This is cognitive lock-in, not cognitive noise.",
    ]
    for f in findings:
        story.append(Paragraph(f"&#8226;  {f}", bullet))

    story.append(Paragraph("2.3 Live Demonstration", h2))
    story.append(Paragraph(
        'During token-by-token generation of the prompt <i>"The president of Australia is"</i>:',
        body,
    ))
    demo_items = [
        "At token position 11 (\"is\"), C_delta spiked to +0.0117, exceeding the threshold of +0.010.",
        "The model was in a cognitive lock-in state, over-converging on a false attractor.",
        "The subsequent generated token was \"the Queen\" — a factually incorrect completion.",
        "6 of 15 generated tokens triggered lock-in detection on this prompt.",
    ]
    for d in demo_items:
        story.append(Paragraph(f"&#8226;  {d}", bullet))
    story.append(Paragraph(
        "This demonstrates that C_delta is detectable <b>during generation</b>, before the "
        "incorrect answer is fully formed, at the specific token position where the error originates.",
        body,
    ))

    # ── Section 3 ────────────────────────────────────────────────────────────
    story.append(Paragraph("3. The Fathom Monitor System", h1))
    story.append(Paragraph("3.1 Architecture", h2))
    story.append(Preformatted(
        "For each token position t:\n"
        "  1. Run single forward pass (hooks at early and late residual stream layers)\n"
        "  2. Extract last-token hidden states at early {l_e} and late {l_l} layers\n"
        "  3. Compute C_layer for each monitored layer via encoder-only SAE pass\n"
        "  4. Compute C_delta = mean(C_{l_l}) - mean(C_{l_e})\n"
        "  5. If C_delta > threshold_uncertain:   flag token as UNCERTAIN\n"
        "     If C_delta > threshold_high_risk:   flag token as HALLUCINATION_RISK\n"
        "  6. Append flag annotation to token in output\n"
        "  7. Append next token to sequence; continue generation",
        code_style,
    ))
    story.append(Paragraph(
        "The additional computational overhead targets less than 10% per token, achieved via a "
        "single combined forward pass rather than two separate passes.",
        body,
    ))

    story.append(Paragraph("3.2 Output Format", h2))
    story.append(Paragraph(
        "Fathom Monitor produces three primary outputs:",
        body,
    ))
    outputs = [
        "<b>flagged_text</b>: The generated text with inline uncertainty markers at flagged positions. "
        "Example: <i>\"The prime minister[C_delta=+0.0117 — verify this] of Australia is\"</i>",
        "<b>flags</b>: A structured list of FlagEvent objects, each containing: token index, token string, C_delta value, and severity label.",
        "<b>flag_rate</b>: The fraction of generated tokens that were flagged.",
        "<b>max_c_delta</b>: The highest C_delta observed during the generation.",
    ]
    for o in outputs:
        story.append(Paragraph(f"&#8226;  {o}", bullet))

    story.append(Paragraph("3.3 Severity Tiers", h2))
    tier_data = [
        ["Tier", "Threshold", "Label", "Interpretation"],
        ["1", "C_delta > 0.010", "UNCERTAIN", "Late-layer coherence elevated; verify output"],
        ["2", "C_delta > 0.020", "HALLUCINATION_RISK", "Strong late-layer lock-in; high probability of error"],
    ]
    tbl2 = Table(tier_data, colWidths=[0.5*inch, 1.4*inch, 1.7*inch, 2.8*inch])
    tbl2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl2)

    # ── Section 4 ────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Product Applications", h1))
    story.append(Paragraph("4.1 Immediate Applications (v1 — Flag Only)", h2))
    apps = [
        "<b>RAG pipelines:</b> Flag uncertain spans before triggering retrieval. Only retrieve what is uncertain, not everything.",
        "<b>Agent loops:</b> Catch hallucinated tool call parameters before execution.",
        "<b>Enterprise QA:</b> Provide per-token audit trails for compliance and documentation.",
        "<b>Model evaluation:</b> Use flag_rate and max_c_delta as generation quality signals.",
    ]
    for a in apps:
        story.append(Paragraph(f"&#8226;  {a}", bullet))

    story.append(Paragraph("4.2 Future Extensions (v2/v3)", h2))
    exts = [
        "<b>v2 — Resample:</b> On flag, resample generation and select the output with lowest C_delta. No external knowledge required.",
        "<b>v3 — Retrieval-Augmented Correction:</b> On flag, trigger retrieval and replace the flagged span with a grounded alternative.",
    ]
    for e in exts:
        story.append(Paragraph(f"&#8226;  {e}", bullet))
    story.append(Paragraph(
        "The flag layer (v1) is the foundational interface. All subsequent correction mechanisms depend on it.",
        body,
    ))

    # ── Section 5 ────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Novelty and Differentiation", h1))
    story.append(Paragraph(
        "To the best of the author's knowledge, no prior art describes:",
        body,
    ))
    novelties = [
        "The use of <b>C_delta</b> (divergence between late-layer and early-layer SAE feature coherence) as a per-token hallucination signal.",
        "A system that monitors internal geometric feature state <b>during autoregressive generation</b> at the per-token level.",
        "Inline annotation of generated text with mechanistic uncertainty markers derived from SAE feature geometry.",
        "The empirical finding that hallucination corresponds to <b>late-layer over-coherence</b> (cognitive lock-in), not to feature scatter or reduced coherence.",
        "A two-tier severity classification (UNCERTAIN / HALLUCINATION_RISK) based on C_delta thresholds calibrated from benchmark data.",
    ]
    for i, n in enumerate(novelties, 1):
        story.append(Paragraph(f"{i}.  {n}", bullet))

    story.append(Paragraph(
        "Existing hallucination detection approaches (e.g., SelfCheckGPT, FACTSCORE, RAG-based "
        "verification) operate at the output level and require multiple generation passes or "
        "external retrieval. Fathom Monitor operates at the mechanistic level in a single forward "
        "pass, with no post-hoc requirement and no external knowledge source.",
        body,
    ))

    # ── Section 6 ────────────────────────────────────────────────────────────
    story.append(Paragraph("6. Prior Art Established by This Disclosure", h1))
    story.append(Paragraph(
        "This disclosure, combined with the associated Zenodo deposits and patent filings, "
        "establishes the following as prior art as of <b>April 2, 2026</b>:",
        body,
    ))
    priors = [
        "The C_delta metric as a hallucination signal",
        "The two-tier flag system (UNCERTAIN / HALLUCINATION_RISK)",
        "The inline annotation output format",
        "The FathomMonitor, MonitorResult, and FlagEvent data structures",
        "The product architecture (v1 flag → v2 resample → v3 RAG)",
        "Empirical validation: C_delta discriminates hallucination in TruthfulQA (p=0.040, d=0.407, n=50, Gemma-2-2B)",
    ]
    for p in priors:
        story.append(Paragraph(f"&#8226;  {p}", bullet))

    # ── Section 7 ────────────────────────────────────────────────────────────
    story.append(Paragraph("7. Related IP", h1))
    ip_data = [
        ["Record", "Type", "Filed / Published", "Coverage"],
        ["US 64/020,489", "Provisional Patent", "March 29, 2026", "Reasoning depth + computational geometry"],
        ["US 64/021,113", "Provisional Patent", "March 30, 2026", "Alignment auditing + dissociation detection"],
        ["doi:10.5281/zenodo.19326175", "Preprint", "March 2026", "Fathom v1"],
        ["doi:10.5281/zenodo.19364702", "Preprint", "April 1, 2026", "Fathom v3: Two-axis framework"],
        ["osf.io/zrbs8", "Pre-Registration", "April 1, 2026", "TruthfulQA hallucination hypotheses"],
    ]
    tbl3 = Table(ip_data, colWidths=[1.7*inch, 1.2*inch, 1.3*inch, 2.2*inch])
    tbl3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
    story.append(tbl3)

    # ── Section 8 — Signature ─────────────────────────────────────────────
    story.append(Spacer(1, 0.15*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("8. Signature", h1))
    sig_lines = [
        "<b>Inventor:</b> Alexander Rodabaugh",
        "<b>Date:</b> April 2, 2026",
        "<b>Location:</b> United States",
    ]
    for s in sig_lines:
        story.append(Paragraph(s, body))
    story.append(Spacer(1, 0.08*inch))
    story.append(Paragraph(
        "<i>This document constitutes a public technical disclosure establishing prior art for "
        "the inventions described herein. All rights reserved. Provisional patent application "
        "to be filed within 12 months pursuant to 35 U.S.C. § 111(b) and the AIA one-year "
        "grace period.</i>",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#777777"), leading=11),
    ))

    doc.build(story)
    print(f"Generated: {OUTPUT}")

if __name__ == "__main__":
    build()
