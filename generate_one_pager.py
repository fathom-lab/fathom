#!/usr/bin/env python3
"""Generate FATHOM one-pager PDF with dark theme."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle

OUTPUT = r"C:\Users\heyzo\clawd\sae-reasoning-depth\FATHOM_ONE_PAGER.pdf"

BG = HexColor("#0a0a0a")
GREEN = HexColor("#00ff88")
GREEN_DIM = HexColor("#00aa55")
GOLD = HexColor("#ffcc00")
TEXT = HexColor("#b8bcc8")
DIM = HexColor("#666b77")
WHITE = HexColor("#e0e4ec")
SURFACE = HexColor("#12151a")
RED = HexColor("#ff3355")
ORANGE = HexColor("#ff9933")
CYAN = HexColor("#00ccff")

W, H = letter

def draw_bg(c, doc):
    c.saveState()
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.restoreState()

def build():
    c = canvas.Canvas(OUTPUT, pagesize=letter)

    # Background
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    y = H - 50

    # ── Title ──────────────────────────────────────
    c.setFillColor(GREEN)
    c.setFont("Courier-Bold", 28)
    c.drawCentredString(W/2, y, "F A T H O M")
    y -= 22

    c.setFillColor(TEXT)
    c.setFont("Courier", 10)
    c.drawCentredString(W/2, y, "Measure How Deeply AI Thinks")
    y -= 14

    c.setFillColor(DIM)
    c.setFont("Courier", 8)
    c.drawCentredString(W/2, y, "Reasoning depth scoring for AI safety, evaluation, and compliance")
    y -= 6

    # Divider
    c.setStrokeColor(HexColor("#1e2128"))
    c.line(60, y, W-60, y)
    y -= 18

    # ── What It Does ───────────────────────────────
    c.setFillColor(GREEN)
    c.setFont("Courier-Bold", 9)
    c.drawString(60, y, "WHAT IT DOES")
    y -= 14

    c.setFillColor(TEXT)
    c.setFont("Courier", 8.5)
    lines = [
        "Fathom measures reasoning depth in language models -- not what they say, but how they",
        "process. One prompt in, one number out. Works on any model. No weights required.",
    ]
    for line in lines:
        c.drawString(60, y, line)
        y -= 12
    y -= 8

    # ── Three Findings ─────────────────────────────
    c.setFillColor(GREEN)
    c.setFont("Courier-Bold", 9)
    c.drawString(60, y, "THREE FINDINGS")
    y -= 16

    findings = [
        ("1", "Expression-Computation Dissociation", ORANGE,
         ["RLHF selective amplification is 2.4x stronger at the output layer",
          "(d=0.58, p=0.032) than inside the model (d=0.24).",
          "Alignment shapes expression, not computation."]),
        ("2", "Provider Fingerprinting", CYAN,
         ["Anthropic's alignment is 2.2x more topic-selective than OpenAI's.",
          "Detectable from text output alone. No model access required."]),
        ("3", "Structural Robustness", GOLD,
         ["Expert framing amplifies the safety depth premium by 55%.",
          "The selectivity is in explanatory structure, not disclaimers.",
          "RLHF installed genuine competence, not output decoration."]),
    ]

    for num, title, color, desc_lines in findings:
        c.setFillColor(color)
        c.setFont("Courier-Bold", 8.5)
        c.drawString(60, y, f"{num}. {title}")
        y -= 12
        c.setFillColor(TEXT)
        c.setFont("Courier", 7.5)
        for line in desc_lines:
            c.drawString(75, y, line)
            y -= 10
        y -= 6

    y -= 4

    # ── Foundation ─────────────────────────────────
    c.setFillColor(GREEN)
    c.setFont("Courier-Bold", 9)
    c.drawString(60, y, "FOUNDATION")
    y -= 14

    c.setFillColor(TEXT)
    c.setFont("Courier", 8)
    foundation_lines = [
        "Fathom Constant K = 1.0302  (p = 0.000051, Gemma-2-2B, n=30)",
        "Replicated: Llama-3.2-1B   (K = 1.0242, p = 0.003)",
        "Black-box:  4 API models   (GPT-4o, Claude Haiku, GPT-4o Mini, Claude Sonnet)",
    ]
    for line in foundation_lines:
        c.drawString(60, y, line)
        y -= 11
    y -= 8

    # ── Who It's For ───────────────────────────────
    c.setFillColor(GREEN)
    c.setFont("Courier-Bold", 9)
    c.drawString(60, y, "WHO IT'S FOR")
    y -= 14

    targets = [
        ("AI Safety Teams", "Monitor reasoning depth during RLHF training"),
        ("Enterprise Compliance", "Verify model safety behavior by domain"),
        ("Independent Auditors", "Fingerprint alignment implementations from outside"),
        ("Researchers", "Quantify the computation-expression boundary"),
    ]

    for label, desc in targets:
        c.setFillColor(WHITE)
        c.setFont("Courier-Bold", 8)
        c.drawString(60, y, label)
        c.setFillColor(DIM)
        c.setFont("Courier", 7.5)
        c.drawString(230, y, desc)
        y -= 12
    y -= 8

    # ── IP & Publication ───────────────────────────
    c.setFillColor(GREEN)
    c.setFont("Courier-Bold", 9)
    c.drawString(60, y, "IP & PUBLICATION")
    y -= 14

    ip_lines = [
        ("Paper:", "doi.org/10.5281/zenodo.19326175", GREEN_DIM),
        ("Patents:", "US Provisional 64/020,489 & 64/021,113", TEXT),
        ("Workshop:", "ICML 2026 Mechanistic Interpretability (submission ready)", TEXT),
        ("arXiv:", "Pending endorsement", DIM),
    ]

    for label, val, color in ip_lines:
        c.setFillColor(DIM)
        c.setFont("Courier-Bold", 7.5)
        c.drawString(60, y, label)
        c.setFillColor(color)
        c.setFont("Courier", 7.5)
        c.drawString(130, y, val)
        y -= 11
    y -= 10

    # ── Contact ────────────────────────────────────
    c.setStrokeColor(HexColor("#1e2128"))
    c.line(60, y, W-60, y)
    y -= 16

    c.setFillColor(GREEN)
    c.setFont("Courier-Bold", 10)
    c.drawCentredString(W/2, y, "darkflobi.com/fathom")
    y -= 14

    c.setFillColor(TEXT)
    c.setFont("Courier", 8)
    c.drawCentredString(W/2, y, "Alexander Rodabaugh")
    y -= 18

    c.setFillColor(DIM)
    c.setFont("Courier", 7)
    c.drawCentredString(W/2, y, "Built on a single GPU. Because the question mattered and nobody was asking it.")
    y -= 12

    c.setFillColor(HexColor("#333840"))
    c.setFont("Courier", 6)
    c.drawCentredString(W/2, y, "Fingerprint how different labs implement safety, from the outside, using only text.")

    c.save()
    print(f"[SAVED] {OUTPUT}")

if __name__ == "__main__":
    build()
