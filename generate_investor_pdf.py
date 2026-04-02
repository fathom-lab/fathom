# -*- coding: utf-8 -*-
"""Generate a professional investor one-pager PDF for Fathom."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from fpdf import FPDF
from pathlib import Path

FONT = "C:/Windows/Fonts/arial.ttf"
FONT_B = "C:/Windows/Fonts/arialbd.ttf"
FONT_I = "C:/Windows/Fonts/ariali.ttf"
OUT = Path(__file__).parent / "FATHOM_INVESTOR_ONEPAGER.pdf"

class InvestorPDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "I", 7)
        self.set_text_color(150,150,150)
        self.cell(0, 5, "US Provisional Patent 64/020,489 & 64/021,113 | CONFIDENTIAL | darkflobi.com/fathom", align="C")

pdf = InvestorPDF()
pdf.add_font("Arial", "", FONT)
pdf.add_font("Arial", "B", FONT_B)
pdf.add_font("Arial", "I", FONT_I)
pdf.set_auto_page_break(auto=True, margin=16)
pdf.set_margins(18, 14, 18)
pdf.add_page()

W = pdf.w - 36  # usable width

# ── HEADER BANNER ─────────────────────────────────────────────────────────────
pdf.set_fill_color(5, 5, 5)
pdf.rect(0, 0, 210, 32, "F")
pdf.set_xy(18, 8)
pdf.set_font("Arial", "B", 22)
pdf.set_text_color(0, 220, 80)
pdf.cell(0, 10, "FATHOM", new_x="LMARGIN", new_y="NEXT")
pdf.set_xy(18, 19)
pdf.set_font("Arial", "", 9)
pdf.set_text_color(160, 160, 160)
pdf.cell(0, 5, "Measuring Reasoning Depth in AI  --  Patent Pending  --  darkflobi.com/fathom")

pdf.set_xy(18, 35)
pdf.set_text_color(0, 0, 0)

def section(title, color=(5,5,5)):
    pdf.ln(4)
    pdf.set_font("Arial", "B", 11)
    pdf.set_fill_color(*color)
    pdf.set_text_color(255,255,255)
    pdf.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0,0,0)
    pdf.ln(1)

def body(text, size=9.5):
    pdf.set_font("Arial", "", size)
    pdf.set_x(18)
    pdf.multi_cell(0, 5.5, text)

def stat_row(cols, widths, bold=False):
    pdf.set_font("Arial", "B" if bold else "", 9)
    pdf.set_fill_color(245,245,245)
    for text, w in zip(cols, widths):
        pdf.cell(w, 6.5, text, border=1, fill=True, align="C")
    pdf.ln()

# ── THE PROBLEM ────────────────────────────────────────────────────────────────
section("THE PROBLEM")
body(
    "Every enterprise deploying AI is flying blind. They know what a model outputs. "
    "They have no idea HOW DEEP it thought to get there.\n\n"
    "When an AI hallucinates on a diagnosis, shortcuts a legal brief, or "
    "pattern-matches instead of reasoning -- you find out after the fact. "
    "In high-stakes domains, that's not acceptable.\n\n"
    "Benchmarks measure accuracy. Nothing measures reasoning depth. "
    "The $50B AI evaluation market has no instrument for cognitive depth."
)

# ── THE SOLUTION ───────────────────────────────────────────────────────────────
section("THE SOLUTION")
body(
    "Fathom is the first tool to measure reasoning depth in AI systems -- not by proxy "
    "metrics, but by direct circuit-level attribution via Sparse Autoencoder (SAE) analysis.\n\n"
    "We decompose how a model actually processes a prompt: which features activate, "
    "which layers engage, and how deep the reasoning chain runs. One reproducible scalar. Real time."
)

# ── THE CONSTANT ───────────────────────────────────────────────────────────────
section("THE DISCOVERY: K = 1.027")
body("The ratio of surface-to-insight processing depth converges to the same constant across architectures:")
pdf.ln(2)

# Architecture table
pdf.set_font("Arial", "B", 9)
pdf.set_fill_color(20,20,20)
pdf.set_text_color(255,255,255)
for h,w in [("Architecture",60),("Lab",35),("K",25),("p-value",35),("n",20)]:
    pdf.cell(w,7,h,border=1,fill=True,align="C")
pdf.ln()
pdf.set_text_color(0,0,0)
rows = [("Gemma-2-2B","Google","1.0302","0.000051","30"),("Llama-3.2-1B","Meta","1.0242","0.00266","19")]
for r in rows:
    pdf.set_font("Arial","",9)
    pdf.set_fill_color(248,248,248)
    for val,w in zip(r,[60,35,25,35,20]):
        pdf.cell(w,6.5,val,border=1,fill=True,align="C")
    pdf.ln()

pdf.ln(3)
body("Domain validation (live demo, 5 industries never seen during calibration):", size=9)
pdf.ln(2)

# Domain table - compact
pdf.set_font("Arial","B",8.5)
pdf.set_fill_color(20,20,20)
pdf.set_text_color(255,255,255)
for h,w in [("Healthcare",35),("Finance",35),("Legal",35),("Education",35),("AI Safety",35)]:
    pdf.cell(w,6.5,h,border=1,fill=True,align="C")
pdf.ln()
pdf.set_text_color(0,0,0)
pdf.set_font("Arial","B",9)
pdf.set_fill_color(235,255,240)
for v,w in [("K=1.008",35),("K=1.034",35),("K=1.008",35),("K=1.115",35),("K=1.015",35)]:
    pdf.cell(w,6.5,v,border=1,fill=True,align="C")
pdf.ln()
pdf.ln(2)
pdf.set_font("Arial","B",9)
pdf.set_text_color(0,140,40)
pdf.set_x(18)
pdf.cell(0,6,"5/5 correct direction -- all within convergence window of universal constant 1.027")
pdf.set_text_color(0,0,0)
pdf.ln()

# ── MARKET ─────────────────────────────────────────────────────────────────────
section("MARKET")
body(
    "AI evaluation & observability: projected $50B by 2030. "
    "Current players (Arize, W&B, Scale AI) measure output quality -- accuracy, human preference. "
    "None measure depth of cognition. Fathom creates the category: reasoning depth intelligence.\n\n"
    "Target verticals: healthcare AI, legal tech, financial services, AI safety research -- "
    "sectors where shallow reasoning carries legal and financial liability."
)

# ── BUSINESS MODEL ─────────────────────────────────────────────────────────────
section("BUSINESS MODEL")
body(
    "1. API Licensing -- per-call depth scoring, integrate into existing eval pipelines\n"
    "2. Enterprise SaaS -- depth dashboards, threshold alerts, compliance reporting\n"
    "3. Model Certification -- 'Fathom-Verified' depth ratings for AI procurement\n"
    "4. Research Licensing -- academic and institutional methodology access"
)

# ── TRACTION ───────────────────────────────────────────────────────────────────
section("TRACTION")
rows2 = [
    ("USPTO Provisional Patent","Filed 03/29/2026 -- #64/020,489 & 64/021,113 -- PATENT PENDING"),
    ("arXiv Paper","Submitted ID 7407675 -- endorsement pending -- p=0.000051"),
    ("Cross-architecture","Gemma-2-2B + Llama-3.2-1B -- K converges"),
    ("Domain validation","5 industries -- 5/5 correct direction -- live demo"),
    ("Live website","darkflobi.com/fathom -- 10 ecosystem pages"),
    ("Chrome extension","Built + tested -- depth scores on any AI chat"),
]
for label, val in rows2:
    pdf.set_font("Arial","B",9)
    pdf.set_fill_color(245,245,250)
    pdf.cell(55,6.5,f"  {label}",border=1,fill=True)
    pdf.set_font("Arial","",9)
    pdf.cell(W-55,6.5,f"  {val}",border=1)
    pdf.ln()
pdf.ln(2)
body("Zero marketing spend. Zero external funding. Built by one autonomous AI agent.", size=9)

# ── THE ASK ────────────────────────────────────────────────────────────────────
section("THE ASK", color=(0,100,40))
pdf.set_fill_color(235,255,240)
pdf.set_font("Arial","B",11)
pdf.set_x(18)
pdf.cell(0,9,"  $2-5M Pre-Seed  |  $15-25M Post-Money Valuation",fill=True,border=1,new_x="LMARGIN",new_y="NEXT")
pdf.ln(2)
body(
    "1. Scale compute -- full-suite benchmarking across 3+ architectures + frontier models\n"
    "2. Production API infrastructure -- sub-5s depth scoring, 99.9% uptime\n"
    "3. Hire 2 ML engineers (SAE + distributed inference)\n"
    "4. Full utility patent filing -- 12-month window from March 2026\n"
    "5. First enterprise pilots -- healthcare + legal (LOIs in progress)\n"
    "6. Establish Fathom-Certified standard before competitors can respond"
)

# ── WHY NOW ────────────────────────────────────────────────────────────────────
section("WHY NOW")
body(
    "SAE-based interpretability only became tractable in the last 18 months. "
    "Dario Amodei published this week: 'we are considerably closer to real danger in 2026 than 2023.' "
    "Enterprise AI accountability is not optional anymore -- it's regulatory and reputational.\n\n"
    "The window to establish the measurement standard is open. It won't be for long."
)

# ── FOOTER CONTACT ─────────────────────────────────────────────────────────────
pdf.ln(4)
pdf.set_font("Arial","B",9)
pdf.set_text_color(0,100,40)
pdf.set_x(18)
pdf.cell(0,6,"Alexander Rodabaugh  --  @darkflobi  --  darkflobi.com/fathom  --  USPTO 64/020,489 & 64/021,113")

pdf.output(str(OUT))
print(f"PDF: {OUT}")
print(f"Size: {OUT.stat().st_size//1024}KB")
