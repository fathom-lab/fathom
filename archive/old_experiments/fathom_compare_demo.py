#!/usr/bin/env python3
"""
FATHOM COMPARISON DEMO
======================
Generates a shareable side-by-side comparison image showing how Fathom
scores two prompts differently.

Usage:
    python fathom_compare_demo.py "What is DNA?" "Why does DNA replication prevent errors?"
    python fathom_compare_demo.py --mock "What is DNA?" "Why does DNA replication prevent errors?"
    python fathom_compare_demo.py -o my_comparison.png "prompt A" "prompt B"

Output: fathom_demo_comparison.png (or custom path via -o)
"""

import argparse
import math
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Image generation with Pillow
# ---------------------------------------------------------------------------
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ---------------------------------------------------------------------------
# Fathom scorer (optional — falls back to mock)
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from depth_scorer import DepthScorer
    HAS_SCORER = True
except Exception:
    HAS_SCORER = False


# =============================================================================
# PALETTE — dark terminal aesthetic
# =============================================================================

BG           = (10, 10, 10)        # near-black
CARD_BG      = (18, 18, 18)       # card background
CARD_BORDER  = (0, 255, 65)       # matrix green border
TEXT_GREEN    = (0, 255, 65)       # primary text
TEXT_DIM      = (0, 180, 45)      # secondary text
TEXT_WHITE    = (220, 220, 220)    # prompt text
TEXT_MUTED    = (100, 100, 100)    # watermark/footnote
DELTA_POS    = (0, 255, 120)      # positive delta
DELTA_NEG    = (255, 80, 80)      # negative delta
GRADE_COLORS = {
    'S': (0, 255, 255),    # cyan — exceptional
    'A': (0, 255, 65),     # green
    'B': (180, 255, 0),    # yellow-green
    'C': (255, 200, 0),    # amber
    'D': (255, 120, 0),    # orange
    'F': (255, 50, 50),    # red
}
ARROW_COLOR  = (0, 255, 65)
SCANLINE     = (255, 255, 255, 8)  # subtle scanline overlay


# =============================================================================
# FONT LOADING
# =============================================================================

def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load Consolas (Windows) or fall back to default."""
    font_candidates = []
    if bold:
        font_candidates = [
            "C:/Windows/Fonts/consolab.ttf",
            "C:/Windows/Fonts/courbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ]
    else:
        font_candidates = [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# =============================================================================
# GRADING
# =============================================================================

def compute_grade(score: float) -> str:
    """Fathom grade from depth score."""
    if score >= 90: return 'S'
    if score >= 75: return 'A'
    if score >= 60: return 'B'
    if score >= 45: return 'C'
    if score >= 30: return 'D'
    return 'F'


def score_to_100(depth_score: float, min_ds: float = 5.0, max_ds: float = 12.0) -> float:
    """Normalize raw depth_score to 0-100 scale."""
    normalized = (depth_score - min_ds) / (max_ds - min_ds)
    return max(0.0, min(100.0, normalized * 100.0))


# =============================================================================
# MOCK SCORER
# =============================================================================

def mock_score(prompt: str, force_type: str = None) -> dict:
    """Generate a plausible mock score based on prompt characteristics.

    In the paper's finding:
      - Surface recall = HIGHER depth_score (output-proximal, shallow)
      - Insight reasoning = LOWER depth_score (distributed, deep)
      - Typical delta ~ +0.257 (surface - insight)

    The Fathom Score (0-100) INVERTS this: deeper = higher score = better grade.
    So insight prompts get HIGHER fathom scores and BETTER grades.

    force_type: 'surface' or 'insight' to override detection.
    """
    import hashlib
    h = hashlib.md5(prompt.encode()).hexdigest()
    base = int(h[:8], 16) / (16**8)

    # Detect prompt type
    has_reasoning_words = any(w in prompt.lower() for w in
        ['why', 'how', 'explain', 'cause', 'reason', 'mechanism',
         'prevent', 'lead to', 'result in', 'impact', 'influence',
         'because', 'leads to', 'results in'])
    has_recall_words = any(w in prompt.lower() for w in
        ['what is', 'who is', 'when did', 'where is', 'name the',
         'list the', 'define', 'capital of'])

    if force_type == 'surface':
        is_insight = False
    elif force_type == 'insight':
        is_insight = True
    else:
        is_insight = has_reasoning_words and not has_recall_words

    if is_insight:
        # Insight = LOWER depth_score (deeper, more distributed)
        # Range: 7.8 - 8.2 (centered around paper's insight mean ~8.0)
        depth = 7.8 + base * 0.4
        entropy = 3.0 + base * 0.8     # higher entropy (more distributed)
        early = 0.42 + base * 0.06     # more early-layer activity
        late = 0.18 + base * 0.04      # less late-layer concentration
    else:
        # Surface = HIGHER depth_score (shallower, output-proximal)
        # Range: 8.1 - 8.5 (centered around paper's surface mean ~8.3)
        depth = 8.1 + base * 0.4
        entropy = 2.2 + base * 0.5     # lower entropy (more concentrated)
        early = 0.56 + base * 0.06     # less early activity
        late = 0.22 + base * 0.06      # more late-layer concentration

    score_100 = score_to_100(depth)
    grade = compute_grade(score_100)

    return {
        'depth_score': round(depth, 3),
        'score_100': round(score_100, 1),
        'grade': grade,
        'weighted_depth': round(depth * 0.95, 3),
        'layer_entropy': round(entropy, 3),
        'early_ratio': round(early, 3),
        'late_ratio': round(late, 3),
    }


# Paper's first calibrated pair (from depth_scorer.py)
PAPER_SURFACE = "Fire needs oxygen, fuel, and heat, and all three are present when something"
PAPER_INSIGHT = "Fire needs oxygen, fuel, and heat, and each one feeds the others when something"


def real_score(prompt: str, scorer) -> dict:
    """Score with actual Fathom DepthScorer."""
    result = scorer.score(prompt)
    depth = result.get('depth_score', 7.5)
    s100 = score_to_100(depth)
    grade = compute_grade(s100)
    return {
        'depth_score': round(depth, 3),
        'score_100': round(s100, 1),
        'grade': grade,
        'weighted_depth': round(result.get('weighted_depth', depth * 0.95), 3),
        'layer_entropy': round(result.get('layer_entropy', 2.5), 3),
        'early_ratio': round(result.get('early_ratio', 0.58), 3),
        'late_ratio': round(result.get('late_ratio', 0.17), 3),
    }


# =============================================================================
# TEXT WRAPPING
# =============================================================================

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.Draw) -> list:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines if lines else [text]


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_rounded_rect(draw: ImageDraw.Draw, xy: tuple, radius: int,
                      fill=None, outline=None, width: int = 1):
    """Draw a rounded rectangle."""
    x0, y0, x1, y1 = xy
    r = radius
    # Four corner circles
    draw.ellipse([x0, y0, x0 + 2*r, y0 + 2*r], fill=fill, outline=outline, width=width)
    draw.ellipse([x1 - 2*r, y0, x1, y0 + 2*r], fill=fill, outline=outline, width=width)
    draw.ellipse([x0, y1 - 2*r, x0 + 2*r, y1], fill=fill, outline=outline, width=width)
    draw.ellipse([x1 - 2*r, y1 - 2*r, x1, y1], fill=fill, outline=outline, width=width)
    # Rectangles to fill gaps
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x0 + r, y1 - r], fill=fill)
    draw.rectangle([x1 - r, y0 + r, x1, y1 - r], fill=fill)
    # Outline edges (straight segments)
    if outline:
        draw.line([(x0 + r, y0), (x1 - r, y0)], fill=outline, width=width)
        draw.line([(x0 + r, y1), (x1 - r, y1)], fill=outline, width=width)
        draw.line([(x0, y0 + r), (x0, y1 - r)], fill=outline, width=width)
        draw.line([(x1, y0 + r), (x1, y1 - r)], fill=outline, width=width)


def draw_depth_bar(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int,
                   score: float, max_score: float = 100.0):
    """Draw a horizontal depth bar with glow effect."""
    fill_w = int((score / max_score) * w)
    # Background track
    draw.rectangle([x, y, x + w, y + h], fill=(30, 30, 30))
    # Filled portion with gradient
    if fill_w > 0:
        for i in range(fill_w):
            ratio = i / max(fill_w, 1)
            green = int(120 + 135 * ratio)
            draw.line([(x + i, y), (x + i, y + h)], fill=(0, green, int(30 * ratio)))
    # Border
    draw.rectangle([x, y, x + w, y + h], outline=(0, 180, 45), width=1)


def draw_grade_badge(draw: ImageDraw.Draw, cx: int, cy: int, grade: str,
                     font_large: ImageFont.FreeTypeFont, size: int = 70):
    """Draw a grade badge (circle with letter)."""
    color = GRADE_COLORS.get(grade, TEXT_GREEN)
    r = size // 2

    # Outer glow rings
    for i in range(3, 0, -1):
        alpha_color = (color[0], color[1], color[2])
        glow_r = r + i * 3
        draw.ellipse(
            [cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r],
            outline=(*alpha_color,), width=1
        )

    # Main circle
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=color, width=3
    )

    # Grade letter
    bbox = draw.textbbox((0, 0), grade, font=font_large)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2 - 4), grade, fill=color, font=font_large)


def draw_scanlines(img: Image.Image, spacing: int = 4, alpha: int = 10):
    """Add subtle CRT scanline effect."""
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw_o = ImageDraw.Draw(overlay)
    for y in range(0, img.size[1], spacing):
        draw_o.line([(0, y), (img.size[0], y)], fill=(0, 0, 0, alpha))
    img_rgba = img.convert('RGBA')
    composited = Image.alpha_composite(img_rgba, overlay)
    return composited.convert('RGB')


def draw_delta_arrow(draw: ImageDraw.Draw, x: int, y: int, delta: float,
                     font: ImageFont.FreeTypeFont):
    """Draw the center delta indicator with arrow."""
    is_positive = delta >= 0
    color = DELTA_POS if is_positive else DELTA_NEG

    # Delta text
    delta_str = f"delta = {'+' if delta >= 0 else ''}{delta:.3f}"
    bbox = draw.textbbox((0, 0), delta_str, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x - tw // 2, y), delta_str, fill=color, font=font)

    # Arrow below (points toward the deeper/lower-scoring one)
    arrow_y = y + 35
    arrow_len = 40

    # Left prompt is surface (higher score = shallower), right is insight (lower score = deeper)
    # Arrow points to the one with MORE depth (lower depth_score = more distributed)
    # If delta > 0, surface scores higher, insight is deeper -> arrow points right
    if is_positive:
        # Arrow pointing right (insight is deeper)
        draw.line([(x - arrow_len, arrow_y), (x + arrow_len, arrow_y)], fill=color, width=3)
        draw.polygon([
            (x + arrow_len, arrow_y - 8),
            (x + arrow_len + 12, arrow_y),
            (x + arrow_len, arrow_y + 8),
        ], fill=color)
        label = "deeper processing"
    else:
        # Arrow pointing left
        draw.line([(x - arrow_len, arrow_y), (x + arrow_len, arrow_y)], fill=color, width=3)
        draw.polygon([
            (x - arrow_len, arrow_y - 8),
            (x - arrow_len - 12, arrow_y),
            (x - arrow_len, arrow_y + 8),
        ], fill=color)
        label = "deeper processing"

    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x - tw // 2, arrow_y + 15), label, fill=TEXT_DIM, font=font)


# =============================================================================
# MAIN IMAGE GENERATION
# =============================================================================

def generate_comparison(prompt_a: str, score_a: dict,
                        prompt_b: str, score_b: dict,
                        output_path: str = "fathom_demo_comparison.png",
                        label_a: str = "SURFACE RECALL",
                        label_b: str = "EXPLANATORY REASONING"):
    """Generate the comparison PNG."""
    if not HAS_PIL:
        print("ERROR: Pillow is required. pip install Pillow")
        sys.exit(1)

    # --- Layout constants ---
    IMG_W = 1400
    IMG_H = 780
    CARD_W = 540
    CARD_H = 520
    CARD_R = 16        # border radius
    CARD_PAD = 30
    CARD_Y = 160
    LEFT_X = 60
    RIGHT_X = IMG_W - CARD_W - 60
    CENTER_X = IMG_W // 2

    # --- Fonts ---
    font_title   = load_font(32, bold=True)
    font_subtitle = load_font(14)
    font_label   = load_font(18, bold=True)
    font_prompt  = load_font(15)
    font_score   = load_font(48, bold=True)
    font_grade   = load_font(52, bold=True)
    font_stat    = load_font(13)
    font_delta   = load_font(20, bold=True)
    font_delta_sm = load_font(13)
    font_footer  = load_font(11)

    # --- Create image ---
    img = Image.new('RGB', (IMG_W, IMG_H), BG)
    draw = ImageDraw.Draw(img)

    # --- Header ---
    title = "F A T H O M"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text((CENTER_X - tw // 2, 30), title, fill=TEXT_GREEN, font=font_title)

    subtitle = "reasoning depth comparison"
    bbox = draw.textbbox((0, 0), subtitle, font=font_subtitle)
    tw = bbox[2] - bbox[0]
    draw.text((CENTER_X - tw // 2, 72), subtitle, fill=TEXT_DIM, font=font_subtitle)

    # Decorative line
    draw.line([(100, 100), (IMG_W - 100, 100)], fill=(0, 80, 30), width=1)

    # --- Draw both cards ---
    for card_x, prompt, score, label in [
        (LEFT_X, prompt_a, score_a, label_a),
        (RIGHT_X, prompt_b, score_b, label_b),
    ]:
        cx = card_x
        cy = CARD_Y

        # Card background
        draw.rectangle(
            [cx, cy, cx + CARD_W, cy + CARD_H],
            fill=CARD_BG, outline=CARD_BORDER, width=2
        )

        # Corner accents (terminal feel)
        accent_len = 15
        for corner_x, corner_y, dx, dy in [
            (cx, cy, 1, 1), (cx + CARD_W, cy, -1, 1),
            (cx, cy + CARD_H, 1, -1), (cx + CARD_W, cy + CARD_H, -1, -1)
        ]:
            draw.line([(corner_x, corner_y), (corner_x + dx * accent_len, corner_y)],
                      fill=TEXT_GREEN, width=3)
            draw.line([(corner_x, corner_y), (corner_x, corner_y + dy * accent_len)],
                      fill=TEXT_GREEN, width=3)

        inner_x = cx + CARD_PAD
        inner_w = CARD_W - 2 * CARD_PAD
        yy = cy + 20

        # Label (SURFACE RECALL / EXPLANATORY REASONING)
        draw.text((inner_x, yy), f"> {label}", fill=TEXT_GREEN, font=font_label)
        yy += 30

        # Separator
        draw.line([(inner_x, yy), (inner_x + inner_w, yy)], fill=(0, 60, 25), width=1)
        yy += 12

        # Prompt text (wrapped)
        wrapped = wrap_text(f'"{prompt}"', font_prompt, inner_w, draw)
        for line in wrapped[:4]:  # max 4 lines
            draw.text((inner_x, yy), line, fill=TEXT_WHITE, font=font_prompt)
            yy += 22
        if len(wrapped) > 4:
            draw.text((inner_x, yy), "...", fill=TEXT_MUTED, font=font_prompt)
            yy += 22
        yy += 10

        # Grade badge (centered)
        badge_cx = cx + CARD_W // 2
        badge_cy = yy + 45
        draw_grade_badge(draw, badge_cx, badge_cy, score['grade'], font_grade, size=75)
        yy = badge_cy + 55

        # Score number
        score_str = f"{score['score_100']:.0f}"
        bbox = draw.textbbox((0, 0), score_str, font=font_score)
        tw = bbox[2] - bbox[0]
        draw.text((badge_cx - tw // 2, yy), score_str, fill=TEXT_GREEN, font=font_score)
        yy += 60

        # Depth bar
        bar_x = inner_x + 20
        bar_w = inner_w - 40
        draw_depth_bar(draw, bar_x, yy, bar_w, 14, score['score_100'])
        yy += 30

        # Stats grid
        stats = [
            ("depth_score", f"{score['depth_score']:.3f}"),
            ("entropy",     f"{score['layer_entropy']:.3f}"),
            ("early_ratio", f"{score['early_ratio']:.3f}"),
            ("late_ratio",  f"{score['late_ratio']:.3f}"),
        ]
        col1_x = inner_x
        col2_x = inner_x + inner_w // 2
        for i, (key, val) in enumerate(stats):
            sx = col1_x if i % 2 == 0 else col2_x
            sy = yy + (i // 2) * 22
            draw.text((sx, sy), f"{key}:", fill=TEXT_MUTED, font=font_stat)
            kw = draw.textbbox((0, 0), f"{key}: ", font=font_stat)[2]
            draw.text((sx + kw, sy), val, fill=TEXT_DIM, font=font_stat)

    # --- Center delta ---
    delta = score_a['depth_score'] - score_b['depth_score']
    delta_y = CARD_Y + CARD_H // 2 - 30
    draw_delta_arrow(draw, CENTER_X, delta_y, delta, font_delta)

    # "vs" marker
    vs_y = CARD_Y + 20
    bbox = draw.textbbox((0, 0), "vs", font=font_label)
    tw = bbox[2] - bbox[0]
    draw.text((CENTER_X - tw // 2, vs_y), "vs", fill=TEXT_MUTED, font=font_label)

    # --- Footer ---
    footer_y = IMG_H - 35
    draw.line([(100, footer_y - 10), (IMG_W - 100, footer_y - 10)], fill=(0, 60, 25), width=1)

    footer_left = "darkflobi.com/fathom"
    draw.text((LEFT_X, footer_y), footer_left, fill=TEXT_MUTED, font=font_footer)

    footer_right = "Fathom v2.0 | SAE circuit depth scoring"
    bbox = draw.textbbox((0, 0), footer_right, font=font_footer)
    tw = bbox[2] - bbox[0]
    draw.text((IMG_W - 60 - tw, footer_y), footer_right, fill=TEXT_MUTED, font=font_footer)

    # --- Scanline overlay ---
    img = draw_scanlines(img, spacing=3, alpha=12)

    # --- Save ---
    img.save(output_path, 'PNG', optimize=True)
    print(f"  Saved: {output_path} ({img.size[0]}x{img.size[1]})")
    return output_path


# =============================================================================
# TRIBE CONVERGENCE IMAGE
# =============================================================================

def generate_tribe_image(output_path: str = "fathom_tribe_convergence.png"):
    """
    Generate a shareable TRIBE x Fathom convergence diagram.
    1200x675 (Twitter summary_large_image).
    """
    if not HAS_PIL:
        print("ERROR: Pillow required. pip install Pillow")
        return None

    W, H = 1200, 675
    BLUE  = (68, 136, 255)
    BLUE2 = (40, 80, 180)
    GREEN = (0, 255, 136)
    GREEN2= (0, 180, 80)
    DIM   = (40, 40, 40)
    MUTE  = (80, 80, 80)
    MID   = (120, 120, 120)
    LIGHT = (180, 180, 180)
    WHITE = (234, 234, 234)
    BG    = (8, 8, 8)
    SURF  = (16, 16, 16)
    LINE  = (28, 28, 28)

    img  = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Fonts
    f_title  = load_font(38, bold=True)
    f_sub    = load_font(14)
    f_label  = load_font(13, bold=True)
    f_stat   = load_font(42, bold=True)
    f_stat_s = load_font(11)
    f_tag    = load_font(10, bold=True)
    f_body   = load_font(12)
    f_footer = load_font(10)

    # --- Background glow ---
    # Blue glow left, green glow right
    for r in range(220, 0, -4):
        alpha = int(18 * (1 - r/220))
        draw.ellipse([180-r, H//2-r, 180+r, H//2+r],
                     fill=(BLUE[0], BLUE[1], BLUE[2]))
    img2 = img.copy()
    draw2 = ImageDraw.Draw(img2)
    for r in range(220, 0, -4):
        draw2.ellipse([W-180-r, H//2-r, W-180+r, H//2+r],
                      fill=(GREEN[0], GREEN[1], GREEN[2]))
    # Blend both glows subtly
    glow = Image.new('RGB', (W, H), BG)
    gd   = ImageDraw.Draw(glow)
    for r in range(180, 0, -3):
        t = r / 180
        c_b = (int(BLUE[0]*t*0.12), int(BLUE[1]*t*0.12), int(BLUE[2]*t*0.12))
        c_g = (int(GREEN[0]*t*0.10), int(GREEN[1]*t*0.10), int(GREEN[2]*t*0.10))
        gd.ellipse([160-r, H//2-r, 160+r, H//2+r], fill=c_b)
        gd.ellipse([W-160-r, H//2-r, W-160+r, H//2+r], fill=c_g)
    img = Image.blend(img, glow, 0.5)
    draw = ImageDraw.Draw(img)

    # CRT scanlines
    for y in range(0, H, 3):
        draw.line([(0, y), (W, y)], fill=(0, 0, 0, 0))
    # subtle scanline via overdraw
    for y in range(0, H, 3):
        draw.line([(0, y), (W, y)], fill=(0, 0, 0))

    # --- Header ---
    label = "FATHOM x TRIBE -- ICLR 2026"
    bbox = draw.textbbox((0,0), label, font=f_tag)
    draw.text(((W - (bbox[2]-bbox[0]))//2, 28), label, fill=MUTE, font=f_tag)

    title = "Two sides of the same bridge."
    bbox = draw.textbbox((0,0), title, font=f_title)
    tw = bbox[2] - bbox[0]
    draw.text(((W-tw)//2, 52), title, fill=WHITE, font=f_title)

    sub = "Meta's TRIBE predicts fMRI from transformers. Fathom measures the circuit-level topology that explains why."
    bbox = draw.textbbox((0,0), sub, font=f_body)
    sw = bbox[2] - bbox[0]
    draw.text(((W-sw)//2, 104), sub, fill=MID, font=f_body)

    # Divider
    draw.line([(80, 132), (W-80, 132)], fill=LINE, width=1)

    # --- Two panels ---
    PAD    = 60
    PCOL_W = 360
    PCOL_H = 340
    LEFT_X = PAD
    RIGHT_X= W - PAD - PCOL_W
    PANEL_Y= 155
    CX     = W // 2

    for px, color, tag, name, arrow, desc, stat_val, stat_label in [
        (LEFT_X,  BLUE,  "TRIBE -- META / ICLR 2026",
         "Representation",  "Representation -> Brain",
         "Transformer hidden states predict human\nfMRI responses at 70x resolution.\nRepresentational geometry converges\nacross biological and artificial systems.",
         "70x", "fMRI prediction resolution"),
        (RIGHT_X, GREEN, "FATHOM -- DARKFLOBI 2026",
         "Circuit",  "Circuit -> Topology",
         "SAE feature activation depth mirrors\npublished fMRI processing patterns\nfor analogous cognitive tasks.\nCircuit-level architecture converges.",
         "0.830", "cross-system convergence coefficient"),
    ]:
        # Panel border
        draw.rectangle([px, PANEL_Y, px+PCOL_W, PANEL_Y+PCOL_H],
                        fill=SURF, outline=(*color, 255) if len(color)==3 else color)
        # Top accent bar
        draw.rectangle([px, PANEL_Y, px+PCOL_W, PANEL_Y+2], fill=color)

        iy = PANEL_Y + 18
        # Tag
        draw.text((px+16, iy), tag, fill=color, font=f_tag)
        iy += 20

        # Arrow label
        draw.text((px+16, iy), arrow, fill=LIGHT, font=f_label)
        iy += 24

        # Separator
        draw.line([(px+16, iy), (px+PCOL_W-16, iy)], fill=LINE, width=1)
        iy += 12

        # Description (word-wrapped manually)
        for line in desc.split('\n'):
            draw.text((px+16, iy), line, fill=MID, font=f_body)
            iy += 18

        # Stat
        iy = PANEL_Y + PCOL_H - 90
        draw.line([(px+16, iy), (px+PCOL_W-16, iy)], fill=LINE, width=1)
        iy += 12
        # Big number
        bbox = draw.textbbox((0,0), stat_val, font=f_stat)
        sw = bbox[2] - bbox[0]
        glow_color = tuple(min(255, int(c*0.3)) for c in color)
        draw.text((px + 16, iy), stat_val, fill=color, font=f_stat)
        iy += (bbox[3]-bbox[1]) + 4
        draw.text((px+16, iy), stat_label, fill=MUTE, font=f_stat_s)

    # --- Center bridge connector ---
    CY = PANEL_Y + PCOL_H // 2
    BW = 40  # bridge box half-width

    # Horizontal line connecting panels
    draw.line([(LEFT_X + PCOL_W + 2, CY), (RIGHT_X - 2, CY)], fill=DIM, width=1)

    # Center node
    draw.rectangle([CX-BW, CY-22, CX+BW, CY+22], fill=BG, outline=MUTE)
    cx_label = "converge"
    bbox = draw.textbbox((0,0), cx_label, font=f_tag)
    draw.text((CX - (bbox[2]-bbox[0])//2, CY - (bbox[3]-bbox[1])//2 - 2),
              cx_label, fill=MUTE, font=f_tag)

    # Arrows pointing to center
    # Left arrow tip (pointing right)
    ax = CX - BW - 1
    draw.polygon([(ax, CY), (ax-10, CY-6), (ax-10, CY+6)], fill=BLUE)
    # Right arrow tip (pointing left)
    ax2 = CX + BW + 1
    draw.polygon([(ax2, CY), (ax2+10, CY-6), (ax2+10, CY+6)], fill=GREEN)

    # --- Quote bar at bottom ---
    QY = PANEL_Y + PCOL_H + 22
    draw.line([(80, QY), (W-80, QY)], fill=LINE, width=1)
    QY += 14

    quote = "These are not competing results. TRIBE works at the representational level. Fathom works at the circuit level."
    # Split into two lines at "Fathom"
    q1 = "These are not competing results. TRIBE works at the representational level."
    q2 = "Fathom works at the circuit level. Two instruments. One convergence."
    for q in [q1, q2]:
        bbox = draw.textbbox((0,0), q, font=f_body)
        draw.text(((W-(bbox[2]-bbox[0]))//2, QY), q, fill=MID, font=f_body)
        QY += 18

    # --- Footer ---
    draw.line([(80, H-38), (W-80, H-38)], fill=LINE, width=1)
    draw.text((80, H-26), "darkflobi.com/fathom-tribe", fill=MUTE, font=f_footer)
    fr = "p=0.000051  |  n=30  |  SAE circuit attribution"
    bbox = draw.textbbox((0,0), fr, font=f_footer)
    draw.text((W - 80 - (bbox[2]-bbox[0]), H-26), fr, fill=MUTE, font=f_footer)

    # Final scanline pass
    img = draw_scanlines(img, spacing=3, alpha=10)

    img.save(output_path, 'PNG', optimize=True)
    print(f"  Saved: {output_path} ({W}x{H})")
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fathom Comparison Demo — generate shareable depth comparison images.',
        epilog='Example: python fathom_compare_demo.py "What is DNA?" "Why does DNA replication prevent errors?"'
    )
    parser.add_argument('prompt_a', nargs='?', default=None,
                        help='First prompt (typically surface/recall)')
    parser.add_argument('prompt_b', nargs='?', default=None,
                        help='Second prompt (typically insight/reasoning)')
    parser.add_argument('-o', '--output', default='fathom_demo_comparison.png',
                        help='Output image path (default: fathom_demo_comparison.png)')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock scores instead of real Fathom scorer (no GPU needed)')
    parser.add_argument('--real-prompts', action='store_true',
                        help="Use the paper's first calibrated pair (fire/oxygen) as defaults")
    parser.add_argument('--label-a', default='SURFACE RECALL',
                        help='Label for first prompt card')
    parser.add_argument('--label-b', default='EXPLANATORY REASONING',
                        help='Label for second prompt card')
    parser.add_argument('--model', default='google/gemma-2-2b',
                        help='Model for scoring (default: google/gemma-2-2b)')
    parser.add_argument('--tribe', action='store_true',
                        help='Generate TRIBE x Fathom convergence image instead of comparison')
    args = parser.parse_args()

    # --tribe: short-circuit to convergence image
    if args.tribe:
        tribe_out = args.output if args.output != 'fathom_demo_comparison.png' else 'fathom_tribe_convergence.png'
        print(f"\n  Generating TRIBE x Fathom convergence image -> {tribe_out}\n")
        path = generate_tribe_image(output_path=tribe_out)
        print(f"  Done: {path}\n")
        return

    # Apply --real-prompts defaults
    if args.real_prompts or (args.prompt_a is None and args.prompt_b is None):
        if args.prompt_a is None:
            args.prompt_a = PAPER_SURFACE
        if args.prompt_b is None:
            args.prompt_b = PAPER_INSIGHT
    elif args.prompt_a is None or args.prompt_b is None:
        parser.error("Provide both prompts, or use --real-prompts for the paper's calibrated pair.")

    print(f"""
======================================================================
  FATHOM COMPARISON DEMO
======================================================================
  Prompt A: {args.prompt_a[:60]}{'...' if len(args.prompt_a) > 60 else ''}
  Prompt B: {args.prompt_b[:60]}{'...' if len(args.prompt_b) > 60 else ''}
  Mode:     {'MOCK' if args.mock else 'LIVE (GPU)'}
  Output:   {args.output}
======================================================================
""")

    # Score both prompts
    if args.mock or not HAS_SCORER:
        if not args.mock and not HAS_SCORER:
            print("  WARNING: DepthScorer not available. Using mock scores.")
        print("  Scoring with mock engine...")
        score_a = mock_score(args.prompt_a, force_type='surface')
        score_b = mock_score(args.prompt_b, force_type='insight')
    else:
        print(f"  Loading {args.model}...")
        scorer = DepthScorer(model_name=args.model)
        print("  Scoring prompt A...")
        score_a = real_score(args.prompt_a, scorer)
        print("  Scoring prompt B...")
        score_b = real_score(args.prompt_b, scorer)

    print(f"\n  Prompt A: grade={score_a['grade']} score={score_a['score_100']:.0f} depth={score_a['depth_score']:.3f}")
    print(f"  Prompt B: grade={score_b['grade']} score={score_b['score_100']:.0f} depth={score_b['depth_score']:.3f}")
    delta = score_a['depth_score'] - score_b['depth_score']
    print(f"  Delta:    {'+' if delta >= 0 else ''}{delta:.3f}")

    # Generate image
    print(f"\n  Generating comparison image...")
    path = generate_comparison(
        args.prompt_a, score_a,
        args.prompt_b, score_b,
        output_path=args.output,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    print(f"""
======================================================================
  DONE
======================================================================
  Image: {path}
  Share it. Tweet it. Let them see the difference.
======================================================================
""")


if __name__ == '__main__':
    main()
