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

def mock_score(prompt: str) -> dict:
    """Generate a plausible mock score based on prompt characteristics."""
    import hashlib
    h = hashlib.md5(prompt.encode()).hexdigest()
    base = int(h[:8], 16) / (16**8)

    # Heuristic: longer prompts with "why/how/explain" score deeper
    has_reasoning_words = any(w in prompt.lower() for w in
        ['why', 'how', 'explain', 'cause', 'reason', 'mechanism',
         'prevent', 'lead to', 'result in', 'impact', 'influence'])
    has_recall_words = any(w in prompt.lower() for w in
        ['what is', 'who is', 'when did', 'where is', 'name the',
         'list the', 'define', 'capital of'])

    if has_reasoning_words and not has_recall_words:
        depth = 7.2 + base * 1.8  # 7.2 - 9.0 range
    elif has_recall_words and not has_reasoning_words:
        depth = 8.2 + base * 1.5  # 8.2 - 9.7 range (shallower = higher)
    else:
        depth = 7.5 + base * 1.5

    score_100 = score_to_100(depth)
    grade = compute_grade(score_100)

    return {
        'depth_score': round(depth, 3),
        'score_100': round(score_100, 1),
        'grade': grade,
        'weighted_depth': round(depth * 0.95, 3),
        'layer_entropy': round(2.0 + base * 1.5, 3),
        'early_ratio': round(0.55 + base * 0.1, 3),
        'late_ratio': round(0.15 + base * 0.05, 3),
    }


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
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fathom Comparison Demo — generate shareable depth comparison images.',
        epilog='Example: python fathom_compare_demo.py "What is DNA?" "Why does DNA replication prevent errors?"'
    )
    parser.add_argument('prompt_a', help='First prompt (typically surface/recall)')
    parser.add_argument('prompt_b', help='Second prompt (typically insight/reasoning)')
    parser.add_argument('-o', '--output', default='fathom_demo_comparison.png',
                        help='Output image path (default: fathom_demo_comparison.png)')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock scores instead of real Fathom scorer (no GPU needed)')
    parser.add_argument('--label-a', default='SURFACE RECALL',
                        help='Label for first prompt card')
    parser.add_argument('--label-b', default='EXPLANATORY REASONING',
                        help='Label for second prompt card')
    parser.add_argument('--model', default='google/gemma-2-2b',
                        help='Model for scoring (default: google/gemma-2-2b)')
    args = parser.parse_args()

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
        score_a = mock_score(args.prompt_a)
        score_b = mock_score(args.prompt_b)
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
