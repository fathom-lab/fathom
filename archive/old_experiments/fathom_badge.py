"""
FATHOM BADGE GENERATOR — Shareable SVG Badges and Report Cards
================================================================

Generates visual artifacts from Fathom scores: SVG badges for embedding,
full report cards for sharing. Pure Python using xml.etree.ElementTree
for SVG generation — zero external dependencies.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional


# ── Grade colors ─────────────────────────────────────────────────────

GRADE_COLORS = {
    "S": {"bg": "#ffd700", "text": "#1a1a00", "glow": "#ffd70066"},
    "A": {"bg": "#00ff00", "text": "#001a00", "glow": "#00ff0044"},
    "B": {"bg": "#00e5ff", "text": "#001a1e", "glow": "#00e5ff44"},
    "C": {"bg": "#ffeb3b", "text": "#1a1800", "glow": "#ffeb3b44"},
    "D": {"bg": "#ff9800", "text": "#1a0e00", "glow": "#ff980044"},
    "F": {"bg": "#ff3d00", "text": "#1a0500", "glow": "#ff3d0044"},
}


def _get_colors(grade: str) -> dict:
    return GRADE_COLORS.get(grade, GRADE_COLORS["F"])


# ── SVG Badge ────────────────────────────────────────────────────────

def generate_badge(fathom_score, output_path: str) -> str:
    """Generate a compact SVG badge showing grade and score.

    Args:
        fathom_score: A FathomScore dataclass instance.
        output_path: Path to write the SVG file.

    Returns:
        The output path.
    """
    grade = fathom_score.grade
    score = int(round(fathom_score.score))
    colors = _get_colors(grade)

    width = 160
    height = 40

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })

    # Background — two-tone pill
    # Left section: "FATHOM" label
    left_w = 70
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(left_w), "height": str(height),
        "rx": "5", "ry": "5",
        "fill": "#1a1a1a",
    })
    # Right section: grade + score
    ET.SubElement(svg, "rect", {
        "x": str(left_w), "y": "0",
        "width": str(width - left_w), "height": str(height),
        "rx": "5", "ry": "5",
        "fill": colors["bg"],
    })
    # Cover the inner corners with overlapping rects
    ET.SubElement(svg, "rect", {
        "x": str(left_w - 5), "y": "0",
        "width": "10", "height": str(height),
        "fill": "#1a1a1a",
    })
    ET.SubElement(svg, "rect", {
        "x": str(left_w), "y": "0",
        "width": "5", "height": str(height),
        "fill": colors["bg"],
    })

    # "FATHOM" text
    label = ET.SubElement(svg, "text", {
        "x": str(left_w // 2),
        "y": str(height // 2 + 1),
        "text-anchor": "middle",
        "dominant-baseline": "central",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "11",
        "font-weight": "700",
        "fill": "#888",
        "letter-spacing": "2",
    })
    label.text = "FATHOM"

    # Grade + Score text
    score_text = ET.SubElement(svg, "text", {
        "x": str(left_w + (width - left_w) // 2),
        "y": str(height // 2 + 1),
        "text-anchor": "middle",
        "dominant-baseline": "central",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "16",
        "font-weight": "900",
        "fill": colors["text"],
    })
    score_text.text = f"{grade} {score}"

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    return str(output_path)


# ── Full Report Card ─────────────────────────────────────────────────

def generate_report_card(prompt: str, fathom_score, output_path: str,
                         depth_result=None) -> str:
    """Generate a full SVG report card with all metrics.

    Args:
        prompt: The analyzed prompt text.
        fathom_score: A FathomScore dataclass instance.
        output_path: Path to write the SVG file.
        depth_result: Optional DepthResult for additional metrics.

    Returns:
        The output path.
    """
    grade = fathom_score.grade
    score = int(round(fathom_score.score))
    colors = _get_colors(grade)

    # Truncate prompt
    prompt_display = prompt[:97] + "..." if len(prompt) > 100 else prompt
    # Escape XML entities
    prompt_display = prompt_display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    width = 480
    height = 340

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })

    # Background
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(width), "height": str(height),
        "rx": "8", "ry": "8",
        "fill": "#0a0a0a",
        "stroke": "#222",
        "stroke-width": "1",
    })

    # Header bar
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(width), "height": "48",
        "rx": "8", "ry": "8",
        "fill": "#141414",
    })
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "40",
        "width": str(width), "height": "8",
        "fill": "#141414",
    })

    # Header text
    h_title = ET.SubElement(svg, "text", {
        "x": "20", "y": "30",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "18", "font-weight": "700",
        "fill": "#00ff00", "letter-spacing": "6",
    })
    h_title.text = "FATHOM"

    h_ver = ET.SubElement(svg, "text", {
        "x": str(width - 20), "y": "30",
        "text-anchor": "end",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "11", "fill": "#444",
    })
    h_ver.text = "v2.0"

    # ── Prompt text ──────────────────────────────────────────────
    y = 74
    p_label = ET.SubElement(svg, "text", {
        "x": "20", "y": str(y),
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "10", "fill": "#555",
        "letter-spacing": "2",
    })
    p_label.text = "PROMPT"

    y += 18
    # Wrap prompt into lines of ~55 chars
    lines = _wrap_text(prompt_display, 55)
    for line in lines[:2]:
        p_text = ET.SubElement(svg, "text", {
            "x": "20", "y": str(y),
            "font-family": "Consolas, 'Courier New', monospace",
            "font-size": "12", "fill": "#999",
        })
        p_text.text = line
        y += 16

    # ── Grade circle ─────────────────────────────────────────────
    y += 10
    cx = 60
    cy = y + 35

    # Glow
    ET.SubElement(svg, "circle", {
        "cx": str(cx), "cy": str(cy), "r": "38",
        "fill": "none", "stroke": colors["glow"],
        "stroke-width": "4",
    })
    # Ring
    ET.SubElement(svg, "circle", {
        "cx": str(cx), "cy": str(cy), "r": "32",
        "fill": "none", "stroke": colors["bg"],
        "stroke-width": "3",
    })
    # Grade letter
    g_text = ET.SubElement(svg, "text", {
        "x": str(cx), "y": str(cy + 2),
        "text-anchor": "middle",
        "dominant-baseline": "central",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "36", "font-weight": "900",
        "fill": colors["bg"],
    })
    g_text.text = grade

    # Score number
    s_num = ET.SubElement(svg, "text", {
        "x": str(cx), "y": str(cy + 50),
        "text-anchor": "middle",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "14", "font-weight": "700",
        "fill": "#ccc",
    })
    s_num.text = f"{score}/100"

    # ── Metrics column ───────────────────────────────────────────
    mx = 140
    my = y - 2

    metrics = [
        ("SCORE", f"{score}/100"),
        ("GRADE", grade),
    ]

    if depth_result:
        ds = depth_result.depth_score
        if not math.isnan(ds):
            metrics.append(("DEPTH", f"{ds:.3f}"))
        if depth_result.weighted_depth_score is not None:
            metrics.append(("W.DEPTH", f"{depth_result.weighted_depth_score:.3f}"))
        if depth_result.layer_entropy is not None:
            metrics.append(("ENTROPY", f"{depth_result.layer_entropy:.3f}"))
        metrics.append(("FEATURES", str(depth_result.feature_count)))
        metrics.append(("EARLY", f"{depth_result.early_ratio:.0%}"))
        metrics.append(("MID", f"{depth_result.mid_ratio:.0%}"))
        metrics.append(("LATE", f"{depth_result.late_ratio:.0%}"))

    col_w = 160
    for i, (label, value) in enumerate(metrics):
        col = i // 5
        row = i % 5
        rx = mx + col * col_w
        ry = my + row * 22

        lbl = ET.SubElement(svg, "text", {
            "x": str(rx), "y": str(ry),
            "font-family": "Consolas, 'Courier New', monospace",
            "font-size": "10", "fill": "#555",
        })
        lbl.text = label

        val = ET.SubElement(svg, "text", {
            "x": str(rx + 72), "y": str(ry),
            "font-family": "Consolas, 'Courier New', monospace",
            "font-size": "12", "font-weight": "700",
            "fill": "#00ff00",
        })
        val.text = value

    # ── Interpretation ───────────────────────────────────────────
    interp_y = height - 60
    ET.SubElement(svg, "rect", {
        "x": "20", "y": str(interp_y - 6),
        "width": str(width - 40), "height": "1",
        "fill": "#222",
    })

    interp_lines = _wrap_text(fathom_score.interpretation, 62)
    for i, line in enumerate(interp_lines[:2]):
        it = ET.SubElement(svg, "text", {
            "x": "20", "y": str(interp_y + 14 + i * 14),
            "font-family": "Consolas, 'Courier New', monospace",
            "font-size": "10", "fill": "#777",
        })
        it.text = line

    # ── Footer ───────────────────────────────────────────────────
    ft = ET.SubElement(svg, "text", {
        "x": str(width - 20), "y": str(height - 12),
        "text-anchor": "end",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "9", "fill": "#333",
    })
    ft.text = "Analyzed by FATHOM v2.0"

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    return str(output_path)


# ── Batch Report ─────────────────────────────────────────────────────

def batch_report(results: List[dict], output_path: str) -> str:
    """Generate an SVG comparison table for multiple scored prompts.

    Args:
        results: List of dicts with keys: prompt, fathom_score (FathomScore),
                 depth_result (optional DepthResult).
        output_path: Path to write the SVG file.

    Returns:
        The output path.
    """
    if not results:
        raise ValueError("No results to report")

    row_h = 36
    header_h = 70
    width = 700
    height = header_h + len(results) * row_h + 20

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })

    # Background
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(width), "height": str(height),
        "rx": "6", "ry": "6",
        "fill": "#0a0a0a",
        "stroke": "#222", "stroke-width": "1",
    })

    # Title
    title = ET.SubElement(svg, "text", {
        "x": "20", "y": "30",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "16", "font-weight": "700",
        "fill": "#00ff00", "letter-spacing": "4",
    })
    title.text = "FATHOM BATCH REPORT"

    count_text = ET.SubElement(svg, "text", {
        "x": str(width - 20), "y": "30",
        "text-anchor": "end",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "11", "fill": "#444",
    })
    count_text.text = f"{len(results)} prompts"

    # Column headers
    cols = [
        (20, "#"),
        (50, "GRADE"),
        (110, "SCORE"),
        (170, "DEPTH"),
        (240, "ENTROPY"),
        (320, "PROMPT"),
    ]
    for cx, label in cols:
        h = ET.SubElement(svg, "text", {
            "x": str(cx), "y": "56",
            "font-family": "Consolas, 'Courier New', monospace",
            "font-size": "10", "fill": "#555",
            "letter-spacing": "1",
        })
        h.text = label

    # Divider
    ET.SubElement(svg, "rect", {
        "x": "20", "y": "62",
        "width": str(width - 40), "height": "1",
        "fill": "#222",
    })

    # Rows
    for i, item in enumerate(results):
        fs = item["fathom_score"]
        dr = item.get("depth_result")
        prompt = item.get("prompt", "")
        colors = _get_colors(fs.grade)

        ry = header_h + i * row_h + 14

        # Alternating row background
        if i % 2 == 0:
            ET.SubElement(svg, "rect", {
                "x": "4", "y": str(ry - 12),
                "width": str(width - 8), "height": str(row_h),
                "fill": "#111", "rx": "2",
            })

        # Row number
        _add_text(svg, 20, ry, str(i + 1), "#444", "10")

        # Grade
        _add_text(svg, 50, ry, fs.grade, colors["bg"], "14", weight="900")

        # Score
        _add_text(svg, 110, ry, str(int(round(fs.score))), "#ccc", "12", weight="700")

        # Depth
        depth_str = "--"
        if dr and hasattr(dr, "depth_score") and not math.isnan(dr.depth_score):
            depth_str = f"{dr.depth_score:.3f}"
        _add_text(svg, 170, ry, depth_str, "#888", "11")

        # Entropy
        ent_str = "--"
        if dr and hasattr(dr, "layer_entropy") and dr.layer_entropy is not None:
            ent_str = f"{dr.layer_entropy:.3f}"
        _add_text(svg, 240, ry, ent_str, "#888", "11")

        # Prompt (truncated)
        p_short = prompt[:42] + "..." if len(prompt) > 45 else prompt
        p_short = p_short.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        _add_text(svg, 320, ry, p_short, "#777", "10")

    # Footer
    ft = ET.SubElement(svg, "text", {
        "x": str(width - 20), "y": str(height - 8),
        "text-anchor": "end",
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": "9", "fill": "#333",
    })
    ft.text = "FATHOM v2.0"

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    return str(output_path)


# ── Helpers ──────────────────────────────────────────────────────────

def _add_text(svg, x, y, text, fill, size, weight="400"):
    t = ET.SubElement(svg, "text", {
        "x": str(x), "y": str(y),
        "font-family": "Consolas, 'Courier New', monospace",
        "font-size": size, "font-weight": weight,
        "fill": fill,
    })
    t.text = text


def _wrap_text(text: str, max_chars: int) -> List[str]:
    """Wrap text into lines of max_chars, breaking at word boundaries."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = current + " " + word if current else word
    if current:
        lines.append(current)
    return lines
