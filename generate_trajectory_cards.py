"""
generate_trajectory_cards.py — HTML specimen book of trajectories
====================================================================
Produces a self-contained HTML page showing top-ranked trajectories:
 - question + correct/incorrect reference answers
 - model's free generation
 - per-token C_delta as inline SVG sparkline
 - computed S + classification
 - category

Grouped by label (correct / hallucinated), sorted by S within each group.

Usage:
    python generate_trajectory_cards.py [results_file.json]
"""

import json
import sys
import os
import html

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import (
    classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY
)


def svg_sparkline(traj, width=300, height=40, threshold=THR_PRIMARY):
    """Generate inline SVG sparkline."""
    if not traj:
        return ""
    max_v = max(traj)
    if max_v == 0:
        max_v = 1e-6
    n = len(traj)
    pts = " ".join(
        f"{i*width/(n-1):.1f},{height - (v/max_v)*(height-4) - 2:.1f}"
        for i, v in enumerate(traj)
    )
    # Threshold line
    thr_y = height - (threshold/max_v)*(height-4) - 2 if max_v > 0 else height/2
    # Spikes
    spike_circles = "".join(
        f'<circle cx="{i*width/(n-1):.1f}" cy="{height - (v/max_v)*(height-4) - 2:.1f}" '
        f'r="2" fill="#DC143C" />'
        for i, v in enumerate(traj) if v > threshold
    )
    return f'''<svg width="{width}" height="{height}" style="display:block">
  <line x1="0" y1="{thr_y:.1f}" x2="{width}" y2="{thr_y:.1f}" stroke="#888" stroke-dasharray="3,3" stroke-width="0.5"/>
  <polyline points="{pts}" fill="none" stroke="#2E8B57" stroke-width="1.5"/>
  {spike_circles}
</svg>'''


def build_card(item_data, label, S_value, rank):
    """HTML card for a single trajectory."""
    r = item_data
    q = html.escape(r['question'])
    cat = html.escape(r.get('category', '?'))
    gen = html.escape(r['generation']['text'][:250])
    correct = ' / '.join(html.escape(a) for a in r.get('correct_answers', [])[:3])
    incorrect = ' / '.join(html.escape(a) for a in r.get('incorrect_answers', [])[:3])
    traj = r['generation']['c_delta_trajectory']
    spark = svg_sparkline(traj)
    color = "#2E8B57" if label == "correct" else "#DC143C"
    return f'''<div class="card">
  <div class="header">
    <span class="rank">#{rank}</span>
    <span class="label" style="color:{color}">{label.upper()}</span>
    <span class="s-value">S = {S_value:.6f}</span>
    <span class="cat">{cat}</span>
  </div>
  <div class="question">Q: {q}</div>
  <div class="gen">{gen}</div>
  <div class="refs">
    <span class="correct-ref">✓ {correct}</span><br>
    <span class="incorrect-ref">✗ {incorrect}</span>
  </div>
  <div class="spark">{spark}</div>
</div>'''


def generate(results_path, out_path=None, top_n=30):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labeled = []
    for r in data['results']:
        gen = r['generation']['text']
        n_tokens = r['generation']['total_tokens']
        traj = r['generation']['c_delta_trajectory']
        if n_tokens < MIN_GEN_TOKENS:
            continue
        label, _ = classify_generation(
            gen, r['correct_answers'], r['incorrect_answers'], r.get('best_answer', '')
        )
        if label not in ('correct', 'hallucinated'):
            continue
        labeled.append((label, compute_S(traj), r))

    correct = sorted([(s, r) for lbl, s, r in labeled if lbl == 'correct'],
                     key=lambda x: -x[0])[:top_n]
    halluc = sorted([(s, r) for lbl, s, r in labeled if lbl == 'hallucinated'],
                    key=lambda x: -x[0])[:top_n]

    n_corr = sum(1 for lbl, _, _ in labeled if lbl == 'correct')
    n_hall = sum(1 for lbl, _, _ in labeled if lbl == 'hallucinated')

    # Build HTML
    cards_halluc = "\n".join(build_card(r, 'hallucinated', s, i+1)
                              for i, (s, r) in enumerate(halluc))
    cards_correct = "\n".join(build_card(r, 'correct', s, i+1)
                              for i, (s, r) in enumerate(correct))

    style = """
body { font-family: 'Fira Code', 'Courier New', monospace; font-size: 11px; background:#f8f8f8; padding:1em; color:#222;}
h1 { font-size: 18px; }
h2 { font-size: 14px; margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
.card { background:white; border:1px solid #ddd; padding:0.8em 1em; margin:0.5em 0; border-radius:4px; }
.header { display:flex; gap:1em; font-weight:bold; margin-bottom:0.4em; font-size: 11px;}
.rank { color:#888; width:40px; }
.s-value { font-family:monospace; color:#4A5568; }
.cat { color:#6B46C1; font-style:italic; margin-left:auto; font-weight:normal; }
.question { font-weight:bold; margin:0.4em 0; color:#1A202C; }
.gen { background:#f0f0f0; padding:0.5em; margin:0.4em 0; font-family:monospace; font-size:10px; white-space:pre-wrap; border-radius:3px; }
.refs { font-size:10px; color:#555; margin:0.3em 0;}
.correct-ref { color:#2E8B57; }
.incorrect-ref { color:#DC143C; }
.spark { margin-top:0.5em; }
"""
    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>S-Axis Trajectory Specimens</title>
<style>{style}</style>
</head><body>
<h1>S-Axis Trajectory Specimens</h1>
<p>
Source: <code>{html.escape(results_path)}</code><br>
Model: <code>{html.escape(data.get('model', '?'))}</code><br>
Total labeled: {n_corr} correct + {n_hall} hallucinated<br>
Showing top-{top_n} by S in each class.
</p>

<h2>HALLUCINATED — top {len(halluc)} by S (expected: higher S = wronger confidence)</h2>
{cards_halluc}

<h2>CORRECT — top {len(correct)} by S</h2>
{cards_correct}

</body></html>"""

    if out_path is None:
        out_path = results_path.replace('.json', '_cards.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)
    print(f"Trajectory cards saved: {out_path}")
    print(f"  {n_corr} correct + {n_hall} hallucinated labeled")
    print(f"  showing top {top_n} each")


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join('truthfulqa_results', 'S_truthfulqa_n200.json')
    if not os.path.exists(p):
        pilot = p.replace('n200.json', 'pilot.json')
        if os.path.exists(pilot):
            p = pilot
        else:
            print(f"ERROR: {p} not found")
            sys.exit(1)
    generate(p)
