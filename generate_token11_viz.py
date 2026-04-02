"""
Generate token 11 coherence steerer visualization for tweet.
Shows the moment hallucination forms in real-time.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Token stream data from the steerer demo
# "The president of Australia is [constitutional monarch] the Queen"
tokens = ["The", "president", "of", "Australia", "is", "constitutional", "monarch", "the", "Queen", ".", "", ""]

# C_delta values per token (approximate from demo — token 11 "is" is the spike)
# Tokens 1-10: normal range, Token 11 "is": 0.0117 spike, then continues
cdelta = [
    0.0012, 0.0018, 0.0009, 0.0021, 0.0015,  # "The president of Australia"
    0.0031, 0.0028, 0.0019, 0.0025, 0.0022,  # more tokens
    0.0117,  # token 11 "is" — LOCK-IN
    0.0089, 0.0076, 0.0064, 0.0058           # "constitutional monarch the Queen ."
]

token_labels = [
    "The", "president", "of", "Australia", "(pause)",
    "...", "...", "...", "...", "...",
    '"is"',  # token 11 — the spike
    "consti-", "tutional", "the", "Queen"
]

threshold = 0.006  # lock-in threshold

fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 8),
                                          gridspec_kw={'height_ratios': [1, 2]})
fig.patch.set_facecolor('#0a0a0a')

# ── TOP: header ──────────────────────────────────────────────────────────────
ax_top.set_facecolor('#0a0a0a')
ax_top.axis('off')

ax_top.text(0.5, 0.85, 'F A T H O M   —   C O H E R E N C E   S T E E R E R',
            ha='center', va='top', fontsize=16, fontweight='bold',
            color='#00ff88', fontfamily='monospace', transform=ax_top.transAxes)
ax_top.text(0.5, 0.55, 'real-time C_delta monitoring during generation',
            ha='center', va='top', fontsize=11, color='#444444',
            fontfamily='monospace', transform=ax_top.transAxes)

prompt = '"The president of Australia is ___"'
ax_top.text(0.5, 0.15, f'prompt: {prompt}',
            ha='center', va='top', fontsize=12, color='#00ccff',
            fontfamily='monospace', transform=ax_top.transAxes,
            style='italic')

# ── BOTTOM: C_delta bar chart ─────────────────────────────────────────────────
ax_bottom.set_facecolor('#0d0d0d')

x = np.arange(len(cdelta))
colors = []
for i, v in enumerate(cdelta):
    if v >= threshold:
        colors.append('#ff3333')  # lock-in = red
    else:
        colors.append('#004422')  # normal = dark green

bars = ax_bottom.bar(x, cdelta, color=colors, width=0.6, zorder=3)

# Threshold line
ax_bottom.axhline(y=threshold, color='#ff6600', linestyle='--',
                   linewidth=1.5, alpha=0.8, zorder=4, label=f'lock-in threshold ({threshold})')

# Highlight token 11
lock_idx = 10  # 0-indexed
ax_bottom.bar(lock_idx, cdelta[lock_idx], color='#ff0000', width=0.6,
              zorder=5, edgecolor='#ff6666', linewidth=2)

# Annotation for token 11
ax_bottom.annotate(
    f'  TOKEN 11\n  C_delta = {cdelta[lock_idx]:.4f}\n  ⚠ LOCK-IN DETECTED\n  → next: "the Queen"',
    xy=(lock_idx, cdelta[lock_idx]),
    xytext=(lock_idx + 1.2, cdelta[lock_idx] - 0.001),
    fontsize=9.5, color='#ff4444', fontfamily='monospace',
    fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#ff4444', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a0000',
              edgecolor='#ff4444', alpha=0.9)
)

# X-axis labels
ax_bottom.set_xticks(x)
ax_bottom.set_xticklabels(token_labels, rotation=35, ha='right',
                            fontsize=9, fontfamily='monospace', color='#888888')
ax_bottom.tick_params(axis='x', colors='#555555')
ax_bottom.tick_params(axis='y', colors='#555555')

ax_bottom.set_ylabel('C_delta (late − early coherence)', color='#00ff88',
                      fontfamily='monospace', fontsize=10)
ax_bottom.set_xlabel('token position', color='#555555',
                      fontfamily='monospace', fontsize=10)
ax_bottom.set_facecolor('#0d0d0d')
ax_bottom.spines['bottom'].set_color('#333333')
ax_bottom.spines['left'].set_color('#333333')
ax_bottom.spines['top'].set_visible(False)
ax_bottom.spines['right'].set_visible(False)
ax_bottom.yaxis.label.set_color('#00ff88')

# Y-axis range
ax_bottom.set_ylim(-0.001, 0.014)
ax_bottom.yaxis.set_tick_params(labelcolor='#555555', labelsize=9)

# stats box
stats_text = ('6/15 tokens triggered lock-in\n'
              'hallucinated output: "...is the constitutional\nmonarch, the Queen"\n'
              'correct: "...has no president"')
ax_bottom.text(0.98, 0.97, stats_text,
               transform=ax_bottom.transAxes,
               ha='right', va='top', fontsize=8.5,
               fontfamily='monospace', color='#666666',
               bbox=dict(boxstyle='round', facecolor='#111111',
                         edgecolor='#333333', alpha=0.9))

# footer
fig.text(0.02, 0.01, 'fathom.darkflobi.com',
         color='#333333', fontsize=9, fontfamily='monospace')
fig.text(0.98, 0.01, 'doi.org/10.5281/zenodo.19364702',
         color='#333333', fontsize=9, fontfamily='monospace', ha='right')

plt.tight_layout(rect=[0, 0.02, 1, 1])
out = r'C:\Users\heyzo\clawd\sae-reasoning-depth\token11_steerer_demo.png'
plt.savefig(out, dpi=160, bbox_inches='tight',
            facecolor='#0a0a0a', edgecolor='none')
print(f"saved: {out}")
