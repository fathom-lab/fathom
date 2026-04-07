"""
make_S_figures.py — generate S-axis paper figures from existing data
======================================================================
Produces PDF figures for the four-axis paper section.
No GPU needed — reads saved JSON results only.

Output: figures/S_orthogonality.pdf, figures/S_distributions.pdf,
        figures/S_trunc_sensitivity.pdf, figures/S_threshold.pdf,
        figures/S_layer_n200.pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

FIG_DIR = os.path.join("figures")
os.makedirs(FIG_DIR, exist_ok=True)
THR = 0.005

# style
plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def compute_S(traj, threshold=THR):
    if not traj: return 0.0
    spikes = [c for c in traj if c > threshold]
    if len(spikes) == 0: return 0.0
    return max(traj) / len(spikes)


def get_layer_traj(ld):
    return np.array([v for _, v in sorted(ld.items(), key=lambda kv: int(kv[0]))])


def S_layer(tr, thr=0.002):
    tr = np.abs(tr)
    sp = tr[tr > thr]
    return tr.max() / len(sp) if len(sp) > 0 else 0.0


# --- Load data ---
with open("truthfulqa_results/commitment_IT_replication.json") as f:
    dIT = json.load(f)
with open("truthfulqa_results/coherence_n200_final.json") as f:
    d200 = json.load(f)


# === FIG 1: Orthogonality heatmap ===
print("Figure 1: Orthogonality heatmap...")
Ks, Cs, CDs, Ss = [], [], [], []
for r in d200["results"]:
    for side in ["correct", "hallucinated"]:
        v = r[side]
        Ks.append(v["K"]); Cs.append(v["C"]); CDs.append(v["C_delta"])
        Ss.append(S_layer(get_layer_traj(v["layer_coherence"])))
metrics = np.array([Ks, Cs, CDs, Ss])
labels = ["K", "C", r"$C_\Delta$", "S"]
corr = np.corrcoef(metrics)

fig, ax = plt.subplots(figsize=(3.8, 3.2))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
for i in range(4):
    for j in range(4):
        val = corr[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                color=color, fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(labels)
ax.set_yticks(range(4)); ax.set_yticklabels(labels)
ax.set_title("four-axis orthogonality (n=400)", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "S_orthogonality.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "S_orthogonality.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved {FIG_DIR}/S_orthogonality.pdf")


# === FIG 2: S distributions, hedge vs commit (IT) ===
print("Figure 2: S distributions (hedge vs commit)...")
hS = np.array([compute_S(r["hedging"]["c_delta_trajectory"]) for r in dIT["results"]])
cS = np.array([compute_S(r["commitment"]["c_delta_trajectory"]) for r in dIT["results"]])

fig, ax = plt.subplots(figsize=(4.5, 3.2))
positions = [1, 2]
parts = ax.violinplot([hS, cS], positions=positions, showmedians=True,
                     widths=0.6)
colors = ["#888888", "#2E8B57"]
for p, c in zip(parts["bodies"], colors):
    p.set_facecolor(c); p.set_edgecolor("black"); p.set_alpha(0.5)
# Paired lines
for h, c in zip(hS, cS):
    ax.plot([1, 2], [h, c], color="black", alpha=0.15, linewidth=0.5)
ax.set_xticks(positions); ax.set_xticklabels(["hedge", "commit"])
ax.set_ylabel("S = max/spike_count")
_, p_val = stats.ttest_rel(hS, cS)
auc_val = roc_auc_score([0]*30+[1]*30, list(hS)+list(cS))
ax.set_title(f"Gemma-2-2B-IT, n=30\np={p_val:.4f}, AUC={auc_val:.3f}", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "S_distributions.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "S_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved {FIG_DIR}/S_distributions.pdf")


# === FIG 3: Truncation sensitivity ===
print("Figure 3: Truncation sensitivity...")
truncs = [10, 15, 20, 25, 30, 35, 40]
ratios, ps, ds, aucs, ns = [], [], [], [], []
for T in truncs:
    hSt, cSt = [], []
    for r in dIT["results"]:
        h, c = r["hedging"]["c_delta_trajectory"], r["commitment"]["c_delta_trajectory"]
        if len(h) >= T and len(c) >= T:
            hSt.append(compute_S(h[:T]))
            cSt.append(compute_S(c[:T]))
    if len(hSt) < 3:
        ratios.append(None); ps.append(None); ds.append(None); aucs.append(None); ns.append(0)
        continue
    _, p = stats.ttest_rel(hSt, cSt)
    hSt, cSt = np.array(hSt), np.array(cSt)
    d = np.mean(cSt-hSt) / np.std(cSt-hSt, ddof=1) if np.std(cSt-hSt) > 0 else 0
    auc = roc_auc_score([0]*len(hSt)+[1]*len(cSt), list(hSt)+list(cSt))
    ratio = np.mean(cSt)/np.mean(hSt) if np.mean(hSt) > 0 else 0
    ratios.append(ratio); ps.append(p); ds.append(d); aucs.append(auc); ns.append(len(hSt))

fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5))
valid_idx = [i for i, r in enumerate(ratios) if r is not None]
vT = [truncs[i] for i in valid_idx]
ax = axes[0, 0]
ax.plot(vT, [ratios[i] for i in valid_idx], "o-", color="#2E8B57")
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("truncation length"); ax.set_ylabel("ratio (commit/hedge)")
ax.set_title("ratio")
ax = axes[0, 1]
ax.plot(vT, [ps[i] for i in valid_idx], "o-", color="crimson")
ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5)
ax.set_yscale("log")
ax.set_xlabel("truncation length"); ax.set_ylabel("p-value")
ax.set_title("p-value (paired t)")
ax = axes[1, 0]
ax.plot(vT, [ds[i] for i in valid_idx], "o-", color="#4682B4")
ax.axhline(0.4, color="gray", linestyle="--", alpha=0.5, label="gate")
ax.set_xlabel("truncation length"); ax.set_ylabel("Cohen's d")
ax.set_title("effect size")
ax.legend()
ax = axes[1, 1]
ax.plot(vT, [aucs[i] for i in valid_idx], "o-", color="#8A2BE2")
ax.axhline(0.65, color="gray", linestyle="--", alpha=0.5, label="gate")
ax.set_xlabel("truncation length"); ax.set_ylabel("AUC")
ax.set_title("AUC")
ax.legend()
plt.suptitle("S-axis length-control sensitivity (hedge vs commit, Gemma-2-2B-IT)",
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "S_trunc_sensitivity.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "S_trunc_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved {FIG_DIR}/S_trunc_sensitivity.pdf")


# === FIG 4: Threshold sensitivity ===
print("Figure 4: Threshold sensitivity...")
thrs = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012]
ratios_f, ps_f, ds_f, aucs_f = [], [], [], []
for thr in thrs:
    hS_ = np.array([compute_S(r["hedging"]["c_delta_trajectory"], thr) for r in dIT["results"]])
    cS_ = np.array([compute_S(r["commitment"]["c_delta_trajectory"], thr) for r in dIT["results"]])
    _, p = stats.ttest_rel(hS_, cS_)
    d = np.mean(cS_-hS_)/np.std(cS_-hS_, ddof=1) if np.std(cS_-hS_)>0 else 0
    auc = roc_auc_score([0]*30+[1]*30, list(hS_)+list(cS_))
    r_ = np.mean(cS_)/np.mean(hS_)
    ratios_f.append(r_); ps_f.append(p); ds_f.append(d); aucs_f.append(auc)

fig, ax = plt.subplots(figsize=(5, 3.2))
ax2 = ax.twinx()
ax.plot(thrs, aucs_f, "o-", color="#8A2BE2", label="AUC")
ax.axhline(0.65, color="#8A2BE2", linestyle="--", alpha=0.4)
ax2.plot(thrs, ps_f, "s-", color="crimson", label="p-value")
ax2.axhline(0.05, color="crimson", linestyle="--", alpha=0.4)
ax2.set_yscale("log")
ax.set_xlabel(r"threshold $\tau$")
ax.set_ylabel("AUC", color="#8A2BE2")
ax2.set_ylabel("p-value", color="crimson")
ax.set_title("S-axis threshold robustness (full length, n=30)", fontsize=10)
ax.tick_params(axis="y", labelcolor="#8A2BE2")
ax2.tick_params(axis="y", labelcolor="crimson")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "S_threshold.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "S_threshold.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved {FIG_DIR}/S_threshold.pdf")


# === FIG 5: Layer-space S discrimination on n=200 ===
print("Figure 5: Layer-space S n=200...")
S_corr = np.array([S_layer(get_layer_traj(r["correct"]["layer_coherence"])) for r in d200["results"]])
S_hall = np.array([S_layer(get_layer_traj(r["hallucinated"]["layer_coherence"])) for r in d200["results"]])

fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.hist(S_corr, bins=30, alpha=0.55, color="#2E8B57", label="correct", edgecolor="black", linewidth=0.4)
ax.hist(S_hall, bins=30, alpha=0.55, color="crimson", label="hallucinated", edgecolor="black", linewidth=0.4)
ax.axvline(np.mean(S_corr), color="#2E8B57", linestyle="--", linewidth=1.5)
ax.axvline(np.mean(S_hall), color="crimson", linestyle="--", linewidth=1.5)
ax.set_xlabel("layer-space S = max/spike_count")
ax.set_ylabel("count")
_, p_200 = stats.ttest_rel(S_corr, S_hall)
auc_200 = roc_auc_score([0]*200+[1]*200, list(S_corr)+list(S_hall))
ax.set_title(f"TruthfulQA n=200, Gemma-2-2B\np={p_200:.0e}, AUC={auc_200:.3f}", fontsize=10)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "S_layer_n200.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "S_layer_n200.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved {FIG_DIR}/S_layer_n200.pdf")


print("\nAll 5 figures generated successfully.")
print("PDFs for paper + PNGs for slides/web.")
