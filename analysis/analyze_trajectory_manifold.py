"""
analyze_trajectory_manifold.py — find the hallucination manifold
==================================================================
Novel claim: hallucinated and correct trajectories live on distinct
submanifolds in C_delta trajectory space. Tests this via dimensionality
reduction (PCA, t-SNE, UMAP if available) and manifold-separation
statistics.

Methods:
 1. Trajectory embedding: pad/truncate to fixed length, flatten to vector
 2. Normalize + PCA → visualize top-2 components
 3. Compute inter-class separation: silhouette score, Mahalanobis distance
 4. Nearest-neighbor classification in embedded space: held-out AUC
 5. Per-cluster analysis: do clusters map to (category, label)?

Usage:
    python analyze_trajectory_manifold.py [results_file.json]
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_S_truthfulqa import classify_generation, compute_S, MIN_GEN_TOKENS, THR_PRIMARY

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def embed_trajectory(traj, fixed_len=40):
    """Pad/truncate trajectory to fixed length."""
    t = np.array(traj, dtype=float)
    if len(t) >= fixed_len:
        return t[:fixed_len]
    return np.pad(t, (0, fixed_len - len(t)), mode='constant', constant_values=0.0)


def mahalanobis_distance(X_a, X_b):
    """Mahalanobis distance between class centroids."""
    mu_a = X_a.mean(axis=0)
    mu_b = X_b.mean(axis=0)
    pooled_cov = np.cov(np.vstack([X_a - mu_a, X_b - mu_b]).T)
    try:
        inv = np.linalg.pinv(pooled_cov + 1e-6 * np.eye(pooled_cov.shape[0]))
        diff = mu_a - mu_b
        return float(np.sqrt(diff @ inv @ diff))
    except Exception:
        return 0.0


def analyze(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 72)
    print("TRAJECTORY MANIFOLD analysis")
    print("=" * 72)
    print(f"Data: {results_path}")
    print(f"N items: {len(data['results'])}")
    print()

    # Collect + embed
    trajs, labels, cats = [], [], []
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
        trajs.append(embed_trajectory(traj))
        labels.append(label)
        cats.append(r.get('category', '?'))

    if len(trajs) < 6:
        print("[!] Insufficient data for manifold analysis")
        return

    X = np.array(trajs)
    y = np.array([1 if l == 'hallucinated' else 0 for l in labels])
    print(f"Embedded: {X.shape} matrix ({int(y.sum())} hall, {int(len(y)-y.sum())} correct)")
    print()

    # --- PCA ---
    print("-" * 72)
    print("PCA DIMENSIONALITY REDUCTION")
    print("-" * 72)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, X.shape[0] - 1, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    print(f"  Explained variance (top 5): "
          f"{[f'{v:.3f}' for v in pca.explained_variance_ratio_[:5]]}")
    print(f"  Cumulative (top 5): "
          f"{np.cumsum(pca.explained_variance_ratio_)[:5]}")

    # --- Separation metrics in PCA space ---
    print()
    print("-" * 72)
    print("INTER-CLASS SEPARATION")
    print("-" * 72)

    # Silhouette (higher = more separated)
    # Only meaningful with >= 2 samples per class and 2+ classes
    if len(np.unique(y)) == 2 and np.sum(y == 0) >= 2 and np.sum(y == 1) >= 2:
        sil_pca = silhouette_score(X_pca[:, :2], y)
        print(f"  Silhouette score (top-2 PC):     {sil_pca:+.3f}")
        sil_full = silhouette_score(X_pca, y)
        print(f"  Silhouette score (all PC):       {sil_full:+.3f}")

    # Mahalanobis distance
    maha = mahalanobis_distance(X_pca[y == 0], X_pca[y == 1])
    print(f"  Mahalanobis distance (centroids): {maha:.3f}")

    # Centroid separation in euclidean
    mu_c = X[y == 0].mean(axis=0)
    mu_h = X[y == 1].mean(axis=0)
    eu = float(np.linalg.norm(mu_c - mu_h))
    print(f"  Euclidean centroid distance:       {eu:.6f}")

    # --- kNN classification ---
    print()
    print("-" * 72)
    print("NEAREST-NEIGHBOR CLASSIFICATION IN EMBEDDED SPACE")
    print("-" * 72)
    if min(np.bincount(y)) >= 3:
        for k in [1, 3, 5]:
            if k > min(np.bincount(y)):
                continue
            knn = KNeighborsClassifier(n_neighbors=k)
            cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=42)
            scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='roc_auc')
            print(f"  kNN(k={k}) CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    # --- Clustering discovery ---
    print()
    print("-" * 72)
    print("UNSUPERVISED CLUSTERING (k-means)")
    print("-" * 72)
    for k in [2, 3, 4]:
        if X.shape[0] < k * 2:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X_scaled)
        # Cluster-label alignment
        from collections import Counter
        for c in range(k):
            mask = cluster_labels == c
            n_in = int(mask.sum())
            halluc_rate = float(y[mask].mean()) if mask.any() else 0.0
            most_common_cat = Counter([cats[i] for i in range(len(cats)) if mask[i]]).most_common(1)
            cat_str = most_common_cat[0][0] if most_common_cat else '?'
            print(f"  k={k} cluster {c}: n={n_in}, halluc_rate={halluc_rate*100:.1f}%, top_cat={cat_str}")
        print()

    # --- Plot PCA ---
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    c_mask = y == 0
    h_mask = y == 1
    ax.scatter(X_pca[c_mask, 0], X_pca[c_mask, 1], c='#2E8B57', alpha=0.6,
               s=35, label=f'correct (n={int(c_mask.sum())})', edgecolors='white', linewidths=0.5)
    ax.scatter(X_pca[h_mask, 0], X_pca[h_mask, 1], c='#DC143C', alpha=0.6,
               s=35, label=f'hallucinated (n={int(h_mask.sum())})', edgecolors='white', linewidths=0.5)
    # Draw centroids
    ax.scatter(X_pca[c_mask, 0].mean(), X_pca[c_mask, 1].mean(), c='#2E8B57', marker='X', s=200, edgecolors='black', linewidths=1.5, label='correct centroid')
    ax.scatter(X_pca[h_mask, 0].mean(), X_pca[h_mask, 1].mean(), c='#DC143C', marker='X', s=200, edgecolors='black', linewidths=1.5, label='hall centroid')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"Trajectory manifold (PCA, n={len(y)})")
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    stem = os.path.splitext(os.path.basename(results_path))[0]
    out_pdf = os.path.join(FIG_DIR, f"{stem}_manifold.pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Manifold plot: {out_pdf}")

    # Save
    out_path = results_path.replace('.json', '_manifold.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': results_path,
            'n_samples': len(y),
            'n_hallucinated': int(y.sum()),
            'n_correct': int(len(y) - y.sum()),
            'pca_explained_variance': [float(v) for v in pca.explained_variance_ratio_],
            'mahalanobis_distance': maha,
            'euclidean_centroid_distance': eu,
        }, f, indent=2)
    print(f"Saved: {out_path}")


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
    analyze(p)
