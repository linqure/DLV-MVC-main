"""
Utility functions for DLC-MVC.

Includes:
  - CIR (Consistency-guided Intra-view Refinement)
  - Target distribution computation (Student's t + sharpening)
  - Seed setup
  - Model checkpoint save/load
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CIR: Consistency-guided Intra-view Refinement
# ---------------------------------------------------------------------------

def cir_refine(H_list, class_num, tau_s=0.5, lam_cir=0.5, rho=1.0):
    """Consistency-guided Intra-view Refinement (CIR).

    For each view:
      1. Compute cosine affinity matrix S^v_ij = softmax(cos(h_i, h_j) / τ_s)
      2. Sparsify: keep top-k_H entries per row, k_H = ⌊N/K · ρ⌋
      3. Residual aggregate: Ĥ^v = H^v + λ_cir · Ŝ^v · H^v

    Args:
        H_list: list of [N, d] consistency features, one per view
        class_num: number of clusters K
        tau_s: temperature for affinity softmax (default 0.5)
        lam_cir: refinement strength λ_cir ∈ [0, 1] (default 0.5)
        rho: scaling factor for neighborhood size (default 1.0)

    Returns:
        H_refined: list of [N, d] refined features Ĥ^v
        S_list: list of [N, N] affinity matrices (for logging)
    """
    N = H_list[0].size(0)
    k_H = max(1, int(np.floor(N / class_num * rho)))

    H_refined = []
    S_list = []

    for H in H_list:
        # Step 1: Cosine similarity matrix
        H_norm = F.normalize(H, dim=1)  # L2-normalize each row
        cos_sim = torch.matmul(H_norm, H_norm.T)  # [N, N]
        S = F.softmax(cos_sim / tau_s, dim=1)  # row-wise softmax

        # Step 2: Top-k sparsification
        _, topk_idx = torch.topk(S, k=min(k_H, N), dim=1)
        S_sparse = torch.zeros_like(S)
        S_sparse.scatter_(1, topk_idx, S.gather(1, topk_idx))
        # Re-normalize rows
        S_sparse = S_sparse / (S_sparse.sum(dim=1, keepdim=True) + 1e-10)

        # Step 3: Residual aggregation
        H_ref = H + lam_cir * torch.matmul(S_sparse, H)
        H_refined.append(H_ref)
        S_list.append(S_sparse)

    return H_refined, S_list


# ---------------------------------------------------------------------------
# Target distribution (DEC-style)
# ---------------------------------------------------------------------------

def compute_target_distribution(B, class_num, nu=1.0):
    """Compute sharpened target distribution P̂ via Student's t kernel + DEC sharpening.

    Args:
        B: fused feature [N, d] (numpy or torch)
        class_num: K (number of clusters)
        nu: degrees of freedom for Student's t (default 1.0)

    Returns:
        P_hat: sharpened target distribution [N, K] (torch.Tensor)
        centroids: cluster centroids [K, d] (torch.Tensor)
    """
    from sklearn.cluster import KMeans

    if isinstance(B, torch.Tensor):
        B_np = B.detach().cpu().numpy()
    else:
        B_np = B

    kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=0)
    kmeans.fit(B_np)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=B.device if isinstance(B, torch.Tensor) else 'cpu')

    if isinstance(B, torch.Tensor):
        B_t = B
    else:
        B_t = torch.tensor(B, dtype=torch.float32)

    # Student's t-distribution kernel
    dist = torch.cdist(B_t, centroids.to(B_t.device))  # [N, K]
    q = (1.0 + dist ** 2 / nu) ** (-(nu + 1.0) / 2.0)
    P = q / q.sum(dim=1, keepdim=True)  # [N, K]

    # DEC sharpening: square and re-normalize
    P_sq = P ** 2
    P_hat = P_sq / P_sq.sum(dim=0, keepdim=True)
    P_hat = P_hat / P_hat.sum(dim=1, keepdim=True)

    return P_hat, centroids


def plot_tsne(features, labels, save_path, title='t-SNE of fused features',
              random_state=42, perplexity=30):
    """Save a 2-D t-SNE visualization of fused features.

    Args:
        features: [N, D] NumPy array or torch.Tensor.
        labels: [N] labels used only for coloring points.
        save_path: output image path.
        title: figure title.
        random_state: deterministic t-SNE seed.

        perplexity: requested t-SNE perplexity; it is capped below N.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError('features must be [N, D] and aligned with labels')
    if features.shape[0] < 3:
        raise ValueError('t-SNE requires at least three samples')

    perplexity = min(float(perplexity), float(features.shape[0] - 1))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init='pca',
    ).fit_transform(features)

    save_path = os.fspath(save_path)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        embedding[:, 0], embedding[:, 1], c=labels,
        cmap='tab10', s=18, alpha=0.85, linewidths=0,
    )
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labelleft=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Seed setup
# ---------------------------------------------------------------------------

def setup_seed(seed):
    """Set global random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model checkpoint
# ---------------------------------------------------------------------------

def save_model(state, save_dir, dataset_name):
    """Save model state dict to disk."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, f'{dataset_name}.pth')
    torch.save(state, path)
    print(f'Model saved to: {path}')


def load_model(model, save_dir, dataset_name, device):
    """Load model state dict from disk."""
    path = os.path.join(save_dir, f'{dataset_name}.pth')
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model loaded from: {path}')
        return checkpoint.get('epoch', 0)
    else:
        print(f'No pretrained model found at: {path}')
        return 0
