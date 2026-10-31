"""
Loss functions for DLC-MVC.

Loss components:
  L_rec     — Reconstruction loss (MSE)
  L_orth    — Orthogonality constraint between H and Z
  L_con     — Instance-level contrastive loss (standard InfoNCE, on H only)
  L_div     — Diversity loss (variance + inter-view decorrelation, on Z only)
  L_clc     — Label-level contrastive loss (between cluster assignments)
  L_clu     — Clustering loss (KL divergence with target distribution)

Total loss: L = L_rec + α·L_con + β·(L_div + L_orth) + γ·(L_clc + L_clu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalLoss(nn.Module):
    """Aggregated loss with configurable weights.

    """

    def __init__(self, class_num, batch_size,
                 temperature_f=0.5, temperature_l=0.5,
                 alpha=1.0, beta=0.1, gamma=0.5,
                 lam_decor=0.1, device='cuda'):
        super(TotalLoss, self).__init__()
        self.class_num = class_num
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam_decor = lam_decor
        self.device = device
        self.mse = nn.MSELoss()
        self.similarity = nn.CosineSimilarity(dim=2)
        self.ce = nn.CrossEntropyLoss(reduction="sum")

    # ---- Mask helpers ------------------------------------------------------

    def _mask_correlated_samples(self, N):
        """Create mask that excludes self-pairs and positive pairs."""
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        return mask

    # ---- Reconstruction ----------------------------------------------------

    def forward_rec(self, xs, xrs):
        """L_rec = Σ_v ||Xv - X̂v||²."""
        loss = sum(self.mse(xs[v], xrs[v]) for v in range(len(xs)))
        return loss

    # ---- Orthogonality -----------------------------------------------------

    def forward_orth(self, Hs, Zs):
        """L_orth = (1/V) Σ_v (1/d²) ||(H̄^v)ᵀ Z̄^v||_F².

        Column-wise L2-normalized H and Z, then penalize their cross-product
        averaged over d² entries to ensure scale-invariance w.r.t. feature dimension.
        """
        loss = 0.0
        V = len(Hs)
        d = Hs[0].size(1)
        for v in range(V):
            H_norm = F.normalize(Hs[v], dim=0)  # column-wise L2
            Z_norm = F.normalize(Zs[v], dim=0)
            cross = torch.matmul(H_norm.T, Z_norm)  # [d, d]
            loss += (cross ** 2).sum() / (d * d)
        return loss / V

    # ---- Instance-level Contrastive (on H only) ---------------------------

    def forward_con(self, H_u, H_v):
        """Standard InfoNCE contrastive loss between two views' H features.

        Positive pair: (h_i^u, h_i^v) — same instance across views.
        Negative pairs: all other cross-view pairs in the batch.
        """
        N = self.batch_size * 2
        H = torch.cat([H_u, H_v], dim=0)  # [2B, d]

        sim = torch.matmul(H, H.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)   # pos: u->v
        sim_j_i = torch.diag(sim, -self.batch_size)  # pos: v->u

        positive = torch.cat([sim_i_j, sim_j_i], dim=0).view(N, 1)
        mask = self._mask_correlated_samples(N).to(self.device)
        negative = sim[mask].view(N, -1)

        labels = torch.zeros(N, dtype=torch.long, device=self.device)
        logits = torch.cat([positive, negative], dim=1)
        loss = self.ce(logits, labels) / N
        return loss

    def forward_con_all(self, Hs):
        """L_con = (1/2) Σ_{v} Σ_{u≠v} ℓ_con^{uv}."""
        V = len(Hs)
        loss = 0.0
        count = 0
        for v in range(V):
            for u in range(v + 1, V):
                loss += self.forward_con(Hs[v], Hs[u])
                count += 1
        return loss / max(count, 1)

    # ---- Diversity (on Z only) --------------------------------------------

    def _off_diagonal(self, x):
        """Flatten off-diagonal elements of a square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward_var(self, Z):
        """Intra-view variance regularization (per-view, per-dimension).

        v(Z) = (1/d) Σ_j max(0, γ - std(z_{·j})), γ = 1.0.
        """
        Z_centered = Z - Z.mean(dim=0)
        std = torch.sqrt(Z_centered.var(dim=0) + 1e-4)
        loss = F.relu(1.0 - std).mean()
        return loss

    def forward_decor(self, Z_u, Z_v):
        """Inter-view decorrelation: ||corr(Z^u, Z^v)||_F².

        Computes cross-correlation matrix between column-wise standardized features.
        """
        d = Z_u.size(1)
        # Standardize columns
        Z_u_std = (Z_u - Z_u.mean(dim=0)) / (Z_u.std(dim=0) + 1e-4)
        Z_v_std = (Z_v - Z_v.mean(dim=0)) / (Z_v.std(dim=0) + 1e-4)

        corr = torch.matmul(Z_u_std.T, Z_v_std) / (self.batch_size - 1)
        loss = (corr ** 2).sum() / d
        return loss

    def forward_div(self, Zs):
        """L_div = Σ_v v(Z^v) + λ_decor · (1/(V(V-1))) Σ_{u≠v} ||corr(Z^u, Z^v)||_F²."""
        V = len(Zs)
        var_loss = sum(self.forward_var(Zs[v]) for v in range(V)) / V

        decor_loss = 0.0
        count = 0
        for u in range(V):
            for v in range(u + 1, V):
                decor_loss += self.forward_decor(Zs[u], Zs[v])
                count += 1
        decor_loss = decor_loss / max(count, 1)

        return var_loss + self.lam_decor * decor_loss

    # ---- Label-level Contrastive ------------------------------------------

    def forward_label_con_pair(self, Q_u, Q_v):
        """Label-level contrastive loss between two views' cluster assignments.

        Uses cosine similarity on transposed Q (class_num × N → class_num dim).
        Includes entropy regularization to prevent cluster collapse.
        """
        K = self.class_num

        # Entropy regularization: H(q̄)
        p_u = Q_u.sum(dim=0) / Q_u.size(0)
        p_v = Q_v.sum(dim=0) / Q_v.size(0)
        ne_u = math.log(K) + (p_u * torch.log(p_u + 1e-10)).sum()
        ne_v = math.log(K) + (p_v * torch.log(p_v + 1e-10)).sum()
        entropy = ne_u + ne_v

        # Transpose: treat K clusters as N samples for contrastive
        q_u = Q_u.T  # [K, B]
        q_v = Q_v.T
        N_k = 2 * K
        q = torch.cat([q_u, q_v], dim=0)  # [2K, B]

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, K)
        sim_j_i = torch.diag(sim, -K)

        positive = torch.cat([sim_i_j, sim_j_i], dim=0).view(N_k, 1)
        mask = self._mask_correlated_samples(N_k).to(self.device)
        negative = sim[mask].view(N_k, -1)

        labels = torch.zeros(N_k, dtype=torch.long, device=self.device)
        logits = torch.cat([positive, negative], dim=1)
        loss = self.ce(logits, labels) / N_k
        return loss + entropy

    def forward_label_con(self, Qs):
        """L_clc = (1/2) Σ_v Σ_{u≠v} ℓ_clc^{uv}."""
        V = len(Qs)
        loss = 0.0
        count = 0
        for v in range(V):
            for u in range(v + 1, V):
                loss += self.forward_label_con_pair(Qs[v], Qs[u])
                count += 1
        return loss / max(count, 1)

    # ---- Clustering Loss (KL divergence) -----------------------------------

    def forward_clu(self, Qs, P_hat):
        """L_clu = Σ_v KL(P̂ || Q^v).

        Args:
            Qs: list of view-specific soft assignments [N, K]
            P_hat: sharpened target distribution [N, K]
        """
        loss = 0.0
        for Qv in Qs:
            loss += F.kl_div(
                torch.log(Qv + 1e-10),
                P_hat.detach(),
                reduction='batchmean'
            )
        return loss

    # ---- Total loss --------------------------------------------------------

    def forward(self, xs, xrs, Hs, Zs, Qs, P_hat):
        """Compute total loss.

        L = L_rec + α·L_con + β·(L_div + L_orth) + γ·(L_clc + L_clu)

        Args:
            xs: list of original inputs
            xrs: list of reconstructed inputs
            Hs: list of raw consistency features
            Zs: list of complementarity features
            Qs: list of soft cluster assignments [N, K]
            P_hat: target distribution [N, K]
        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual loss values (for logging)
        """
        l_rec = self.forward_rec(xs, xrs)
        l_con = self.forward_con_all(Hs)
        l_div = self.forward_div(Zs)
        l_orth = self.forward_orth(Hs, Zs)
        l_clc = self.forward_label_con(Qs)
        l_clu = self.forward_clu(Qs, P_hat)

        total = (l_rec
                 + self.alpha * l_con
                 + self.beta * l_div
                 + self.beta * l_orth
                 + self.gamma * l_clc
                 + self.gamma * l_clu)

        loss_dict = {
            'rec': l_rec.item(),
            'con': l_con.item(),
            'div': l_div.item(),
            'orth': l_orth.item(),
            'clc': l_clc.item(),
            'clu': l_clu.item(),
            'total': total.item(),
        }
        return total, loss_dict
