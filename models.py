"""
DLC-MVC: Decoupled Feature Learning for Contrastive Multi-View Clustering

Network Architecture (updated with DACVE):
  Shared Encoder -> Ev
    -> Consistency Head (phi_cons) -> Hv
    -> Complementarity Head (phi_comp) -> Zv
  DACVE: {Zv} -> {Zv_out}  (difference-aware cross-view enhancement)
  Integration: [Zv_out; Hv] -> MLP -> Bv
  CIR: Bv -> Bv_refined  (optional, controlled by use_cir)
  AttentionFusion: {Bv_refined} -> B
  ClusterHead: Bv_refined -> Qv
  Decoder: [Hv; Zv] -> X_hat_v  (reconstruction from original features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cir_refine


# ---------------------------------------------------------------------------
# Encoder / Decoder / ProjectionHead (unchanged)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Shared backbone encoder: dv -> 500 -> 500 -> 2000 -> d_enc."""

    def __init__(self, input_dim, enc_dim=256):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, enc_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class ProjectionHead(nn.Module):
    """Lightweight projection head: d_enc -> 128 -> d."""

    def __init__(self, input_dim=256, hidden_dim=128, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.head(x)


class Decoder(nn.Module):
    """Symmetric decoder: 2*d -> 2000 -> 500 -> 500 -> dv (reconstructs from [H; Z])."""

    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2 * feature_dim, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, input_dim),
        )

    def forward(self, x):
        return self.decoder(x)


# ---------------------------------------------------------------------------
# Attention Fusion (unchanged)
# ---------------------------------------------------------------------------

class AttentionFusion(nn.Module):
    """Learnable attention-based multi-view fusion.

    Bv -> pool -> MLP_attn -> scalar weight alpha_v.
    B = sum alpha_v * Bv.
    """

    def __init__(self, feat_dim):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, B_list):
        weights = []
        for Bv in B_list:
            pooled = F.normalize(Bv.mean(dim=0, keepdim=True), dim=1)
            w = self.attn(pooled)
            weights.append(w)
        weights = torch.cat(weights, dim=0)
        weights = F.softmax(weights, dim=0)

        fused = sum(w * Bv for w, Bv in zip(weights, B_list))
        return fused, weights.squeeze(-1)


# ---------------------------------------------------------------------------
# DACVE: Difference-Aware Cross-View Enhancement
# ---------------------------------------------------------------------------

class DACVE(nn.Module):
    """Difference-Aware Cross-View Enhancement with learned attention.

    Replaces the fixed cosine-distance heuristic with learnable QKV projections.
    Each view learns what constitutes useful cross-view complementary information.

    For each view v:
      1. Project:        Q_v = W_q(Z^v),  K_u = W_k(Z^u),  V_u = W_v(Z^u)
      2. Attention:      α_{v→u} = softmax(Q_v @ K_u^T / √d),  u≠v
      3. Aggregation:    c^v = Σ_{u≠v} α_{v→u} · V_u
      4. Residual:       Z^v_out = LayerNorm(Z^v + λ · W_o(c^v))

    Key properties:
      - Per-view output preserves view specificity (not a single fused Z)
      - Learnable QKV projections replace the cosine-distance heuristic
      - Output projection W_o transforms the aggregated context
      - Scalar λ controls enhancement magnitude

    Args:
        feat_dim: dimension of each Z^v
        lam: residual enhancement strength λ (default 0.1)
        dropout: attention dropout rate (default 0.0)
    """

    def __init__(self, feat_dim, lam=0.1, dropout=0.0):
        super(DACVE, self).__init__()
        self.lam = lam
        self.scale = feat_dim ** -0.5

        self.W_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_k = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_v = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_o = nn.Linear(feat_dim, feat_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, Z_list):
        """Enhance each view's Z^v with attention-weighted cross-view context.

        Args:
            Z_list: list of [N, d] tensors, one per view

        Returns:
            Z_out_list: list of [N, d] enhanced features, one per view
        """
        V = len(Z_list)
        N, d = Z_list[0].shape
        Z_stack = torch.stack(Z_list, dim=0)  # [V, N, d]

        Q = self.W_q(Z_stack)  # [V, N, d]
        K = self.W_k(Z_stack)  # [V, N, d]
        V_val = self.W_v(Z_stack)  # [V, N, d]

        Z_out_list = []
        for v in range(V):
            other_idx = [u for u in range(V) if u != v]

            q_v = Q[v]  # [N, d]
            k_other = K[other_idx]  # [V-1, N, d]
            v_other = V_val[other_idx]  # [V-1, N, d]

            # Cross-view attention: Q_v attends to K_u (u≠v)
            attn = torch.einsum('nd,mnd->nm', q_v, k_other) * self.scale  # [N, V-1]
            attn = F.softmax(attn, dim=1)
            attn = self.dropout(attn)

            # Weighted aggregation: Σ α_{v→u} · V_u
            c = (attn.unsqueeze(-1) * v_other.permute(1, 0, 2)).sum(dim=1)  # [N, d]

            Z_out = Z_stack[v] + self.lam * self.W_o(c)
            Z_out_list.append(Z_out)

        return Z_out_list


# ---------------------------------------------------------------------------
# DLC-MVC (updated)
# ---------------------------------------------------------------------------

class DLC_MVC(nn.Module):
    """Decoupled Feature Learning for Contrastive Multi-View Clustering.

    Architecture (updated):
        Xv -> SharedEncoder -> Ev
            -> ConsHead  -> Hv  (consistency features)
            -> CompHead  -> Zv  (complementarity features)
        DACVE: {Zv} -> {Zv_out}  (difference-aware cross-view enhancement)
        Integration: [Zv_out; Hv] -> FuseMLP -> Bv
        CIR: Bv -> Bv_refined  (optional, use_cir=True)
        AttentionFusion: {Bv_refined} -> B
        ClusterHead: Bv_refined -> Qv
        Decoder: [Hv; Zv] -> X_hat_v  (original Zv for reconstruction)

    Args:
        view: number of views
        dims: list of input dimensions per view
        enc_dim: shared encoder output dimension (default 256)
        proj_dim: projection head output dimension (default 128)
        fuse_dim: fusion feature dimension (default 128)
        class_num: number of clusters K
        use_cir: enable CIR refinement on Bv (default True)
        tau_s: CIR affinity temperature (default 0.5)
        lam_cir: CIR refinement strength (default 0.5)
        rho: CIR neighborhood scaling factor (default 1.0)
        dacve_lam: DACVE residual strength (default 0.1)
        dacve_dropout: DACVE attention dropout (default 0.0)
    """

    def __init__(self, view, dims, enc_dim=256, proj_dim=128, fuse_dim=128,
                 class_num=10, use_cir=True,
                 tau_s=0.5, lam_cir=0.5, rho=1.0, dacve_lam=0.01, dacve_dropout=0.5):
        super(DLC_MVC, self).__init__()
        self.view = view
        self.enc_dim = enc_dim
        self.proj_dim = proj_dim
        self.fuse_dim = fuse_dim
        self.class_num = class_num

        # CIR control
        self.use_cir = use_cir
        self.tau_s = tau_s
        self.lam_cir = lam_cir
        self.rho = rho

        # Shared encoders (one per view)
        self.encoders = nn.ModuleList([
            Encoder(dims[v], enc_dim) for v in range(view)
        ])

        # Consistency projection heads (one per view)
        self.cons_heads = nn.ModuleList([
            ProjectionHead(enc_dim, 128, proj_dim) for v in range(view)
        ])

        # Complementarity projection heads (one per view)
        self.comp_heads = nn.ModuleList([
            ProjectionHead(enc_dim, 128, proj_dim) for v in range(view)
        ])

        # Decoders (one per view, reconstructs from [Hv; Zv] — original Zv)
        self.decoders = nn.ModuleList([
            Decoder(dims[v], proj_dim) for v in range(view)
        ])

        # DACVE: difference-aware cross-view enhancement (learned attention)
        self.dacve = DACVE(proj_dim, lam=dacve_lam, dropout=dacve_dropout)

        # Fusion MLP: [Zv_out; Hv] -> Bv  (2 * proj_dim -> fuse_dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * proj_dim, fuse_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fuse_dim, fuse_dim),
        )

        # Attention-based fusion
        self.attn_fusion = AttentionFusion(fuse_dim)

        # Cluster head: Bv_refined -> Qv (soft cluster assignments)
        self.cluster_head = nn.Sequential(
            nn.Linear(fuse_dim, class_num),
            nn.Softmax(dim=1),
        )

    # ---- Encoding & Decoding ------------------------------------------------

    def forward_enc(self, xs):
        """Encode: produce Hv, Zv for all views.

        Returns:
            Hs: list of [N, proj_dim] — raw consistency features
            Zs: list of [N, proj_dim] — complementarity features
            Evs: list of [N, enc_dim] — shared encoder outputs
        """
        Hs, Zs, Evs = [], [], []
        for v in range(self.view):
            Ev = self.encoders[v](xs[v])
            Hv = self.cons_heads[v](Ev)
            Zv = self.comp_heads[v](Ev)
            Evs.append(Ev)
            Hs.append(Hv)
            Zs.append(Zv)
        return Hs, Zs, Evs

    def decode(self, Hs, Zs):
        """Decode from original (pre-DACVE, pre-CIR) features [Hv; Zv] -> X_hat_v.

        Uses per-view Zv (not fused Z) to preserve view-specific reconstruction.
        """
        Xrs = []
        for v in range(self.view):
            cat_feat = torch.cat([Hs[v], Zs[v]], dim=1)
            Xr = self.decoders[v](cat_feat)
            Xrs.append(Xr)
        return Xrs

    # ---- DACVE + Integration -------------------------------------------------

    def integrate(self, Hs, Zs):
        """Integrate enhanced per-view Z with each view's consistency Hv.

        [Zv_out; Hv] -> FuseMLP -> Bv

        Args:
            Hs: list of [N, d] raw consistency features
            Zs: list of [N, d] enhanced complementarity features (DACVE output)

        Returns:
            B_list: list of [N, fuse_dim] integrated features
        """
        B_list = []
        for v in range(self.view):
            cat_feat = torch.cat([Zs[v], Hs[v]], dim=1)
            Bv = self.fuse_mlp(cat_feat)
            B_list.append(Bv)
        return B_list

    # ---- Clustering ---------------------------------------------------------

    def get_cluster_assignments(self, B_list):
        """Get soft cluster assignments from per-view features.

        Args:
            B_list: list of Bv tensors [N, fuse_dim] (refined or raw)

        Returns:
            Qs: list of [N, K] soft assignments
        """
        Qs = []
        for Bv in B_list:
            Qv = self.cluster_head(Bv)
            Qs.append(Qv)
        return Qs

    # ---- Main forward passes ------------------------------------------------

    def forward(self, xs):
        """Full forward pass for pretraining (encode + decode only).

        Returns:
            Hs: raw consistency features
            Zs: complementarity features (per-view, pre-DACVE)
            Xrs: reconstructed inputs
        """
        Hs, Zs, _ = self.forward_enc(xs)
        Xrs = self.decode(Hs, Zs)
        return Hs, Zs, Xrs

    def forward_fuse(self, xs):
        """Full forward pass for fine-tuning: DACVE -> integrate -> CIR -> fuse -> cluster.

        Pipeline:
            1. Encode: Xv -> Hv, Zv
            2. DACVE: {Zv} -> {Zv_out}  (per-view enhancement)
            3. Integrate: [Zv_out; Hv] -> Bv
            4. CIR (optional): Bv -> Bv_refined
            5. AttentionFusion: {Bv_refined} -> B
            6. ClusterHead: Bv_refined -> Qv

        Args:
            xs: list of input tensors [N, dv]

        Returns:
            B: fused consensus feature [N, fuse_dim]
            B_list: integrated features (pre-CIR) [V][N, fuse_dim]
            B_refined_list: refined features (post-CIR, or clone if use_cir=False)
            Qs: cluster assignments [V][N, K]
            Zs: original complementarity features (for L_div, L_orth)
            weights: attention weights [V]
        """
        Hs, Zs, _ = self.forward_enc(xs)

        # DACVE: difference-aware cross-view enhancement (per-view output)
        Zs_enhanced = self.dacve(Zs)

        # Integrate per-view enhanced Z with each Hv
        B_list = self.integrate(Hs, Zs_enhanced)

        # CIR on Bv (with on/off switch for ablation)
        if self.use_cir:
            B_refined_list, _ = cir_refine(
                B_list, self.class_num,
                tau_s=self.tau_s, lam_cir=self.lam_cir, rho=self.rho,
            )
        else:
            B_refined_list = [Bv.clone() for Bv in B_list]

        # Attention fusion on refined features
        B, weights = self.attn_fusion(B_refined_list)

        # Cluster assignments from refined features
        Qs = self.get_cluster_assignments(B_refined_list)

        return B, B_list, B_refined_list, Qs, Zs, weights,Hs

    def forward_eval(self, xs):
        """End-to-end evaluation forward pass.

        Returns:
            Qs: cluster assignments [V][N, K]
            B: fused consensus feature [N, fuse_dim]
            Hs: raw consistency features [V][N, d]
            Zs: original complementarity features (per-view) [V][N, d]
            B_refined_list: refined per-view features [V][N, fuse_dim]
        """
        Hs, Zs, _ = self.forward_enc(xs)
        Zs_enhanced = self.dacve(Zs)
        B_list = self.integrate(Hs, Zs_enhanced)

        if self.use_cir:
            B_refined_list, _ = cir_refine(
                B_list, self.class_num,
                tau_s=self.tau_s, lam_cir=self.lam_cir, rho=self.rho,
            )
        else:
            B_refined_list = [Bv.clone() for Bv in B_list]

        B, weights = self.attn_fusion(B_refined_list)
        Qs = self.get_cluster_assignments(B_refined_list)

        return Qs, B, Hs, Zs, B_refined_list
