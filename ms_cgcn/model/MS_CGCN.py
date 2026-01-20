"""Multi-Scale CGCNN with cross-scale attention.

Contains:
- CrossScaleAttention: keys-only masking for stability; dropped scales output zeroed.
- MultiScaleCGCNN: per-scale CGCNN encoders + cross-scale enhancement + per-sample fusion.

This file is a comment/formatting-only normalization of the user's working code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .cgcnn_gate import CGCNN


class CrossScaleAttention(nn.Module):
    """Cross-scale attention over per-scale embeddings.

    Stability rules (kept):
    - Mask keys only (not queries) to avoid all-masked rows -> NaN softmax.
    - key_mask determines which scales can be used as K/V.
    - Dropped scales (scale_mask=False) outputs are set to zero.
    - lambda_gate init = -2.0 -> sigmoid ~= 0.12.

    Inputs:
        emb_list:   list of length S, each [B, D]
        scale_mask: None or bool tensor [S] / [B, S]
        key_mask:   None or bool tensor [S] / [B, S]

    Returns:
        list of length S, each [B, D]
    """

    def __init__(
        self,
        emb_dim: int,
        num_scales: int = 3,
        num_heads: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.head_dim = max(emb_dim // num_heads, 16)

        self.pre_norm = nn.LayerNorm(emb_dim)
        self.q = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.k = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.v = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.out = nn.Linear(num_heads * self.head_dim, emb_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        self.lambda_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        emb_list: List[torch.Tensor],
        scale_mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        S = self.num_scales
        assert len(emb_list) == S, f"Expected {S} scales, got {len(emb_list)}"

        # [B, S, D]
        E = torch.stack([F.layer_norm(e, e.shape[-1:]) for e in emb_list], dim=1)
        B, _, _ = E.shape
        En = self.pre_norm(E)
        device = E.device

        # scale_mask -> [B, S]
        if scale_mask is None:
            scale_mask_b = torch.ones(B, S, dtype=torch.bool, device=device)
        else:
            if scale_mask.dim() == 1:
                scale_mask_b = scale_mask.view(1, S).expand(B, S).to(device=device)
            elif scale_mask.dim() == 2:
                scale_mask_b = scale_mask.to(device=device)
            else:
                raise ValueError("scale_mask must have shape [S] or [B, S]")

        # key_mask -> [B, S]
        if key_mask is None:
            key_mask_b = scale_mask_b
        else:
            if key_mask.dim() == 1:
                key_mask_b = key_mask.view(1, S).expand(B, S).to(device=device)
            elif key_mask.dim() == 2:
                key_mask_b = key_mask.to(device=device)
            else:
                raise ValueError("key_mask must have shape [S] or [B, S]")

        # Ensure at least one key per sample
        all_key_dropped = ~key_mask_b.any(dim=1)
        if all_key_dropped.any():
            key_mask_b = key_mask_b.clone()
            key_mask_b[all_key_dropped, 0] = True

        # Q from all scales; K/V from allowed keys only
        Q_in = En
        kv_f = key_mask_b.float().unsqueeze(-1)  # [B, S, 1]
        KV_in = En * kv_f

        Q = self.q(Q_in).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(KV_in).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(KV_in).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Mask keys only (last dim)
        key_mask_4d = key_mask_b.view(B, 1, 1, S)
        scores = scores.masked_fill(~key_mask_4d, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        o = torch.matmul(attn, V)
        o = o.transpose(1, 2).contiguous().view(B, S, -1)
        o = self.out_drop(self.out(o))

        # Zero-out dropped scale outputs
        out_f = scale_mask_b.float().unsqueeze(-1)
        o = o * out_f

        lam = torch.sigmoid(self.lambda_gate)
        fused = (E + lam * o) * out_f

        return [fused[:, i, :] for i in range(S)]


class MultiScaleCGCNN(nn.Module):
    """Multi-scale model with per-sample fusion.

    Key logic (kept):
    - Scale dropout builds `active_mask`.
    - `key_mask` is derived from gate effective ratio (if available).
    - Cross-scale attention enhances per-scale embeddings.
    - Per-sample weights from MLP are softmax-normalized and used to fuse.
    """

    def __init__(
        self,
        dataset,
        num_scales: int = 3,
        cross_att_heads: int = 2,
        out_hidden: int = 128,
        drop_scale_p: float = 0.0,
        pre_fc_count: int = 1,
        post_fc_count: int = 1,
        radius_init_list=None,
        dim1: int = 64,
        dim2: int = 64,
        gc_count: int = 3,
        pool: str = "global_mean_pool",
        pool_order: str = "early",
        batch_norm: str = "True",
        batch_track_stats: str = "True",
        act: str = "relu",
        dropout_rate: float = 0.0,
        use_radius_gate: bool = True,
        alpha: float = 10.0,
        thr_eff: float = 0.30,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.drop_scale_p = float(drop_scale_p)
        self.pre_fc_count = pre_fc_count
        self.post_fc_count = post_fc_count
        self.thr_eff = float(thr_eff)

        if radius_init_list is None:
            radius_init_list = [
                (0.0, 3.5),
                (3.0, 5.5),
                (4.5, 8.0),
            ]
        assert len(radius_init_list) == num_scales

        self.gnns = nn.ModuleList(
            [
                CGCNN(
                    dataset,
                    dim1=dim1,
                    dim2=dim2,
                    pre_fc_count=pre_fc_count,
                    gc_count=gc_count,
                    post_fc_count=post_fc_count,
                    pool=pool,
                    pool_order=pool_order,
                    batch_norm=batch_norm,
                    batch_track_stats=batch_track_stats,
                    act=act,
                    dropout_rate=dropout_rate,
                    use_radius_gate=use_radius_gate,
                    r_min_init=r0,
                    r_max_init=r1,
                    alpha=alpha,
                )
                for (r0, r1) in radius_init_list
            ]
        )

        emb_dim = self.gnns[0].embedding_dim
        self.emb_dim = emb_dim

        self.cross_att = CrossScaleAttention(
            emb_dim=emb_dim,
            num_scales=num_scales,
            num_heads=cross_att_heads,
        )

        hidden_w = max(emb_dim // 4, 8)
        self.scale_weight_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_w),
            nn.ReLU(),
            nn.Linear(hidden_w, 1),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(emb_dim, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, 1),
        )

    def forward(self, batch):
        device = batch.x.device

        B = getattr(batch, "num_graphs", None)
        if B is None:
            B = int(batch.batch.max().item()) + 1 if hasattr(batch, "batch") else 1

        # Scale dropout mask
        if self.training and self.drop_scale_p > 0.0 and self.num_scales > 1:
            keep = (torch.rand(self.num_scales, device=device) > self.drop_scale_p)
            if keep.sum().item() == 0:
                keep[torch.randint(0, self.num_scales, (1,), device=device)] = True
            active_mask = keep
        else:
            active_mask = torch.ones(self.num_scales, dtype=torch.bool, device=device)

        embeddings: List[torch.Tensor] = []
        gate_eff_list = [None] * self.num_scales

        for i, gnn in enumerate(self.gnns):
            if not active_mask[i]:
                embeddings.append(torch.zeros(B, self.emb_dim, device=device))
                continue

            _, emb = gnn(batch)  # [B, D]
            embeddings.append(emb)

            if hasattr(batch, "edge_dist"):
                gate = getattr(gnn, "radius_gate", None)
                if gate is not None:
                    g = gate(batch.edge_dist)
                    if g.dim() > 1:
                        g = g.squeeze(-1)
                    gate_eff_list[i] = (g > 1e-3).float().mean()

        # key_mask for K/V only
        key_mask = active_mask.clone()
        for i in range(self.num_scales):
            if not active_mask[i]:
                continue
            eff = gate_eff_list[i]
            if eff is None:
                continue
            if eff.item() < self.thr_eff:
                key_mask[i] = False

        if key_mask.sum().item() == 0:
            idx = int(active_mask.nonzero(as_tuple=False)[0].item())
            key_mask[idx] = True

        enhanced = self.cross_att(
            embeddings,
            scale_mask=active_mask,
            key_mask=key_mask,
        )

        # Per-sample scale weights
        weights = torch.cat([self.scale_weight_mlp(e) for e in enhanced], dim=1)  # [B, S]
        weights = F.softmax(weights, dim=1)

        if self.training and self.drop_scale_p > 0.0 and self.num_scales > 1:
            mask = active_mask.float().unsqueeze(0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        fused = sum(enhanced[i] * weights[:, i].unsqueeze(1) for i in range(self.num_scales))
        pred = self.out_mlp(fused).view(-1)
        return pred, fused

    def overlap_regularization(self, delta_overlap=0.3, lambda_gap=0.1):
        """Overlap/gap regularization across scale radius ranges (kept)."""
        ranges = []
        for gnn in self.gnns:
            gate = getattr(gnn, "radius_gate", None)
            if gate is None:
                return torch.tensor(0.0, device=next(self.parameters()).device)

            r_min = F.softplus(gate.r_min_raw)
            r_max = r_min + F.softplus(gate.r_delta_raw)
            ranges.append((r_min, r_max))

        loss = 0.0
        count = 0
        for i in range(len(ranges)):
            rmin_i, rmax_i = ranges[i]
            width_i = rmax_i - rmin_i
            for j in range(i + 1, len(ranges)):
                rmin_j, rmax_j = ranges[j]
                width_j = rmax_j - rmin_j

                overlap = torch.relu(torch.min(rmax_i, rmax_j) - torch.max(rmin_i, rmin_j))
                overlap_penalty = torch.relu(overlap - delta_overlap)
                overlap_penalty = overlap_penalty / (torch.min(width_i, width_j) + 1e-6)

                gap = torch.relu(torch.max(rmin_j - rmax_i, rmin_i - rmax_j))
                gap_penalty = gap / (torch.min(width_i, width_j) + 1e-6)

                loss = loss + overlap_penalty + lambda_gap * gap_penalty
                count += 1

        if count > 0:
            loss = loss / count
        return loss

    def get_all_radius_ranges(self):
        out = []
        for gnn in self.gnns:
            if hasattr(gnn, "get_radius_range"):
                out.append(gnn.get_radius_range())
            else:
                out.append(None)
        return out
