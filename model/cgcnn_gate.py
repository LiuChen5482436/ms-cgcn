"""CGCNN with an optional learnable radius-range gate.

This module keeps the original computation logic intact:
- Uses torch_geometric.nn.CGConv for message passing.
- Optionally gates edge_attr by a learnable distance-range gate using `data.edge_dist`.
- Supports early/late pooling and optional Set2Set.

Expected PyG Data fields:
    data.x         : [N, num_node_features]
    data.edge_index: [2, E]
    data.edge_attr : [E, num_edge_features]
    data.batch     : [N]
Optional:
    data.edge_dist : [E] or [E, 1]

Return:
    If self.get_embedding is True:
        (pred, embedding)
    else:
        pred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import torch_geometric
from torch_geometric.nn import Set2Set, CGConv
from torch import Tensor


class RadiusRangeGate(nn.Module):
    """Learnable soft gate over edge distances.

    The gate defines a differentiable window [r_min, r_max] with learnable bounds:
        r_min = softplus(r_min_raw)
        r_max = r_min + softplus(r_delta_raw)

    For each edge distance d:
        g(d) = sigmoid(alpha * (d - r_min)) * sigmoid(alpha * (r_max - d))

    Input:
        edge_dist: [E] or [E, 1]

    Output:
        gate: [E, 1]
    """

    def __init__(self, r_min_init=2.5, r_max_init=6.0, alpha=10.0):
        super().__init__()
        self.r_min_raw = nn.Parameter(torch.tensor(float(r_min_init)))
        self.r_delta_raw = nn.Parameter(torch.tensor(float(r_max_init - r_min_init)))
        self.alpha = float(alpha)

    def forward(self, edge_dist: Tensor) -> Tensor:
        if edge_dist.dim() == 1:
            edge_dist = edge_dist.view(-1, 1)

        r_min = F.softplus(self.r_min_raw)
        r_max = r_min + F.softplus(self.r_delta_raw)

        g1 = torch.sigmoid(self.alpha * (edge_dist - r_min))
        g2 = torch.sigmoid(self.alpha * (r_max - edge_dist))
        return g1 * g2

    def get_radius_range(self):
        with torch.no_grad():
            r_min = F.softplus(self.r_min_raw).item()
            r_max = r_min + F.softplus(self.r_delta_raw).item()
        return r_min, r_max


class GatedCGConvWrapper(nn.Module):
    """Thin wrapper around torch_geometric.nn.CGConv.

    It does NOT change CGConv internals; it only gates edge_attr:
        edge_attr <- edge_attr * gate(edge_dist)
    """

    def __init__(self, channels, dim, aggr="mean", batch_norm=False, radius_gate: RadiusRangeGate = None):
        super().__init__()
        self.conv = CGConv(channels, dim, aggr=aggr, batch_norm=batch_norm)
        self.radius_gate = radius_gate

    def forward(self, x, edge_index, edge_attr, edge_dist=None):
        if (self.radius_gate is not None) and (edge_dist is not None):
            gate = self.radius_gate(edge_dist)  # [E, 1]
            edge_attr = edge_attr * gate        # broadcast to [E, dim]
        return self.conv(x, edge_index, edge_attr)


class CGCNN(torch.nn.Module):
    """CGCNN backbone with optional radius gating.

    Important:
        This class preserves legacy string flags (e.g., batch_norm="True").
    """

    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        use_radius_gate=True,
        r_min_init=2.5,
        r_max_init=6.0,
        alpha=10.0,
        **kwargs,
    ):
        super(CGCNN, self).__init__()

        if batch_track_stats == "False":
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True

        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.get_embedding = True

        assert gc_count > 0, "Need at least 1 GC layer"
        sample = data[0] if hasattr(data, "__getitem__") else data
        if sample.y.dim() == 0:
            output_dim = 1
        else:
            output_dim = sample.y.numel()

        if pre_fc_count == 0:
            gc_dim = data.num_features
            post_fc_dim = data.num_features
        else:
            gc_dim = dim1
            post_fc_dim = dim1

        self.use_radius_gate = bool(use_radius_gate)
        self.radius_gate = None
        if self.use_radius_gate:
            self.radius_gate = RadiusRangeGate(r_min_init=r_min_init, r_max_init=r_max_init, alpha=alpha)

        # Pre-GNN FC layers
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(data.num_features, dim1)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                self.pre_lin_list.append(lin)
        else:
            self.pre_lin_list = torch.nn.ModuleList()

        # GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for _ in range(gc_count):
            conv = GatedCGConvWrapper(
                gc_dim,
                data.num_edge_features,
                aggr="mean",
                batch_norm=False,
                radius_gate=self.radius_gate,
            )
            self.conv_list.append(conv)

            if self.batch_norm == "True":
                bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        # Post-GNN dense layers
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)
        else:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim * 2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)

        # Set2Set pooling
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)

        # embedding_dim
        if self.pool_order == "early":
            if len(self.post_lin_list) > 0:
                self.embedding_dim = dim2
            else:
                if self.pool == "set2set":
                    self.embedding_dim = post_fc_dim * 2
                else:
                    self.embedding_dim = post_fc_dim
        else:
            self.embedding_dim = output_dim

    def forward(self, data):
        embedding = None
        edge_dist = getattr(data, "edge_dist", None)

        # Pre-GNN
        for i in range(len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        # GNN stack
        for i in range(len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                base_x = data.x
            else:
                base_x = out

            if self.batch_norm == "True":
                out = self.conv_list[i](base_x, data.edge_index, data.edge_attr, edge_dist=edge_dist)
                out = self.bn_list[i](out)
            else:
                out = self.conv_list[i](base_x, data.edge_index, data.edge_attr, edge_dist=edge_dist)

            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        # Post + pooling
        if self.pool_order == "early":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

            for i in range(len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)

            embedding = out
            out = self.lin_out(out)

        elif self.pool_order == "late":
            for i in range(len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)

            out = self.lin_out(out)

            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

            embedding = out

        if self.get_embedding:
            if out.shape[1] == 1:
                return out.view(-1), embedding
            return out, embedding

        if out.shape[1] == 1:
            return out.view(-1)
        return out

    def get_radius_range(self):
        if self.radius_gate is None:
            return None
        return self.radius_gate.get_radius_range()
