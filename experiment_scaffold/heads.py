"""Graph-level regression model: GCN backbone + pooling + optional @head side-channel.

``GraphRegressor`` wraps a ``models.build_gcn`` backbone, mean-pools node
embeddings to a graph embedding, optionally concatenates an MLP-encoded
graph-level side channel (``data.u`` -- age scalar for C1, GLM zmap for C2),
and maps the result to a scalar with a small MLP head.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool


class GraphRegressor(nn.Module):
    """Backbone -> mean-pool -> [optional u-encoder ++] -> MLP head -> scalar."""

    def __init__(self, backbone, pooled_dim: int, global_dim: int = 0,
                 global_hidden: int = 32, head_hidden: int = 64):
        super().__init__()
        self.backbone = backbone
        self.global_dim = global_dim
        if global_dim > 0:
            self.global_enc = nn.Sequential(
                nn.Linear(global_dim, global_hidden), nn.ReLU())
            head_in = pooled_dim + global_hidden
        else:
            self.global_enc = None
            head_in = pooled_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))

    def forward(self, x, edge_index, edge_weight, batch, u=None):
        h = self.backbone(x, edge_index, edge_weight=edge_weight)   # [n_nodes, pooled_dim]
        h = global_mean_pool(h, batch)                              # [B, pooled_dim]
        if self.global_enc is not None:
            if u is None:
                raise ValueError("global_dim>0 requires u (the @head side channel)")
            h = torch.cat([h, self.global_enc(u)], dim=-1)
        return self.head(h).squeeze(-1)                             # [B]
