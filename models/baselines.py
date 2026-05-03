"""
Baseline models for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_backbone import GNNEncoder
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class VanillaGNN(nn.Module):
    """
    Standard GNN baseline without any role information.
    Simply: encoder -> global pooling -> MLP head.
    """

    def __init__(
        self,
        encoder: GNNEncoder,
        hidden_dim: int,
        pool_type: str = "mean"
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type

        self.pools = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool
        }

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        h = self.encoder(
            data.x,
            data.edge_index,
            data.batch,
            getattr(data, 'edge_attr', None)
        )
        hg = self.pools[self.pool_type](h, data.batch)
        return self.pred_head(hg).view(-1)

class AtomAttentionGNN(nn.Module):
    """
    Node-level attention baseline.
    Injects role as attention weights (not as structural bias).
    """

    def __init__(
        self,
        encoder: GNNEncoder,
        hidden_dim: int,
        pool_type: str = "mean"
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim

        from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
        self.pools = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool
        }

        # Attention over nodes ( learns to weight backbone vs side-chain)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        h = self.encoder(
            data.x,
            data.edge_index,
            data.batch,
            getattr(data, 'edge_attr', None)
        )

        # Node-level attention (uses node_roles implicitly via learned weights)
        attn_weights = torch.sigmoid(self.attention(h))
        h_weighted = h * attn_weights

        hg = self.pools["mean"](h_weighted, data.batch)
        return self.pred_head(hg).view(-1)


# Alias for Table 1
BaselineGNN = VanillaGNN
