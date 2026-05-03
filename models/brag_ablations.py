"""
BRAG ablation variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_backbone import GNNEncoder
from pooling.role_pool import role_pool, role_pool_with_interaction

try:
    from torch_geometric.nn import global_mean_pool
except ImportError:
    global_mean_pool = None


class BRAGOnlyBackbone(nn.Module):
    """
    BRAG variant: only backbone representation.
    """

    def __init__(self, encoder: GNNEncoder, hidden_dim: int, pool: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.pool = pool

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
        h_bb, _ = role_pool(h, data.node_roles, data.batch, self.pool)
        return self.pred_head(h_bb).view(-1)


class BRAGOnlySidechain(nn.Module):
    """
    BRAG variant: only side-chain representation.
    """

    def __init__(self, encoder: GNNEncoder, hidden_dim: int, pool: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.pool = pool

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
        _, h_sc = role_pool(h, data.node_roles, data.batch, self.pool)
        return self.pred_head(h_sc).view(-1)


class BRAGSharedPool(nn.Module):
    """
    BRAG variant: standard pooling without role separation.
    """

    def __init__(self, encoder: GNNEncoder, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim

        from torch_geometric.nn import global_mean_pool

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
        hg = global_mean_pool(h, data.batch)
        return self.pred_head(hg).view(-1)


class BRAGAddPlusDiff(nn.Module):
    """
    BRAG variant: uses both add and diff interactions.
    Pool = add, Interaction = add + diff
    """

    def __init__(self, encoder: GNNEncoder, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim

        # Pool type = add (preserves mass-like contributions)
        # Interaction = add + diff (combined)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # bb + sc + (bb+sc) + (bb-sc)
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
        h_bb, h_sc = role_pool(h, data.node_roles, data.batch, pool="add")

        # Combined interactions
        h_add = h_bb + h_sc
        h_diff = h_bb - h_sc

        # Concatenate all representations
        hg = torch.cat([h_bb, h_sc, h_add, h_diff], dim=1)
        return self.pred_head(hg).view(-1)


class BRAGConcatOnly(nn.Module):
    """
    BRAG variant: simple concatenation without interaction.
    """

    def __init__(self, encoder: GNNEncoder, hidden_dim: int, pool: str = "add"):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.pool = pool

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # bb + sc only
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
        hg = role_pool_with_interaction(
            h, data.node_roles, data.batch, self.pool, interaction="cat"
        )
        return self.pred_head(hg).view(-1)


class BRAGAbsDiff(nn.Module):
    """
    BRAG variant: absolute difference interaction.
    """

    def __init__(self, encoder: GNNEncoder, hidden_dim: int, pool: str = "add"):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.pool = pool

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # bb + sc + |bb-sc|
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
        hg = role_pool_with_interaction(
            h, data.node_roles, data.batch, self.pool, interaction="abs_diff"
        )
        return self.pred_head(hg).view(-1)
