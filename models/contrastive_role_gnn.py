"""
Contrastive Role GNN.
Uses contrastive learning to separate backbone and side-chain representations.
"""

import torch
import torch.nn as nn

from .gnn_backbone import GNNEncoder
from torch_geometric.nn import global_add_pool


class ContrastiveRoleGNN(nn.Module):
    """
    Contrastive Role GNN model.
    """
    
    def __init__(self, encoder: GNNEncoder, hidden_dim: int):
        """
        Args:
            encoder: Shared GNN encoder
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        
        # Regression head (uses standard global pooling)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data, return_node_emb=False):
        h = self.encoder(
            data.x,
            data.edge_index,
            data.batch,
            getattr(data, 'edge_attr', None)
        )

        # size-invariant graph representation
        hg = global_add_pool(h, data.batch)
        counts = torch.bincount(data.batch)
        hg = hg / counts.unsqueeze(1).clamp(min=1)

        pred = self.reg_head(hg).view(-1)

        if return_node_emb:
            return pred, h
        return pred


    
    def get_role_repr(self, h, data):
        num_graphs = data.batch.max().item() + 1 if data.batch.numel() > 0 else 0

        bb_list, sc_list = [], []

        for g in range(num_graphs):
            mask = data.batch == g
            h_g = h[mask]
            r_g = data.node_roles[mask]

            if h_g.size(0) == 0:
                bb_list.append(torch.zeros(self.hidden_dim, device=h.device))
                sc_list.append(torch.zeros(self.hidden_dim, device=h.device))
                continue

            bb_mask = r_g == 0
            sc_mask = r_g == 1

            # --- mean pooling instead of sum pooling ---
            if bb_mask.any():
                bb = h_g[bb_mask].mean(0)
            else:
                bb = torch.zeros(self.hidden_dim, device=h.device)

            if sc_mask.any():
                sc = h_g[sc_mask].mean(0)
            else:
                sc = torch.zeros(self.hidden_dim, device=h.device)

            bb_list.append(bb)
            sc_list.append(sc)

        bb = torch.stack(bb_list)
        sc = torch.stack(sc_list)

        # normalize BEFORE contrastive
        bb = nn.functional.normalize(bb, dim=1)
        sc = nn.functional.normalize(sc, dim=1)

        return bb, sc
