"""
BRAG (Backbone-Aware Graph) Model.
Introduces role-aware inductive bias at graph-level representation construction.
"""

import torch
import torch.nn as nn

from .gnn_backbone import GNNEncoder
from pooling.role_pool import role_pool_with_interaction


class BRAG(nn.Module):
    """
    BRAG: Backbone-Aware Graph model.
    """
    
    def __init__(
        self,
        encoder: GNNEncoder,
        hidden_dim: int,
        pool: str = "mean",
        interaction: str = "abs_diff"
    ):
        """
        Args:
            encoder: Shared GNN encoder
            hidden_dim: Hidden dimension
            pool: Pooling type ("mean", "add", "max")
            interaction: Interaction type ("diff", "abs_diff", "cat")
        """
        super().__init__()
        
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.pool = pool
        self.interaction = interaction
        
        # Calculate output dimension based on interaction type
        if interaction in ["diff", "abs_diff"]:
            out_dim = hidden_dim * 3  # bb, sc, interaction
        elif interaction == "cat":
            out_dim = hidden_dim * 2  # bb, sc
        else:
            raise ValueError(f"Unsupported interaction: {interaction}")
        
        # Prediction head
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data object with attributes:
                - x: Node features
                - edge_index: Edge indices
                - node_roles: Node roles (0=backbone, 1=side-chain)
                - batch: Batch indices

        Returns:
            pred: Predicted property [batch_size]
        """
        # Node encoding (shared, vanilla GNN)
        h = self.encoder(
            data.x,
            data.edge_index,
            data.batch,
            getattr(data, 'edge_attr', None)  # Use edge_attr if available
        )

        # Role-aware pooling with interaction (inductive bias)
        hg = role_pool_with_interaction(
            h,
            data.node_roles,
            data.batch,
            self.pool,
            self.interaction
        )

        # Prediction (use view(-1) to ensure correct shape for all batch sizes)
        pred = self.mlp(hg).view(-1)

        return pred
