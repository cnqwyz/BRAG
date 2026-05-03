from torch_geometric.nn import GCNConv, SAGEConv, GINEConv, GraphNorm
import torch.nn.functional as F
import torch.nn as nn


class GNNEncoder(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        residual_alpha: float = 0.3,
        use_edge_attr: bool = True
    ):
        """
        GNN encoder with optional edge attributes support.

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            gnn_type: GNN type ("gcn", "sage", or "gine")
            residual_alpha: Residual connection coefficient (anti-oversmoothing)
            use_edge_attr: Whether to use edge attributes (bond types)

        Note on edge attributes:
            - GINE: Uses edge features directly via GINEConv architecture
            - GCN/SAGE: Do not natively support edge features in message passing
              Edge features are projected but not used to maintain consistency
              across all GNN types. For GCN/SAGE, bond type information is
              indirectly captured through node features (atomic environment)
        """
        super().__init__()

        self.residual_alpha = residual_alpha
        self.use_edge_attr = use_edge_attr
        self.gnn_type = gnn_type

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Edge feature projection (only if using edge_attr)
        if use_edge_attr:
            # from_smiles() returns edge_attr with 3 dims (bond type encoding)
            self.edge_proj = nn.Linear(3, hidden_dim)
        else:
            self.edge_proj = None

        for _ in range(num_layers):

            if gnn_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif gnn_type == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim)
            elif gnn_type == "gine":
                # GINEConv uses MLP for edge features
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                conv = GINEConv(mlp, eps=0, train_eps=False)
            else:
                raise ValueError(gnn_type)

            self.layers.append(conv)
            self.norms.append(GraphNorm(hidden_dim))

    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        Forward pass with optional edge attributes.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            edge_attr: Edge features [num_edges, edge_dim] (optional)

        Returns:
            h: Node embeddings [num_nodes, hidden_dim]
        """
        # initial embedding (important: feature space alignment)
        h = self.input_proj(x)

        # Process edge features if provided
        if self.use_edge_attr and edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr.float())
        else:
            edge_feat = None

        for conv, norm in zip(self.layers, self.norms):

            # Message passing with edge features
            if self.gnn_type == "gine" and edge_feat is not None:
                # GINEConv handles edge features internally
                m = conv(h, edge_index, edge_feat)
            else:
                # Standard GNN without edge features (GCN, SAGE)
                # Note: For GCN/SAGE, edge features are not used directly
                # as these architectures don't support edge attribute input
                m = conv(h, edge_index)

            m = norm(m, batch)
            m = F.relu(m)

            # anti-oversmoothing residual update
            h = h + self.residual_alpha * m

        return h
