"""
Data transforms for ablation studies (e.g., role shuffle).
"""

import torch
import random
from typing import Optional


class RoleShuffleTransform:
    """
    Shuffle node roles randomly.
    
    This is used for sanity check: if performance drops after shuffling,
    then the model is actually using role information.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
    
    def __call__(self, data):
        """
        Shuffle node roles in place.

        Args:
            data: PyG Data object with node_roles attribute

        Returns:
            data: Data with shuffled node_roles
        """
        if hasattr(data, "node_roles"):
            node_roles = data.node_roles.clone()

            # Check if data is batched or single graph
            if hasattr(data, "batch") and data.batch is not None:
                # Batched data
                num_graphs = data.batch.max().item() + 1 if data.batch.numel() > 0 else 0

                if num_graphs > 1:
                    for g in range(num_graphs):
                        mask = data.batch == g
                        indices = torch.where(mask)[0]
                        shuffled_roles = node_roles[indices][torch.randperm(len(indices))]
                        data.node_roles[indices] = shuffled_roles
                else:
                    # Single graph (but has batch attribute with size 1)
                    indices = torch.randperm(len(node_roles))
                    data.node_roles = node_roles[indices]
            else:
                # Single graph without batch attribute
                indices = torch.randperm(len(node_roles))
                data.node_roles = node_roles[indices]

        return data


class RoleRandomAssignTransform:
    """
    Randomly assign backbone/side-chain roles (ignoring actual structure).
    
    This is a stronger sanity check than shuffling.
    """
    
    def __init__(self, backbone_ratio: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            backbone_ratio: Ratio of backbone nodes
            seed: Random seed
        """
        self.backbone_ratio = backbone_ratio
        if seed is not None:
            torch.manual_seed(seed)
    
    def __call__(self, data):
        """
        Randomly assign roles.
        
        Args:
            data: PyG Data object
            
        Returns:
            data: Data with randomly assigned node_roles
        """
        if hasattr(data, "node_roles"):
            num_nodes = data.num_nodes
            num_bb = int(num_nodes * self.backbone_ratio)
            
            # Create random assignment
            roles = torch.zeros(num_nodes, dtype=torch.long)
            bb_indices = torch.randperm(num_nodes)[:num_bb]
            roles[bb_indices] = 1  # side-chain
            data.node_roles = roles
        
        return data
