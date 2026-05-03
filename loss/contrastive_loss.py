"""
Contrastive loss(Contrastive Role GNN).
Uses batch-level contrastive learning with negative samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoleContrastiveLoss(nn.Module):
    """
    Stable symmetric InfoNCE loss between backbone and sidechain views.
    """

    def __init__(self, tau: float = 0.2):
        super().__init__()
        self.tau = tau
        self.ce = nn.CrossEntropyLoss()

    def forward(self, h_bb, h_sc):
        """
        h_bb : [B, D]
        h_sc : [B, D]
        """

        B = h_bb.size(0)
        device = h_bb.device

        # cosine similarity
        logits = torch.matmul(h_bb, h_sc.T) / self.tau  # [B, B]

        labels = torch.arange(B, device=device)

        # bb -> sc
        loss_bb = self.ce(logits, labels)

        # sc -> bb
        loss_sc = self.ce(logits.T, labels)

        return (loss_bb + loss_sc) * 0.5

