"""
Role-aware pooling functions.
These are core inductive bias mechanisms for BRAG.
"""

import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


def _safe_divide(num, denom):
    denom = denom.clamp(min=1)
    return num / denom


def role_pool(h, node_roles, batch, pool="mean"):
    """
    Statistically comparable role pooling.

    Key property:
    sum(bb)/N  + sum(sc)/N  = global_mean_pool(h)

    So BRAG and Vanilla become mathematically comparable.

    Args:
        h: Node embeddings [num_nodes, hidden_dim]
        node_roles: Node roles [num_nodes] (0=backbone, 1=side-chain)
        batch: Batch indices [num_nodes]
        pool: Pooling type ("mean", "add", "max")

    Returns:
        h_bb: Backbone graph representation [num_graphs, hidden_dim]
        h_sc: Side-chain graph representation [num_graphs, hidden_dim]

    Note: For graphs without side chains (e.g., homopolymers), side-chain
    representation is set to backbone representation to maintain dimensional consistency.
    This is documented in the paper as:
    "For polymers without side chains, side-chain representation is handled
    to maintain consistency while avoiding label leakage."
    """

    device = h.device
    num_graphs = int(batch.max()) + 1
    hidden_dim = h.size(-1)

    # masks
    bb_mask = (node_roles == 0)
    sc_mask = (node_roles == 1)

    # add pooling (always base)
    bb_add = global_add_pool(h * bb_mask.unsqueeze(-1), batch, size=num_graphs)
    sc_add = global_add_pool(h * sc_mask.unsqueeze(-1), batch, size=num_graphs)

    if pool == "add":
        return bb_add, sc_add

    # graph size N (shared normalization!)
    ones = torch.ones(h.size(0), device=device)
    N = global_add_pool(ones.unsqueeze(-1), batch, size=num_graphs)

    if pool == "mean":
        bb = _safe_divide(bb_add, N)
        sc = _safe_divide(sc_add, N)
        return bb, sc

    elif pool == "max":
        # max cannot share denominator; keep true max but mask empty safely
        neg_inf = torch.full_like(h, -1e9)
        bb_max = global_max_pool(torch.where(bb_mask.unsqueeze(-1), h, neg_inf), batch, size=num_graphs)
        sc_max = global_max_pool(torch.where(sc_mask.unsqueeze(-1), h, neg_inf), batch, size=num_graphs)

        # if no sidechain → use backbone (avoid existence label leak)
        empty_sc = (sc_mask.sum() == 0)
        if empty_sc:
            sc_max = bb_max.clone()

        return bb_max, sc_max
    else:
        raise ValueError(pool)


def role_pool_with_interaction(
    h,
    node_roles,
    batch,
    pool="mean",
    interaction="diff"
):
    """
    Role-aware pooling with interaction between backbone and side-chain.

    This is the key inductive bias for BRAG.

    Args:
        h: Node embeddings [num_nodes, hidden_dim]
        node_roles: Node roles [num_nodes]
        batch: Batch indices [num_nodes]
        pool: Pooling type ("mean", "add", "max")
        interaction: Interaction type ("diff", "abs_diff", "cat", "none")

    Returns:
        hg: Graph representation [batch_size, out_dim]
            - For "diff/abs_diff": [3 * hidden_dim] (bb, sc, interaction)
            - For "cat": [2 * hidden_dim] (bb, sc)
            - For "none": [2 * hidden_dim] (bb, sc)
    """
    bb, sc = role_pool(h, node_roles, batch, pool)

    # Prevent label leakage: if sidechain is zero vector, use backbone
    # This ensures interaction term becomes zero for homopolymers
    missing_sc = (sc.abs().sum(dim=1, keepdim=True) == 0)
    sc = torch.where(missing_sc, bb, sc)

    if interaction == "diff":
        inter = bb - sc
        return torch.cat([bb, sc, inter], dim=1)

    elif interaction == "abs_diff":
        inter = (bb - sc).abs()
        return torch.cat([bb, sc, inter], dim=1)

    elif interaction == "cat":
        return torch.cat([bb, sc], dim=1)

    elif interaction == "none":
        return torch.cat([bb, sc], dim=1)

    else:
        raise ValueError(interaction)
