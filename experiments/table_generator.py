"""
Unified experiment runner for all tables.

Usage:
    python experiments/table_generator.py --table 1 --epochs 50 --seeds 3
    python experiments/table_generator.py --table 4 --pool add --interaction diff
"""

import argparse
import os
import sys
import json
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

from data.dataset import PolymerDataset
from data.transforms import RoleShuffleTransform, RoleRandomAssignTransform
from models.gnn_backbone import GNNEncoder
from models.baselines import VanillaGNN, AtomAttentionGNN
from models.brag import BRAG
from models.contrastive_role_gnn import ContrastiveRoleGNN
from models.brag_ablations import (
    BRAGOnlyBackbone, BRAGOnlySidechain, BRAGSharedPool,
    BRAGAddPlusDiff, BRAGConcatOnly, BRAGAbsDiff
)
from loss.contrastive_loss import RoleContrastiveLoss
from pooling.role_pool import role_pool_with_interaction
from utils.seed import set_seed
from utils.eval_utils import compute_metrics_with_bias


class ExperimentConfig:
    """Base experiment configuration."""
    def __init__(
        self,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        device: str = "cuda"
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.device = device


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: ExperimentConfig,
    use_contrastive: bool = False,
    lambda_c: float = 0.1,
    high_tg_threshold: Optional[float] = None
) -> Dict:
    """
    Train a model and return metrics.

    Note: Training logic is identical to train/train_brag.py and train/train_contrastive.py
    This ensures reproducibility across different experiment scripts.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Experiment configuration
        use_contrastive: Whether to use contrastive loss
        lambda_c: Contrastive loss weight
        high_tg_threshold: Fixed threshold for High-Tg bias analysis (TEST PHASE ONLY)
                          Used only for reporting bias metrics, NOT for early stopping

    Returns:
        dict: Test metrics (including bias analysis)

    CRITICAL PROTOCOL:
    - Early stopping uses GLOBAL MAE on validation set (prevents model selection bias)
    - High-Tg threshold is only used during TEST phase for bias reporting
    - This proves architecture genuinely reduces bias, not hyperparameter tuning
    """
    device = torch.device(config.device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Loss function
    # Note: Contrastive loss is only applied during training.
    # Validation and test use pure regression outputs for fair comparison.
    if use_contrastive:
        criterion = RoleContrastiveLoss(tau=0.2)
    else:
        criterion = nn.MSELoss(reduction='mean')


    # CRITICAL: Early stopping must use GLOBAL MAE, NOT High-Tg MAE
    # Using High-Tg MAE for model selection creates evaluation leakage:
    # - Model is selected to optimize the paper's primary metric
    # - Table 2 results become self-fulfilling prophecy
    # - Cannot prove bias reduction is intrinsic to architecture
    #
    # Correct protocol: Select model based on global MAE, then report High-Tg bias
    # This proves architecture genuinely reduces bias, not just hyperparameter tuning
    best_val_score = float("inf")
    print(f"  Early stopping on GLOBAL MAE (prevents model selection bias)")

    for epoch in tqdm(range(config.epochs), desc=f"Training {model.__class__.__name__}", leave=False):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            if use_contrastive:
                # Get node embeddings first
                pred, h = model(batch, return_node_emb=True)
                # Get role representations using node embeddings
                h_bb, h_sc = model.get_role_repr(h, batch)
                # batch.y has shape [batch_size]
                reg_loss = nn.MSELoss(reduction='mean')(pred, batch.y)

                # valid contrastive pairs (必须用 detach，否则梯度会通过 mask 反向传播)
                valid_mask = (h_sc.detach().norm(dim=1) > 0)

                if valid_mask.sum().item() > 1:
                    cont_loss = criterion(h_bb[valid_mask], h_sc[valid_mask])
                else:
                    cont_loss = torch.zeros(1, device=device)

                loss = reg_loss + lambda_c * cont_loss
            else:
                pred = model(batch)
                # batch.y has shape [batch_size]
                # Strict shape check - any mismatch indicates a structural bug
                assert pred.shape[0] == batch.y.shape[0], \
                    f"Shape mismatch: pred={pred.shape}, y={batch.y.shape}"
                loss = criterion(pred, batch.y)

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_preds.append(pred)
                val_targets.append(batch.y)

            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)

            # Compute validation score based on early stopping metric
            # CRITICAL: Always use global MAE for model selection (no evaluation leakage)
            val_score = torch.mean(torch.abs(val_preds - val_targets)).item()

            if val_score < best_val_score:
                best_val_score = val_score
                # Deep copy to avoid reference sharing
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            all_preds.append(pred)
            all_targets.append(batch.y)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

    # Compute metrics with bias analysis
    # If high_tg_threshold is None, use Q75 of targets; otherwise use fixed threshold
    metrics = compute_metrics_with_bias(all_preds, all_targets, high_tg_threshold=high_tg_threshold)

    return metrics


def get_fixed_splits(dataset, train_size, val_size, test_size, seed):
    """
    Create fixed train/val/test splits with reproducible random state.

    This ensures all models in the same experiment use the same data split.

    DEPRECATED: This function creates DIFFERENT splits for each seed.
    Use create_global_data_split() for multi-seed experiments.
    """
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )
    return train_dataset, val_dataset, test_dataset


def create_global_data_split(dataset, train_ratio=0.8, val_ratio=0.1, split_seed=2024):
    """
    Create ONE GLOBAL train/val/test split for ALL seeds.

    This is the CORRECT protocol for multi-seed experiments:
    - Seed 1, 2, 3, ... all use the SAME train/val/test indices
    - Only model initialization changes (set_seed at training time)
    - Reported std reflects model stability, NOT data split variance

    Args:
        dataset: Full dataset
        train_ratio: Ratio for training split
        val_ratio: Ratio for validation split
        split_seed: Fixed random seed for data splitting (same across all experiments)

    Returns:
        dict: {"train": Subset, "val": Subset, "test": Subset, "indices": dict}
    """
    from torch.utils.data import Subset

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    # Split ONCE with fixed seed
    generator = torch.Generator().manual_seed(split_seed)
    train_split, val_split, test_split = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )

    # Save indices for reproducibility
    indices = {
        "train": train_split.indices,
        "val": val_split.indices,
        "test": test_split.indices
    }

    # Create Subsets (allows reuse without re-splitting)
    train_dataset = Subset(dataset, indices["train"])
    val_dataset = Subset(dataset, indices["val"])
    test_dataset = Subset(dataset, indices["test"])

    print(f"\nGlobal data split (seed={split_seed}):")
    print(f"  Train: {len(indices['train'])} ({len(indices['train'])/total*100:.1f}%)")
    print(f"  Val: {len(indices['val'])} ({len(indices['val'])/total*100:.1f}%)")
    print(f"  Test: {len(indices['test'])} ({len(indices['test'])/total*100:.1f}%)")

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "indices": indices
    }


def run_table2_experiments(config: ExperimentConfig, seeds: List[int]) -> Dict:
    """
    Table 2: High-Tg Region Bias Analysis (The "Soul Table")

    Purpose: Prove that the problem is not just average error, but structural bias.
    This table provides the core justification for BRAG.

    Table Structure:
    - MAE (All test set)
    - MAE (High-Tg region, >= Q75 or threshold like 450K)
    - Delta Bias (High-Tg MAE - Low-Tg MAE)

    Key message: BRAG significantly reduces high-Tg bias despite comparable average error.

    Note:
    - CRITICAL: All seeds use the SAME train/val/test split
    - Reported std reflects model stability (optimization variance), NOT split variance
    - High-Tg threshold is computed from FULL DATASET (not per-split)
    - Bias Δ and Normalized Bias account for heteroscedastic noise
    """
    print("=" * 60)
    print("Table 2: High-Tg Region Bias Analysis")
    print("=" * 60)

    results = {}

    # Load dataset once
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=os.path.join(project_root, "data", "openpoly.csv"),
        target_column="Tg_K"
    )

    in_dim = dataset[0].x.shape[-1]

    # CRITICAL: Create GLOBAL split ONCE for all seeds
    data_split = create_global_data_split(dataset, train_ratio=0.8, val_ratio=0.1, split_seed=2024)
    train_dataset = data_split["train"]
    val_dataset = data_split["val"]
    test_dataset = data_split["test"]

    # CRITICAL: Compute global high-Tg threshold from FULL DATASET (dataset-intrinsic property)
    # This prevents distribution-aware model selection AND makes threshold a scientific constant
    print("\nComputing global high-Tg threshold from FULL DATASET...")
    all_targets = torch.cat([dataset[i].y for i in range(len(dataset))])
    global_threshold = float(np.percentile(all_targets.numpy(), 75))
    print(f"  Global High-Tg threshold (Q75 of FULL dataset): {global_threshold:.1f} K")
    print(f"  This is a dataset-intrinsic property (frozen constant across all seeds and models)")
    print(f"  (Same as Table 1 - ensures cross-table consistency)")

    # Use same models as Table 1 for fair comparison
    model_configs = [
        ("VanillaGNN", lambda: VanillaGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), False),
        ("AtomAttentionGNN", lambda: AtomAttentionGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), False),
        ("BRAG", lambda: BRAG(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim, pool="mean", interaction="abs_diff"
        ), False),
        ("Contrastive", lambda: ContrastiveRoleGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), True),
    ]

    for model_name, model_fn, use_contrastive in model_configs:
        print(f"\nRunning {model_name}...")

        seed_results = []

        for seed in seeds:
            set_seed(seed)  # Only affects model initialization

            # Use SAME split for all seeds, but INDEPENDENT RNG per seed
            train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=True, seed=seed)
            val_loader = create_dataloader(val_dataset, config.batch_size, shuffle=False, seed=None)
            test_loader = create_dataloader(test_dataset, config.batch_size, shuffle=False, seed=None)

            # Create model
            model = model_fn().to(config.device)

            # Train and evaluate using GLOBAL threshold for all seeds
            metrics = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                config, use_contrastive=use_contrastive,
                high_tg_threshold=global_threshold
            )

            seed_results.append(metrics)
            print(f"  Seed {seed}: MAE(All)={metrics['mae']:.2f}, MAE(High-Tg)={metrics['high_tg_mae']:.2f}, Bias Δ={metrics['bias_delta']:+.2f}, Norm. Bias={metrics['normalized_bias']:+.3f}")

        # Aggregate results
        results[model_name] = aggregate_results(seed_results)

    print("\n" + "=" * 60)
    print("Table 2 Results Summary")
    print("=" * 60)
    print_table2_summary(results)

    return results


def run_table3_experiments(config: ExperimentConfig, seeds: List[int]) -> Dict:
    """
    Table 3: Where to Inject Role Information?

    Purpose: Answer the reviewer's question: Why at architecture level?
    Why not attention or contrastive?

    Models:
    - Atom Attention (node-level role injection)
    - BRAG (graph-level pooling)
    - Contrastive Role GNN (representation-level via loss)

    Key message: Different injection levels lead to fundamentally different
    trade-offs between accuracy and bias control.

    Note:
    - All seeds use the SAME train/val/test split (split_seed=2024)
    """
    print("=" * 60)
    print("Table 3: Role Injection Level Comparison")
    print("=" * 60)

    results = {}

    # Load dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=os.path.join(project_root, "data", "openpoly.csv"),
        target_column="Tg_K"
    )

    in_dim = dataset[0].x.shape[-1]

    # CRITICAL: Create GLOBAL split ONCE for all seeds
    data_split = create_global_data_split(dataset, train_ratio=0.8, val_ratio=0.1, split_seed=2024)
    train_dataset = data_split["train"]
    val_dataset = data_split["val"]
    test_dataset = data_split["test"]

    # CRITICAL: Compute global high-Tg threshold from FULL DATASET (dataset-intrinsic property)
    # Ensures consistent evaluation across all tables
    print("\nComputing global high-Tg threshold from FULL DATASET...")
    all_targets = torch.cat([dataset[i].y for i in range(len(dataset))])
    global_threshold = float(np.percentile(all_targets.numpy(), 75))
    print(f"  Global High-Tg threshold (Q75 of FULL dataset): {global_threshold:.1f} K")
    print(f"  This ensures consistency with Table 1 and Table 2 (dataset-intrinsic property)")

    # Model configurations with injection level
    model_configs = [
        ("Atom Attention (node-level)", lambda: AtomAttentionGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), False),
        ("BRAG (graph-level)", lambda: BRAG(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim, pool="mean", interaction="abs_diff"
        ), False),
        ("Contrastive (representation-level)", lambda: ContrastiveRoleGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), True),
    ]

    for model_name, model_fn, use_contrastive in model_configs:
        print(f"\nRunning {model_name}...")

        seed_results = []
        for seed in seeds:
            set_seed(seed)  # Only affects model initialization

            # Use SAME split for all seeds, but INDEPENDENT RNG per seed
            train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=True, seed=seed)
            val_loader = create_dataloader(val_dataset, config.batch_size, shuffle=False, seed=None)
            test_loader = create_dataloader(test_dataset, config.batch_size, shuffle=False, seed=None)

            model = model_fn().to(config.device)
            metrics = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                config, use_contrastive=use_contrastive,
                high_tg_threshold=global_threshold  # Use global threshold for consistency
            )

            seed_results.append(metrics)

        results[model_name] = aggregate_results(seed_results)

    print("\n" + "=" * 60)
    print("Table 3 Results Summary")
    print("=" * 60)
    print_table3_summary(results)

    return results


def run_table6_experiments(config: ExperimentConfig, seeds: List[int]) -> Dict:
    """
    Table 6: Optimization Stability Analysis

    Purpose: Make reviewers confident you're not "just lucky".

    Design: Multiple random seeds, report mean ± std.

    Key message: Role-aware models exhibit improved optimization stability
    under random initialization.

    Note:
    - CRITICAL: All seeds use the SAME train/val/test split (split_seed=2024)
    - Reported std reflects model optimization stability, NOT split variance
    - Table measures optimization stability (not robustness to data perturbations)
    """
    print("=" * 60)
    print("Table 6: Optimization Stability Analysis")
    print("=" * 60)

    results = {}

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=os.path.join(project_root, "data", "openpoly.csv"),
        target_column="Tg_K"
    )

    in_dim = dataset[0].x.shape[-1]

    # CRITICAL: Create GLOBAL split ONCE for all seeds
    data_split = create_global_data_split(dataset, train_ratio=0.8, val_ratio=0.1, split_seed=2024)
    train_dataset = data_split["train"]
    val_dataset = data_split["val"]
    test_dataset = data_split["test"]

    # CRITICAL: Compute global high-Tg threshold from FULL DATASET (dataset-intrinsic property)
    # Ensures consistent evaluation across all tables
    print("\nComputing global high-Tg threshold from FULL DATASET...")
    all_targets = torch.cat([dataset[i].y for i in range(len(dataset))])
    global_threshold = float(np.percentile(all_targets.numpy(), 75))
    print(f"  Global High-Tg threshold (Q75 of FULL dataset): {global_threshold:.1f} K")
    print(f"  This ensures consistency with Table 1 and Table 2 (dataset-intrinsic property)")

    # Core models for stability analysis
    model_configs = [
        ("VanillaGNN", lambda: VanillaGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), False),
        ("BRAG", lambda: BRAG(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim, pool="mean", interaction="abs_diff"
        ), False),
        ("Contrastive", lambda: ContrastiveRoleGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), True),
    ]

    for model_name, model_fn, use_contrastive in model_configs:
        print(f"\nRunning {model_name} across {len(seeds)} seeds...")

        seed_results = []
        for seed in seeds:
            set_seed(seed)  # Only affects model initialization

            # Use SAME split for all seeds, but INDEPENDENT RNG per seed
            train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=True, seed=seed)
            val_loader = create_dataloader(val_dataset, config.batch_size, shuffle=False, seed=None)
            test_loader = create_dataloader(test_dataset, config.batch_size, shuffle=False, seed=None)

            model = model_fn().to(config.device)
            metrics = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                config, use_contrastive=use_contrastive,
                high_tg_threshold=global_threshold  # Use global threshold for consistency
            )

            seed_results.append(metrics)

        results[model_name] = aggregate_results(seed_results)

    print("\n" + "=" * 60)
    print("Table 6 Results Summary")
    print("=" * 60)
    print_table6_summary(results)

    return results


def generate_table_s1(dataset_path: str) -> Dict:
    """
    Table S1: Dataset Statistics

    Displays:
    - Total number of graphs
    - Tg distribution (mean, std, min, max, percentiles)
    - Backbone / side-chain ratio statistics
    """
    import pandas as pd

    print("=" * 60)
    print("Table S1: Dataset Statistics")
    print("=" * 60)

    # Load dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=os.path.join(project_root, "data", "openpoly.csv"),
        target_column="Tg_K"
    )

    # Collect statistics
    tg_values = []
    num_backbone = []
    num_sidechain = []
    num_nodes = []

    for data in dataset:
        tg_values.append(data.y.item())
        num_backbone.append((data.node_roles == 0).sum().item())
        num_sidechain.append((data.node_roles == 1).sum().item())
        num_nodes.append(data.num_nodes)

    tg_values = np.array(tg_values)
    num_backbone = np.array(num_backbone)
    num_sidechain = np.array(num_sidechain)
    num_nodes = np.array(num_nodes)

    # High-Tg threshold (Q75)
    high_tg_threshold = np.percentile(tg_values, 75)
    high_tg_mask = tg_values >= high_tg_threshold
    high_tg_ratio = float(high_tg_mask.sum() / len(tg_values))

    # Count homopolymers (no side-chain)
    is_homopolymer = num_sidechain == 0
    homopolymer_ratio = float(is_homopolymer.sum() / len(num_nodes))

    stats = {
        "total_graphs": len(dataset),
        "tg_mean": float(np.mean(tg_values)),
        "tg_std": float(np.std(tg_values)),
        "tg_min": float(np.min(tg_values)),
        "tg_max": float(np.max(tg_values)),
        "tg_q25": float(np.percentile(tg_values, 25)),
        "tg_q50": float(np.percentile(tg_values, 50)),
        "tg_q75": float(np.percentile(tg_values, 75)),
        "avg_num_nodes": float(np.mean(num_nodes)),
        "avg_backbone_nodes": float(np.mean(num_backbone)),
        "avg_sidechain_nodes": float(np.mean(num_sidechain)),
        "avg_backbone_ratio": float(np.mean(num_backbone / (num_nodes + 1e-8))),
        "avg_sidechain_ratio": float(np.mean(num_sidechain / (num_nodes + 1e-8))),
        "high_tg_threshold": float(high_tg_threshold),
        "high_tg_ratio": high_tg_ratio,
        "homopolymer_ratio": homopolymer_ratio,
    }

    print_table_s1_summary(stats)
    return stats


def generate_table_s2(config: ExperimentConfig) -> Dict:
    """
    Table S2: Hyperparameters

    Lists all hyperparameters used in BRAG experiments.
    Note: These reflect the actual defaults used in the codebase.
    """
    print("=" * 60)
    print("Table S2: Hyperparameters")
    print("=" * 60)

    hyperparams = {
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "gnn_type": config.gnn_type,
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "contrastive_lambda_c": 0.1,
        "contrastive_temperature": 0.2,  # CRITICAL: Fixed to match actual default
        "pool_type": "mean",               # CRITICAL: Fixed to match BRAG default
        "interaction_type": "abs_diff",    # CRITICAL: Fixed to match BRAG default
    }

    print_table_s2_summary(hyperparams)
    return hyperparams


def run_table1_experiments(config: ExperimentConfig, seeds: List[int]) -> Dict:
    """
    Table 1: Overall Performance Comparison

    Models:
    - Vanilla GNN
    - Atom Attention GNN
    - BRAG
    - Contrastive Role GNN

    Note:
    - All models share the same encoder (GNNEncoder) and hidden dimension
    - All seeds use the SAME train/val/test split (split_seed=2024)
    - Reported std reflects model stability, NOT split variance
    - High-Tg metrics use GLOBAL Q75 threshold (no evaluation leakage)
    """
    print("=" * 60)
    print("Table 1: Overall Performance Comparison")
    print("=" * 60)
    print("\nNote: High-Tg metrics use GLOBAL Q75 threshold (no evaluation leakage)")

    results = {}

    # Load dataset once
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=os.path.join(project_root, "data", "openpoly.csv"),
        target_column="Tg_K"
    )

    in_dim = dataset[0].x.shape[-1]

    # CRITICAL: Create GLOBAL split ONCE for all seeds
    data_split = create_global_data_split(dataset, train_ratio=0.8, val_ratio=0.1, split_seed=2024)
    train_dataset = data_split["train"]
    val_dataset = data_split["val"]
    test_dataset = data_split["test"]

    # CRITICAL: Compute global high-Tg threshold from FULL DATASET (not just training set)
    # This makes threshold a dataset-intrinsic property, not evaluation-dependent
    # Reviewer standard: "High-Tg region definition is a dataset property, not experimental protocol"
    print("\nComputing global high-Tg threshold from FULL DATASET...")
    all_targets = torch.cat([dataset[i].y for i in range(len(dataset))])
    global_threshold = float(np.percentile(all_targets.numpy(), 75))
    print(f"  Global High-Tg threshold (Q75 of FULL dataset): {global_threshold:.1f} K")
    print(f"  This is a dataset-intrinsic property (frozen constant across all experiments)")

    # Model definitions
    # Note: All models share the same GNNEncoder and hidden_dim for fair comparison
    model_configs = [
        ("VanillaGNN", lambda: VanillaGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), False),
        ("AtomAttentionGNN", lambda: AtomAttentionGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), False),
        ("BRAG", lambda: BRAG(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim, pool="mean", interaction="abs_diff"
        ), False),
        ("Contrastive", lambda: ContrastiveRoleGNN(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), True),
    ]

    for model_name, model_fn, use_contrastive in model_configs:
        print(f"\nRunning {model_name}...")

        seed_results = []
        for seed in seeds:
            set_seed(seed)  # Only affects model initialization

            # Use SAME split for all seeds, but INDEPENDENT RNG per seed
            train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=True, seed=seed)
            val_loader = create_dataloader(val_dataset, config.batch_size, shuffle=False, seed=None)
            test_loader = create_dataloader(test_dataset, config.batch_size, shuffle=False, seed=None)

            # Create model
            model = model_fn().to(config.device)

            # Train and evaluate using GLOBAL threshold for all seeds
            metrics = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                config, use_contrastive=use_contrastive,
                high_tg_threshold=global_threshold
            )

            seed_results.append(metrics)
            print(f"  Seed {seed}: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.3f}, High-Tg MAE={metrics['high_tg_mae']:.2f}")

        # Aggregate results
        results[model_name] = aggregate_results(seed_results)

    print("\n" + "=" * 60)
    print("Table 1 Results Summary")
    print("=" * 60)
    print_table1_summary(results)

    return results


def run_table4_experiments(config: ExperimentConfig, seeds: List[int]) -> Dict:
    """
    Table 4: BRAG Architecture Ablation

    Models:
    - BRAG-full (backbone + side-chain)
    - BRAG (shuffled roles) - controls for parameter equivalence (weak control)
    - BRAG (random roles) - destroys topology-correlated signal (strong control)
    - w/o backbone (only side-chain)
    - w/o side-chain (only backbone)
    - shared pool (no role separation)

    Note:
    - Shuffled roles: preserves ratio, destroys correspondence (weak)
    - Random roles: random assignment, destroys topology signal (strong)
    - CRITICAL: Control experiments apply transforms BEFORE split to ensure
      the same graph has consistent fake roles across train/val/test
    - CRITICAL: Use global High-Tg threshold (Q75 of full dataset)
      for consistency with Tables 1-3,5,6
    """
    print("=" * 60)
    print("Table 4: BRAG Architecture Ablation")
    print("=" * 60)

    results = {}

    # Load full dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=os.path.join(project_root, "data", "openpoly.csv"),
        target_column="Tg_K"
    )

    in_dim = dataset[0].x.shape[-1]

    # CRITICAL: Compute global high-Tg threshold from FULL DATASET
    # This ensures consistency with Tables 1-3,5,6
    print("\nComputing global high-Tg threshold from FULL DATASET...")
    all_targets = torch.cat([dataset[i].y for i in range(len(dataset))])
    global_threshold = float(np.percentile(all_targets.numpy(), 75))
    print(f"  Global High-Tg threshold (Q75 of FULL dataset): {global_threshold:.1f} K")
    print(f"  This is a dataset-intrinsic property (frozen constant across all seeds and models)")
    print(f"  (Same as Tables 1-3,5,6 - ensures cross-table consistency)")

    # CRITICAL: Compute split indices ONCE from original dataset
    # This ensures all models see EXACTLY the same graph indices in train/val/test
    # Ablation must only vary architecture, NOT data sampling
    print("\nCreating GLOBAL split (reused for all model variants)...")
    data_split_original = create_global_data_split(dataset, train_ratio=0.8, val_ratio=0.1, split_seed=2024)
    global_indices = data_split_original["indices"]  # Save indices for reuse
    print(f"  Train indices: {len(global_indices['train'])} samples")
    print(f"  Val indices: {len(global_indices['val'])} samples")
    print(f"  Test indices: {len(global_indices['test'])} samples")
    print(f"  These indices are IDENTICAL across all model variants")

    # CRITICAL: Create FIXED transform datasets for control experiments
    # Transform is applied to FULL DATASET, then split using GLOBAL indices
    print("\nCreating fixed control datasets (pre-split transforms)...")

    class FixedTransformDataset:
        """Dataset with pre-applied, fixed transformation."""
        def __init__(self, dataset, transform):
            # Apply transform ONCE during initialization
            self.data_list = [transform(dataset[i].clone()) for i in range(len(dataset))]

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]

    # Apply transforms to FULL DATASET (before split)
    # Then apply GLOBAL indices to ensure same graph indices across all variants
    shuffled_transform = RoleShuffleTransform(seed=42)
    random_transform = RoleRandomAssignTransform(backbone_ratio=0.5, seed=42)

    dataset_shuffled = FixedTransformDataset(dataset, shuffled_transform)
    dataset_random = FixedTransformDataset(dataset, random_transform)

    # Helper to create subsets using GLOBAL indices
    from torch.utils.data import Subset
    def create_dataset_from_indices(dataset, indices):
        """Create Subset using pre-computed global indices."""
        return Subset(dataset, indices)

    # Model definitions with their corresponding datasets
    # CRITICAL: All ablation variants use IDENTICAL pool="mean" for fair comparison
    # Only architecture varies, pooling strategy is held constant
    model_configs = [
        ("BRAG-full", lambda: BRAG(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim, pool="mean", interaction="abs_diff"
        ), dataset),  # Original dataset
        ("BRAG (shuffled roles)", lambda: BRAG(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim, pool="mean", interaction="abs_diff"
        ), dataset_shuffled),  # Pre-transformed full dataset
        ("BRAG (random roles)", lambda: BRAG(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim, pool="mean", interaction="abs_diff"
        ), dataset_random),  # Pre-transformed full dataset
        ("w/o backbone", lambda: BRAGOnlySidechain(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim, pool="mean"
        ), dataset),  # Original dataset
        ("w/o side-chain", lambda: BRAGOnlyBackbone(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim, pool="mean"
        ), dataset),  # Original dataset
        ("shared pool", lambda: BRAGSharedPool(
            GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type),
            config.hidden_dim
        ), dataset),  # Original dataset (no pool parameter for BRAGSharedPool)
    ]

    for model_name, model_fn, dataset_variant in model_configs:
        print(f"\nRunning {model_name}...")

        # Use GLOBAL indices for all models (fair comparison)
        # This eliminates sampling variance - only architecture differs
        train_dataset_fixed = create_dataset_from_indices(dataset_variant, global_indices["train"])
        val_dataset_fixed = create_dataset_from_indices(dataset_variant, global_indices["val"])
        test_dataset_fixed = create_dataset_from_indices(dataset_variant, global_indices["test"])

        seed_results = []
        for seed in seeds:
            set_seed(seed)  # Only affects model initialization

            # Use SAME split for all seeds, but INDEPENDENT RNG per seed
            train_loader = create_dataloader(train_dataset_fixed, config.batch_size, shuffle=True, seed=seed)
            val_loader = create_dataloader(val_dataset_fixed, config.batch_size, shuffle=False, seed=None)
            test_loader = create_dataloader(test_dataset_fixed, config.batch_size, shuffle=False, seed=None)

            model = model_fn().to(config.device)

            # Train and evaluate using GLOBAL threshold for consistency
            metrics = train_and_evaluate(
                model, train_loader, val_loader, test_loader, config,
                high_tg_threshold=global_threshold  # CRITICAL: Use global threshold
            )
            seed_results.append(metrics)

        results[model_name] = aggregate_results(seed_results)

    print("\n" + "=" * 60)
    print("Table 4 Results Summary")
    print("=" * 60)
    print_table4_summary(results)

    return results


def run_table5_experiments(config: ExperimentConfig, seeds: List[int]) -> Dict:
    """
    Table 5: Pooling & Interaction Ablation

    Variables:
    - Pool: add, mean, max
    - Interaction: cat, diff, abs_diff, add+diff

    Note:
    - CRITICAL: All seeds use the SAME train/val/test split (split_seed=2024)
    """
    print("=" * 60)
    print("Table 5: Pooling & Interaction Ablation")
    print("=" * 60)

    results = {}

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=os.path.join(project_root, "data", "openpoly.csv"),
        target_column="Tg_K"
    )

    in_dim = dataset[0].x.shape[-1]

    # CRITICAL: Create GLOBAL split ONCE for all seeds
    data_split = create_global_data_split(dataset, train_ratio=0.8, val_ratio=0.1, split_seed=2024)
    train_dataset = data_split["train"]
    val_dataset = data_split["val"]
    test_dataset = data_split["test"]

    # CRITICAL: Compute global high-Tg threshold from FULL DATASET (dataset-intrinsic property)
    # Ensures consistent evaluation across all tables
    print("\nComputing global high-Tg threshold from FULL DATASET...")
    all_targets = torch.cat([dataset[i].y for i in range(len(dataset))])
    global_threshold = float(np.percentile(all_targets.numpy(), 75))
    print(f"  Global High-Tg threshold (Q75 of FULL dataset): {global_threshold:.1f} K")
    print(f"  This ensures consistency with Table 1 and Table 2 (dataset-intrinsic property)")

    # Pool types
    pools = ["add", "mean", "max"]

    # Interaction types - pool will be passed from outer loop
    def create_interaction_model(interaction_name, pool_type, encoder):
        if interaction_name == "cat":
            return BRAGConcatOnly(encoder, config.hidden_dim, pool=pool_type)
        elif interaction_name == "diff":
            return BRAG(encoder, config.hidden_dim, pool=pool_type, interaction="diff")
        elif interaction_name == "abs_diff":
            return BRAGAbsDiff(encoder, config.hidden_dim, pool=pool_type)
        elif interaction_name == "add+diff":
            # NOTE: BRAGAddPlusDiff uses pool="add" by design
            # For pool types other than "add", we still use add pooling internally
            # to preserve the add+diff interaction semantics
            if pool_type != "add":
                print(f"  [NOTE] {pool_type}_add+diff uses add pooling internally "
                      f"(BRAGAddPlusDiff design constraint)")
            return BRAGAddPlusDiff(encoder, config.hidden_dim)
        else:
            raise ValueError(f"Unknown interaction: {interaction_name}")

    # Run experiments
    interaction_types = ["cat", "diff", "abs_diff", "add+diff"]
    for pool in pools:
        for int_name in interaction_types:
            model_name = f"{pool}_{int_name}"
            print(f"\nRunning {model_name}...")

            seed_results = []
            for seed in seeds:
                set_seed(seed)  # Only affects model initialization

                # Use SAME split for all seeds, but INDEPENDENT RNG per seed
                train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=True, seed=seed)
                val_loader = create_dataloader(val_dataset, config.batch_size, shuffle=False, seed=None)
                test_loader = create_dataloader(test_dataset, config.batch_size, shuffle=False, seed=None)

                encoder = GNNEncoder(in_dim, config.hidden_dim, config.num_layers, config.gnn_type)
                model = create_interaction_model(int_name, pool, encoder).to(config.device)

                metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, config, high_tg_threshold=global_threshold)
                seed_results.append(metrics)

            results[model_name] = aggregate_results(seed_results)

    print("\n" + "=" * 60)
    print("Table 5 Results Summary")
    print("=" * 60)
    print_table5_summary(results)

    return results


def aggregate_results(seed_results: List[Dict]) -> Dict:
    """
    Aggregate results across multiple seeds.

    Uses unbiased estimator for std (ddof=1) to match paper reproducibility standards.
    When only 1 seed is used, std is set to 0 instead of NaN.
    """
    aggregated = {}

    for key in seed_results[0].keys():
        values = [r[key] for r in seed_results]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        # CRITICAL: Use ddof=1 for unbiased estimator of std
        # Handle single-seed case to avoid NaN
        if len(values) > 1:
            aggregated[f"{key}_std"] = float(np.std(values, ddof=1))
        else:
            aggregated[f"{key}_std"] = 0.0

    return aggregated


def create_dataloader(dataset, batch_size, shuffle=True, seed=None):
    """
    Create DataLoader with independent RNG for true multi-seed experiments.

    CRITICAL: Without generator=..., DataLoader shuffle is not fully reproducible
    across seeds, which breaks the "independent trials" assumption.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle
        seed: Random seed for DataLoader's internal generator

    Returns:
        DataLoader with independent RNG
    """
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator
    )


def print_table1_summary(results: Dict):
    """Print Table 1 in a formatted way.

    Note:
    - All metrics computed on test set only (not validation)
    - Contrastive loss is only used during training for representation learning
    - All models use the same data splits per seed for fair comparison
    """
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R^2':>10} {'High-Tg MAE':>15}")
    print("-" * 65)

    for model_name, metrics in results.items():
        mae = f"{metrics['mae_mean']:.2f}±{metrics['mae_std']:.2f}"
        rmse = f"{metrics['rmse_mean']:.2f}±{metrics['rmse_std']:.2f}"
        r2 = f"{metrics['r2_mean']:.3f}±{metrics['r2_std']:.3f}"
        high_mae = f"{metrics['high_tg_mae_mean']:.2f}±{metrics['high_tg_mae_std']:.2f}"

        print(f"{model_name:<20} {mae:>10} {rmse:>10} {r2:>10} {high_mae:>15}")


def print_table4_summary(results: Dict):
    """Print Table 4 in a formatted way."""
    print(f"{'Model':<20} {'MAE':>10} {'R^2':>10} {'High-Tg MAE':>15}")
    print("-" * 55)

    for model_name, metrics in results.items():
        mae = f"{metrics['mae_mean']:.2f}±{metrics['mae_std']:.2f}"
        r2 = f"{metrics['r2_mean']:.3f}±{metrics['r2_std']:.3f}"
        high_mae = f"{metrics['high_tg_mae_mean']:.2f}±{metrics['high_tg_mae_std']:.2f}"

        print(f"{model_name:<20} {mae:>10} {r2:>10} {high_mae:>15}")


def print_table5_summary(results: Dict):
    """Print Table 5 in a formatted way."""
    print(f"{'Pool':<10} {'Interaction':<15} {'R^2':>10} {'High-Tg MAE':>15}")
    print("-" * 50)

    for model_name, metrics in results.items():
        parts = model_name.split("_")
        pool = parts[0]
        interaction = "_".join(parts[1:])

        r2 = f"{metrics['r2_mean']:.3f}±{metrics['r2_std']:.3f}"
        high_mae = f"{metrics['high_tg_mae_mean']:.2f}±{metrics['high_tg_mae_std']:.2f}"

        print(f"{pool:<10} {interaction:<15} {r2:>10} {high_mae:>15}")


def print_table2_summary(results: Dict):
    """Print Table 2 in a formatted way.

    Note:
    - **Normalized Bias** is the PRIMARY metric (accounts for heteroscedastic noise)
    - Bias Δ = MAE(High-Tg) - MAE(Low-Tg) in Kelvin (absolute bias gap)
    - Normalized Bias = (MAE_High/σ_High) - (MAE_Low/σ_Low) accounts for heteroscedastic noise
    - High-Tg threshold is fixed across all seeds for consistency
    - High-Tg region typically represents ~25% of test samples
    """
    print(f"{'Model':<20} {'MAE (All)':>12} {'Norm. Bias★':>15} {'MAE (High-Tg)':>15} {'Bias Δ':>12}")
    print("-" * 74)
    print("★ Primary metric - Normalized Bias accounts for heteroscedastic noise")

    # Print threshold info if available
    if results and 'high_tg_threshold_mean' in next(iter(results.values())):
        threshold = next(iter(results.values()))['high_tg_threshold_mean']
        high_ratio = next(iter(results.values()))['high_tg_ratio_mean']
        print(f"\nHigh-Tg Threshold: {threshold:.1f} K (Q75 of full dataset)")
        print(f"High-Tg Sample Ratio: {high_ratio*100:.1f}% of test set\n")
        print("-" * 74)

    for model_name, metrics in results.items():
        mae_all = f"{metrics['mae_mean']:.2f}±{metrics['mae_std']:.2f}"
        norm_bias = f"{metrics['normalized_bias_mean']:+.3f}±{metrics['normalized_bias_std']:.3f}"
        mae_high = f"{metrics['high_tg_mae_mean']:.2f}±{metrics['high_tg_mae_std']:.2f}"
        bias = f"{metrics['bias_delta_mean']:+.2f}±{metrics['bias_delta_std']:.2f}"

        print(f"{model_name:<20} {mae_all:>12} {norm_bias:>15} {mae_high:>15} {bias:>12}")


def print_table3_summary(results: Dict):
    """Print Table 3 in a formatted way.

    Note:
    - This table isolates the "where to inject role info" question
    - Atom Attention: node-level role injection via attention weights
    - BRAG: graph-level role injection via separate pooling and interaction
    - Contrastive: representation-level role injection via contrastive loss
    - All metrics computed on test set only
    """
    print(f"{'Model (Injection Level)':<35} {'R^2':>10} {'High-Tg MAE':>15}")
    print("-" * 60)

    for model_name, metrics in results.items():
        r2 = f"{metrics['r2_mean']:.3f}±{metrics['r2_std']:.3f}"
        high_mae = f"{metrics['high_tg_mae_mean']:.2f}±{metrics['high_tg_mae_std']:.2f}"

        print(f"{model_name:<35} {r2:>10} {high_mae:>15}")


def print_table6_summary(results: Dict):
    """Print Table 6 in a formatted way."""
    print(f"{'Model':<20} {'MAE':>15} {'R^2':>15} {'High-Tg MAE':>18}")
    print("-" * 68)

    for model_name, metrics in results.items():
        mae = f"{metrics['mae_mean']:.2f}±{metrics['mae_std']:.2f}"
        r2 = f"{metrics['r2_mean']:.3f}±{metrics['r2_std']:.3f}"
        high_mae = f"{metrics['high_tg_mae_mean']:.2f}±{metrics['high_tg_mae_std']:.2f}"

        print(f"{model_name:<20} {mae:>15} {r2:>15} {high_mae:>18}")


def print_table_s1_summary(stats: Dict):
    """Print Table S1 in a formatted way."""
    print("\n" + "=" * 60)
    print("Table S1: Dataset Statistics")
    print("=" * 60)

    print("\nDataset Overview:")
    print(f"  Total graphs: {stats['total_graphs']}")
    print(f"  Avg nodes per graph: {stats['avg_num_nodes']:.1f}")

    print("\nTg Distribution (K):")
    print(f"  Mean ± Std: {stats['tg_mean']:.2f} ± {stats['tg_std']:.2f}")
    print(f"  Range: [{stats['tg_min']:.2f}, {stats['tg_max']:.2f}]")
    print(f"  Percentiles: Q25={stats['tg_q25']:.2f}, Q50={stats['tg_q50']:.2f}, Q75={stats['tg_q75']:.2f}")

    print("\nNode Role Statistics:")
    print(f"  Avg backbone nodes: {stats['avg_backbone_nodes']:.1f} ({stats['avg_backbone_ratio']*100:.1f}%)")
    print(f"  Avg side-chain nodes: {stats['avg_sidechain_nodes']:.1f} ({stats['avg_sidechain_ratio']*100:.1f}%)")
    print(f"  Homopolymer ratio (no side-chain): {stats['homopolymer_ratio']*100:.1f}%")

    print("\nHigh-Tg Region (Reviewer Critical):")
    print(f"  High-Tg threshold (Q75): {stats['high_tg_threshold']:.1f} K")
    print(f"  High-Tg sample ratio: {stats['high_tg_ratio']*100:.1f}%")
    print(f"  Low-Tg sample ratio: {(1-stats['high_tg_ratio'])*100:.1f}%")
    print("=" * 60)


def print_table_s2_summary(hyperparams: Dict):
    """Print Table S2 in a formatted way."""
    print("\nModel Architecture:")
    print(f"  GNN type: {hyperparams['gnn_type'].upper()}")
    print(f"  Hidden dimension: {hyperparams['hidden_dim']}")
    print(f"  Number of GNN layers: {hyperparams['num_layers']}")

    print("\nTraining Configuration:")
    print(f"  Batch size: {hyperparams['batch_size']}")
    print(f"  Learning rate: {hyperparams['learning_rate']}")
    print(f"  Weight decay: {hyperparams['weight_decay']}")
    print(f"  Epochs: {hyperparams['epochs']}")

    print("\nBRAG Configuration:")
    print(f"  Pooling type: {hyperparams['pool_type']}")
    print(f"  Interaction type: {hyperparams['interaction_type']}")

    print("\nContrastive Learning:")
    print(f"  Lambda (contrastive weight): {hyperparams['contrastive_lambda_c']}")
    print(f"  Temperature: {hyperparams['contrastive_temperature']}")


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {k: float(v) for k, v in value.items()}
        else:
            json_results[key] = value

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main(args):
    config = ExperimentConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        device=args.device
    )

    seeds = list(range(args.seeds))

    results = None

    if args.table == 1:
        results = run_table1_experiments(config, seeds)
        output_path = "results/table1_results.json"
    elif args.table == 2:
        results = run_table2_experiments(config, seeds)
        output_path = "results/table2_results.json"
    elif args.table == 3:
        results = run_table3_experiments(config, seeds)
        output_path = "results/table3_results.json"
    elif args.table == 4:
        results = run_table4_experiments(config, seeds)
        output_path = "results/table4_results.json"
    elif args.table == 5:
        results = run_table5_experiments(config, seeds)
        output_path = "results/table5_results.json"
    elif args.table == 6:
        results = run_table6_experiments(config, seeds)
        output_path = "results/table6_results.json"
    elif args.table == 101:  # Table S1
        results = generate_table_s1("data/openpoly.csv")
        output_path = "results/table_s1_results.json"
    elif args.table == 102:  # Table S2
        results = generate_table_s2(config)
        output_path = "results/table_s2_results.json"
    else:
        print(f"Table {args.table} not yet implemented.")
        return

    if results and args.save:
        save_results(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for paper tables")

    parser.add_argument("--table", type=int, default=1,
                       choices=[1, 2, 3, 4, 5, 6, 101, 102],
                       help="Which table to generate (1-6 for main, 101-102 for supplementary)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--seeds", type=int, default=10,
                       help="Number of random seeds to run (≥10 for stability analysis)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "sage"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")

    args = parser.parse_args()
    main(args)
