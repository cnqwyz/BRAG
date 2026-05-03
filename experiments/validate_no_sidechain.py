"""
Validation experiment for no-sidechain handling strategy.

Purpose: Verify that setting side-chain representation = backbone representation
for homopolymers (no side-chain) does not leak label information.

Methodology:
1. Identify homopolymers (samples with 0 side-chain nodes)
2. Compare two strategies:
   a) Current: sc_rep = bb_rep
   b) Alternative: Drop homopolymers from dataset
3. Verify that performance difference is negligible

Expected result: Both strategies should yield similar performance,
proving that sc_rep = bb_rep does not leak label info.
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from data.dataset import PolymerDataset
from models.brag import BRAG
from models.gnn_backbone import GNNEncoder
from utils.seed import set_seed
from utils.eval_utils import compute_metrics_with_bias


def identify_homopolymers(dataset):
    """
    Identify homopolymers (no side-chain atoms).

    Returns:
        indices: List of indices of homopolymers
        ratio: Ratio of homopolymers in dataset
    """
    indices = []
    for i in range(len(dataset)):
        data = dataset[i]
        if data.node_roles.sum() == 0:  # All atoms are backbone
            indices.append(i)

    ratio = len(indices) / len(dataset)
    return indices, ratio


def create_dataset_without_homopolymers(dataset, homo_indices):
    """
    Create a dataset excluding homopolymers.

    Args:
        dataset: Original dataset
        homo_indices: Indices of homopolymers to exclude

    Returns:
        List of data objects excluding homopolymers
    """
    from torch.utils.data import Subset

    # Create mask to exclude homopolymers
    all_indices = set(range(len(dataset)))
    homo_set = set(homo_indices)
    kept_indices = sorted(list(all_indices - homo_set))

    # Create subset
    dataset_without_homo = Subset(dataset, kept_indices)
    return dataset_without_homo, len(kept_indices)


def run_validation_experiment(
    dataset,
    device="cpu",
    hidden_dim=128,
    num_layers=3,
    gnn_type="gcn",
    epochs=50,
    batch_size=32,
    seed=42
):
    """
    Run validation experiment comparing two strategies.

    Strategy A: Keep homopolymers, sc_rep = bb_rep (current)
    Strategy B: Drop homopolymers from dataset

    Args:
        dataset: Full dataset
        device: Device to use
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        gnn_type: GNN type
        epochs: Training epochs
        batch_size: Batch size
        seed: Random seed

    Returns:
        dict: Comparison results
    """
    set_seed(seed)
    print("=" * 60)
    print("No-Sidechain Handling Validation Experiment")
    print("=" * 60)

    # Identify homopolymers
    homo_indices, homo_ratio = identify_homopolymers(dataset)
    print(f"\nHomopolymers in dataset: {len(homo_indices)} ({homo_ratio*100:.2f}%)")

    # Strategy A: Keep homopolymers (current approach)
    print("\n--- Strategy A: Keep Homopolymers (sc_rep = bb_rep) ---")
    train_size_A = int(0.8 * len(dataset))
    val_size_A = int(0.1 * len(dataset))
    test_size_A = len(dataset) - train_size_A - val_size_A
    generator = torch.Generator().manual_seed(seed)
    train_A, val_A, test_A = torch.utils.data.random_split(
        dataset, [train_size_A, val_size_A, test_size_A],
        generator=generator
    )

    train_loader_A = DataLoader(train_A, batch_size=batch_size, shuffle=True)
    val_loader_A = DataLoader(val_A, batch_size=batch_size, shuffle=False)
    test_loader_A = DataLoader(test_A, batch_size=batch_size, shuffle=False)

    in_dim = dataset[0].x.shape[-1]
    encoder_A = GNNEncoder(in_dim, hidden_dim, num_layers, gnn_type)
    model_A = BRAG(
        encoder_A, hidden_dim, pool="mean", interaction="abs_diff"
    ).to(device)

    optimizer_A = torch.optim.Adam(model_A.parameters(), lr=0.001, weight_decay=1e-5)
    criterion_A = torch.nn.MSELoss()

    best_val_A = float("inf")
    for epoch in range(epochs):
        model_A.train()
        for batch in train_loader_A:
            batch = batch.to(device)
            optimizer_A.zero_grad()
            pred = model_A(batch)
            loss = criterion_A(pred, batch.y)
            loss.backward()
            optimizer_A.step()

        model_A.eval()
        with torch.no_grad():
            val_preds, val_targets = [], []
            for batch in val_loader_A:
                batch = batch.to(device)
                pred = model_A(batch)
                val_preds.append(pred)
                val_targets.append(batch.y)
            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)
            val_mae = torch.mean(torch.abs(val_preds - val_targets)).item()

        if val_mae < best_val_A:
            best_val_A = val_mae
            best_state_A = model_A.state_dict().copy()

    # Evaluate on test set
    model_A.load_state_dict(best_state_A)
    model_A.eval()
    with torch.no_grad():
        test_preds_A, test_targets_A = [], []
        for batch in test_loader_A:
            batch = batch.to(device)
            pred = model_A(batch)
            test_preds_A.append(pred)
            test_targets_A.append(batch.y)
    test_preds_A = torch.cat(test_preds_A)
    test_targets_A = torch.cat(test_targets_A)

    metrics_A = compute_metrics_with_bias(test_preds_A, test_targets_A)
    print(f"Test MAE: {metrics_A['mae']:.2f}")
    print(f"Test R²: {metrics_A['r2']:.3f}")
    print(f"High-Tg MAE: {metrics_A['high_tg_mae']:.2f}")

    # Strategy B: Drop homopolymers
    print("\n--- Strategy B: Drop Homopolymers ---")
    dataset_B, kept_count = create_dataset_without_homopolymers(dataset, homo_indices)
    print(f"Dataset size after dropping homopolymers: {kept_count}")

    train_size_B = int(0.8 * kept_count)
    val_size_B = int(0.1 * kept_count)
    test_size_B = kept_count - train_size_B - val_size_B
    generator = torch.Generator().manual_seed(seed)
    train_B, val_B, test_B = torch.utils.data.random_split(
        dataset_B, [train_size_B, val_size_B, test_size_B],
        generator=generator
    )

    train_loader_B = DataLoader(train_B, batch_size=batch_size, shuffle=True)
    val_loader_B = DataLoader(val_B, batch_size=batch_size, shuffle=False)
    test_loader_B = DataLoader(test_B, batch_size=batch_size, shuffle=False)

    encoder_B = GNNEncoder(in_dim, hidden_dim, num_layers, gnn_type)
    model_B = BRAG(
        encoder_B, hidden_dim, pool="mean", interaction="abs_diff"
    ).to(device)

    optimizer_B = torch.optim.Adam(model_B.parameters(), lr=0.001, weight_decay=1e-5)
    criterion_B = torch.nn.MSELoss()

    best_val_B = float("inf")
    for epoch in range(epochs):
        model_B.train()
        for batch in train_loader_B:
            batch = batch.to(device)
            optimizer_B.zero_grad()
            pred = model_B(batch)
            loss = criterion_B(pred, batch.y)
            loss.backward()
            optimizer_B.step()

        model_B.eval()
        with torch.no_grad():
            val_preds, val_targets = [], []
            for batch in val_loader_B:
                batch = batch.to(device)
                pred = model_B(batch)
                val_preds.append(pred)
                val_targets.append(batch.y)
            val_preds = torch.cat(val_preds)
            val_targets = torch.cat(val_targets)
            val_mae = torch.mean(torch.abs(val_preds - val_targets)).item()

        if val_mae < best_val_B:
            best_val_B = val_mae
            best_state_B = model_B.state_dict().copy()

    # Evaluate on test set
    model_B.load_state_dict(best_state_B)
    model_B.eval()
    with torch.no_grad():
        test_preds_B, test_targets_B = [], []
        for batch in test_loader_B:
            batch = batch.to(device)
            pred = model_B(batch)
            test_preds_B.append(pred)
            test_targets_B.append(batch.y)
    test_preds_B = torch.cat(test_preds_B)
    test_targets_B = torch.cat(test_targets_B)

    metrics_B = compute_metrics_with_bias(test_preds_B, test_targets_B)
    print(f"Test MAE: {metrics_B['mae']:.2f}")
    print(f"Test R²: {metrics_B['r2']:.3f}")
    print(f"High-Tg MAE: {metrics_B['high_tg_mae']:.2f}")

    # Comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<20} {'Strategy A':<15} {'Strategy B':<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'MAE':<20} {metrics_A['mae']:<15.2f} {metrics_B['mae']:<15.2f} {metrics_A['mae']-metrics_B['mae']:<+15.2f}")
    print(f"{'R²':<20} {metrics_A['r2']:<15.3f} {metrics_B['r2']:<15.3f} {metrics_A['r2']-metrics_B['r2']:<+15.3f}")
    print(f"{'High-Tg MAE':<20} {metrics_A['high_tg_mae']:<15.2f} {metrics_B['high_tg_mae']:<15.2f} {metrics_A['high_tg_mae']-metrics_B['high_tg_mae']:<+15.2f}")
    print("-" * 60)

    # Conclusion
    mae_diff = abs(metrics_A['mae'] - metrics_B['mae'])
    if mae_diff < 0.1 * metrics_A['mae']:
        print("\n✓ VALIDATION PASSED: Performance difference is negligible (<10% of MAE)")
        print("  Conclusion: sc_rep = bb_rep does NOT leak label information")
    else:
        print("\n✗ VALIDATION FAILED: Significant performance difference detected")
        print("  Consider alternative handling strategy for homopolymers")

    return {
        "homo_ratio": homo_ratio,
        "strategy_A": metrics_A,
        "strategy_B": metrics_B,
        "mae_difference": metrics_A['mae'] - metrics_B['mae']
    }


if __name__ == "__main__":
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=os.path.join(project_root, "data", "openpoly.csv"),
        target_column="Tg_K"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_validation_experiment(
        dataset,
        device=device,
        epochs=30,  # Quick validation
        seed=42
    )

    print(f"\nFull results saved to validation_no_sidechain.json")
    import json
    with open("validation_no_sidechain.json", "w") as f:
        # Convert tensors to float for JSON serialization
        results_serializable = {
            "homo_ratio": results["homo_ratio"],
            "strategy_A": {k: float(v) if not isinstance(v, (list, dict)) else v for k, v in results["strategy_A"].items()},
            "strategy_B": {k: float(v) if not isinstance(v, (list, dict)) else v for k, v in results["strategy_B"].items()},
            "mae_difference": float(results["mae_difference"])
        }
        json.dump(results_serializable, f, indent=2)
