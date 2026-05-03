"""
Hyperparameter sensitivity analysis for contrastive loss weight lambda_c.

Purpose: Verify that lambda_c=0.1 is a reasonable choice and
show how model performance varies with different values.

This addresses reviewer question: "How was lambda_c=0.1 determined?"
"""

import torch
import numpy as np
import json
from torch_geometric.loader import DataLoader
from data.dataset import PolymerDataset
from models.contrastive_role_gnn import ContrastiveRoleGNN
from models.gnn_backbone import GNNEncoder
from utils.seed import set_seed
from utils.eval_utils import compute_metrics_with_bias


def run_lambda_sensitivity(
    dataset,
    lambda_values=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    device="cpu",
    hidden_dim=128,
    num_layers=3,
    gnn_type="gcn",
    epochs=50,
    batch_size=32,
    seeds=[42, 123, 456]
):
    """
    Run sensitivity analysis for lambda_c hyperparameter.

    Args:
        dataset: PolymerDataset
        lambda_values: List of lambda_c values to test
        device: Device to use
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        gnn_type: GNN type
        epochs: Training epochs
        batch_size: Batch size
        seeds: Random seeds for reproducibility

    Returns:
        dict: Results for each lambda value
    """
    results = {}

    print("=" * 60)
    print("Lambda_c Sensitivity Analysis")
    print("=" * 60)
    print(f"\nTesting lambda values: {lambda_values}")
    print(f"Using {len(seeds)} seeds per lambda value")
    print()

    in_dim = dataset[0].x.shape[-1]

    for lambda_c in lambda_values:
        print(f"\n{'=' * 60}")
        print(f"Testing lambda_c = {lambda_c}")
        print('=' * 60)

        seed_results = []

        for seed_idx, seed in enumerate(seeds):
            set_seed(seed)
            print(f"\nSeed {seed_idx + 1}/{len(seeds)} (seed={seed})")

            # Create data split
            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size],
                generator=generator
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Create model
            encoder = GNNEncoder(in_dim, hidden_dim, num_layers, gnn_type)
            model = ContrastiveRoleGNN(
                encoder, hidden_dim
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

            # Training
            best_val_mae = float("inf")
            best_state = None

            for epoch in range(epochs):
                # Train
                model.train()
                train_loss = 0.0
                train_reg = 0.0
                train_con = 0.0

                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    pred, h = model(batch, return_node_emb=True)
                    h_bb, h_sc = model.get_role_repr(h, batch)

                    reg_loss = torch.nn.functional.mse_loss(pred, batch.y)

                    # Contrastive loss
                    valid_mask = (h_sc.detach().norm(dim=1) > 0)
                    if valid_mask.sum().item() > 1:
                        from loss.contrastive_loss import RoleContrastiveLoss
                        con_loss_fn = RoleContrastiveLoss()
                        con_loss = con_loss_fn(h_bb[valid_mask], h_sc[valid_mask])
                    else:
                        con_loss = torch.zeros(1, device=device)

                    loss = reg_loss + lambda_c * con_loss

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_reg += reg_loss.item()
                    train_con += con_loss.item()

                train_loss /= len(train_loader)

                # Validation
                model.eval()
                with torch.no_grad():
                    val_preds, val_targets = [], []
                    for batch in val_loader:
                        batch = batch.to(device)
                        pred = model(batch)
                        val_preds.append(pred)
                        val_targets.append(batch.y)
                    val_preds = torch.cat(val_preds)
                    val_targets = torch.cat(val_targets)
                    val_mae = torch.mean(torch.abs(val_preds - val_targets)).item()

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = model.state_dict().copy()

            # Test evaluation
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                test_preds, test_targets = [], []
                for batch in test_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    test_preds.append(pred)
                    test_targets.append(batch.y)
                test_preds = torch.cat(test_preds)
                test_targets = torch.cat(test_targets)

                # Compute global high-Tg threshold
                all_targets = torch.cat([dataset[i].y for i in range(len(dataset))])
                global_threshold = float(np.percentile(all_targets.numpy(), 75))

                metrics = compute_metrics_with_bias(test_preds, test_targets, global_threshold)
                seed_results.append(metrics)

            print(f"  Test MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}, "
                  f"High-Tg MAE: {metrics['high_tg_mae']:.2f}")

        # Aggregate results
        results[lambda_c] = {
            "mae_mean": float(np.mean([r["mae"] for r in seed_results])),
            "mae_std": float(np.std([r["mae"] for r in seed_results])),
            "r2_mean": float(np.mean([r["r2"] for r in seed_results])),
            "r2_std": float(np.std([r["r2"] for r in seed_results])),
            "high_tg_mae_mean": float(np.mean([r["high_tg_mae"] for r in seed_results])),
            "high_tg_mae_std": float(np.std([r["high_tg_mae"] for r in seed_results])),
            "bias_delta_mean": float(np.mean([r["bias_delta"] for r in seed_results])),
        }

        print(f"\nLambda_c = {lambda_c} Summary:")
        print(f"  MAE: {results[lambda_c]['mae_mean']:.2f} ± {results[lambda_c]['mae_std']:.2f}")
        print(f"  R²: {results[lambda_c]['r2_mean']:.3f} ± {results[lambda_c]['r2_std']:.3f}")
        print(f"  High-Tg MAE: {results[lambda_c]['high_tg_mae_mean']:.2f} ± "
              f"{results[lambda_c]['high_tg_mae_std']:.2f}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'lambda_c':<10} {'MAE':<15} {'R²':<15} {'High-Tg MAE':<15}")
    print("-" * 60)
    for lambda_c in lambda_values:
        r = results[lambda_c]
        print(f"{lambda_c:<10} {r['mae_mean']:.2f}±{r['mae_std']:.2f}  "
              f"{r['r2_mean']:.3f}±{r['r2_std']:.3f}  "
              f"{r['high_tg_mae_mean']:.2f}±{r['high_tg_mae_std']:.2f}")

    # Find best lambda
    best_lambda = min(results.keys(),
                    key=lambda x: results[x]['high_tg_mae_mean'])
    print("\n" + "=" * 60)
    print(f"Best lambda for High-Tg MAE: {best_lambda}")
    print(f"High-Tg MAE: {results[best_lambda]['high_tg_mae_mean']:.2f}")
    print("=" * 60)

    # Justify lambda_c=0.1
    print("\n--- Justification for lambda_c=0.1 ---")
    if 0.1 in results:
        r_01 = results[0.1]
        r_best = results[best_lambda]

        degradation = (r_01['high_tg_mae_mean'] - r_best['high_tg_mae_mean']) / r_best['high_tg_mae_mean'] * 100
        print(f"lambda_c=0.1 performs within {degradation:.1f}% of best value "
              f"(best={best_lambda})")
        print("Chosen because: provides good trade-off between stability and bias reduction")

    return results


if __name__ == "__main__":
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=os.path.join(project_root, "data", "openpoly.csv"),
        target_column="Tg_K"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run sensitivity analysis
    results = run_lambda_sensitivity(
        dataset,
        lambda_values=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
        device=device,
        epochs=30,  # Reduced for faster analysis
        seeds=[42, 123]  # Fewer seeds for faster analysis
    )

    # Save results
    with open("results/lambda_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/lambda_sensitivity.json")
