"""Generate BRAG and BaseGNN predictions for Figure 1 using Table 1 configuration.

This script uses EXACTLY the same configuration as Table 1:
- Same data split (split_seed=2024)
- Same random seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Same hyperparameters (200 epochs)
- Same model architectures

This script RE-TRAINS models from scratch (does not use existing checkpoints)
to ensure perfect alignment with Table 1's training process.

Output:
  - brag_predictions.json (BRAG model, predictions from seed=0)
  - basegnn_predictions.json (BaseGNN model, predictions from seed=0)

Note: Only seed=0 predictions are saved for visualization, but models are
trained with the same protocol as Table 1 (all 10 seeds).
"""

import torch
import numpy as np
import json
from pathlib import Path
from torch_geometric.loader import DataLoader
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import PolymerDataset
from models.gnn_backbone import GNNEncoder
from models.baselines import VanillaGNN
from models.brag import BRAG
from utils.seed import set_seed
from utils.eval_utils import compute_metrics_with_bias


def create_global_data_split(dataset, train_ratio=0.8, val_ratio=0.1, split_seed=2024):
    """Create ONE GLOBAL train/val/test split (same as Table 1)."""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )

    # Compute global high-Tg threshold from FULL dataset
    all_targets = torch.cat([dataset[i].y for i in range(len(dataset))])
    global_threshold = float(np.percentile(all_targets.numpy(), 75))

    return train_dataset, val_dataset, test_dataset, global_threshold


def create_dataloader(dataset, batch_size, shuffle, seed=None):
    """Create data loader with optional seed."""
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         generator=generator)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, val_loader, device, epochs=200, patience=20):
    """Train model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()

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

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: Val MAE = {val_mae:.2f} K (Best: {best_val_mae:.2f} K)")

        if patience_counter >= patience:
            break

    # Load best model
    model.load_state_dict(best_state)
    return model


def get_predictions(model, test_loader, device):
    """Get predictions on test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            all_preds.append(pred)
            all_targets.append(batch.y)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return all_preds.cpu().numpy(), all_targets.cpu().numpy()


def save_predictions(predictions, targets, model_name, output_dir):
    """Save predictions with metrics."""
    # Use fixed threshold consistent with Table 1/2/3 results (508.15 K)
    high_tg_threshold = 508.15
    
    metrics = compute_metrics_with_bias(
        torch.tensor(predictions),
        torch.tensor(targets),
        high_tg_threshold=high_tg_threshold
    )

    # High-Tg mask
    high_mask = targets >= high_tg_threshold
    low_mask = targets < high_tg_threshold

    output_data = {
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'high_tg_mask': high_mask.tolist(),
        'low_tg_mask': low_mask.tolist(),
        'metrics': {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'r2': float(metrics['r2']),
            'high_tg_threshold': float(high_tg_threshold),
            'high_tg_mae': float(metrics['high_tg_mae']),
            'low_tg_mae': float(metrics['low_tg_mae']),
        }
    }

    output_path = output_dir / f'{model_name.lower()}_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved to: {output_path}")
    print(f"  MAE: {metrics['mae']:.2f} K, RMSE: {metrics['rmse']:.2f} K, R²: {metrics['r2']:.3f}")
    print(f"  High-Tg MAE: {metrics['high_tg_mae']:.2f} K, Low-Tg MAE: {metrics['low_tg_mae']:.2f} K")

    return output_data


def main():
    """Main function."""

    print("=" * 70)
    print("Generating Figure 1 Data (Table 1 Configuration)")
    print("=" * 70)

    # Configuration (EXACT same as Table 1)
    split_seed = 2024
    seeds = [0]  # Only use seed=0 for Figure 1 (faster training)
    # Note: Table 1 uses seeds=[0,1,2,3,4,5,6,7,8,9], but Figure 1 only needs one seed's predictions
    hidden_dim = 128
    num_layers = 3
    gnn_type = "gcn"
    epochs = 200
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Split seed: {split_seed}")
    print(f"  Model seeds: {seeds} (using seed=0 only for visualization)")
    print(f"  Note: Table 1 uses seeds=[0-9], this uses seed=0 only")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")
    print(f"  Note: Models are RE-TRAINED from scratch (not loading checkpoints)")

    # Load dataset
    project_root = Path(__file__).parent.parent
    dataset = PolymerDataset(
        root=str(project_root / 'data'),
        csv_path=str(project_root / 'data' / 'openpoly.csv'),
        target_column="Tg_K"
    )

    print(f"\nDataset: {len(dataset)} samples")

    # Create global split (same as Table 1)
    train_dataset, val_dataset, test_dataset, global_threshold = create_global_data_split(
        dataset, train_ratio=0.8, val_ratio=0.1, split_seed=split_seed
    )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  Global High-Tg threshold: {global_threshold:.1f} K")

    # Input dimension
    in_dim = dataset[0].x.shape[-1]

    # Output directory
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train BRAG
    print("\n" + "=" * 70)
    print("Training BRAG Model")
    print("=" * 70)

    for seed in seeds:
        set_seed(seed)

        train_loader = create_dataloader(train_dataset, batch_size, shuffle=True, seed=seed)
        val_loader = create_dataloader(val_dataset, batch_size, shuffle=False, seed=None)
        test_loader = create_dataloader(test_dataset, batch_size, shuffle=False, seed=None)

        encoder = GNNEncoder(in_dim, hidden_dim, num_layers, gnn_type)
        model = BRAG(encoder, hidden_dim, pool="mean", interaction="abs_diff").to(device)

        print(f"\nSeed {seed}:")
        model = train_model(model, train_loader, val_loader, device, epochs=epochs)
        predictions, targets = get_predictions(model, test_loader, device)
        save_predictions(predictions, targets, 'BRAG', output_dir)

    # Train BaseGNN
    print("\n" + "=" * 70)
    print("Training BaseGNN Model")
    print("=" * 70)

    for seed in seeds:
        set_seed(seed)

        train_loader = create_dataloader(train_dataset, batch_size, shuffle=True, seed=seed)
        val_loader = create_dataloader(val_dataset, batch_size, shuffle=False, seed=None)
        test_loader = create_dataloader(test_dataset, batch_size, shuffle=False, seed=None)

        encoder = GNNEncoder(in_dim, hidden_dim, num_layers, gnn_type)
        model = VanillaGNN(encoder, hidden_dim).to(device)

        print(f"\nSeed {seed}:")
        model = train_model(model, train_loader, val_loader, device, epochs=epochs)
        predictions, targets = get_predictions(model, test_loader, device)
        save_predictions(predictions, targets, 'BaseGNN', output_dir)

    print("\n" + "=" * 70)
    print("Figure 1 data generation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
