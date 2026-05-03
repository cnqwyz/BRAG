"""Generate BaseGNN predictions for Figure 1 comparison.

This script trains BaseGNN model and saves predictions for visualization.
Output: basegnn_predictions.json with 'predicted' and 'actual' values
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
from utils.seed import set_seed


def load_or_train_basegnn(dataset_path='data/openpoly.csv', target_column='Tg_K',
                          checkpoint_path='checkpoints/basegnn_best.pt',
                          device='cuda', epochs=200, seed=42):
    """Load existing checkpoint or train new BaseGNN model."""

    set_seed(seed)

    # Load dataset
    project_root = Path(__file__).parent.parent
    dataset = PolymerDataset(
        root=str(project_root / 'data'),
        csv_path=str(project_root / dataset_path),
        target_column=target_column
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Split dataset (same as BRAG: 80/10/10, same seed)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )

    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model (same hyperparameters as Table 1)
    in_dim = dataset[0].x.shape[-1]
    hidden_dim = 128
    num_layers = 3
    gnn_type = "gcn"

    encoder = GNNEncoder(in_dim, hidden_dim, num_layers, gnn_type)
    model = VanillaGNN(encoder, hidden_dim).to(device)

    # Force retrain to ensure compatibility
    print(f"\nForce retraining BaseGNN model...")
    model = train_basegnn(model, train_loader, val_loader, device,
                          epochs=epochs, checkpoint_path=checkpoint_path)

    return model, test_dataset, test_loader, device


def train_basegnn(model, train_loader, val_loader, device,
                  epochs=200, checkpoint_path='checkpoints/basegnn_best.pt'):
    """Train BaseGNN model with early stopping."""

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_mae = float('inf')
    best_state = None
    patience = 20  # More patience for 200 epochs
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred, batch.y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

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

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0

            # Save checkpoint
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                  f"Val MAE: {val_mae:.2f} K (Best: {best_val_mae:.2f} K)")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_state)
    print(f"\nTraining completed. Best Val MAE: {best_val_mae:.2f} K")

    return model


def generate_predictions(model, test_loader, device):
    """Generate predictions on test set."""

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            predictions.extend(pred.cpu().numpy().tolist())
            targets.extend(batch.y.cpu().numpy().tolist())

    return predictions, targets


def main():
    """Main function to generate and save predictions."""

    print("=" * 70)
    print("Generating BaseGNN Predictions for Figure 1 Comparison")
    print("=" * 70)

    # Load or train model
    model, test_dataset, test_loader, device = load_or_train_basegnn(
        dataset_path='data/openpoly.csv',
        target_column='Tg_K',
        checkpoint_path='checkpoints/basegnn_best.pt',
        device='cuda',
        epochs=200,
        seed=42
    )

    # Generate predictions
    print("\nGenerating predictions on test set...")
    predictions, targets = generate_predictions(model, test_loader, device)

    # Compute metrics
    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

    # High-Tg threshold
    high_tg_threshold = np.percentile(targets, 75)
    high_mask = targets >= high_tg_threshold
    low_mask = targets < high_tg_threshold

    high_tg_mae = np.mean(np.abs(predictions[high_mask] - targets[high_mask]))
    low_tg_mae = np.mean(np.abs(predictions[low_mask] - targets[low_mask]))

    print(f"\nTest Set Metrics:")
    print(f"  Samples: {len(targets)}")
    print(f"  MAE: {mae:.2f} K")
    print(f"  RMSE: {rmse:.2f} K")
    print(f"  R²: {r2:.3f}")
    print(f"  High-Tg threshold: {high_tg_threshold:.1f} K")
    print(f"  High-Tg MAE: {high_tg_mae:.2f} K")
    print(f"  Low-Tg MAE: {low_tg_mae:.2f} K")

    # Save predictions
    output_data = {
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'high_tg_mask': high_mask.tolist(),
        'low_tg_mask': low_mask.tolist(),
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'high_tg_threshold': float(high_tg_threshold),
            'high_tg_mae': float(high_tg_mae),
            'low_tg_mae': float(low_tg_mae),
        }
    }

    output_path = Path('results/basegnn_predictions.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nPredictions saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
