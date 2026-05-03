"""
Training script for BRAG.
"""

import argparse
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.gnn_backbone import GNNEncoder
from models.brag import BRAG
from data.dataset import PolymerDataset
from utils.seed import set_seed
from utils.metrics import compute_metrics, format_metrics


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(batch)
        # batch.y has shape [batch_size]
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        pred = model(batch)
        all_preds.append(pred)
        all_targets.append(batch.y)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets)
    return metrics


def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create dataset
    dataset = PolymerDataset(
        root=os.path.join(project_root, "data"),
        csv_path=args.csv_path,
        target_column=args.target_column
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Get node feature dimension
    sample_data = dataset[0]
    in_dim = sample_data.x.shape[-1]
    
    # Create model
    encoder = GNNEncoder(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type
    )
    model = BRAG(
        encoder=encoder,
        hidden_dim=args.hidden_dim,
        pool=args.pool,
        interaction=args.interaction
    ).to(device)
    
    print(f"Model: BRAG with {args.interaction} interaction")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_mae = float("inf")
    best_model_state = None
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val: {format_metrics(val_metrics, prefix='val_')}")
        print(f"  Test: {format_metrics(test_metrics, prefix='test_')}")
        
        # Save best model
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_model_state = model.state_dict().copy()
            print(f"  *** New best model (val_mae: {best_val_mae:.4f}) ***")
    
    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        test_metrics = evaluate(model, test_loader, device)
        print(f"\nBest Test Results: {format_metrics(test_metrics)}")
    
    # Save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--root", type=str, default="data/processed")
    parser.add_argument("--csv_path", type=str, default="data/openpoly.csv")
    parser.add_argument("--target_column", type=str, default="Tg_K")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "sage"])
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "add", "max"])
    parser.add_argument("--interaction", type=str, default="abs_diff",
                       choices=["diff", "abs_diff", "cat"])
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    
    # Save
    parser.add_argument("--save_path", type=str, default="checkpoints/brag_best.pt")
    
    args = parser.parse_args()
    main(args)
