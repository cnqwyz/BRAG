"""
Training script for Contrastive Role GNN.
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
from models.contrastive_role_gnn import ContrastiveRoleGNN
from loss.contrastive_loss import RoleContrastiveLoss
from data.dataset import PolymerDataset
from utils.seed import set_seed
from utils.metrics import compute_metrics, format_metrics


def train_epoch(model, dataloader, optimizer, device, lambda_c):
    model.train()
    total_loss = 0.0
    total_reg = 0.0
    total_con = 0.0

    contrastive_loss_fn = RoleContrastiveLoss().to(device)

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # forward
        pred, h = model(batch, return_node_emb=True)
        h_bb, h_sc = model.get_role_repr(h, batch)

        # regression loss
        reg_loss = nn.functional.mse_loss(pred, batch.y.view_as(pred))

        # valid contrastive pairs (必须用 detach，否则梯度会通过 mask 反向传播)
        valid_mask = (h_sc.detach().norm(dim=1) > 0)

        if valid_mask.sum().item() > 1:
            con_loss = contrastive_loss_fn(
                h_bb[valid_mask], h_sc[valid_mask]
            )
        else:
            con_loss = torch.zeros(1, device=device)

        # final loss
        loss = reg_loss + lambda_c * con_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reg += reg_loss.item()
        total_con += con_loss.item()

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "reg": total_reg / n,
        "con": total_con / n
    }




@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model (inference only, no contrastive loss).
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
    generator = torch.Generator().manual_seed(args.seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
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
    model = ContrastiveRoleGNN(
        encoder=encoder,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    print(f"Model: Contrastive Role GNN with lambda_c={args.lambda_c}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    best_val_mae = float("inf")
    best_model_state = None

    for epoch in range(args.epochs):
        # Train
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lambda_c
        )   
        from split_and_sanity_utils import debug_embedding_geometry
        debug_embedding_geometry(model, train_loader, device)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_losses['loss']:.4f} (reg: {train_losses['reg']:.4f}, con: {train_losses['con']:.4f})")
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
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    # Save
    parser.add_argument("--save_path", type=str, default="checkpoints/contrastive_best.pt")
    
    args = parser.parse_args()
    main(args)
